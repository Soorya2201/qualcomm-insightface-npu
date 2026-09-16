from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import uuid

import qai_hub as hub


def prepare_external_data_bundle(source: Path, external_data: Path, output: Path) -> None:
    """Create the two-file external-data layout required by AI Hub Workbench.

    The downloaded model refers to ``model.onnx_data``. Workbench requires the
    weights file to end in ``.data``. Replacing it with an equally sized name
    keeps the serialized protobuf lengths valid without requiring the ONNX
    Python package in the Workbench environment.
    """
    original_name = external_data.name.encode("utf-8")
    workbench_name = b"model_data.data"
    if len(original_name) != len(workbench_name):
        raise RuntimeError(
            "Cannot safely rewrite external ONNX data filename: "
            f"{external_data.name!r} and {workbench_name.decode()!r} differ in length"
        )

    graph = source.read_bytes()
    reference_count = graph.count(original_name)
    if reference_count == 0:
        raise RuntimeError(
            f"The ONNX graph does not refer to expected weights file {external_data.name!r}"
        )

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"Workbench upload directory is not empty: {output}")
    (output / "model.onnx").write_bytes(graph.replace(original_name, workbench_name))
    shutil.copyfile(external_data, output / workbench_name.decode("utf-8"))
    print(
        "Prepared Workbench ONNX bundle with "
        f"{reference_count} external-weight reference(s): {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile DINOv3 ViT-S/16 for Snapdragon X Elite ONNX Runtime"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models/dinov3-vits16-onnx/onnx/model.onnx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/dinov3-vits16-workbench/model.onnx"),
    )
    parser.add_argument("--device", default="Snapdragon X Elite CRD")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    external_data = source.with_name(source.name + "_data")
    if not external_data.is_file():
        raise FileNotFoundError(
            f"External ONNX weights must remain beside the graph: {external_data}"
        )

    client = hub.Client()
    upload_bundle = source.parent / f".workbench-upload-{uuid.uuid4().hex}.onnx"
    upload_bundle.mkdir()
    try:
        prepare_external_data_bundle(source, external_data, upload_bundle)
        job = client.submit_compile_job(
            model=upload_bundle,
            device=hub.Device(args.device),
            name="dinov3-vits16-224-onnx",
            input_specs={"pixel_values": ((1, 3, 224, 224), "float32")},
            options="--target_runtime onnx",
        )
        print(f"Submitted Workbench compile job: {job.url}")
        status = job.wait()
        if not status.success:
            raise RuntimeError(f"Workbench compilation failed: {status}")
    finally:
        shutil.rmtree(upload_bundle, ignore_errors=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    downloaded = job.download_target_model(str(args.output.resolve()))
    if downloaded is None:
        raise RuntimeError("Workbench returned no compiled model")
    print(f"Downloaded compiled model to: {downloaded}")


if __name__ == "__main__":
    main()
