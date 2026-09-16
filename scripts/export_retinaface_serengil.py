from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _add_repo_to_path(repo: Path) -> None:
    repo = repo.resolve()
    if not (repo / "retinaface").exists():
        raise FileNotFoundError(f"RetinaFace repo not found at {repo}")
    sys.path.insert(0, str(repo))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export serengil/retinaface as a fixed-shape TensorFlow SavedModel."
    )
    parser.add_argument("--repo", type=Path, default=Path("external/retinaface"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/retinaface_serengil/saved_model_640"),
    )
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    _add_repo_to_path(args.repo)

    import tensorflow as tf  # noqa: PLC0415
    from retinaface.model import retinaface_model  # noqa: PLC0415

    model = retinaface_model.build_model()

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[1, args.height, args.width, 3],
                dtype=tf.float32,
                name="data",
            )
        ]
    )
    def serve(data: tf.Tensor) -> dict[str, tf.Tensor]:
        outputs = model(data, training=False)
        return {f"output_{idx}": tensor for idx, tensor in enumerate(outputs)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(args.out), signatures={"serving_default": serve})
    print(f"Saved fixed-shape RetinaFace model to {args.out}")


if __name__ == "__main__":
    main()
