from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class QnnSession:
    """Small ONNX Runtime wrapper that can require complete QNN/NPU assignment."""

    def __init__(self, model_path: Path, config: RuntimeConfig) -> None:
        started = time.perf_counter()
        LOGGER.info("Loading NPU model: %s", model_path)
        try:
            import onnxruntime as ort
            import onnxruntime_qnn as ort_qnn
        except ImportError as exc:
            raise RuntimeError("onnxruntime-qnn is not installed") from exc

        available = ort.get_available_providers()
        if config.provider not in available:
            ort.register_execution_provider_library(
                ort_qnn.get_ep_name(), ort_qnn.get_library_path()
            )
            available = ort.get_available_providers()
        if config.provider not in available:
            raise RuntimeError(
                f"{config.provider} is unavailable; providers={available}. "
                "Use native Windows ARM64 Python 3.11 and install onnxruntime-qnn."
            )
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

        npu_devices = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == config.provider
            and device.device.type == ort.OrtHardwareDeviceType.NPU
        ]
        if not npu_devices:
            raise RuntimeError(
                f"No NPU device was exposed by {config.provider}; "
                f"devices={ort.get_ep_devices()}"
            )

        options = ort.SessionOptions()
        # AI Hub's quantized ONNX exports include small input/output QDQ wrapper
        # nodes that ORT may keep on CPU. QNN still owns the neural-network
        # partition, so disabling every CPU node would reject the official asset.
        if config.context_cache:
            options.add_session_config_entry("ep.context_enable", "1")
            options.add_session_config_entry("ep.context_embed_mode", "1")

        # Plugin EPs must be attached through their discovered OrtEpDevice.
        # Passing a registered plugin by its legacy provider name can silently
        # create a CPU-only session with current ONNX Runtime releases.
        options.add_provider_for_devices(npu_devices, {})
        self.session = ort.InferenceSession(str(model_path), sess_options=options)
        active = self.session.get_providers()
        if config.require_npu and not active_strictly_qnn(active, config.provider):
            raise RuntimeError(f"NPU was required but active providers are {active}")

        self._input_quantization, self._output_quantization = _load_quantization_metadata(
            model_path
        )
        self._run_options = ort.RunOptions()
        self._run_options.add_run_config_entry("qnn.perf_mode", config.performance_mode)
        LOGGER.info(
            "NPU session ready model=%s load_ms=%.1f providers=%s inputs=%s outputs=%s",
            model_path.name,
            (time.perf_counter() - started) * 1000,
            active,
            [(item.name, item.shape, item.type) for item in self.inputs],
            [(item.name, item.shape, item.type) for item in self.outputs],
        )

    @property
    def inputs(self):
        return self.session.get_inputs()

    @property
    def outputs(self):
        return self.session.get_outputs()

    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        quantized_feeds = {
            name: _quantize(value, self._input_quantization.get(name))
            for name, value in feeds.items()
        }
        raw = self.session.run(None, quantized_feeds, self._run_options)
        return [
            _dequantize(value, self._output_quantization.get(meta.name))
            for meta, value in zip(self.outputs, raw)
        ]


def _load_quantization_metadata(
    model_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load the quantization contract shipped beside AI Hub ONNX assets."""
    metadata_path = model_path.with_name("metadata.json")
    if not metadata_path.is_file():
        return {}, {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    model = metadata.get("model_files", {}).get(model_path.name, {})
    inputs = {
        name: spec
        for name, spec in model.get("inputs", {}).items()
        if "quantization_parameters" in spec
    }
    outputs = {
        name: spec
        for name, spec in model.get("outputs", {}).items()
        if "quantization_parameters" in spec
    }
    return inputs, outputs


def _quantize(value: np.ndarray, spec: dict[str, Any] | None) -> np.ndarray:
    if spec is None:
        return value
    params = spec["quantization_parameters"]
    dtype = np.dtype(spec["dtype"])
    limits = np.iinfo(dtype)
    quantized = np.rint(value / float(params["scale"]) + int(params["zero_point"]))
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def _dequantize(value: np.ndarray, spec: dict[str, Any] | None) -> np.ndarray:
    if spec is None:
        return value
    params = spec["quantization_parameters"]
    return (value.astype(np.float32) - int(params["zero_point"])) * float(
        params["scale"]
    )


def active_strictly_qnn(active: list[str], provider: str) -> bool:
    # ORT automatically adds CPU after the explicitly selected QNN provider.
    return bool(active) and active[0] == provider
