#!/usr/bin/env python3
"""Validate a SmolVLA checkpoint and optionally run an OpenVINO model.

This is the Intel-side entry point.  A raw ``model.safetensors`` file is not
enough to run a policy: the checkpoint directory must also contain the model
configuration, tokenizer, and LeRobot preprocessing metadata.

The complete SmolVLA-to-OpenVINO adapter is model-version dependent.  This
script deliberately validates the checkpoint first and only executes an
OpenVINO graph when ``--openvino-model`` is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


def find_checkpoint_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = [path]
    candidates.extend(sorted(path.rglob("config.json")))
    for candidate in candidates:
        root = candidate.parent if candidate.name == "config.json" else candidate
        if (root / "model.safetensors").exists():
            return root
    raise FileNotFoundError(
        f"Could not find a complete checkpoint below {path}. "
        "Expected a directory containing model.safetensors and config.json."
    )


def checkpoint_report(root: Path) -> dict:
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    tokenizer = root / "tokenizer"
    weights = root / "model.safetensors"
    return {
        "checkpoint": str(root),
        "complete": not missing,
        "missing": missing,
        "weights_bytes": weights.stat().st_size if weights.exists() else 0,
        "tokenizer_present": tokenizer.is_dir(),
        "files": sorted(p.name for p in root.iterdir()),
    }


def inspect_openvino(model_path: Path, device: str) -> dict:
    try:
        from openvino import Core
    except ImportError as exc:
        raise RuntimeError("Install OpenVINO first: pip install openvino") from exc

    core = Core()
    available = list(core.available_devices)
    compiled = core.compile_model(str(model_path), device)
    return {
        "openvino_model": str(model_path.resolve()),
        "requested_device": device,
        "available_devices": available,
        "inputs": [
            {"name": port.any_name, "shape": str(port.partial_shape),
             "type": str(port.element_type)}
            for port in compiled.inputs
        ],
        "outputs": [
            {"name": port.any_name, "shape": str(port.partial_shape),
             "type": str(port.element_type)}
            for port in compiled.outputs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="checkpoint folder or its parent directory")
    parser.add_argument("--openvino-model", type=Path,
                        help="optional .xml/.onnx OpenVINO graph to inspect")
    parser.add_argument("--device", default="CPU",
                        help="OpenVINO device, e.g. CPU, GPU, NPU, AUTO")
    parser.add_argument("--report", type=Path,
                        help="optional JSON report output")
    args = parser.parse_args()

    root = find_checkpoint_root(Path(args.checkpoint))
    report = checkpoint_report(root)
    if not report["complete"]:
        raise SystemExit(json.dumps(report, indent=2))

    if args.openvino_model:
        report["openvino"] = inspect_openvino(args.openvino_model, args.device)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
