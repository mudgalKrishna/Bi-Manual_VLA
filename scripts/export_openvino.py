#!/usr/bin/env python3
"""Export a supported inference graph to OpenVINO IR.

SmolVLA is a multimodal policy and may contain custom policy code that cannot
be exported by a generic Hugging Face command.  This tool supports the safe,
explicit paths: convert an ONNX graph, convert a TorchScript graph, or invoke
Optimum Intel when the installed model adapter supports it.  It never claims
that a raw safetensors file was converted without a computation graph.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def validate_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not (path / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing model.safetensors in {path}")
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="complete pretrained_model directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="directory for OpenVINO XML/BIN output")
    parser.add_argument("--onnx", type=Path,
                        help="explicit ONNX graph to convert")
    parser.add_argument("--torchscript", type=Path,
                        help="explicit TorchScript graph to convert")
    parser.add_argument("--optimum", action="store_true",
                        help="try Optimum Intel export for this checkpoint")
    args = parser.parse_args()

    checkpoint = validate_checkpoint(Path(args.checkpoint))
    if sum(bool(x) for x in (args.onnx, args.torchscript, args.optimum)) != 1:
        parser.error("choose exactly one of --onnx, --torchscript, or --optimum")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.optimum:
        command = [
            "optimum-cli", "export", "openvino",
            "--model", str(checkpoint), str(args.output),
        ]
        print("Running:", " ".join(command))
        completed = subprocess.run(command)
        if completed.returncode:
            raise SystemExit(
                "Optimum Intel could not export this SmolVLA checkpoint. "
                "Create an explicit policy wrapper/ONNX graph and rerun with --onnx."
            )
        return

    try:
        import openvino as ov
    except ImportError as exc:
        raise SystemExit("Install OpenVINO first: pip install openvino") from exc

    source = args.onnx or args.torchscript
    try:
        model = ov.convert_model(str(source))
    except Exception as exc:
        raise SystemExit(
            f"OpenVINO could not convert {source}. Export a deterministic "
            "policy graph with the required image/state/language inputs first.\n"
            f"Original error: {exc}"
        ) from exc
    output_xml = args.output / "policy.xml"
    ov.save_model(model, str(output_xml), compress_to_fp16=False)
    print(f"OpenVINO IR written to {output_xml}")


if __name__ == "__main__":
    main()
