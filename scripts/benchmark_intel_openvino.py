#!/usr/bin/env python3
"""Measure OpenVINO graph latency and throughput on an Intel device."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def dtype_for(element_type):
    name = str(element_type).lower()
    if "i64" in name:
        return np.int64
    if "i32" in name:
        return np.int32
    if "boolean" in name or name == "bool":
        return np.bool_
    return np.float32


def concrete_shape(port, batch: int) -> list[int]:
    dims = []
    for index, dim in enumerate(port.partial_shape):
        if dim.is_static:
            dims.append(int(dim))
        elif index == 0:
            dims.append(batch)
        else:
            raise ValueError(
                f"Dynamic non-batch dimension in {port.any_name}: {port.partial_shape}. "
                "Provide a fixed exported policy graph before benchmarking."
            )
    return dims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path,
                        help="OpenVINO .xml model")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/intel_openvino_benchmark.json"))
    args = parser.parse_args()

    try:
        from openvino import Core
    except ImportError as exc:
        raise SystemExit("Install OpenVINO first: pip install openvino") from exc

    core = Core()
    compiled = core.compile_model(str(args.model), args.device)
    request = compiled.create_infer_request()
    inputs = {}
    for port in compiled.inputs:
        shape = concrete_shape(port, args.batch)
        inputs[port.any_name] = np.zeros(shape, dtype=dtype_for(port.element_type))

    for _ in range(args.warmup):
        request.infer(inputs)
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        request.infer(inputs)
        samples.append(time.perf_counter() - start)

    values = np.asarray(samples, dtype=np.float64)
    report = {
        "model": str(args.model.resolve()),
        "device": args.device,
        "available_devices": list(core.available_devices),
        "iterations": args.iterations,
        "batch": args.batch,
        "mean_ms": float(values.mean() * 1000),
        "median_ms": float(np.median(values) * 1000),
        "p95_ms": float(np.percentile(values, 95) * 1000),
        "throughput_samples_per_second": float(args.batch / values.mean()),
        "inputs": {name: list(array.shape) for name, array in inputs.items()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
