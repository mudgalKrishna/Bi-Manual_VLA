#!/usr/bin/env python3
"""Record measured OpenVINO device availability and optional graph timings."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path)
    p.add_argument('--iterations', type=int, default=50)
    p.add_argument('--report', type=Path, default=Path('outputs/device_benchmark.json'))
    a = p.parse_args()
    out = {'iterations': a.iterations}
    try:
        import openvino as ov
        core = ov.Core()
        out['openvino_version'] = ov.__version__
        out['available_devices'] = list(core.available_devices)
    except Exception as e:
        core = None
        out['openvino_error'] = f'{type(e).__name__}: {e}'
    try:
        import torch
        out['pytorch_version'] = torch.__version__
        out['cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available(): out['cuda_device'] = torch.cuda.get_device_name(0)
    except Exception as e:
        out['pytorch_error'] = f'{type(e).__name__}: {e}'
    if a.model:
        if core is None or not a.model.exists():
            out['graph_status'] = 'not_benchmarked_model_missing_or_openvino_unavailable'
        else:
            out['graph'] = str(a.model.resolve()); results = {}
            import numpy as np
            for device in core.available_devices:
                try:
                    compiled = core.compile_model(str(a.model), device); req = compiled.create_infer_request()
                    inputs = {port.any_name: np.zeros([int(d) for d in port.shape], np.float32) for port in compiled.inputs}
                    for _ in range(5): req.infer(inputs)
                    ts = []
                    for _ in range(a.iterations):
                        t = time.perf_counter(); req.infer(inputs); ts.append((time.perf_counter()-t)*1000)
                    ts.sort(); mean = sum(ts)/len(ts)
                    results[device] = {'mean_ms': mean, 'p95_ms': ts[min(len(ts)-1, int(.95*len(ts)))], 'throughput_per_second': 1000/mean}
                except Exception as e:
                    results[device] = {'status': 'unavailable', 'error': f'{type(e).__name__}: {e}'}
            out['graph_devices'] = results
    a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))
if __name__ == '__main__': main()
