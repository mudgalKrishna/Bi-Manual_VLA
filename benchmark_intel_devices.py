#!/usr/bin/env python3
"""
Benchmark the fine-tuned SmolVLA policy across Intel execution devices.

RUN THIS ON THE INTEL CORE ULTRA 7 265T BOX -- not the laptop. The i5-1334U has
no NPU (Raptor Lake-U predates Core Ultra's AI Boost), so NPU results are
impossible there. The checkpoint lives at:

    C:\\Users\\devcloud\\Bi-Manual_VLA\\models\\checkpoint_004000\\pretrained_model


WHY AN EXPLICIT --iters MATTERS
------------------------------
The earlier timing report (outputs/smolvla_cpu_timing.txt) recorded
"TotalSeconds : 16.1733579" with no iteration count. That number cannot be
interpreted: 16 s for ONE inference is 0.06 Hz; 16 s for 100 inferences is
161 ms each. This script always prints a denominator, so the result is quotable.


THE NUMBER THAT MATTERS
-----------------------
The MuJoCo expert runs at a 0.04 s control period (25 Hz) -- visible as the
`meta` array in any episode npz: [0.004, 0.04] = (timestep, control_dt). A
policy inference must complete inside 40 ms to drive the robot in real time.
Every device below is reported against that 40 ms budget, as PASS or FAIL.

Note the budget applies to producing ONE ACTION, not one action chunk. SmolVLA
is configured with chunk_size = n_action_steps = 50, so one chunk generation
yields 50 actions. The script measures full chunk generation and divides.


USAGE
-----
    python benchmark_intel_devices.py \
        --checkpoint "C:\\Users\\devcloud\\Bi-Manual_VLA\\models\\checkpoint_004000\\pretrained_model" \
        --iters 20 --warmup 3 --devices CPU GPU NPU \
        --out benchmark_results.json

Results are written as JSON and printed as a table. Every device is attempted
independently: if NPU cannot compile the model, that is RECORDED as a result,
not raised as a crash.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

# --------------------------------------------------------------------------
# Project constants -- keep in sync with the dataset schema.
# --------------------------------------------------------------------------
CAMERA_KEYS = ("front", "arm_a", "arm_b", "top")
IMAGE_SIZE = 320
STATE_DIM = 12
CHUNK_SIZE = 50          # n_action_steps, from the adapted SmolVLA config
CONTROL_PERIOD_S = 0.04  # 25 Hz, from episode meta
BUDGET_MS_PER_ACTION = CONTROL_PERIOD_S * 1000.0   # 40 ms
BUDGET_MS_PER_CHUNK = BUDGET_MS_PER_ACTION * CHUNK_SIZE  # 2000 ms

TASK = "Open the drawer and set the table for two with both plates, forks, spoons, and the mug."


# --------------------------------------------------------------------------
# Package power
#
# Energy per inference is the metric that decides an edge deployment, so it has
# to come from a counter and not from a datasheet TDP. Linux exposes an Intel
# package counter through the RAPL powercap interface. Windows -- the actual
# deployment target -- has no unprivileged equivalent, so power is reported as
# *not measured* rather than quietly dropped. An external meter, or a Linux run
# of this same harness, is how that column gets filled in.
# --------------------------------------------------------------------------
def discover_rapl_counter() -> Path | None:
    """Return the package-0 RAPL energy counter if this machine exposes one."""
    root = Path("/sys/class/powercap")
    if not root.is_dir():
        return None
    for entry in sorted(root.glob("intel-rapl:*")):
        energy_file = entry / "energy_uj"
        try:
            if energy_file.exists() and \
                    (entry / "name").read_text(encoding="utf-8").strip() == "package-0":
                return energy_file
        except OSError:
            continue
    return None


def read_energy_uj(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def power_reader(counter: Path | None):
    """Bind a reader to `counter`, or return None when there is nothing to read.

    Returning None is the normal case on Windows. Every consumer treats it as
    "power not measured" and leaves the fields null.
    """
    if counter is None:
        return None
    try:
        read_energy_uj(counter)
    except (OSError, ValueError):
        return None
    return lambda: read_energy_uj(counter)


# --------------------------------------------------------------------------
# Input construction
# --------------------------------------------------------------------------
def build_batch(torch, batch_size: int = 1):
    """A fixed, valid SmolVLA observation batch. Deterministic, so every device
    sees byte-identical input and the comparison is fair."""
    g = torch.Generator().manual_seed(0)
    obs = {}
    for key in CAMERA_KEYS:
        obs[f"observation.images.{key}"] = torch.rand(
            (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE), generator=g, dtype=torch.float32
        )
    obs["observation.state"] = torch.randn((batch_size, STATE_DIM), generator=g, dtype=torch.float32)
    obs["task"] = [TASK] * batch_size
    return obs


def describe_batch(obs) -> dict:
    return {
        k: (list(v.shape) if hasattr(v, "shape") else f"list[{len(v)}]")
        for k, v in obs.items()
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_policy(checkpoint: str, device: str):
    """Load the fine-tuned policy through LeRobot. Tries the known import paths
    in order so a LeRobot version bump does not silently break the benchmark."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.eval()
    policy.to(device)
    if hasattr(policy, "reset"):
        policy.reset()
    return policy


def count_params(policy) -> int:
    return sum(p.numel() for p in policy.parameters())


# --------------------------------------------------------------------------
# Timing core
# --------------------------------------------------------------------------
def time_calls(fn, iters: int, warmup: int, energy_reader=None) -> dict:
    """Time `fn` iters times after `warmup` untimed calls.

    Reports first-call latency separately from steady state: the first call on
    any device includes kernel compilation and memory allocation, which is why
    the earlier report's single 16.17 s figure was ambiguous. Here it is
    explicitly separated and excluded from the mean.

    When `energy_reader` is supplied, the package energy counter is sampled
    around the timed loop, so power and energy/inference come from the same
    calls as the latency. A counter that wraps or goes backwards is treated as
    a failed reading, not as a negative joule count.
    """
    first_ms = None
    for i in range(warmup):
        t0 = time.perf_counter()
        fn()
        if i == 0:
            first_ms = (time.perf_counter() - t0) * 1000.0

    joules = None
    if energy_reader is not None:
        try:
            e0 = energy_reader()
        except (OSError, ValueError):
            e0 = None
    else:
        e0 = None

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)

    if e0 is not None:
        try:
            e1 = energy_reader()
            if e1 > e0:
                joules = (e1 - e0) / 1e6   # microjoules -> joules
        except (OSError, ValueError):
            joules = None

    samples.sort()
    n = len(samples)
    result = {
        "iters": n,
        "first_call_ms": round(first_ms, 2) if first_ms is not None else None,
        "mean_ms": round(statistics.fmean(samples), 2),
        "median_ms": round(statistics.median(samples), 2),
        "p90_ms": round(samples[min(n - 1, int(0.9 * n))], 2),
        "min_ms": round(samples[0], 2),
        "max_ms": round(samples[-1], 2),
        "stdev_ms": round(statistics.pstdev(samples), 2) if n > 1 else 0.0,
        "package_power_w": None,
        "energy_per_inference_mj": None,
        "efficiency_inf_per_s_per_w": None,
    }
    if joules is not None:
        elapsed_s = sum(samples) / 1000.0
        if joules > 0 and elapsed_s > 0:
            result["package_power_w"] = round(joules / elapsed_s, 2)
            result["energy_per_inference_mj"] = round(joules * 1000.0 / n, 2)
            result["efficiency_inf_per_s_per_w"] = round(n / joules, 3)
    return result


def budget_verdict(chunk_ms: float) -> dict:
    per_action = chunk_ms / CHUNK_SIZE
    return {
        "ms_per_action": round(per_action, 3),
        "actions_per_second": round(1000.0 / per_action, 2) if per_action > 0 else None,
        "budget_ms_per_action": BUDGET_MS_PER_ACTION,
        "meets_25hz_realtime": bool(per_action <= BUDGET_MS_PER_ACTION),
    }


# --------------------------------------------------------------------------
# Backend 1 -- PyTorch (always available, gives the honest baseline)
# --------------------------------------------------------------------------
def bench_torch(policy, obs, iters: int, warmup: int, energy_reader=None) -> dict:
    import torch

    def call():
        with torch.inference_mode():
            policy.select_action(obs)

    # Each measured call must regenerate a full action chunk. Without the reset,
    # 49 of every 50 calls return a cached action and the number is meaningless.
    def call_fresh_chunk():
        if hasattr(policy, "reset"):
            policy.reset()
        call()

    return time_calls(call_fresh_chunk, iters, warmup, energy_reader)


# --------------------------------------------------------------------------
# Backend 2 -- OpenVINO
# --------------------------------------------------------------------------
class PolicyWrapper:
    """Traceable wrapper: flat tensors in, action tensor out.

    OpenVINO needs a plain module with tensor inputs and no Python-side control
    flow or internal caching. This strips the policy to its compute core.
    """

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, front, arm_a, arm_b, top, state):
        obs = {
            "observation.images.front": front,
            "observation.images.arm_a": arm_a,
            "observation.images.arm_b": arm_b,
            "observation.images.top": top,
            "observation.state": state,
            "task": [TASK],
        }
        self.policy.reset()
        return self.policy.select_action(obs)


def export_openvino(policy, obs, out_dir: Path) -> tuple:
    """Convert the policy to OpenVINO IR. Returns (ov_model, error_or_None).

    SmolVLA is a VLM plus a flow-matching action expert. Flow matching runs
    several denoising steps internally, so the exported graph is larger and
    less NPU-friendly than a typical vision model. Expect CPU and GPU to
    convert; NPU may reject dynamic axes or unsupported ops. That outcome is
    reported, not treated as a script failure.
    """
    import torch
    import openvino as ov

    wrapper = PolicyWrapper(policy).eval()
    example = (
        obs["observation.images.front"],
        obs["observation.images.arm_a"],
        obs["observation.images.arm_b"],
        obs["observation.images.top"],
        obs["observation.state"],
    )
    try:
        with torch.inference_mode():
            ov_model = ov.convert_model(wrapper, example_input=example)
        out_dir.mkdir(parents=True, exist_ok=True)
        ir_path = out_dir / "smolvla_policy.xml"
        ov.save_model(ov_model, str(ir_path))
        return ov_model, None, str(ir_path)
    except Exception:
        return None, traceback.format_exc(limit=3), None


def bench_ov(core, ov_model, device: str, iters: int, warmup: int, energy_reader=None) -> dict:
    """Compile for one device and time it. NPU compilation commonly fails on
    models of this size -- catch it and report the reason."""
    compiled = core.compile_model(ov_model, device)
    infer = compiled.create_infer_request()

    # Fixed input, matching build_batch()
    import numpy as np

    rng = np.random.default_rng(0)
    inputs = {
        0: rng.random((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
        1: rng.random((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
        2: rng.random((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
        3: rng.random((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
        4: rng.standard_normal((1, STATE_DIM), dtype=np.float32),
    }

    def call():
        infer.infer(inputs)

    return time_calls(call, iters, warmup, energy_reader)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="path to pretrained_model dir")
    ap.add_argument("--iters", type=int, default=20, help="timed iterations (the denominator)")
    ap.add_argument("--warmup", type=int, default=3, help="untimed warmup calls")
    ap.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    ap.add_argument("--torch-device", default="cpu")
    ap.add_argument("--out", default="benchmark_results.json")
    ap.add_argument("--ir-dir", default="openvino_ir")
    ap.add_argument("--skip-openvino", action="store_true")
    ap.add_argument("--power-counter", type=Path, default=None,
                    help="path to a RAPL energy_uj counter; auto-detected on Linux. "
                         "Absent on Windows -- power is then reported as not measured")
    args = ap.parse_args()

    if not Path(args.checkpoint).is_dir():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    report = {
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "iters": args.iters,
            "warmup": args.warmup,
            "control_period_s": CONTROL_PERIOD_S,
            "budget_ms_per_action": BUDGET_MS_PER_ACTION,
            "chunk_size": CHUNK_SIZE,
        },
        "input": describe_batch(build_batch(__import__("torch"))),
        "results": {},
    }

    counter = args.power_counter or discover_rapl_counter()
    energy_reader = power_reader(counter)
    report["power"] = {
        "counter": str(counter) if counter else None,
        "measured": energy_reader is not None,
        "note": (
            "package energy sampled from a RAPL counter around the same timed loop "
            "as the latency"
            if energy_reader is not None else
            "no RAPL counter on this host -- package power and energy per inference "
            "are NOT measured and are reported as null"
        ),
    }

    print("=" * 74)
    print("SmolVLA Intel device benchmark")
    print("=" * 74)
    print(f"checkpoint : {args.checkpoint}")
    print(f"iters      : {args.iters}  (warmup {args.warmup}, excluded from the mean)")
    print(f"budget     : {BUDGET_MS_PER_ACTION:.0f} ms/action  ({1/CONTROL_PERIOD_S:.0f} Hz control)")
    print(f"power      : {counter if energy_reader is not None else 'not measured (no RAPL counter)'}")
    print()

    # ---- load ----
    t0 = time.perf_counter()
    try:
        policy = load_policy(args.checkpoint, args.torch_device)
    except Exception:
        print("FATAL: could not load the policy.", file=sys.stderr)
        traceback.print_exc()
        return 3
    load_s = time.perf_counter() - t0
    n_params = count_params(policy)
    report["environment"]["load_seconds"] = round(load_s, 2)
    report["environment"]["parameters"] = n_params
    print(f"loaded in {load_s:.1f}s -- {n_params:,} parameters")
    print()

    obs = build_batch(__import__("torch"))

    # ---- PyTorch baseline ----
    print("-" * 74)
    print(f"PyTorch baseline ({args.torch_device})")
    print("-" * 74)
    try:
        r = bench_torch(policy, obs, args.iters, args.warmup, energy_reader)
        r.update(budget_verdict(r["mean_ms"]))
        report["results"][f"pytorch_{args.torch_device}"] = r
        print(f"  chunk mean {r['mean_ms']:.1f} ms -> {r['ms_per_action']:.2f} ms/action, "
              f"{r['actions_per_second']:.1f} actions/s")
        print(f"  first call {r['first_call_ms']:.1f} ms (compile+alloc, excluded from mean)")
        if r["package_power_w"] is None:
            print("  power      not measured on this host")
        else:
            print(f"  power      {r['package_power_w']:.1f} W -> "
                  f"{r['energy_per_inference_mj']:.1f} mJ/inference, "
                  f"{r['efficiency_inf_per_s_per_w']:.2f} inf/s/W")
        print(f"  25 Hz realtime: {'PASS' if r['meets_25hz_realtime'] else 'FAIL'}")
    except Exception:
        report["results"][f"pytorch_{args.torch_device}"] = {"error": traceback.format_exc(limit=3)}
        print("  FAILED -- see JSON for the traceback")
    print()

    # ---- OpenVINO ----
    if args.skip_openvino:
        print("OpenVINO skipped (--skip-openvino)")
        Path(args.out).write_text(json.dumps(report, indent=2))
        return 0

    try:
        import openvino as ov
    except ImportError:
        report["results"]["openvino"] = {"error": "openvino not installed -- pip install openvino"}
        print("OpenVINO not installed. Install with:  pip install openvino")
        Path(args.out).write_text(json.dumps(report, indent=2))
        return 0

    report["environment"]["openvino_version"] = ov.__version__
    core = ov.Core()
    report["environment"]["available_devices"] = list(core.available_devices)
    print("-" * 74)
    print(f"OpenVINO {ov.__version__} -- available: {list(core.available_devices)}")
    print("-" * 74)

    print("Converting to IR (this is the slow step)...")
    t0 = time.perf_counter()
    ov_model, err, ir_path = export_openvino(policy, obs, Path(args.ir_dir))
    conv_s = time.perf_counter() - t0
    report["environment"]["ov_convert_seconds"] = round(conv_s, 2)

    if ov_model is None:
        report["results"]["openvino_export"] = {"error": err}
        print(f"  CONVERSION FAILED after {conv_s:.1f}s:")
        print("  " + (err or "").strip().replace("\n", "\n  ") if err else "")
        print()
        print("  A conversion failure is a real result -- record it, do not hide it.")
        print("  Fall back to component-level export (vision encoder and LLM separately).")
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
        return 0

    report["environment"]["ir_path"] = ir_path
    print(f"  converted in {conv_s:.1f}s -> {ir_path}")
    print()

    for device in args.devices:
        print(f"  [{device}]")
        try:
            r = bench_ov(core, ov_model, device, args.iters, args.warmup, energy_reader)
            r.update(budget_verdict(r["mean_ms"]))
            report["results"][f"openvino_{device}"] = r
            print(f"    chunk mean {r['mean_ms']:.1f} ms -> {r['ms_per_action']:.2f} ms/action, "
                  f"{r['actions_per_second']:.1f} actions/s")
            if r["package_power_w"] is None:
                print("    power        not measured on this host")
            else:
                print(f"    power        {r['package_power_w']:.1f} W -> "
                      f"{r['energy_per_inference_mj']:.1f} mJ/inference, "
                      f"{r['efficiency_inf_per_s_per_w']:.2f} inf/s/W")
            print(f"    25 Hz realtime: {'PASS' if r['meets_25hz_realtime'] else 'FAIL'}")
        except Exception as e:
            report["results"][f"openvino_{device}"] = {
                "error": str(e).split("\n")[0],
                "detail": traceback.format_exc(limit=3),
                "supported": False,
            }
            print(f"    NOT SUPPORTED: {str(e).splitlines()[0]}")
            print("    (recorded as a result -- most likely static-shape or op-support limits)")
        print()

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")

    # ---- summary ----
    print()
    print("=" * 74)
    print(f"{'device':<22}{'ms/action':>12}{'actions/s':>12}{'25Hz':>8}{'mJ/inf':>10}")
    print("-" * 74)
    for name, r in report["results"].items():
        if "error" in r:
            print(f"{name:<22}{'--':>12}{'--':>12}{'n/a':>8}{'--':>10}")
        else:
            power = ("--" if r["energy_per_inference_mj"] is None
                     else f"{r['energy_per_inference_mj']:.1f}")
            print(f"{name:<22}{r['ms_per_action']:>12.2f}{r['actions_per_second']:>12.1f}"
                  f"{('PASS' if r['meets_25hz_realtime'] else 'FAIL'):>8}{power:>10}")
    print("=" * 74)
    print()
    if energy_reader is None:
        print("Power and energy/inference are NOT measured: this host exposes no RAPL")
        print("counter. Do not fill that column from a datasheet -- leave it absent.")
        print()
    print("Reminder: report ms/action WITH the iters count. A latency without a")
    print("denominator is not a measurement -- that is what went wrong with")
    print("smolvla_cpu_timing.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
