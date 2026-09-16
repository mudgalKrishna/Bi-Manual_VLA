#!/usr/bin/env python3
"""Run the ten-seed evaluation entry point and save its log.

By default this invokes the repository's scripted environment evaluator.  A
learned-policy evaluator can be supplied with ``--command``; use ``{seed}``
as a placeholder for the per-seed integer.  This keeps scripted-expert
validation separate from the final SmolVLA inference claim.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def split_command(command: str) -> list:
    """Split a command template portably.

    `shlex.split` in POSIX mode eats the backslashes in `C:\\path\\to\\python.exe`,
    which is the normal shape of a command line on the deployment target. On
    Windows, split without POSIX escaping and drop the quotes it leaves behind.
    """
    if os.name != "nt":
        return shlex.split(command)
    return [token.strip('"') for token in shlex.split(command, posix=False)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/intel_10_seed_evaluation.log"))
    parser.add_argument(
        "--command",
        help=("optional learned-policy command template, e.g. "
              "'python scripts/run_intel_inference.py --checkpoint <dir> --device CPU'. "
              "Use {seed} as the placeholder for the per-seed integer"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Each seed is one process. task_demo.py reads DEMO_EPISODE_SEED from the
    # environment, which is what makes the sweep reproducible from the shell:
    #   DEMO_EPISODE_SEED=3 python scripts/task_demo.py
    #
    # It also writes a status JSON when DEMO_STATUS_OUT is set, and exits 2 on a
    # failed episode under DEMO_STRICT_SUCCESS. Driving both means a seed's exit
    # code carries its verdict, instead of only saying that the process ran.
    status_dir = args.output.resolve().parent
    if args.command:
        runs = [(seed, split_command(args.command.format(seed=seed)), {})
                for seed in range(args.seeds)]
    else:
        runs = [(seed, [sys.executable, "scripts/task_demo.py"], {
            "DEMO_EPISODE_SEED": str(seed),
            "DEMO_STRICT_SUCCESS": "1",
            "DEMO_STATUS_OUT": str(status_dir / f"seed_{seed}.json"),
        }) for seed in range(args.seeds)]

    records = []
    with args.output.open("w", encoding="utf-8") as log:
        for seed, command, overrides in runs:
            env = {**os.environ, **overrides}
            shown = " ".join(f"{k}={v}" for k, v in overrides.items())
            log.write(f"\n=== seed {seed}: {shown} {' '.join(command)} ===\n")
            completed = subprocess.run(command, text=True, capture_output=True, env=env)
            log.write(completed.stdout)
            log.write(completed.stderr)

            record = {"seed": seed, "returncode": completed.returncode, "success": None}
            status_path = status_dir / f"seed_{seed}.json"
            if status_path.exists():
                try:
                    record["success"] = bool(json.loads(
                        status_path.read_text(encoding="utf-8")).get("success"))
                except (OSError, ValueError):
                    record["success"] = None
            records.append(record)
            log.write(f"=== seed {seed}: exit {completed.returncode}, "
                      f"success={record['success']} ===\n")
            if record["success"] is False or completed.returncode:
                print(f"seed {seed}: exit {completed.returncode}, "
                      f"success={record['success']}; see {args.output}")

    judged = [r for r in records if r["success"] is not None]
    succeeded = sum(r["success"] for r in judged)
    print(f"Episodes reported success: {succeeded}/{len(judged)}")
    if len(judged) < len(records):
        print(f"  ({len(records) - len(judged)} seed(s) wrote no status file -- "
              f"count them as unmeasured, not as passes)")
    print(f"Log: {args.output.resolve()}")


if __name__ == "__main__":
    main()
