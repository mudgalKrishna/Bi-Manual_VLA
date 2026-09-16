#!/usr/bin/env python3
"""Run the ten-seed evaluation entry point and save its log.

By default this invokes the repository's scripted environment evaluator.  A
learned-policy evaluator can be supplied with ``--command``; use ``{seed}``
as a placeholder for the per-seed integer.  This keeps scripted-expert
validation separate from the final SmolVLA inference claim.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/intel_10_seed_evaluation.log"))
    parser.add_argument(
        "--command",
        help=("optional learned-policy command template, e.g. "
              "'python scripts/run_intel_inference.py --seed {seed}'"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.command:
        commands = [shlex.split(args.command.format(seed=seed)) for seed in range(args.seeds)]
    else:
        commands = [[
            sys.executable, "scripts/eval_seeds.py",
            "--seeds", str(args.seeds), "--mode", "task", "--save-videos", "3",
        ]]

    records = []
    with args.output.open("w", encoding="utf-8") as log:
        for seed, command in enumerate(commands):
            log.write(f"\n=== seed {seed}: {' '.join(command)} ===\n")
            completed = subprocess.run(command, text=True, capture_output=True)
            log.write(completed.stdout)
            log.write(completed.stderr)
            records.append({"seed": seed, "returncode": completed.returncode})
            if completed.returncode:
                print(f"seed {seed} failed; see {args.output}")

    passed = sum(record["returncode"] == 0 for record in records)
    print(f"Evaluation commands completed: {passed}/{len(records)}")
    print(f"Log: {args.output.resolve()}")


if __name__ == "__main__":
    main()
