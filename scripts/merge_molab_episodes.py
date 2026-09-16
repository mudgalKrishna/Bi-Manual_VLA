"""Merge finalized Molab episode checkpoints into one trainable LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from lerobot.datasets import LeRobotDataset, merge_datasets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--expected", type=int, default=150)
    args = p.parse_args()
    base = args.output_root / "dinner_table_dataset"
    roots = sorted((base / "episodes").glob("episode_*/lerobot_dataset"))
    if not roots:
        roots = sorted((base / "workers").glob(
            "worker_*/episodes/episode_*/lerobot_dataset"))
    if len(roots) != args.expected:
        raise RuntimeError(f"expected {args.expected} finalized episodes, found {len(roots)}")
    datasets = []
    total_frames = 0
    for root in roots:
        gid = root.parent.name.split("_")[-1]
        ds = LeRobotDataset(repo_id=f"local/dinner-table-molab-{gid}", root=root)
        if ds.meta.total_episodes != 1 or ds.meta.fps != 25:
            raise ValueError(f"invalid checkpoint {root}")
        datasets.append(ds)
        total_frames += ds.meta.total_frames
    output = base / "merged" / "dinner_table_smolvla_v1"
    if output.exists():
        shutil.rmtree(output)
    merged = merge_datasets(
        datasets, output_repo_id="local/dinner-table-smolvla-v1",
        output_dir=output, concatenate_videos=False, concatenate_data=False,
    )
    if merged.meta.total_episodes != args.expected:
        raise RuntimeError("merged episode count mismatch")
    report = {"episodes": merged.meta.total_episodes,
              "frames": merged.meta.total_frames, "fps": merged.meta.fps,
              "expected_source_frames": total_frames, "output": str(output)}
    (output.parent / "MERGE_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
