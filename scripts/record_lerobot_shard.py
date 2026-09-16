"""Generate one resumable Kaggle worker shard from the validated v21 expert.

Each successful rollout is first converted into its own finalized LeRobot v3
dataset.  This makes every episode an atomic checkpoint that can be copied to
Drive without allowing multiple workers to touch the same metadata.  At the
end, the episode datasets are merged into one worker-level LeRobot dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


WORKERS = {
    0: (0, 38, 11000),
    1: (38, 76, 21000),
    2: (76, 113, 31000),
    3: (113, 150, 41000),
}

JOINT_NAMES = [
    f"{side}_{joint}.pos"
    for side in ("left", "right")
    for joint in (
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    )
]

FEATURES = {
    "observation.images.front": {
        "dtype": "video", "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.arm_a": {
        "dtype": "video", "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.arm_b": {
        "dtype": "video", "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.top": {
        "dtype": "video", "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32", "shape": (12,), "names": JOINT_NAMES,
    },
    "action": {
        "dtype": "float32", "shape": (12,), "names": JOINT_NAMES,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--worker-id", type=int, choices=range(4), required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--episode-limit", type=int, default=None,
                   help="Pilot only: cap the number assigned to this worker")
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--drive-remote", default="",
                   help="Configured rclone remote, for example gdrive:")
    p.add_argument("--drive-root-folder-id", default="")
    p.add_argument("--rclone-config", type=Path, default=None)
    p.add_argument("--keep-npz", action="store_true")
    return p.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def rclone_copy(local: Path, remote_rel: str, args: argparse.Namespace) -> None:
    if not args.drive_remote:
        return
    cmd = ["rclone", "copy", str(local), f"{args.drive_remote}{remote_rel}",
           "--transfers", "4", "--checkers", "8", "--retries", "5",
           "--low-level-retries", "10", "--stats", "30s"]
    if args.drive_root_folder_id:
        cmd += ["--drive-root-folder-id", args.drive_root_folder_id]
    if args.rclone_config:
        cmd += ["--config", str(args.rclone_config)]
    subprocess.run(cmd, check=True)


def run_expert(global_id: int, seed: int, attempt: int, temp_dir: Path,
               args: argparse.Namespace) -> tuple[Path, dict, Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    npz = temp_dir / f"episode_{global_id:06d}.npz"
    status = temp_dir / f"episode_{global_id:06d}.status.json"
    log = temp_dir / f"episode_{global_id:06d}.attempt_{attempt}.log"
    env = os.environ.copy()
    env.update({
        "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
        "PYOPENGL_PLATFORM": env.get("PYOPENGL_PLATFORM", "egl"),
        "DEMO_GRASP_STABILIZE": "0",
        "DEMO_NO_RENDER": "0",
        "DEMO_NO_DEMO_VIDEO": "1",
        "DEMO_RANDOMIZE": "1",
        "DEMO_STRICT_SUCCESS": "1",
        "DEMO_EPISODE_SEED": str(seed),
        "DEMO_GLOBAL_EPISODE_ID": str(global_id),
        "DEMO_EPISODE_OUT": str(npz),
        "DEMO_STATUS_OUT": str(status),
    })
    command = [sys.executable, "-u", str(args.repo_root / "scripts" / "task_demo.py")]
    started = time.time()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=args.repo_root, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if not status.exists():
        raise RuntimeError(f"expert produced no status file; inspect {log}")
    info = json.loads(status.read_text(encoding="utf-8"))
    info.update({"attempt": attempt, "returncode": result.returncode,
                 "wall_seconds": round(time.time() - started, 2)})
    if result.returncode != 0 or not info.get("success") or not npz.exists():
        raise RuntimeError(f"rollout failed; inspect {log}")
    return npz, info, log


def convert_npz(npz_path: Path, dataset_root: Path, repo_id: str) -> dict:
    from lerobot.datasets import LeRobotDataset

    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    with np.load(npz_path, allow_pickle=False) as ep:
        required = ("img_front", "img_a", "img_b", "img_top", "state", "action")
        lengths = {key: len(ep[key]) for key in required}
        if len(set(lengths.values())) != 1 or next(iter(lengths.values())) == 0:
            raise ValueError(f"unsynchronized or empty episode: {lengths}")
        if ep["state"].shape[1:] != (12,) or ep["action"].shape[1:] != (12,):
            raise ValueError("state/action must both be 12-dimensional")
        if any(ep[key].shape[1:] != (224, 224, 3)
               for key in ("img_front", "img_a", "img_b", "img_top")):
            raise ValueError("all camera streams must be HWC uint8 224x224x3")
        instruction = str(ep["instruction"].item())

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=25,
            features=FEATURES,
            root=dataset_root,
            robot_type="bimanual_so101_mujoco",
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            batch_encoding_size=1,
        )
        for i in range(lengths["state"]):
            dataset.add_frame({
                "observation.images.front": ep["img_front"][i],
                "observation.images.arm_a": ep["img_a"][i],
                "observation.images.arm_b": ep["img_b"][i],
                "observation.images.top": ep["img_top"][i],
                "observation.state": ep["state"][i].astype(np.float32, copy=False),
                "action": ep["action"][i].astype(np.float32, copy=False),
                "task": instruction,
            })
        dataset.save_episode(parallel_encoding=True)
        dataset.finalize()
        return {"frames": lengths["state"], "instruction": instruction}


def merge_worker_datasets(episode_roots: list[Path], output_root: Path,
                          worker_id: int) -> None:
    from lerobot.datasets import LeRobotDataset, merge_datasets

    if output_root.exists():
        shutil.rmtree(output_root)
    sources = [
        LeRobotDataset(repo_id=f"local/dinner-table-episode-{root.parent.name.split('_')[-1]}",
                       root=root)
        for root in episode_roots
    ]
    merge_datasets(
        sources,
        output_repo_id=f"local/dinner-table-worker-{worker_id:02d}",
        output_dir=output_root,
        concatenate_videos=False,
        concatenate_data=False,
    )


def main() -> None:
    args = parse_args()
    start, stop, seed_base = WORKERS[args.worker_id]
    ids = list(range(start, stop))
    if args.episode_limit is not None:
        ids = ids[:args.episode_limit]

    worker_root = args.output_root / "dinner_table_dataset" / "shards" / f"worker_{args.worker_id:02d}"
    episodes_root = worker_root / "episodes"
    logs_root = worker_root / "logs"
    episodes_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    manifest_path = worker_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "worker_id": args.worker_id,
        "assigned_global_ids": ids,
        "seed_base": seed_base,
        "fps": 25,
        "camera_keys": ["front", "arm_a", "arm_b", "top"],
        "episodes": [],
        "failures": [],
    }
    completed_ids = {int(x["global_episode_id"]) for x in manifest["episodes"]}

    for global_id in ids:
        episode_dir = episodes_root / f"episode_{global_id:06d}"
        success_marker = episode_dir / "SUCCESS.json"
        if global_id in completed_ids and success_marker.exists():
            print(f"[skip] episode {global_id} already checkpointed", flush=True)
            continue

        accepted = False
        for attempt in range(args.max_attempts):
            seed = seed_base + (global_id - start) + attempt * 100_000
            temp = args.output_root / "temp" / f"worker_{args.worker_id:02d}"
            try:
                print(f"[run] episode={global_id} seed={seed} attempt={attempt}", flush=True)
                npz, run_info, log = run_expert(global_id, seed, attempt, temp, args)
                dataset_root = episode_dir / "lerobot_dataset"
                converted = convert_npz(
                    npz, dataset_root,
                    repo_id=f"local/dinner-table-episode-{global_id:06d}",
                )
                record = {
                    **run_info, **converted,
                    "global_episode_id": global_id,
                    "seed": seed,
                    "dataset": str(dataset_root.relative_to(worker_root)),
                }
                episode_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log, episode_dir / "generation.log")
                atomic_json(success_marker, record)
                manifest["episodes"] = [
                    x for x in manifest["episodes"]
                    if int(x["global_episode_id"]) != global_id
                ] + [record]
                manifest["episodes"].sort(key=lambda x: x["global_episode_id"])
                atomic_json(manifest_path, manifest)
                if not args.keep_npz:
                    npz.unlink(missing_ok=True)
                remote = f"dinner_table_dataset/shards/worker_{args.worker_id:02d}/episodes/episode_{global_id:06d}"
                rclone_copy(episode_dir, remote, args)
                rclone_copy(manifest_path, f"dinner_table_dataset/shards/worker_{args.worker_id:02d}", args)
                print(f"[saved] episode={global_id} frames={converted['frames']}", flush=True)
                accepted = True
                break
            except Exception as exc:
                failure = {"global_episode_id": global_id, "seed": seed,
                           "attempt": attempt, "error": str(exc)}
                manifest["failures"].append(failure)
                atomic_json(manifest_path, manifest)
                print(f"[retry] {failure}", flush=True)
        if not accepted:
            raise RuntimeError(f"episode {global_id} failed {args.max_attempts} attempts")

    selected = [
        episodes_root / f"episode_{global_id:06d}" / "lerobot_dataset"
        for global_id in ids
    ]
    if not all(path.exists() for path in selected):
        raise RuntimeError("not all assigned episode checkpoints exist; refusing final merge")
    shard_root = worker_root / "lerobot_dataset"
    print(f"[merge] {len(selected)} episodes -> {shard_root}", flush=True)
    merge_worker_datasets(selected, shard_root, args.worker_id)
    atomic_json(worker_root / "COMPLETE.json", {
        "worker_id": args.worker_id, "episodes": len(selected),
        "global_id_start": ids[0], "global_id_stop_exclusive": ids[-1] + 1,
        "dataset": "lerobot_dataset",
    })
    rclone_copy(shard_root,
                f"dinner_table_dataset/shards/worker_{args.worker_id:02d}/lerobot_dataset",
                args)
    rclone_copy(worker_root / "COMPLETE.json",
                f"dinner_table_dataset/shards/worker_{args.worker_id:02d}", args)
    print("[complete] worker shard finalized, validated, and synced", flush=True)


if __name__ == "__main__":
    main()
