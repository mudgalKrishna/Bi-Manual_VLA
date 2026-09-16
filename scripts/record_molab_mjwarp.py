"""Molab collector: v21 CPU contact physics + CUDA batch image rendering.

The expert is deliberately simulated by MuJoCo CPU because the validated v21
grasp uses noslip contact iterations, which MJWarp does not currently support.
Only rendering is moved to the NVIDIA GPU; it does not use EGL or OpenGL.
Each accepted episode is finalized as an independent LeRobot v3 checkpoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from copy import deepcopy

import numpy as np

from record_lerobot_shard import FEATURES as BASE_FEATURES, atomic_json, rclone_copy


CAMERAS = ("cam_front", "cam_arm_a", "cam_arm_b", "cam_top")
CAMERA_KEYS = ("front", "arm_a", "arm_b", "top")


def features_for_size(image_size: int) -> dict:
    features = deepcopy(BASE_FEATURES)
    for key in CAMERA_KEYS:
        features[f"observation.images.{key}"]["shape"] = (
            image_size, image_size, 3)
    return features


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--episode-start", type=int, default=0)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--seed-base", type=int, default=51000)
    p.add_argument("--physics-workers", type=int, default=3)
    p.add_argument("--render-batch", type=int, default=64)
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--drive-remote", default="")
    p.add_argument("--drive-root-folder-id", default="")
    p.add_argument("--rclone-config", type=Path, default=None)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--drive-prefix", default="dinner_table_dataset/molab_v2")
    p.add_argument("--keep-trajectories", action="store_true")
    return p.parse_args()


def run_trajectory(gid: int, seed: int, attempt: int, args: argparse.Namespace) -> dict:
    work = args.output_root / "work" / f"episode_{gid:06d}"
    work.mkdir(parents=True, exist_ok=True)
    npz = work / "trajectory.npz"
    model = work / "scene.mjb"
    status = work / "status.json"
    log = work / f"attempt_{attempt}.log"
    env = os.environ.copy()
    env.update({
        "DEMO_GRASP_STABILIZE": "0",
        "DEMO_NO_RENDER": "1",
        "DEMO_TRAJECTORY_ONLY": "1",
        "DEMO_NO_DEMO_VIDEO": "1",
        "DEMO_RANDOMIZE": "1",
        "DEMO_STRICT_SUCCESS": "1",
        "DEMO_EPISODE_SEED": str(seed),
        "DEMO_GLOBAL_EPISODE_ID": str(gid),
        "DEMO_EPISODE_OUT": str(npz),
        "DEMO_MODEL_OUT": str(model),
        "DEMO_STATUS_OUT": str(status),
    })
    started = time.time()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, "-u", str(args.repo_root / "scripts/task_demo.py")],
            cwd=args.repo_root, env=env, stdout=stream, stderr=subprocess.STDOUT,
        )
    info = json.loads(status.read_text()) if status.exists() else {}
    if result.returncode or not info.get("success") or not npz.exists() or not model.exists():
        raise RuntimeError(f"trajectory failed gid={gid}; inspect {log}")
    return {"gid": gid, "seed": seed, "attempt": attempt, "npz": npz,
            "model": model, "log": log,
            "physics_seconds": round(time.time() - started, 2)}


class WarpRenderer:
    def __init__(self, model_path: Path, batch: int, image_size: int):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        self.mjw, self.wp, self.batch, self.image_size = mjw, wp, batch, image_size
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("Warp cannot see CUDA; verify the Molab GPU runtime")
        wp.set_device("cuda:0")
        self.mjm = mujoco.MjModel.from_binary_path(str(model_path))
        # Render-only device model. Physics has already been computed by the
        # exact CPU v21 solver; MJWarp does not support MuJoCo noslip.
        self.mjm.opt.noslip_iterations = 0
        # MuJoCo-Warp adds the model headlight to the explicit scene lights.
        # With the v21 two-light rig that clipped most pixels to white/yellow.
        # Keep lighting deterministic but reduce it to a useful camera range.
        self.mjm.vis.headlight.active = 0
        self.mjm.light_diffuse[:] *= 0.55
        self.mjm.light_specular[:] *= 0.10
        self.m = mjw.put_model(self.mjm)
        self.d = mjw.make_data(self.mjm, nworld=batch)
        self.rc = mjw.create_render_context(
            self.mjm, nworld=batch, cam_res=(image_size, image_size), render_rgb=True,
            render_depth=False, use_textures=True, use_shadows=False,
            cam_active=list(CAMERAS), enabled_geom_groups=[0, 1, 2, 3, 4, 5],
        )
        self.rgb = [wp.empty((batch, image_size, image_size), dtype=wp.vec3f, device="cuda:0")
                    for _ in CAMERAS]

    def render(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        n = len(qpos)
        padded = np.empty((self.batch, qpos.shape[1]), np.float32)
        padded[:n] = qpos
        padded[n:] = qpos[-1]
        self.wp.copy(self.d.qpos, self.wp.array(padded, dtype=self.wp.float32,
                                                device="cuda:0"))
        # Offline replay needs transforms only.  Calling full forward dynamics
        # unnecessarily compiles collision/solver kernels and currently fails
        # on some Blackwell + gVisor runtimes.  fwd_kinematics computes body,
        # geom, camera, and light transforms without collision detection.
        self.mjw.fwd_kinematics(self.m, self.d)
        self.mjw.refit_bvh(self.m, self.d, self.rc)
        self.mjw.render(self.m, self.d, self.rc)
        result = {}
        for index, (key, out) in enumerate(zip(CAMERA_KEYS, self.rgb)):
            self.mjw.get_rgb(self.rc, index, out)
            result[key] = np.clip(out.numpy()[:n] * 255.0, 0, 255).astype(np.uint8)
        return result


def convert_episode(job: dict, renderer: WarpRenderer, args: argparse.Namespace) -> dict:
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    gid = job["gid"]
    episode = args.output_root / "dinner_table_dataset" / "episodes" / f"episode_{gid:06d}"
    dataset_root = episode / "lerobot_dataset"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    with np.load(job["npz"], allow_pickle=False) as src:
        qpos = src["qpos"]
        state = src["state"]
        action = src["action"]
        task = str(src["instruction"].item())
        if not (len(qpos) == len(state) == len(action) > 0):
            raise ValueError("trajectory/state/action streams are not synchronized")

        ds = LeRobotDataset.create(
            repo_id=f"local/dinner-table-molab-{gid:06d}", root=dataset_root,
            fps=25, features=features_for_size(renderer.image_size),
            robot_type="bimanual_so101_mujoco",
            use_videos=True, image_writer_threads=8, image_writer_processes=0,
            batch_encoding_size=1,
            # CRF 30 was visibly soft, particularly for thin forks/spoons.
            # CRF 18 preserves grasp-relevant edges; g=25
            # gives one random-access keyframe per second.
            rgb_encoder=RGBEncoderConfig(
                vcodec="libsvtav1", pix_fmt="yuv420p", g=25,
                crf=18, preset=10,
            ),
        )
        rendered = {key: [] for key in CAMERA_KEYS}
        t0 = time.time()
        for begin in range(0, len(qpos), renderer.batch):
            batch = renderer.render(qpos[begin:begin + renderer.batch])
            for key in CAMERA_KEYS:
                rendered[key].append(batch[key])
        rendered = {key: np.concatenate(parts, axis=0) for key, parts in rendered.items()}
        for i in range(len(qpos)):
            ds.add_frame({
                "observation.images.front": rendered["front"][i],
                "observation.images.arm_a": rendered["arm_a"][i],
                "observation.images.arm_b": rendered["arm_b"][i],
                "observation.images.top": rendered["top"][i],
                "observation.state": state[i].astype(np.float32, copy=False),
                "action": action[i].astype(np.float32, copy=False),
                "task": task,
            })
        ds.save_episode(parallel_encoding=True)
        ds.finalize()
    info = {"global_episode_id": gid, "seed": job["seed"], "attempt": job["attempt"],
            "frames": len(qpos), "task": task, "physics_seconds": job["physics_seconds"],
            "render_encode_seconds": round(time.time() - t0, 2), "success": True}
    atomic_json(episode / "SUCCESS.json", info)
    return info


def main() -> None:
    args = arguments()
    args.output_root.mkdir(parents=True, exist_ok=True)
    dataset = args.output_root / "dinner_table_dataset"
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "format": "LeRobot v3", "fps": 25, "cameras": list(CAMERA_KEYS),
        "episodes": [], "rejected": []}
    manifest.setdefault("rejected", [])
    completed = {int(x["global_episode_id"]) for x in manifest["episodes"]}
    rejected = {int(x["global_episode_id"]) for x in manifest["rejected"]}
    todo = [args.episode_start + i for i in range(args.episodes)
            if args.episode_start + i not in completed | rejected]
    renderer = None
    started = time.time()

    def accepted_job(gid: int) -> dict:
        last = None
        for attempt in range(args.max_attempts):
            seed = args.seed_base + gid + attempt * 100_000
            try:
                return run_trajectory(gid, seed, attempt, args)
            except Exception as exc:  # retry only this episode
                last = exc
        raise RuntimeError(f"episode {gid} exhausted retries") from last

    with ThreadPoolExecutor(max_workers=args.physics_workers) as pool:
        futures = {pool.submit(accepted_job, gid): gid for gid in todo}
        for future in as_completed(futures):
            gid = futures[future]
            try:
                job = future.result()
            except Exception as exc:
                failure = {
                    "global_episode_id": gid,
                    "attempts": args.max_attempts,
                    "reason": str(exc),
                    "accepted_for_training": False,
                }
                manifest["rejected"].append(failure)
                manifest["rejected"].sort(key=lambda x: x["global_episode_id"])
                atomic_json(manifest_path, manifest)
                remote = f"{args.drive_prefix}/workers/worker_{args.worker_id:02d}"
                rclone_copy(manifest_path, remote, args)
                print(f"[skipped] {failure}", flush=True)
                continue
            if renderer is None:
                renderer = WarpRenderer(job["model"], args.render_batch, args.image_size)
                # One warm-up is intentionally included in the first episode ETA.
            info = convert_episode(job, renderer, args)
            manifest["episodes"].append(info)
            manifest["episodes"].sort(key=lambda x: x["global_episode_id"])
            elapsed = time.time() - started
            processed = len(manifest["episodes"]) + len(manifest["rejected"])
            manifest["elapsed_seconds"] = round(elapsed, 2)
            manifest["eta_seconds"] = round(
                elapsed / max(1, processed) * max(0, args.episodes - processed), 2)
            atomic_json(manifest_path, manifest)
            # Each parallel Molab session owns a separate manifest directory.
            # Only the globally unique episode IDs are merged later.
            remote = f"{args.drive_prefix}/workers/worker_{args.worker_id:02d}"
            rclone_copy(dataset / "episodes" / f"episode_{job['gid']:06d}",
                        f"{remote}/episodes/episode_{job['gid']:06d}", args)
            rclone_copy(manifest_path, remote, args)
            if not args.keep_trajectories:
                shutil.rmtree(job["npz"].parent, ignore_errors=True)
            print(f"[accepted] {info} ETA={manifest['eta_seconds']/3600:.2f}h", flush=True)

    completion = {
        "worker_id": args.worker_id,
        "episode_start": args.episode_start,
        "episode_count": args.episodes,
        "completed_episode_ids": sorted(
            int(x["global_episode_id"]) for x in manifest["episodes"]),
        "rejected_episode_ids": sorted(
            int(x["global_episode_id"]) for x in manifest["rejected"]),
        "successful_count": len(manifest["episodes"]),
        "rejected_count": len(manifest["rejected"]),
        "complete": len(manifest["episodes"]) + len(manifest["rejected"]) == args.episodes,
    }
    complete_path = dataset / "COMPLETE.json"
    atomic_json(complete_path, completion)
    remote = f"{args.drive_prefix}/workers/worker_{args.worker_id:02d}"
    rclone_copy(complete_path, remote, args)


if __name__ == "__main__":
    main()
