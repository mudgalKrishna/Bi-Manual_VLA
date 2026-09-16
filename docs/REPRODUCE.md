# Reproduction

Everything needed to go from a clean checkout to a merged LeRobot v3 dataset. Follow the
steps in order — step 3 is the one that matters most and the one most likely to be
skipped.

---

## Step 0 — Prerequisites

| Need | For |
| --- | --- |
| Python 3.10–3.12 | everything |
| CPU MuJoCo | the expert (no GPU required) |
| CUDA GPU visible to Warp | the GPU collector only |
| `rclone` + a `gdrive:` remote | uploading checkpoints (optional) |

Python 3.13 has no `labmaze` wheel, which `dm_control` depends on. If you need 3.13,
install `dm-control --no-deps` after installing its other dependencies. Otherwise use
3.12 and avoid the problem.

```bash
git clone https://github.com/mudgalKrishna/Bi-Manual_VLA.git
cd Bi-Manual_VLA
python -m venv .venv
. .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify the checkout is complete:

```bash
python tests/test_repo_structure.py
```

That checks structure only — it does not touch MuJoCo. Step 1 is the physics check.

---

## Step 1 — Confirm the expert still reproduces

The reference rollout is seed 0 with randomization off. This is the configuration the
v21 grasp was validated in, and it is the fastest way to tell whether a fresh
environment can reproduce the physics at all.

```bash
python scripts/task_demo.py
```

Expect it to finish and report success. **If this fails, stop.** Everything downstream
inherits its correctness, and a collector run on a broken expert produces a large,
expensive, worthless dataset.

> `task_demo.py` reads `notebooks/01_environment_cells.py` at import and `exec`s cells
> 2–7 to obtain `cfg`, `build_arena` and `lookat_xyaxes`. It also expects the SO-101 model
> at `third_party/SO-ARM100/Simulation/SO101/scene.xml`. If either moved, you will get a
> `KeyError` on `ns["cfg"]` rather than a clear message.

---

## Step 2 — Install the GPU rendering stack

Only if you are using `record_molab_mjwarp.py`. On a machine with working EGL you do not
need this path at all — use `record_lerobot_shard.py`.

```bash
pip install mujoco-warp warp-lang
python -c "import warp as wp; wp.init(); assert wp.is_cuda_available(); print(wp.get_device('cuda:0'))"
```

If that assertion fails, fix it before running the collector. The collector checks the
same thing and will raise `RuntimeError("Warp cannot see CUDA")`.

---

## Step 3 — Verify the pilot

**Do not skip this.** A collector can exit zero and still produce a dataset that trains to
a policy that does nothing.

```bash
python scripts/record_molab_mjwarp.py \
    --output-root /tmp/v21-pilot \
    --episode-start 0 --episodes 1 \
    --physics-workers 1 --render-batch 64
```

Now inspect `dinner_table_dataset/episodes/episode_000000/lerobot_dataset/` and confirm
**all four** of these:

1. **Four camera streams exist** and each frame is 320×320×3.
2. **The arms move.** Decode `arm_a` or `arm_b` and compare the first and last frame. A
   grasp that never closes means the physics did not run.
3. **The cameras move.** This is the one that silently breaks. In the Warp path the
   renderer calls `fwd_kinematics`, and camera transforms come from that same call — if a
   future version changes which forward function is used, the cameras can freeze at their
   initial pose while the arms still animate. Every episode then contains four static
   views and the dataset trains on nothing. Compare `arm_a` frame 0 against frame N.
4. **The `top` view is not blown out.** MuJoCo Warp adds a headlight to the explicit scene
   lights; with the v21 two-light rig that clipped most pixels to white. The collector
   compensates by disabling the headlight and scaling `light_diffuse`/`light_specular`.
   If your frames are white or yellow, that compensation is not taking effect.

Record the rejection rate from the pilot before launching the full run. If the expert is
failing more than a small fraction of rollouts, fix that first — the collector will
happily spend hours retrying.

---

## Step 4 — Full collection

### Option A — GPU, 10 parallel sessions (Molab or equivalent)

Run one session per worker. Each needs a disjoint episode range:

```bash
python scripts/record_molab_mjwarp.py \
    --output-root /tmp/v21-out \
    --episode-start 0 --episodes 12 \
    --seed-base 51000 --physics-workers 3 --render-batch 64 \
    --image-size 320 --max-attempts 4 \
    --worker-id 0 --drive-prefix dinner_table_dataset/molab_v2 \
    --drive-remote gdrive: --rclone-config /path/to/rclone.conf
```

Worker *N* takes `--episode-start $((N*12))`. Ten workers × 12 episodes = 120.

Notes:

- `--physics-workers` is threads **within one session**. Size it to the container's CPU
  count, not to the number of workers. It is not a speed dial once the GPU renderer is
  the bottleneck.
- `--render-batch` batches frames of one episode, not independent worlds. Raising it does
  not increase parallel throughput; it only trades memory for fewer kernel launches.
- Each worker writes to `.../workers/worker_NN/` on Drive. Workers never share a manifest.

### Option B — CPU, sharded (Kaggle or any 4-core box)

```bash
python scripts/record_lerobot_shard.py \
    --worker-id 0 --output-root /tmp/v21-out \
    --drive-remote gdrive: --rclone-config /path/to/rclone.conf
```

`WORKERS` in that script hardcodes four ranges: `(0,38)`, `(38,76)`, `(76,113)`,
`(113,150)`. Edit it to change the split. This path renders with the OpenGL backend, so
it needs working EGL/GLFW and a GPU or a fast CPU rasteriser.

Each episode is checkpointed and uploaded the moment it is accepted. A preempted session
resumes by re-running the same command — completed episodes are skipped.

### Resuming

Both collectors skip episodes already present in `manifest.json`, whether accepted or
rejected. Re-running the same command is always safe and is the intended resume path.

---

## Step 5 — Merge

```bash
python scripts/merge_molab_episodes.py --output-root /tmp/v21-out --expected 120
```

The script locates either layout automatically:

- `dinner_table_dataset/episodes/episode_*/lerobot_dataset` (single-worker)
- `dinner_table_dataset/workers/worker_*/episodes/episode_*/lerobot_dataset` (fan-out)

It validates every checkpoint, merges, re-checks the count, and writes
`MERGE_REPORT.json`. A count mismatch is a hard error, deliberately — a silently partial
merge is worse than a loud failure.

Read `MERGE_REPORT.json` and confirm `frames == expected_source_frames` before training.

---

## Known non-reproducibility

Be explicit about these when publishing results:

- **Different renderer, different pixels.** A dataset built with the Warp path will not
  match one built with `MUJOCO_GL=egl` at the same seed. Physics is identical; images are
  not.
- **GPU nondeterminism.** MuJoCo Warp's batch renderer is not bit-reproducible across
  driver versions. Episode *acceptance* is determined by the CPU physics and should be
  stable; the pixels will not be.
- **Warm-up is charged to episode one.** The first episode of any session includes
  renderer construction and kernel compilation. Its `render_encode_seconds` is an
  outlier — exclude it from any per-episode timing claim.
- **`--image-size` changes the schema.** A 320 dataset and a 224 dataset are not
  interchangeable; the recorded feature shapes differ.
