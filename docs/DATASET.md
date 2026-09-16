# Dataset

## Format

[LeRobot v3](https://github.com/huggingface/lerobot), one episode per accepted rollout,
25 Hz. Every accepted rollout is first written as its own single-episode dataset
(`episodes/episode_NNNNNN/lerobot_dataset/`), then merged.

## Features

| Key | dtype | Shape | Source |
| --- | --- | --- | --- |
| `observation.images.front` | video | 320×320×3 | `cam_front` |
| `observation.images.arm_a` | video | 320×320×3 | `cam_arm_a` (left wrist) |
| `observation.images.arm_b` | video | 320×320×3 | `cam_arm_b` (right wrist) |
| `observation.images.top` | video | 320×320×3 | `cam_top` |
| `observation.state` | float32 | (12,) | measured joint positions |
| `action` | float32 | (12,) | joint position targets |
| `task` | string | — | instruction (default "set the dinner table") |

Joint order is `["<side>_<joint>.pos"]` for `side` in `(left, right)` and `joint` in
`(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)` — the order
defined by `JOINT_NAMES` in `scripts/record_lerobot_shard.py`. **Both arms' joints are in
one 12-vector**; there is no separate left/right action head.

Image size is set by `--image-size` (default 320) and flows into the feature definition
through `features_for_size()`, so the recorded schema always matches the rendered frames.

## Video encoding

```python
RGBEncoderConfig(vcodec="libsvtav1", pix_fmt="yuv420p", g=25, crf=18, preset=10)
```

- **CRF 18, not 30.** CRF 30 was the original setting and was visibly soft on thin
  objects — forks and spoons lost exactly the edges the grasp depends on. If you change
  this, re-check a fork frame before and after.
- **`g=25`** gives one keyframe per second at 25 Hz, which keeps random access cheap
  during training dataloading.
- `yuv420p` is required for broad decoder compatibility.

## Accepted vs rejected

`manifest.json` has two lists:

```json
{
  "format": "LeRobot v3", "fps": 25,
  "cameras": ["front", "arm_a", "arm_b", "top"],
  "episodes": [{"global_episode_id": 0, "seed": 51000, "frames": 2367, "success": true, ...}],
  "rejected": [{"global_episode_id": 7, "attempts": 4, "reason": "...", "accepted_for_training": false}]
}
```

**Only `episodes` goes into training.** Rejections are retained so the failure rate stays
auditable. Do not filter them out of the published manifest — a dataset that reports only
its successes has no measurable quality.

## Merging

```bash
python scripts/merge_molab_episodes.py --output-root <root> --expected <N>
```

Checks before writing anything:

1. Every checkpoint has exactly one episode and `fps == 25`.
2. The number of checkpoints equals `--expected`.
3. After merging, the merged episode count equals `--expected`.

Any mismatch is a hard error. The merge writes `dinner_table_dataset/MERGE_REPORT.json`:

```json
{"episodes": N, "frames": M, "fps": 25, "expected_source_frames": M, "output": "..."}
```

If `frames != expected_source_frames`, episodes were dropped — investigate before
training.

## Reporting results

When publishing dataset numbers, cite these and nothing else:

- `MERGE_REPORT.json` → `episodes`, `frames`
- `manifest.json` → `len(rejected)` and the distinct `reason` strings
- The exact `--seed-base` and episode range used

Do not report an intended episode count as an achieved one. If the run produced 118 of
120, the dataset has 118 episodes.

---

## SmolVLA fine-tune

**In progress — 4,000 of 6,826 steps (58.6%).** The checkpoint loads and emits a valid
12-D action on the Intel target. This is the recipe being used, with the reasoning
recorded so it can be critiqued rather than guessed at. Measured figures:
[`RESULTS.md`](RESULTS.md).

### Camera mismatch — read this first

**SmolVLA expects 3 cameras. This dataset has 4.** The stock SmolVLA config names
`observation.images.camera1/2/3`. You must either:

- **drop one view** by remapping three of the four keys to `camera1/2/3`, or
- **extend the config** to accept a fourth image key.

Keeping all four is ~33% more vision compute per sample. Do not drop the `top` view to
save it — the overhead view is where object positions are least occluded by the arms.

### Starting configuration

| Setting | Value | Note |
| --- | --- | --- |
| `batch_size` | 64 | 128–256 fits comfortably in 96 GB for a ~450M model |
| `steps` | 6826 | 2 epochs at 71 episodes — 3,413 steps/epoch |
| `scheduler_warmup_steps` | 200 | LeRobot ships 1000 — see below |
| `num_workers` | 8+ | four video streams is a heavy decode load |

**Change `scheduler_warmup_steps`.** LeRobot's SmolVLA config defaults to 1000. Over a
20 000-step run that is 5% and harmless. Over a short run it is not: at 2 800 steps the
schedule spends 36% of training still ramping up, and the loss curve looks like a
modelling failure when it is a schedule artefact. If you cut the run short, cut warmup
with it.

### Sizing expectations

These are estimates from SmolVLA's parameter count scaled off A100-class throughput.
**They are not measurements.** Measure before committing:

```bash
# time the first 100 logged steps, then multiply by (steps / 100)
time python -m lerobot.scripts.train --config_path=<your config>
```

Rough guide for ~284 000 frames (120 episodes), batch 64: **~30 min/epoch, ~2–3 h for a
20 000-step run.** Treat any number in this section as unverified until the timing run
above replaces it.

### Data loading is the likely bottleneck

Four 320×320 video streams is a large decode load. If the GPU sits below ~70% utilisation
during the first few hundred steps, raise `num_workers`, not `batch_size`. Blaming the
model for a starved dataloader is the most common way to waste a training run here.

Also budget 15–40 minutes **once** before step 1: LeRobot computes normalization
statistics over every frame of every video stream. It is cached afterwards.

### Overfitting

With this little data, the policy will overfit the common object placements and get
*worse* at the tail. If you extend past ~40 000 steps, hold out episodes and check
placement error on them — not just training loss.
