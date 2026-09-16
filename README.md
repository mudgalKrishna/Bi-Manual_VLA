# Bi-Manual VLA

**A reproducible pipeline for generating a bimanual SO-101 dinner-table manipulation dataset, and fine-tuning a Vision-Language-Action policy on it.**

Two SO-101 arms set a dinner table in MuJoCo: they place a plate, mug, cutlery and a
bottle, and open a drawer. A scripted expert produces physically valid demonstrations;
those are recorded into a [LeRobot v3](https://github.com/huggingface/lerobot) dataset
and used to fine-tune [SmolVLA](https://huggingface.co/blog/smolvla).

The distribution backend is deliberately split:-

| Stage | Runs on | Why |
| --- | --- | --- |
| Contact physics | CPU MuJoCo | The v21 grasp needs `noslip` contact iterations; MJWarp does not implement them |
| Image rendering | GPU via MuJoCo Warp | Ray-traced batch rendering, **no EGL and no OpenGL** |

That split is the point of the project. Container runtimes that expose a CUDA GPU but
no working GPU OpenGL — which is most of them, including Molab — cannot use the
standard `MUJOCO_GL=egl` path. MuJoCo Warp renders through CUDA instead, so the
pipeline runs unchanged where an OpenGL-based collector would fall back to a CPU
rasteriser. See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

---

## Status

| Component | State |
| --- | --- |
| v21 scripted expert (`scripts/task_demo.py`) | Verified — physics tested, produces validated rollouts |
| Molab GPU collector (`scripts/record_molab_mjwarp.py`) | Working — 10 parallel workers, 320×320, AV1 |
| Kaggle CPU collector (`scripts/record_lerobot_shard.py`) | Working — 4-worker sharded fallback |
| Merge + validation (`scripts/merge_molab_episodes.py`) | Working |
| Dataset | **Generation in progress — no published episode count yet** |
| SmolVLA fine-tune | **Not started** |
| Physical robot deployment | Out of scope |

No success rate, frame count, or training result is claimed here. When the dataset is
final, replace this section with the measured numbers and the merge report — see
[`docs/DATASET.md`](docs/DATASET.md#reporting-results) for exactly which fields to cite.

---

## Repository layout

```
Bi-Manual_VLA/
├── notebooks/
│   └── 01_environment_cells.py    marimo cells 1–13; cells 2–7 build the scene
├── scripts/
│   ├── task_demo.py               v21 scripted expert — the only physics authority
│   ├── record_molab_mjwarp.py     GPU collector: CPU physics + Warp rendering
│   ├── record_lerobot_shard.py    CPU collector: sharded Kaggle fallback
│   └── merge_molab_episodes.py    episode checkpoints → one LeRobot v3 dataset
├── assets/meshes/                 tableware; coacd/ holds collision decompositions
├── third_party/SO-ARM100/         vendored SO-101 model (Apache-2.0)
├── docs/                          architecture, dataset schema, reproduction
└── tests/                         smoke checks against a real MuJoCo build
```

**`notebooks/01_environment_cells.py` is not documentation.** `task_demo.py` reads it at
import time, `exec`s cells 2–7, and pulls `cfg`, `build_arena` and `lookat_xyaxes` out of
the resulting namespace. Editing those cells changes the physics. Cells 8–13 are the
standalone marimo environment and rendering checks.

---

## Quickstart

Requires Python 3.10–3.12 and a working MuJoCo. **No GPU is needed for the expert.**

```bash
git clone https://github.com/mudgalKrishna/Bi-Manual_VLA.git
cd Bi-Manual_VLA
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Reproduce the reference rollout — one episode, no rendering, no randomization:

```bash
python scripts/task_demo.py
```

That is the exact configuration used to validate the v21 grasp. If it completes and
reports success, the physics half of the pipeline is working. To confirm the GPU render
path before committing to a long collection run:

```bash
python scripts/record_molab_mjwarp.py \
    --output-root /tmp/v21-pilot \
    --episode-start 0 --episodes 1 --physics-workers 1
```

Then inspect `dinner_table_dataset/episodes/episode_000000/` — the episode must contain
four camera streams, and **the arms and the camera views must both move**. A static
camera across all frames is the failure mode that produces a dataset which trains but
does not learn. See [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

Full instructions, including the 10-worker Molab fan-out and the Kaggle path:
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

---

## Dataset

LeRobot v3, 25 Hz, one episode per accepted rollout.

| Key | Shape | Notes |
| --- | --- | --- |
| `observation.images.front` | 320×320×3 | overhead |
| `observation.images.arm_a` | 320×320×3 | left wrist |
| `observation.images.arm_b` | 320×320×3 | right wrist |
| `observation.images.top` | 320×320×3 | top-down |
| `observation.state` | (12,) | 6 joints × 2 arms |
| `action` | (12,) | joint position targets |

Videos are AV1 (`libsvtav1`), CRF 18, keyframe interval 25 (one per second). CRF 30 was
tried first and visibly softened thin objects — forks and spoons lost the edges the
grasp depends on.

Schema, normalization, and the SmolVLA fine-tune recipe:
[`docs/DATASET.md`](docs/DATASET.md).

---

## Design notes

**Rejected episodes are kept, not deleted.** A rollout that fails the expert's own
success check is written to `manifest.json` under `rejected` with
`"accepted_for_training": false`. The training set contains only accepted episodes, but
the rejection rate stays visible and auditable.

**Every episode is an atomic checkpoint.** Each accepted rollout is finalised as its own
single-episode LeRobot dataset, uploaded immediately, and merged only at the end. A
crashed or preempted worker loses at most one episode, and parallel workers never touch
shared metadata.

**Rendering is not physics.** The Warp renderer calls `fwd_kinematics`, not `forward` —
it needs transforms, not collision detection. Contact behaviour comes exclusively from
the CPU MuJoCo solve in `task_demo.py`.

---

## Licensing and attribution

Original code and assets: [MIT](LICENSE).

The vendored SO-101 robot model under `third_party/SO-ARM100/` is **Apache-2.0** from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) and is not
covered by the MIT license. See [`third_party/NOTICE.md`](third_party/NOTICE.md).

Third-party Python dependencies (MuJoCo, MuJoCo Warp, dm_control, LeRobot) retain their
own licenses.
