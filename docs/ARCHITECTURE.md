# Architecture

## Pipeline

```
notebooks/01_environment_cells.py        cells 2–7: cfg, build_arena, lookat_xyaxes
                │
                │  exec'd at import by task_demo.py
                ▼
        scripts/task_demo.py             CPU MuJoCo · contact-only expert
                │
                │  DEMO_TRAJECTORY_ONLY=1  →  qpos, state, action, scene.mjb
                ▼
   ┌────────────┴─────────────┐
   │                          │
   ▼                          ▼
record_molab_mjwarp.py    record_lerobot_shard.py
CPU physics + GPU render   CPU render (sharded)
   │                          │
   └────────────┬─────────────┘
                ▼
      episode_000000/lerobot_dataset/     atomic, single-episode checkpoints
                │
                ▼
      merge_molab_episodes.py
                │
                ▼
      merged/dinner_table_smolvla_v1/     the trainable LeRobot v3 dataset
```

## Why physics and rendering are split

The v21 grasp depends on MuJoCo's `noslip` contact iteration — a solver pass that makes
the stock jaw pads hold an object by friction alone, without any pose attachment. MuJoCo
Warp does not implement `noslip`, so MJWarp cannot reproduce the grasp.

Rendering, however, is pure kinematics: given `qpos`, produce pixels. MJWarp's ray-traced
batch renderer does that on CUDA without touching contact or the solver.

So the split is not a performance compromise. It is the only arrangement in which both
halves are correct:

- **Contacts, friction, jaw force, drawer drag** — CPU MuJoCo, the validated solver.
- **Cameras** — GPU MuJoCo Warp, `fwd_kinematics` + `render`.

`WarpRenderer.__init__` sets `mjm.opt.noslip_iterations = 0` to make the render-only
intent explicit: the device model carries no solver state that matters.

## The expert

`scripts/task_demo.py` is a **scripted** expert. It is not a policy and not a controller
for deployment — it exists to generate demonstrations.

Every acquisition is a guarded physical pinch:

```
reach → pads straddle the object → jaw closes → lift
      → VERIFY the object followed (honest DROPPED reporting)
      → carry → place → open
```

Grasp targets are chosen per object so that a two-finger pinch is physically possible:

| Object | Grasp target |
| --- | --- |
| Plate | Rim **tube** — the plate's collision is a ring of capsules; pads close radially |
| Cutlery | Raised centre-handle grip on the free utensil body |
| Mug | The physical handle capsule |
| Drawer | Arm A pinches the handle bar and pulls; the actuator only follows measured joint position |

Transport uses **only** contacts, jaw force and friction. No object pose follows the
gripper. If the object slips out, the expert reports failure rather than hiding it — this
is what `DEMO_STRICT_SUCCESS` enforces and what populates `manifest["rejected"]`.

Mesh files under `assets/meshes/` are **visual only**. Collision geometry is analytic and
built in the notebook's object-builder cell; the `coacd/` convex decompositions are
retained for reference and future use.

## Parallelism model

Two independent axes, and it is worth keeping them straight:

- **`--physics-workers`** — threads *within one session*. Each runs a separate
  `task_demo.py` subprocess. Default 3, sized for a small container.
- **`--worker-id`** — separate sessions running concurrently. The Molab deployment runs
  10, each with a disjoint `--episode-start` / `--episodes` range and its own
  `--drive-prefix .../workers/worker_NN`.

Workers never share a manifest. Because each episode is an atomic checkpoint keyed by a
globally unique id, the merge step is a set union, not a reconciliation.

Rendering happens on the collector's main thread after each trajectory completes, in
batches of `--render-batch`, using a single lazily-created `WarpRenderer` reused across
episodes. The first episode therefore absorbs shader/kernel warm-up — treat its timing as
an outlier, not as the per-episode cost.

## Failure handling

| Failure | Response |
| --- | --- |
| Expert rollout fails | Retry up to `--max-attempts`, seed offset by `attempt * 100_000` |
| Retries exhausted | Record in `manifest["rejected"]`, continue with other episodes |
| Worker crash | Lose at most the in-flight episode; completed ones are already on Drive |
| Merge finds wrong episode count | Hard error — never merge a partial dataset |

`merge_molab_episodes.py` validates each checkpoint (`total_episodes == 1`, `fps == 25`)
and re-checks the merged episode count against `--expected` before writing a report. A
silent partial merge is the failure it exists to prevent.

## Known constraints

- The task preset distribution is **fixed**. Object start positions are randomized within
  a calibrated region, not over the full reachable workspace.
- Rendering is MuJoCo Warp's rasteriser, not the OpenGL path. Pixels will not match a
  dataset generated with `MUJOCO_GL=egl` at the same seed.
- The Docker/container path assumes a CUDA GPU visible to Warp. On a machine with working
  EGL, the standard MuJoCo renderer is simpler — see `docs/TROUBLESHOOTING.md`.
