# Roadmap

Honest state of the project, including what does not work.

## Done

- [x] v21 scripted expert — contact-only bimanual grasping, no pose attachment
- [x] MuJoCo scene: two SO-101 arms, table, plate, mug, cutlery, pull drawer
- [x] GPU render path via MuJoCo Warp, no EGL/OpenGL required
- [x] LeRobot v3 recording with per-episode atomic checkpoints
- [x] Parallel collectors: 10-worker GPU fan-out, 4-worker CPU shards
- [x] Merge with hard validation and a written merge report
- [x] Rejection accounting — failed rollouts retained, never silently dropped

## In progress

- [ ] Dataset generation — no published episode count yet
- [ ] Pilot verification on the target hardware (see `docs/REPRODUCE.md` step 3)

## Not started

- [ ] SmolVLA fine-tune. The recipe sketch and its open questions are in
      `docs/DATASET.md`. The unresolved items are the four-camera-to-three-camera
      remap and the warmup schedule.
- [ ] Held-out evaluation of the trained policy. Planning to report per-object
      placement error against the 8 mm threshold the expert already uses, not
      just training loss.
- [ ] Published dataset card with measured frame counts and rejection rate.

## Known limitations

These are properties of the current design, not bugs to be filed.

**Fixed task distribution.** Object start positions are randomized within a
calibrated region, not over the full reachable workspace. A policy trained on
this data will not generalize to arbitrary object placement. Reporting otherwise
would require an evaluation over the full workspace, which has not been run.

**Renderer-specific pixels.** The Warp renderer and the OpenGL renderer produce
different images from identical physics. A dataset built here will not match one
built with `MUJOCO_GL=egl` at the same seed. Compare policies only within one
render path.

**Simulation only.** No physical SO-101 has run this policy. Sim-to-real transfer
is out of scope and untested.

**No language conditioning.** The task instruction is a fixed string. The dataset
carries it as the `task` field for schema compatibility, but there is no language
variation to learn from. SmolVLA will accept the field and learn nothing from it.

**The expert is not a controller.** `task_demo.py` is a scripted demonstration
generator. It reads simulator state freely and is not a deployable policy. Do not
cite its success rate as a policy result.

## Deliberately out of scope

- Pouring, hand-to-hand transfers, deformable objects
- More than two place settings
- Real-time inference or latency work
- Multi-task or multi-scene generalization
