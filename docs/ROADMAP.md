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
- [x] Demonstration dataset — 71 episodes, 218,384 frames, 25 Hz, LeRobot v3
- [x] Ten language paraphrases over the task instruction
- [x] SmolVLA fine-tune to step 4,000 of 6,826 — checkpoint loads and runs
- [x] Policy inference on Intel hardware — emits a valid 12-D action
- [x] OpenVINO runtime sees CPU, GPU and NPU on the target platform

## In progress

- [ ] SmolVLA fine-tune — 4,000 / 6,826 steps (58.6%)
- [ ] CPU / GPU / NPU benchmark sweep with energy per inference
- [ ] Demonstration generator stability — 7 of 10 seeds diverge during collection

## Not started

- [ ] OpenVINO IR conversion and quantization. Device detection is confirmed; the
      model has not been converted.
- [ ] Held-out evaluation of the trained policy. Planning to report per-object
      placement error against the 8 mm threshold the generator already uses, not
      just training loss.
- [ ] Rendered video capture for the accepted episodes.
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

**Language variation is template-level.** The dataset carries ten paraphrases of the
task instruction, so the `task` field is not constant. The paraphrases are authored,
not collected, and they describe the same task — a policy trained on them should be
robust to phrasing, not to a genuinely different goal. Reporting generalization to
novel tasks would require instructions this dataset does not contain.

**The expert is not a controller.** `task_demo.py` is a scripted demonstration
generator. It reads simulator state freely and is not a deployable policy. Do not
cite its success rate as a policy result.

## Deliberately out of scope

- Pouring, hand-to-hand transfers, deformable objects
- More than two place settings
- Multi-task or multi-scene generalization
- Sim-to-real transfer
