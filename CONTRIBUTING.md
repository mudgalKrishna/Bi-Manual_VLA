# Contributing

## Before you open a PR

```bash
python tests/test_repo_structure.py     # structure; runs anywhere
python scripts/task_demo.py             # physics; needs MuJoCo, no GPU
```

Both must pass. The second one is not optional if you touched anything under
`notebooks/` — including comments, because `task_demo.py` splits that file on literal
marker strings.

## Things that will break the pipeline

Named so you can recognize them in review:

| Change | What breaks |
| --- | --- |
| Renaming or reflowing `# CELL N of 13` markers | `task_demo.py` gets a namespace without `cfg` and dies with a bare `KeyError` |
| Defining one global in two notebook cells | marimo rejects the notebook; the split may also pick up the wrong definition |
| Replacing the plate's capsule-ring collision with a mesh | The v21 rim-tube grasp stops working; every dataset generated after is invalid |
| Changing `--image-size` | The dataset schema changes; old and new episodes are no longer mergeable |
| Using `forward` instead of `fwd_kinematics` in the Warp renderer | Collision/solver kernel compilation can fail on Blackwell + gVisor runtimes |
| Adding a camera to one collector only | The two paths produce incompatible datasets that fail to merge |
| Raising `crf` in the video encoder | Thin objects lose the edges the grasp depends on |

## Changing the expert

`scripts/task_demo.py` is the physics authority. Any change to it invalidates every
previously generated episode — you cannot append new episodes to an old dataset.

When you change it:

1. Re-run the full `python scripts/task_demo.py` reference rollout (seed 0, no
   randomization).
2. Regenerate the pilot and complete the four-frame check in
   `docs/REPRODUCE.md`.
3. Start a **new** dataset version. Do not extend an existing one.

## Changing the dataset schema

Any of these is a breaking change and needs a new version directory, not an edit:

- camera count, names, or image size
- `observation.state` / `action` width or joint ordering
- fps

Record the change in `docs/DATASET.md` and state clearly whether previously published
datasets remain valid.

## Reporting results

When you add measured numbers anywhere in the docs:

- Numbers come from `MERGE_REPORT.json` and `manifest.json`, not from a planned run.
- Do not drop rejected episodes from a reported count — report both accepted and
  rejected.
- Do not present an estimate as a measurement. If a figure is extrapolated, say so in
  the same sentence.

A dataset that reports only its successes has no measurable quality, and a benchmark
that reports a constant as a measurement is worse than no benchmark.

## Style

Match the surrounding code. The existing scripts are compact and comment where the
reason is non-obvious — why `fwd_kinematics` rather than `forward`, why CRF 18 rather
than 30. Keep those comments; they are the expensive part of the file.
