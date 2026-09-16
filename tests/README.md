# Tests

## What is here

`test_repo_structure.py` — structural checks. No MuJoCo, no GPU, no network. Runs
anywhere Python runs.

```bash
python tests/test_repo_structure.py
```

It verifies:

1. Every required file is present.
2. Every `.py` file in the repository parses.
3. The `# CELL 2 of 13` / `# CELL 8 of 13` markers still exist, cells 2–7 still parse,
   and they still define `cfg`, `build_arena` and `lookat_xyaxes` — the three names
   `task_demo.py` reads out of the exec namespace.
4. The two collectors agree on the camera schema, and the Warp script's `CAMERAS` and
   `CAMERA_KEYS` tuples are parallel and equal in length.
5. `observation.state` and `action` are both 12-wide.
6. The Warp collector still derives its features from `record_lerobot_shard.FEATURES`
   rather than duplicating them, which is what keeps the two schemas from drifting.

## What is NOT here

**Nothing verifies physics.** That is deliberate — a structural test that pretends to
check simulation behaviour is worse than no test, because it produces false confidence.

The physics check is running the expert:

```bash
python scripts/task_demo.py
```

It must complete and report success. If it does not, everything downstream inherits the
failure.

Rendering is checked by the pilot in
[`../docs/REPRODUCE.md`](../docs/REPRODUCE.md#step-3-verify-the-pilot), which requires a
human to look at four frames. It is not automatable in a meaningful way: the failure mode
is a dataset that is structurally valid, trains without error, and teaches the policy
nothing because the cameras never moved.

## Why not pytest

The checks are cheap, order-independent and produce a readable report. If you want them
under pytest, the `check()` calls map to asserts directly — but keep the "physics is not
covered here" line in the output. It is the most important thing this file says.
