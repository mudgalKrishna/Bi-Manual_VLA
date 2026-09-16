# Notebooks

## `01_environment_cells.py`

A 13-cell marimo notebook stored as plain Python with `# CELL N of 13` markers. It is
not a `.ipynb` — the markers are the cell boundaries.

**This file is a runtime dependency, not just documentation.** `scripts/task_demo.py`
opens it at import, cuts cells 2–7 out by splitting on the literal strings
`"# CELL 2 of 13"` and `"# CELL 8 of 13"`, `exec`s them, and reads `cfg`,
`build_arena` and `lookat_xyaxes` out of the resulting namespace.

Consequences:

- **Do not rename or reflow those two markers.** `tests/test_repo_structure.py` checks
  for exactly this; a change produces a `KeyError` on `ns["cfg"]` at runtime rather than
  a clear error.
- **Do not define the same global in two cells.** marimo rejects that, and it is why the
  cells use distinct names throughout.
- Editing cells 2–7 changes the physics of every dataset generated afterwards.

### Cell map

| Cells | Role |
| --- | --- |
| 1 | Install packages, clone SO-ARM100 if absent |
| 2–5 | Graphics backend, imports, config, camera math + arm loading |
| 6–7 | Object builders and scene assembly — **the physics authority** |
| 8–9 | Build/settle/render the start state; slow drawer open |
| 10 | Gym environment with LeRobot-shaped observations |
| 11 | Smoke test: random actions must physically move the arms |
| 12 | Validation sweep |
| 13 | Goal-frame render — the finished table |

Cells 1–7 and 8–13 are independent of each other in practice: the expert uses 2–7, and
the marimo environment checks use 8–13.

### Running it in marimo

```bash
pip install marimo
marimo edit notebooks/01_environment_cells.py
```

Cell 1 clones SO-ARM100 into the working directory and installs packages on first run.
In this repository that clone is unnecessary — the model is already vendored under
`third_party/SO-ARM100/`, and `task_demo.py` points at it directly.

### Exporting reference renders

Cells 8, 9 and 13 render proof images (closed drawer, open drawer, goal state). They
need a working GL backend. On a headless GPU container without EGL, either set
`MUJOCO_GL=osmesa` or accept that these cells will not run — the collector path does not
use them. See [`../docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md).
