# Mesh assets

## Layout

```
meshes/
├── plate.stl   mug.stl   fork.stl   spoon.stl   table.stl
└── coacd/
    ├── plate_coacd_0..7.stl
    ├── mug_coacd_0..7.stl
    ├── fork_coacd_0..2.stl
    └── spoon_coacd_0..5.stl
```

## These are visual meshes only

The expert does **not** use these files for collision. Collision geometry is analytic and
constructed in the scene builder (`notebooks/01_environment_cells.py`, object builders
cell).

This matters for the plate specifically. The plate's collision is a **ring of capsules**
forming its rim, not a mesh. That is what makes the v21 grasp possible: the finger pads
close radially on the rim tube, and the inboard pad sits inside the plate's open middle
without colliding with it. Swapping in a mesh collider will break the grasp.

Grasp geometry per object is documented in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## `coacd/`

Convex decompositions produced by [CoACD](https://github.com/SarahWeiii/CoACD), retained
for reference and for future collision work. **Not loaded by the current pipeline.**

If you do wire them in, re-validate the grasp before generating any dataset — the plate
decomposition in particular will not reproduce the rim-tube behaviour the expert depends
on.

## Regenerating

The `coacd/` files were generated from the corresponding `*.stl` with default CoACD
settings and approximately 8 parts per object (fewer for the thin fork). Keep the
`_coacd_N` naming if you regenerate; nothing in the pipeline globs these, so the count is
free to change.

## Licensing

Original to this project, MIT — see [`LICENSE`](../../LICENSE). The SO-101 robot meshes
are separate and Apache-2.0; see [`third_party/NOTICE.md`](../../third_party/NOTICE.md).
