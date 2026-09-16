# Third-party notices

The MIT license in the repository root covers the original code and assets in this
project. It does **not** cover the third-party components below, which retain their own
terms.

---

## SO-ARM100 — robot model (Apache-2.0)

**Vendored path:** `third_party/SO-ARM100/`
**Upstream:** https://github.com/TheRobotStudio/SO-ARM100
**License:** Apache License 2.0 — full text at `third_party/LICENSE-APACHE-2.0`

Vendored contents: the SO-101 MJCF/URDF descriptions, the STL and `.part` meshes they
reference, `scene.xml`, and `joints_properties.xml`.

Modifications made for this project:

- The robot model is loaded from a fixed relative path under `third_party/`; upstream
  paths are not edited, but the loader in `task_demo.py` rewrites the reference to
  `SO-ARM100/Simulation/SO101/scene.xml` to point at this vendored copy.
- Two instances are placed and oriented for a bimanual tabletop task.

Upstream notes worth keeping in view (from their own README):

- Model files were generated with `onshape-to-robot` from an Onshape CAD model.
- Motor properties for the STS3215 are adapted from the
  [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini).
- Base collision meshes were removed upstream due to problematic collision behaviour.
- LeRobot represents the gripper as a linear joint where 0 is closed and 100 is fully
  open; that mapping is **not** reflected in the URDF/MuJoCo files.

### Required attribution

Per Apache-2.0 §4, redistributions must retain this notice and state changes. If you
redistribute this repository or a derivative, keep `third_party/LICENSE-APACHE-2.0` and
this file intact.

---

## Python dependencies

Not vendored — installed from PyPI. Each retains its own license.

| Package | License | Used for |
| --- | --- | --- |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 | CPU contact physics |
| [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | Apache-2.0 | GPU batch rendering |
| [dm_control](https://github.com/google-deepmind/dm_control) | Apache-2.0 | MJCF authoring |
| [LeRobot](https://github.com/huggingface/lerobot) | Apache-2.0 | Dataset format and recording |
| [Warp](https://github.com/NVIDIA/warp) | Apache-2.0 | CUDA kernel runtime |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | MIT | Environment interface |
| [SciPy](https://github.com/scipy/scipy) | BSD-3-Clause | Bounded numerical optimization |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause | arrays |
| [mediapy](https://github.com/google/mediapy) | Apache-2.0 | video/image helpers |
| [PyAV](https://github.com/PyAV-Org/PyAV) | BSD-3-Clause | video encoding |
| [PyArrow](https://github.com/apache/arrow) | Apache-2.0 | LeRobot metadata |

`rclone` is used as a binary for Drive upload and is MIT licensed, but is not bundled.

---

## Tableware meshes

`assets/meshes/*.stl` and `assets/meshes/coacd/*.stl` — original to this project, MIT,
same as the root license.

They are **visual only**. Collision geometry for the expert is analytic and constructed
in the scene builder; the `coacd/` convex decompositions are retained for reference and
are not loaded by the current pipeline.
