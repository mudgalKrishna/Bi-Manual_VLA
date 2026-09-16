"""Structural checks — run these first, they need no MuJoCo and no GPU.

These catch the failure modes that have actually cost time on this project:
a missing asset, a renamed notebook marker that breaks the split in
``task_demo.py``, and a camera schema that drifted between the two collectors.

They do NOT verify physics. The physics check is running the expert itself:

    python scripts/task_demo.py

Run:  python tests/test_repo_structure.py
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]

# task_demo.py splits the notebook on these two markers. If either is edited,
# reflowed or renamed, the exec produces a namespace without `cfg` and the
# expert dies with a bare KeyError.
MARKER_EARLY = "# CELL 2 of 13"
MARKER_LATE = "# CELL 8 of 13"

REQUIRED = [
    "scripts/task_demo.py",
    "scripts/record_molab_mjwarp.py",
    "scripts/record_lerobot_shard.py",
    "scripts/merge_molab_episodes.py",
    "notebooks/01_environment_cells.py",
    "assets/meshes/plate.stl",
    "assets/meshes/mug.stl",
    "assets/meshes/fork.stl",
    "assets/meshes/spoon.stl",
    "assets/meshes/table.stl",
    "third_party/SO-ARM100/Simulation/SO101/scene.xml",
    "third_party/SO-ARM100/Simulation/SO101/so101_new_calib.xml",
    "third_party/LICENSE-APACHE-2.0",
]

failures: list[str] = []
notes: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


# --- 1. required files ------------------------------------------------------
missing = [rel for rel in REQUIRED if not (REPO / rel).is_file()]
check(not missing, f"missing required files: {missing}")

# --- 2. every Python file compiles ------------------------------------------
for path in sorted(REPO.rglob("*.py")):
    if ".venv" in path.parts or "__pycache__" in path.parts:
        continue
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        failures.append(f"{path.relative_to(REPO)}: syntax error: {exc}")

# --- 3. the notebook split still works --------------------------------------
notebook = REPO / "notebooks/01_environment_cells.py"
if notebook.is_file():
    text = notebook.read_text(encoding="utf-8")
    check(MARKER_EARLY in text, f"notebook lost its {MARKER_EARLY!r} marker")
    check(MARKER_LATE in text, f"notebook lost its {MARKER_LATE!r} marker")
    if MARKER_EARLY in text and MARKER_LATE in text:
        body = MARKER_EARLY + text.split(MARKER_LATE)[0].split(MARKER_EARLY)[1]
        try:
            tree = ast.parse(body)
        except SyntaxError as exc:
            failures.append(f"notebook cells 2-7 do not parse: {exc}")
        else:
            defined = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            }
            defined |= {
                target.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            for needed in ("cfg", "build_arena", "lookat_xyaxes"):
                check(
                    needed in defined,
                    f"cells 2-7 no longer define {needed!r}, which task_demo.py "
                    f"reads out of the exec namespace",
                )

# --- 4. camera schema agrees across collectors ------------------------------
def camera_keys(path: Path, name: str) -> list[str] | None:
    """The tuple bound to `name` in a collector, without importing it."""
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            if isinstance(node.value, (ast.Tuple, ast.List)):
                values = [
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant)
                ]
                if len(values) == len(node.value.elts):
                    return sorted(values)
    return None


def image_keys_in_features(path: Path) -> list[str] | None:
    """Keys of the FEATURES dict starting with 'observation.images.'."""
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "FEATURES" for t in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            return None
        keys = [
            k.value
            for k in node.value.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        ]
        return sorted(k for k in keys if k.startswith("observation.images."))
    return None


def feature_shape(path: Path, key: str) -> tuple[int, ...] | None:
    """The 'shape' tuple of one entry in the FEATURES dict."""
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "FEATURES" for t in node.targets
        ):
            continue
        for entry_key, entry_value in zip(node.value.keys, node.value.values):
            if not (
                isinstance(entry_key, ast.Constant)
                and entry_key.value == key
                and isinstance(entry_value, ast.Dict)
            ):
                continue
            for k, v in zip(entry_value.keys, entry_value.values):
                if (
                    isinstance(k, ast.Constant)
                    and k.value == "shape"
                    and isinstance(v, ast.Tuple)
                ):
                    return tuple(
                        e.value for e in v.elts if isinstance(e, ast.Constant)
                    )
    return None


warp_script = REPO / "scripts/record_molab_mjwarp.py"
# CAMERAS holds MuJoCo camera names; CAMERA_KEYS holds the LeRobot feature
# suffixes. They are parallel tuples and must stay the same length and order.
mujoco_cams = camera_keys(warp_script, "CAMERAS")
lerobot_cams = camera_keys(warp_script, "CAMERA_KEYS")
shard_cams = image_keys_in_features(REPO / "scripts/record_lerobot_shard.py") or []

check(
    lerobot_cams is not None,
    "record_molab_mjwarp.py no longer defines CAMERA_KEYS",
)
if lerobot_cams and mujoco_cams:
    check(
        len(mujoco_cams) == len(lerobot_cams),
        f"CAMERAS ({len(mujoco_cams)}) and CAMERA_KEYS ({len(lerobot_cams)}) "
        f"are parallel tuples and must have equal length",
    )
    check(
        len(lerobot_cams) == 4,
        f"expected 4 cameras on the Warp path, found {len(lerobot_cams)}: {lerobot_cams}",
    )

if lerobot_cams and shard_cams:
    warp_features = sorted(f"observation.images.{c}" for c in lerobot_cams)
    check(
        warp_features == shard_cams,
        f"collector camera schemas disagree: warp={warp_features} shard={shard_cams}",
    )
    notes.append(f"cameras agree across collectors: {warp_features}")

# The Warp collector deep-copies FEATURES from the shard collector and only
# rewrites the image size, so the joint width has to be checked at the source.
for key in ("observation.state", "action"):
    shape = feature_shape(REPO / "scripts/record_lerobot_shard.py", key)
    check(
        shape == (12,),
        f"{key} should be 12-wide (two 6-joint arms), found shape {shape}",
    )
    if shape == (12,):
        notes.append(f"{key} shape is (12,)")

warp_text = (REPO / "scripts/record_molab_mjwarp.py").read_text(encoding="utf-8")
check(
    "BASE_FEATURES" in warp_text,
    "record_molab_mjwarp.py no longer derives its features from "
    "record_lerobot_shard.FEATURES — the two schemas can now drift apart",
)

# --- report -----------------------------------------------------------------
print(f"repository root : {REPO}")
print(f"checks failed   : {len(failures)}")
for note in notes:
    print(f"  note: {note}")
for failure in failures:
    print(f"  FAIL: {failure}")

if failures:
    print("\nSTRUCTURE CHECK FAILED")
    sys.exit(1)

print("\nStructure OK.")
print("Physics is not covered here. Verify it with:  python scripts/task_demo.py")
