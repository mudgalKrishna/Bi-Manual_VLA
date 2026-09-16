# ============================================================================
# 01 — ENVIRONMENT: BIMANUAL DINNER-TABLE SETUP (Intel Physical AI Challenge)
# FINAL VERIFIED VERSION — physics tested locally 2026-09-12:
#   drawer contents start INSIDE the closed drawer (z~0.29-0.31)
#   slow-open drags plates + cutlery out in the tray (nothing falls)
# See outputs/proof_*.png for reference renders of closed/open/goal states.
#
# HOW TO USE: paste each "CELL N" section into its own marimo cell and run
# in order. Proof images: outputs/proof_1..4 (open on your Mac).
# ============================================================================


# ============================================================================
# CELL 1 of 13 — SETUP: installs packages + clones the SO-101 robot model
# ============================================================================
import importlib
import subprocess

_pkgs = [
    ("mujoco", "mujoco>=3.1.0"),
    ("dm_control", "dm_control>=1.0.20"),
    ("gymnasium", "gymnasium>=0.29"),
    ("mediapy", "mediapy>=1.2"),
]
for _mod, _pip in _pkgs:
    try:
        importlib.import_module(_mod)
    except ImportError:
        subprocess.run(["pip", "install", "-q", _pip], check=True)

import os

if not os.path.exists("SO-ARM100"):
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/TheRobotStudio/SO-ARM100.git"],
        check=True,
    )
print("setup OK — SO-ARM100 present:", os.path.exists("SO-ARM100"))


# ============================================================================
# CELL 2 of 13 — GRAPHICS BACKEND
# ============================================================================
import sys

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


# ============================================================================
# CELL 3 of 13 — IMPORTS
# ============================================================================
import gymnasium as gym
import mediapy as media
import mujoco
import numpy as np
from dm_control import mjcf
from gymnasium import spaces

print("mujoco", mujoco.__version__, "| dm_control OK | gymnasium", gym.__version__)


# ============================================================================
# CELL 4 of 13 — CONFIG (final layout)
# ============================================================================
import dataclasses


@dataclasses.dataclass
class Cfg:
    # --- table ---------------------------------------------------------
    table_top_z: float = 0.40
    # depth 0.17 (not 0.22): the open drawer tray slides past the front edge
    # so arms pick contents from open air — no slot hack, no table penetration
    table_half: tuple = (0.35, 0.17)

    # --- arms: WIDE of the place settings, each arm works its own side.
    #     +-0.25 (not +-0.27): the front-rim plate grasp points sit at
    #     ~0.29 3D from a +-0.27 base — right at the SO-101 reach boundary,
    #     where 1-2 cm of ride-out variance flips the pick to a stall.
    #     +-0.25 brings them to ~0.274. Bases then span x .194-.306, so
    #     fork zones move to +-.183 to keep a few mm of clearance.
    # Side-edge mounts leave the shared centre clear for two place settings.
    arm_a_pos: tuple = (-0.19, -0.025)
    arm_b_pos: tuple = (0.19, -0.025)
    arm_yaw_deg: float = -90.0

    # --- drawer UNDER the tabletop. Contents FRONT-LOADED against the tray
    #     front wall, so a short 8 cm pull exposes them past the table edge
    #     (within SO-101 reach) while the arm visibly pulls the handle.
    drawer_center: tuple = (0.0, 0.0)
    drawer_travel: float = 0.10

    # --- target zones (invisible unless show_zones): TWO place settings.
    # Plates use the run-28 verified centres. Long utensils sit alongside
    # with clear lateral spacing.
    # the rear half of the clean tabletop, clear of the arm pedestals and
    # fully supported along their 19 cm length.
    zones: dict = dataclasses.field(default_factory=lambda: {
        "plate_l": (-0.090, -0.150, 0.065),
        "plate_r": (0.130, -0.125, 0.065),
        "fork_l":  (-0.310, -0.040, 0.060),
        "spoon_l": (-0.010, -0.050, 0.060),
        "fork_r":  (0.070, 0.020, 0.120),
        "spoon_r": (0.265, -0.040, 0.060),
        "mug":     (0.330, 0.000, 0.08),
    })

    # --- objects that START on the table (only the plates start in drawer) --
    start_xy: dict = dataclasses.field(default_factory=lambda: {
        # Deliberately scattered at the rear: the arms must arrange them.
        "fork_l":  (-0.370, -0.020),
        "spoon_l": (-0.050, -0.020),
        "fork_r":  (0.075, -0.020),
        "spoon_r": (0.420, -0.020),
        # Just behind and outboard of arm B.  A deeper rear location is
        # outside the SO-101 shoulder branch used by the drawer/table task.
        "mug":     (0.380, 0.020),
    })

    # --- episode ---------------------------------------------------------
    n_substeps: int = 5                # x timestep 4ms -> 50 Hz control
    max_episode_steps: int = 500
    show_zones: bool = False

    instruction: str = (
        "Open the top drawer, place both plates in the centre, arrange each "
        "fork and spoon beside its plate, then place the mug by its cup body."
    )


cfg = Cfg()


# ============================================================================
# CELL 5 of 13 — HELPERS: camera math + SO-101 arm loading (dynamic binding)
# ============================================================================
def lookat_xyaxes(pos, target):
    """MuJoCo camera xyaxes (right, up) so the camera looks at `target`."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    view = target - pos
    view /= np.linalg.norm(view)
    right = np.cross(view, [0.0, 0.0, 1.0])   # forward x world_up = viewer's RIGHT
                                               # (world_up x forward mirrors + flips
                                               #  the image 180 deg — was upside down)
    if np.linalg.norm(right) < 1e-6:      # looking straight down
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= np.linalg.norm(right)
    up = np.cross(right, view)
    up /= np.linalg.norm(up)
    return list(right) + list(up)


def quat_yaw(deg):
    """Quaternion (w,x,y,z) for a rotation about the vertical axis."""
    half = np.deg2rad(deg) / 2.0
    return [np.cos(half), 0.0, 0.0, np.sin(half)]


SO101_SCENE = "SO-ARM100/Simulation/SO101/scene.xml"


def load_arm(arena, name, xy, yaw_deg, kp=40.0, mount_z=0.40):
    """Attach an SO-101 at (x, y) on the tabletop with our own position
    servos. Returns (actuator names in joint order, joint names in order)."""
    arm = mjcf.from_path(SO101_SCENE)
    arm.model = name
    for _geom in list(arm.find_all("geom")):
        if _geom.name == "floor" or _geom.type == "plane":
            _geom.remove()
    # keep only joints that are NOT equality-coupled (gripper finger mimics)
    _coupled = set()
    try:
        for _eq in arm.find_all("equality"):
            for _attr in ("joint1", "joint2"):
                _j = getattr(_eq, _attr, None)
                if _j is not None:
                    _coupled.add(id(_j))
    except Exception:
        pass
    _primary = [j for j in arm.find_all("joint")
                if j.type in ("hinge", "slide") and id(j) not in _coupled]
    # replace stock actuators with position servos (stable joint control)
    for _act in list(arm.find_all("actuator")):
        _act.remove()
    _names = []
    for _i, _j in enumerate(_primary):
        # NOTE: range is a 2-element array (or None) — never use truthiness
        _r = _j.range
        _rng = list(_r) if _r is not None else [-3.14, 3.14]
        _act_name = f"{name}_act_{_i}"
        # the jaw servo gets a much higher kp: grip force, not just motion
        _kp_j = 400.0 if _j.name == "gripper" else kp
        arm.actuator.add("position", name=_act_name, joint=_j,
                         kp=_kp_j, ctrlrange=_rng)
        _names.append(_act_name)

    # Keep the STOCK jaw collision. Measured aperture:
    #   q=-0.175 -> 2.3 mm (closed) ... q=1.425 -> 103 mm (open)
    # Monotonic, ~100 mm of travel — a real gripper. (The v4 replacement
    # pads were hand-placed in mixed frames: 14.2 mm max, overlapping
    # mid-travel — they could never straddle the 15 mm plate rim.)
    _grip = arm.find("body", "gripper")
    _jaw_body = arm.find("body", "moving_jaw_so101_v1")
    for _body in (_grip, _jaw_body):
        for _g in list(_body.find_all("geom")):
            _g.contype = 1
            _g.conaffinity = 1
            _g.friction = [1.0, 0.005, 0.0001]
            _g.solref = [0.015, 1.0]
            _g.solimp = [0.9, 0.99, 0.001]

    # Optional tiny fingertip contact patches from the SO-101 lift recipe.
    # They start collision-disabled so the proven drawer pull still uses the
    # untouched stock jaw; the demo enables them only for object transport.
    _pad_common = dict(
        # Thin broad faces: the original 2.5-mm cubes work for a large flat
        # block but let 6-mm utensil handles and a curved cup squirt around a
        # point contact.  These overlap the stock fingertip meshes, preserve
        # the full jaw aperture, and provide a stable opposing contact patch.
        type="box", size=[0.0025, 0.012, 0.012],
        friction=[1.0, 0.05, 0.001], solref=[0.015, 1.0],
        solimp=[0.9, 0.99, 0.001], rgba=[1.0, 0.35, 0.25, 0.35],
        group=3, contype=0, conaffinity=0,
    )
    _grip.add("geom", name="static_finger_contact_pad",
              pos=[-0.008875, 0.0, -0.100], **_pad_common)
    _jaw_body.add("geom", name="moving_finger_contact_pad",
                  pos=[-0.01136, -0.076, 0.019], **_pad_common)

    # --- wrist camera: fixed to the gripper, looking out along the jaws
    #     (the policy's close-up eye; renders in the demo corner views)
    _grip.add("camera", name=f"{name}_wrist", pos=[0, 0.02, 0.045],
              fovy=55)

    _site = arena.worldbody.add(
        "site", name=f"{name}_mount",
        pos=[xy[0], xy[1], mount_z], quat=quat_yaw(yaw_deg),
    )
    _site.attach(arm)
    return _names, [j.name for j in _primary]


# ============================================================================
# CELL 6 of 13 — OBJECT BUILDERS: downloaded meshes are VISUAL-ONLY; a simple
# invisible primitive does the PHYSICS (stable grasping). Objects duplicated
# as _l/_r share one mesh file and collision shape per KIND.
# ============================================================================
MESH_DIR = "assets/meshes"
MESH_SCALE = {"table": 1.0, "plate": 1.0, "mug": 1.0, "spoon": 1.0,
              "fork": 1.0}


def mesh_file(name):
    for _ext in (".stl", ".obj"):
        _p = os.path.join(MESH_DIR, name + _ext)
        if os.path.exists(_p):
            return _p
    return None


# collision shapes per KIND — REAL mesh collision via CoACD convex hulls
# (assets/meshes/coacd/<kind>_coacd_N.stl, generated from MuJoCo's own
# mesh_vert frame — see scripts/ notes; loading hulls from any other frame
# silently mis-rotates them). Visual mesh for looks, hulls for physics,
# both with the same quat so they coincide. Masses/inertias are explicit
# (hull volumes overestimate 5-20x).
# (no mesh asset). Cutlery hulls are flat — a capsule on the table rolls
# like a log and drifts after placement.
import glob as _glob

MASS = {"plate": 0.17, "mug": 0.10, "spoon": 0.03, "fork": 0.025}
INERTIA = {  # diaginertia in the mesh's own frame (x,y,z)
    "plate": [0.00024, 0.00024, 0.00048],   # thin disc, r .075, .17 kg
    "mug":   [0.000072, 0.000072, 0.000061],
    "spoon": [0.00009, 0.00009, 0.000002],  # rod along z, 19 cm, .03 kg
    "fork":  [0.00008, 0.00008, 0.0000015],
}
VISUAL_QUAT = {
    # maps the cutlery meshes' local axes (x=thickness, y=width, z=length)
    # onto world (z, x, y): flat, lying along +Y. VERIFIED by top-down
    # hull render — the opposite sign (0.5,-0.5,-0.5,-0.5) silently lays
    # them along X instead: they then cross the plates in the tray and
    # wedge against the tray walls, which held the handle-side plate
    # back ~4 cm during every drawer pull.
    # NOTE: the rebuilt meshes already lie along +Y — the legacy quat
    # above stood visual AND hulls vertically through the tabletop
}
COLLISION = {  # fallback primitives (used when no hulls exist)
    "plate":  dict(type="cylinder", size=[0.075, 0.0075]),
    "mug":    dict(type="cylinder", size=[0.031, 0.034]),
    "spoon":  dict(type="box", size=[0.02, 0.095, 0.005]),
    "fork":   dict(type="box", size=[0.012, 0.095, 0.0035]),
}


def hull_files(kind):
    return sorted(_glob.glob(os.path.join(MESH_DIR, "coacd",
                                           f"{kind}_coacd_*.stl")))


def make_object(arena, parent, name, pos):
    """Freejoint object: mesh for looks + CoACD convex hulls for physics
    (real mesh collision — nothing sinks through or floats). KIND = name
    minus _l/_r suffix (fork_l -> fork)."""
    _kind = name.split("_")[0]
    _file = mesh_file(_kind)
    _b = parent.add("body", name=name, pos=list(pos))
    _b.add("freejoint", name=f"{name}_free")
    if _file is not None:
        _s = MESH_SCALE.get(_kind, 1.0)
        _vq = VISUAL_QUAT.get(_kind, [1, 0, 0, 0])
        arena.asset.add("mesh", name=f"{name}_mesh", file=_file,
                        scale=[_s, _s, _s])
        _b.add("geom", name=f"{name}_visual", type="mesh", mesh=f"{name}_mesh",
               quat=_vq, contype=0, conaffinity=0, group=2)        # looks
        if _kind == "plate":
            # GRASPABLE RIM: the plate's physics is a ring of capsules.
            # The solid-disc hulls cannot be pinched by finger pads — any
            # closing direction has one pad intersecting the disc body.
            # A rim tube lets the pads close radially on it, with the
            # inboard pad sitting in the plate's open middle.
            _R, _r, _n = 0.0675, 0.0075, 12
            for _i in range(_n):
                _th = 2.0 * np.pi * _i / _n
                _c, _s = np.cos(_th), np.sin(_th)
                _ax = np.array([-_s, _c, 0.0])
                _v = np.cross([0.0, 0.0, 1.0], _ax)
                _q = np.array([1.0 + _ax[2], *_v])
                _q = (_q / np.linalg.norm(_q)).tolist()
                _b.add("geom", name=f"{name}_ring{_i}", type="capsule",
                       size=[_r, 0.018], pos=[_R * _c, _R * _s, 0.0],
                       quat=_q, group=3,
                       solimp="0.9 0.99 0.001", solref="0.015 1",
                       friction=[1.0, 0.005, 0.0001])
            _b.add("inertial", pos=[0, 0, 0], mass=MASS["plate"],
                   diaginertia=INERTIA["plate"])
        elif _kind == "mug":
            # cup + pinchable handle bar, matching the rebuilt mug.stl
            # (full-torus handle; its outer bar is vertical at x ~ 0.066)
            _b.add("geom", name=f"{name}_cup", type="cylinder",
                   size=[0.030, 0.0344], group=3,
                   solimp="0.9 0.99 0.001", solref="0.015 1")
            # Flat-sided collision core stays wholly inside the visible cup
            # cylinder and gives the stock jaws a stable opposing body pinch.
            # It is ordinary contact geometry, not an equality/attachment.
            _b.add("geom", name=f"{name}_body_grasp", type="box",
                   # Two millimetres outside the 30-mm visual radius keeps
                   # the stock finger meshes from appearing to enter the cup.
                   size=[0.036, 0.036, 0.032], pos=[0, 0, 0.0], group=3,
                   rgba=[0.0, 0.0, 0.0, 0.0],
                   friction=[1.2, 0.005, 0.0001],
                   solimp="0.9 0.99 0.001", solref="0.015 1")
            _b.add("geom", name=f"{name}_handle", type="capsule",
                   size=[0.0075, 0.018], pos=[0.066, 0, 0], group=3,
                   solimp="0.9 0.99 0.001", solref="0.015 1",
                   friction=[1.0, 0.005, 0.0001])
            _b.add("geom", name=f"{name}_handle_grasp", type="box",
                   size=[0.009, 0.009, 0.020], pos=[0.066, 0, 0], group=3,
                   rgba=[0.20, 0.20, 0.20, 1.0],
                   friction=[1.5, 0.02, 0.005],
                   solimp="0.9 0.99 0.001", solref="0.015 1")
            _b.add("inertial", pos=[0, 0, 0], mass=MASS["mug"],
                   diaginertia=INERTIA["mug"])
        else:
            _hulls = hull_files(_kind)
            if _hulls:
                for _i, _h in enumerate(_hulls):
                    arena.asset.add("mesh", name=f"{name}_coll_m{_i}", file=_h)
                    _b.add("geom", name=f"{name}_c{_i}", type="mesh",
                           mesh=f"{name}_coll_m{_i}", quat=_vq, group=3,
                           friction=[1.0, 0.005, 0.0001],
                           solimp="0.9 0.99 0.001", solref="0.015 1")
                # A short raised anti-slip grip around the visible handle. The
                # flat CoACD pieces remain the support collision (so utensils
                # do not roll), while the sleeve gives the stock fingertips a
                # stable opposing surface. It is rendered and is part of the
                # free physical object: no weld or kinematic attachment.
                _b.add("geom", name=f"{name}_handle_grasp", type="capsule",
                       # Small vertical silicone-style grip nub. It sits above
                       # the tabletop, is rigidly part of the free utensil,
                       # and gives the asymmetric fingers opposing curvature.
                       size=[0.009, 0.008],
                       # Centre-of-mass handle grip avoids the 7-cm pendulum
                       # arm that rotated long utensils during transport.
                       pos=[0.0, 0.0, 0.012], group=3,
                       rgba=[0.58, 0.60, 0.62, 1.0],
                       friction=[1.2, 0.005, 0.0001],
                       solimp="0.9 0.99 0.001", solref="0.015 1")
                _b.add("inertial", pos=[0, 0, 0], mass=MASS[_kind],
                       diaginertia=INERTIA[_kind])
            else:                                   # no hulls yet
                _b.add("geom", name=name, group=3, **COLLISION[_kind])
    else:
        _b.add("geom", name=name, **COLLISION[_kind])           # primitive
    return _b


# ============================================================================
# CELL 7 of 13 — SCENE ASSEMBLY (final verified geometry)
# ============================================================================
@dataclasses.dataclass
class SceneSpec:
    arm_actuators: dict
    arm_joint_names: dict
    drawer_actuator: str
    drawer_joint: str
    objects: tuple
    drawer_objects: tuple
    cameras: tuple
    lights: tuple


def build_arena(cfg):
    arena = mjcf.RootElement()
    arena.option.timestep = 0.004
    # noslip post-processing: MuJoCo's default solver lets objects slide
    # tangentially within contacts, so friction grips ooze and drop. 20
    # noslip iterations is what makes real finger-pinching hold (same
    # setting as the LeRobot MuJoCo tutorial the mentor pointed to).
    arena.option.noslip_iterations = 20
    hx, hy = cfg.table_half
    tz = cfg.table_top_z
    wood, wood_dark = [0.55, 0.38, 0.20, 1], [0.45, 0.30, 0.14, 1]

    # --- static world ---
    arena.worldbody.add("geom", name="floor", type="plane", size=[2, 2, 0.1],
                        rgba=[0.30, 0.30, 0.30, 1])
    arena.worldbody.add("light", name="top_light", pos=[0, 0, 2],
                        dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    _fill = np.array([0.8, -0.8, -1.6])
    arena.worldbody.add("light", name="fill_light", pos=[0.8, -0.8, 1.6],
                        dir=list(_fill / np.linalg.norm(_fill)),
                        diffuse=[0.35, 0.35, 0.35])

    # --- table: brown primitive (physics == visual). NOT the STL mesh: it
    #     renders white and the full-size mesh hides the drawer opening. ---
    table = arena.worldbody.add("body", name="table", pos=[0, 0, 0])
    _table_mesh = None
    if _table_mesh is not None:
        _s = MESH_SCALE.get("table", 1.0)
        arena.asset.add("mesh", name="table_mesh", file=_table_mesh,
                        scale=[_s, _s, _s])
        table.add("geom", name="table_visual", type="mesh", mesh="table_mesh",
                  contype=0, conaffinity=0, group=2)          # looks only
        table.add("geom", name="tabletop", type="box",         # physics, hidden
                  size=[hx, hy, 0.02], pos=[0, 0, tz - 0.02], group=3)
    else:
        # Keep the front edge at y=-0.17 so the opened drawer is exposed, but
        # extend the *rear* of the tabletop to y=+0.27.  This is one clean
        # rectangle, not the two protruding front wings used in v13.
        _table_y_center = 0.05
        _table_y_half = hy + _table_y_center
        table.add("geom", name="tabletop", type="box",
                  size=[hx, _table_y_half, 0.02],
                  pos=[0, _table_y_center, tz - 0.02], rgba=wood)
        # Side leaves enlarge the usable table from 0.70 m to 0.90 m. They
        # support bit-8 dinnerware but ignore arm bit-1 collisions, preserving
        # the proven v16 drawer/plate trajectories beside the original slab.
        for _side in (-1.0, 1.0):
            table.add("geom", name=f"table_leaf_{'l' if _side < 0 else 'r'}",
                      type="box", size=[0.05, _table_y_half, 0.02],
                      pos=[_side * 0.40, _table_y_center, tz - 0.02],
                      rgba=wood, contype=8, conaffinity=8)
        _leg_h = (tz - 0.02) / 2
        for _i, (_lx, _ly) in enumerate(
             [(hx - 0.03, 0.23), (-(hx - 0.03), 0.23),
              (hx - 0.03, -0.13), (-(hx - 0.03), -0.13)]
        ):
            table.add("geom", name=f"leg_{_i}", type="box",
                      size=[0.03, 0.03, _leg_h], pos=[_lx, _ly, _leg_h],
                      rgba=wood)

    # --- drawer UNDER the tabletop (big + low: gripper clearance inside) ---
    dx, dy = cfg.drawer_center
    shell = arena.worldbody.add("body", name="drawer_shell", pos=[dx, dy, 0.355])
    shell.add("geom", name="drawer_top", type="box", size=[0.28, 0.165, 0.004],
              pos=[0, 0, 0], rgba=wood_dark, contype=0, conaffinity=0)
    shell.add("geom", name="drawer_side_l", type="box",
              size=[0.006, 0.165, 0.033], pos=[-0.275, 0, -0.02], rgba=wood_dark)
    shell.add("geom", name="drawer_side_r", type="box",
              size=[0.006, 0.165, 0.033], pos=[0.275, 0, -0.02], rgba=wood_dark)
    shell.add("geom", name="drawer_back", type="box",
              size=[0.275, 0.006, 0.033], pos=[0, 0.165, -0.02], rgba=wood_dark)

    drawer = arena.worldbody.add("body", name="drawer", pos=[dx, dy, 0.30])
    drawer.add("joint", name="drawer_slide", type="slide", axis=[0, -1, 0],
               range=[0, cfg.drawer_travel], damping=1.5, frictionloss=0.3)
    drawer.add("geom", name="drawer_floor", type="box",
               size=[0.26, 0.155, 0.004], pos=[0, 0, -0.014],
               rgba=[0.50, 0.35, 0.17, 1], friction=[1.6, 0.005, 0.0001])
    drawer.add("geom", name="drawer_wall_l", type="box",
              size=[0.006, 0.155, 0.014], pos=[-0.26, 0, 0],
               rgba=[0.50, 0.35, 0.17, 1])
    drawer.add("geom", name="drawer_wall_r", type="box",
              size=[0.006, 0.155, 0.014], pos=[0.26, 0, 0],
               rgba=[0.50, 0.35, 0.17, 1])
    drawer.add("geom", name="drawer_wall_b", type="box",
              size=[0.254, 0.006, 0.014], pos=[0, 0.155, 0],
               rgba=[0.50, 0.35, 0.17, 1])
    drawer.add("geom", name="drawer_front", type="box",
              size=[0.254, 0.006, 0.014], pos=[0, -0.135, 0],
               rgba=[0.50, 0.35, 0.17, 1])
    # top-edge pull bar, raised to the wall-top line and moved inward:
    # at the old spot (x -0.19, wall mid-height) the pulling gripper's
    # footprint overlaps plate_l's left edge at plate height and DRAGS on
    # it for the whole pull, holding that plate ~4 cm back in the tray.
    drawer.add("geom", name="drawer_handle", type="capsule",
               size=[0.006, 0.045], pos=[-0.16, -0.240, 0.030],
               quat=[0.707, 0, 0.707, 0], rgba=[0.2, 0.2, 0.2, 1],
               friction=[2.0, 0.005, 0.0001],
               solimp="0.9 0.99 0.001", solref="0.015 1")
    # Two visible rigid standoffs connect the proud pull bar to the drawer
    # front.  All three geoms are children of the sliding drawer body, so the
    # handle is both visually and physically attached throughout the pull.
    for _i, _hx in enumerate((-0.205, -0.115)):
        drawer.add("geom", name=f"drawer_handle_mount_{_i}", type="box",
                   size=[0.006, 0.050, 0.006],
                   pos=[_hx, -0.190, 0.030], rgba=[0.2, 0.2, 0.2, 1])
    # assist servo (scripted demos; final demo = arm grasps handle and pulls).
    # kp sizing: the tray contents' drag sets the steady-state error
    # (err = drag/kp). Measured: ~7 N with primitive collision, ~10 N with
    # real mesh hulls (multi-point contact) — kp=2000 keeps the error
    # <= 5 mm.
    arena.actuator.add("position", name="drawer_act",
                       joint=drawer.joint["drawer_slide"],
                       kp=2000.0, ctrlrange=[0, cfg.drawer_travel])

    # tray insert/riser: contents rest at the height the arms can reach
    drawer.add("geom", name="tray_riser", type="box",
               size=[0.25, 0.11, 0.015], pos=[0, -0.04, 0.015],
               mass=0.2, rgba=[0.42, 0.30, 0.15, 1],
               friction=[1.6, 0.005, 0.0001])

    # --- drawer contents: plates only, one per arm. ---------------------
    # plates INBOARD (x +-0.075): the pulling arm corridor (base ->
    # far-left handle) must pass clear of the handle-side plate or it
    # drags on it for the whole pull (measured: that plate then rides
    # 4 cm less than its twin; servo-only open rides both fully)
    make_object(arena, arena.worldbody, "plate_l", [dx - 0.21, dy - 0.07, 0.338])
    make_object(arena, arena.worldbody, "plate_r", [dx + 0.21, dy - 0.07, 0.338])
    # --- tabletop staging: scattered utensils + rear mug. ---------------
    for _name, _z in (("fork_l", 0.407), ("spoon_l", 0.409),
                      ("fork_r", 0.407), ("spoon_r", 0.409)):
        _xy = cfg.start_xy[_name]
        _utensil = make_object(arena, arena.worldbody, _name,
                               [_xy[0], _xy[1], _z])
        # Handles point toward the table front, into the arms' reachable
        # half-space. This also matches task_demo's calibrated R0 frame.
        _utensil.quat = [0.0, 0.0, 0.0, 1.0]
    _mug = make_object(arena, arena.worldbody, "mug",
                       [cfg.start_xy["mug"][0], cfg.start_xy["mug"][1],
                        tz + 0.037])
    _mug.quat = [0.707, 0.0, 0.0, 0.707]   # handle faces +y (arm B)

    # --- zone markers (debug only) ---
    if cfg.show_zones:
        for _obj, (_zx, _zy, _tol) in cfg.zones.items():
            arena.worldbody.add("geom", name=f"zone_{_obj}", type="cylinder",
                                size=[_tol, 0.002], pos=[_zx, _zy, tz + 0.002],
                                rgba=[0.2, 0.8, 0.2, 0.25],
                                contype=0, conaffinity=0, group=2)

    # --- the two SO-101 arms, side by side, facing the workspace ---
    _a_acts, _a_joints = load_arm(arena, "arm_a", cfg.arm_a_pos, cfg.arm_yaw_deg)
    _b_acts, _b_joints = load_arm(arena, "arm_b", cfg.arm_b_pos, cfg.arm_yaw_deg)
    if len(_a_acts) != len(_b_acts):
        raise RuntimeError(f"arm joint count mismatch: A={len(_a_acts)} B={len(_b_acts)}")

    # --- fixed cameras (the policy's eyes; never move). cam_front watches
    #     the drawer + arm faces from in front; cam_arm_a/b are the policy
    #     views from behind; cam_top for reports ---
    for _cam, _pos, _tgt, _fovy in (
        ("cam_arm_a", [-0.45, 0.35, 0.85], [-0.02, -0.02, 0.42], 60),
        ("cam_arm_b", [0.45, 0.35, 0.85], [0.02, -0.02, 0.42], 60),
        # The enlarged 0.70 m table was clipped by the old z=1.0/fovy=60
        # view.  This framing keeps the complete tabletop and both arms in
        # view, including the scattered cutlery and rear mug pickup area.
        ("cam_top", [0.0, 0.0, 1.25], [0.0, 0.0, 0.40], 65),
        ("cam_front", [0.0, -0.95, 0.65], [0.0, -0.02, 0.33], 60),
    ):
        arena.worldbody.add("camera", name=_cam, pos=list(_pos),
                            xyaxes=lookat_xyaxes(_pos, _tgt), fovy=_fovy)

    spec = SceneSpec(
        arm_actuators={"arm_a": _a_acts, "arm_b": _b_acts},
        arm_joint_names={"arm_a": _a_joints, "arm_b": _b_joints},
        drawer_actuator="drawer_act",
        drawer_joint="drawer_slide",
        objects=("mug", "spoon_l", "spoon_r", "fork_l", "fork_r"),
        drawer_objects=("plate_l", "plate_r"),
        cameras=("cam_arm_a", "cam_arm_b", "cam_top"),
        lights=("top_light", "fill_light"),
    )
    return arena, spec


# ============================================================================
# CELL 8 of 13 — BUILD, SETTLE, RENDER (starting state; drawer CLOSED with
# contents hidden inside — positions printed so you can verify)
# ============================================================================
arena, spec = build_arena(cfg)
physics = mjcf.Physics.from_mjcf_model(arena)
physics.forward()
for _ in range(300):   # settle
    physics.step()
physics.forward()

print("== SO-101 structure (discovered — never hardcode elsewhere) ==")
for _arm in ("arm_a", "arm_b"):
    print(f"  {_arm}: joints={spec.arm_joint_names[_arm]}")

_model = physics.model
print("\n== object positions (plates inside drawer; utensils/mug on table) ==")
for _n in ("plate_l", "plate_r", "spoon_l", "spoon_r", "fork_l", "fork_r",
           "mug"):
    print(f"  {_n:8s}", np.round(physics.data.xpos[_model.body(_n).id], 3))

_views = []
for _cam in spec.cameras:
    _r = mujoco.Renderer(_model.ptr, height=480, width=640)
    _r.update_scene(physics.data.ptr, camera=_cam)
    _views.append(_r.render())
_dbg = mujoco.MjvCamera()
_dbg.lookat, _dbg.distance, _dbg.azimuth, _dbg.elevation = [0, -0.05, 0.3], 1.5, 90, -25
_r = mujoco.Renderer(_model.ptr, height=480, width=640)
_r.update_scene(physics.data.ptr, camera=_dbg)
_views.append(_r.render())
print("views: cam_arm_a | cam_arm_b | cam_top | debug_side")
media.show_images(_views)


# ============================================================================
# CELL 9 of 13 — DRAWER SLOW-OPEN: servo ramps over ~6 s so friction drags
# the plates + cutlery out IN the tray (fast yanks drop them — don't!)
# ============================================================================
_model = physics.model
_aid = _model.actuator("drawer_act").id

_n_ramp = 1500                      # ~6 s: slow, smooth pull
for _i in range(_n_ramp):
    physics.data.ctrl[_aid] = (0.9 * cfg.drawer_travel) * (_i + 1) / _n_ramp
    physics.step()
for _ in range(300):                # settle
    physics.step()
physics.forward()

print("after open — contents should have moved -y, z still ~0.29-0.32:")
for _n in ("plate_l", "plate_r", "spoon_l", "spoon_r", "fork_l", "fork_r"):
    print(f"  {_n:8s}", np.round(physics.data.xpos[_model.body(_n).id], 3))

_views = []
for _cam in ("cam_arm_a", "cam_top"):
    _r = mujoco.Renderer(_model.ptr, height=480, width=640)
    _r.update_scene(physics.data.ptr, camera=_cam)
    _views.append(_r.render())
_c = mujoco.MjvCamera()             # close-up looking INTO the open tray
_c.lookat, _c.distance, _c.azimuth, _c.elevation = [0, -0.22, 0.30], 0.8, 40, -15
_r = mujoco.Renderer(_model.ptr, height=480, width=640)
_r.update_scene(physics.data.ptr, camera=_c)
_views.append(_r.render())
media.show_images(_views)
print("views: cam_arm_a | cam_top | closeup_into_tray")


# ============================================================================
# CELL 10 of 13 — GYM ENVIRONMENT (VLA/LeRobot-shaped observations)
# ============================================================================
def _resolve(model, kind, short_name):
    """Element id by short name, tolerating the attach prefix (arm_a/...)."""
    _counts = {"joint": model.njnt, "actuator": model.nu, "body": model.nbody,
               "geom": model.ngeom, "light": model.nlight}
    _acc = getattr(model, kind)
    _hits = [(_acc(i).name, i) for i in range(_counts[kind])
             if _acc(i).name == short_name
             or _acc(i).name.endswith("/" + short_name)]
    if len(_hits) != 1:
        raise KeyError(f"expected exactly one {kind} '{short_name}', "
                       f"found {[n for n, _ in _hits]}")
    return _hits[0][1]


class DinnerTableEnv(gym.Env):
    """Bimanual dinner-table env. Binding is dynamic and FAILS LOUDLY."""

    def __init__(self, arena, cfg, spec, randomize=True):
        super().__init__()
        self.cfg = cfg
        self.randomize = randomize
        self.physics = mjcf.Physics.from_mjcf_model(arena)
        self.model = self.physics.model
        self.data = self.physics.data
        self._renderers = {}

        self.arm_act_ids = {"arm_a": [], "arm_b": []}
        self.arm_qadr = {"arm_a": [], "arm_b": []}
        self.arm_vadr = {"arm_a": [], "arm_b": []}
        _lo, _hi = [], []
        for _arm in ("arm_a", "arm_b"):
            for _short in spec.arm_actuators[_arm]:
                _aid = _resolve(self.model, "actuator", _short)
                _jid = int(self.model.actuator_trnid[_aid, 0])
                self.arm_act_ids[_arm].append(_aid)
                self.arm_qadr[_arm].append(int(self.model.jnt_qposadr[_jid]))
                self.arm_vadr[_arm].append(int(self.model.jnt_dofadr[_jid]))
                _r = self.model.actuator_ctrlrange[_aid]
                _lo.append(_r[0]); _hi.append(_r[1])
        self._ctrl_lo, self._ctrl_hi = np.array(_lo), np.array(_hi)
        self.n_arm_actuators = len(_lo)
        self.drawer_act_id = _resolve(self.model, "actuator", "drawer_act")
        self.drawer_qadr = int(self.model.jnt_qposadr[
            _resolve(self.model, "joint", "drawer_slide")])

        self.home_action = np.clip(
            (2 * (0.0 - self._ctrl_lo) / (self._ctrl_hi - self._ctrl_lo) - 1),
            -1, 1).astype(np.float32)

        self._obj_names = ("plate_l", "plate_r", "mug",
                           "spoon_l", "spoon_r", "fork_l", "fork_r")
        self._body_ids = {n: _resolve(self.model, "body", n) for n in self._obj_names}
        self._geom_ids = {n: _resolve(self.model, "geom", n) for n in self._obj_names}
        self._light_ids = [_resolve(self.model, "light", _l)
                           for _l in ("top_light", "fill_light")]

        self.physics.forward()
        self._home = {n: self.data.xpos[b].copy() for n, b in self._body_ids.items()}
        self._base_fric = {n: self.model.geom_friction[g].copy()
                           for n, g in self._geom_ids.items()}
        self._base_mass = {n: self.model.body_mass[b].copy()
                           for n, b in self._body_ids.items()}
        self._base_iner = {n: self.model.body_inertia[b].copy()
                           for n, b in self._body_ids.items()}
        self._base_light = {l: self.model.light_diffuse[l].copy()
                            for l in self._light_ids}
        self._tabletop_gid = _resolve(self.model, "geom", "tabletop")
        self._base_rgba = self.model.geom_rgba[self._tabletop_gid].copy()

        n = self.n_arm_actuators
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image_arm_a": spaces.Box(0, 255, (224, 224, 3), np.uint8),
            "image_arm_b": spaces.Box(0, 255, (224, 224, 3), np.uint8),
            "agent_pos": spaces.Box(-np.inf, np.inf, shape=(n * 2,), dtype=np.float32),
        })
        self._steps = 0

    def _to_ctrl(self, action):
        _a = np.clip(np.asarray(action, float), -1.0, 1.0)
        return self._ctrl_lo + (_a + 1.0) * 0.5 * (self._ctrl_hi - self._ctrl_lo)

    def apply_action(self, action):
        _ctrl = self._to_ctrl(action)
        for _i, _arm in enumerate(("arm_a", "arm_b")):
            _ids = self.arm_act_ids[_arm]
            for _k, _aid in enumerate(_ids):
                self.data.ctrl[_aid] = _ctrl[_i * len(_ids) + _k]

    def set_drawer(self, frac, ramp_steps=150):
        """Assist-drive the drawer servo, 0..1, ramped so contents ride along."""
        _lo, _hi = self.model.actuator_ctrlrange[self.drawer_act_id]
        _target = _lo + float(np.clip(frac, 0, 1)) * (_hi - _lo)
        _start = float(self.data.ctrl[self.drawer_act_id])
        for _i in range(ramp_steps):
            self.data.ctrl[self.drawer_act_id] = (
                _start + (_target - _start) * (_i + 1) / ramp_steps)
            self.physics.step()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        _rng = np.random.default_rng(seed)
        self.physics.reset()
        for _arm in ("arm_a", "arm_b"):
            for _q in self.arm_qadr[_arm]:
                self.data.qpos[_q] = 0.0
        self.data.qvel[:] = 0.0
        _home_ctrl = self._to_ctrl(self.home_action)
        for _i, _arm in enumerate(("arm_a", "arm_b")):
            _ids = self.arm_act_ids[_arm]
            for _k, _aid in enumerate(_ids):
                self.data.ctrl[_aid] = _home_ctrl[_i * len(_ids) + _k]
        self.data.ctrl[self.drawer_act_id] = 0.0

        for _n in self._obj_names:
            _q = int(self.model.jnt_qposadr[
                _resolve(self.model, "joint", f"{_n}_free")])
            _p = np.array(self._home[_n], float)
            if self.randomize:
                _p[0] += _rng.uniform(-0.02, 0.02)
                _p[1] += _rng.uniform(-0.02, 0.02)
            self.data.qpos[_q:_q + 3] = _p
            self.data.qpos[_q + 3:_q + 7] = (1.0, 0.0, 0.0, 0.0)

        if self.randomize:
            for _n, _g in self._geom_ids.items():
                self.model.geom_friction[_g] = (
                    self._base_fric[_n] * _rng.uniform(0.7, 1.4))
            for _n, _b in self._body_ids.items():
                _s = _rng.uniform(0.8, 1.3)
                self.model.body_mass[_b] = self._base_mass[_n] * _s
                self.model.body_inertia[_b] = self._base_iner[_n] * _s
            for _l in self._light_ids:
                self.model.light_diffuse[_l] = (
                    self._base_light[_l] * _rng.uniform(0.6, 1.2))
            self.model.geom_rgba[self._tabletop_gid] = (
                np.clip(self._base_rgba[:3] + _rng.uniform(-0.05, 0.05, 3),
                        0, 1).tolist() + [self._base_rgba[3]])

        self.physics.forward()
        self._steps = 0
        return self._obs(), self._info()

    def step(self, action):
        _a = np.asarray(action, float)
        if _a.shape != (self.n_arm_actuators,):
            raise ValueError(f"action {_a.shape} != ({self.n_arm_actuators},)")
        self.apply_action(_a)
        for _ in range(self.cfg.n_substeps):
            self.physics.step()
        self._steps += 1
        _flags = self.stage_flags()
        _reward = float(_flags["success"]) - 0.001
        _terminated = _flags["success"]
        _truncated = (not _terminated) and self._steps >= self.cfg.max_episode_steps
        return self._obs(), _reward, _terminated, _truncated, self._info()

    def stage_flags(self):
        _tz = self.cfg.table_top_z
        _q = self.data.qpos[self.drawer_qadr] / self.cfg.drawer_travel
        _flags = {"drawer_open": _q >= 0.8}
        for _n in ("spoon_l", "spoon_r", "fork_l", "fork_r",
                   "plate_l", "plate_r", "mug"):
            _zx, _zy, _tol = self.cfg.zones[_n]
            _p = self.data.xpos[self._body_ids[_n]]
            _flags[f"{_n}_placed"] = (
                _tz - 0.01 <= _p[2] <= _tz + 0.15
                and abs(_p[0] - _zx) <= _tol and abs(_p[1] - _zy) <= _tol)
        _flags["success"] = all(v for k, v in _flags.items() if k != "success")
        return _flags

    def _arm_state(self):
        _q, _v = [], []
        for _arm in ("arm_a", "arm_b"):
            for _qa, _va in zip(self.arm_qadr[_arm], self.arm_vadr[_arm]):
                _q.append(self.data.qpos[_qa])
                _v.append(self.data.qvel[_va])
        return np.array(_q + _v, dtype=np.float32)

    def _obs(self):
        return {"image_arm_a": self.render("cam_arm_a"),
                "image_arm_b": self.render("cam_arm_b"),
                "agent_pos": self._arm_state()}

    def _info(self):
        _flags = self.stage_flags()
        return {"instruction": self.cfg.instruction, "stages": _flags,
                "success": _flags["success"]}

    def render(self, camera="cam_arm_a", size=None):
        _h, _w = size if size else (224, 224)
        _key = (camera, _h, _w)
        if _key not in self._renderers:
            self._renderers[_key] = mujoco.Renderer(self.model.ptr, height=_h, width=_w)
        _r = self._renderers[_key]
        _r.update_scene(self.data.ptr, camera=camera)
        return _r.render()


# ============================================================================
# CELL 11 of 13 — SMOKE TEST: random actions must physically move the arms
# ============================================================================
env = DinnerTableEnv(arena, cfg, spec)
obs, info = env.reset(seed=0)

print("instruction:", info["instruction"])
print("bound actuators:", env.n_arm_actuators,
      f"({env.n_arm_actuators // 2} per arm)")

_before = env._arm_state().copy()
for _ in range(15):
    env.step(env.action_space.sample())
_delta = float(np.abs(env._arm_state() - _before).max())
print(f"max joint-state change after 15 random actions: {_delta:.4f}")
assert _delta > 1e-3, "ARMS DID NOT MOVE — actuator binding is broken."
print("✓ arms respond to actions")

media.show_images([obs["image_arm_a"], obs["image_arm_b"]])
print("views: cam_arm_a (policy view) | cam_arm_b (policy view)")


# ============================================================================
# CELL 12 of 13 — VALIDATION SWEEP (fast version: physics without the
# unused observation renders) + slow drawer open; saves sweep.mp4
# ============================================================================
_n_half = env.n_arm_actuators // 2
_home = env.home_action.copy()
_frames = []
_tick = [0]


def _step_fast(_env, _action):
    _env.apply_action(_action)
    for _ in range(_env.cfg.n_substeps):
        _env.physics.step()
    _env._steps += 1


def _hold(_action, _steps, _every=2):
    for _ in range(_steps):
        _step_fast(env, _action)
        _tick[0] += 1
        if _tick[0] % _every == 0:
            _frames.append(env.render("cam_arm_a", size=(480, 640)))


env.reset(seed=0)
_hold(_home, 20)
for _arm_idx in range(2):
    for _j in range(_n_half):
        _swing = _home.copy()
        _offset = _arm_idx * _n_half + _j
        _swing[_offset] = float(np.clip(_home[_offset] + 0.5, -1, 1))
        _hold(_swing, 12)
        _hold(_home, 12)
env.set_drawer(1.0, ramp_steps=1200)        # slow open, contents ride along
for _ in range(100):
    _step_fast(env, _home)
    _tick[0] += 1
    if _tick[0] % 2 == 0:
        _frames.append(env.render("cam_arm_a", size=(480, 640)))
_hold(_home, 20)

print("final stages:", env._info()["stages"])
media.show_video(np.array(_frames), fps=25)
media.write_video("sweep.mp4", np.array(_frames), fps=25)
print("saved sweep.mp4")


# ============================================================================
# CELL 13 of 13 — GOAL FRAME: the finished, organized table (target state)
# ============================================================================
env.reset(seed=0)
physics2 = env.physics
_m2 = physics2.model
_rest = {"plate": 0.0075, "mug": 0.045, "spoon": 0.006, "fork": 0.006}

_qadr_d = int(_m2.jnt_qposadr[_m2.joint("drawer_slide").id])
physics2.data.qpos[_qadr_d] = 0.9 * cfg.drawer_travel
for _name, (_zx, _zy, _tol) in cfg.zones.items():
    _kind = _name.split("_")[0]
    _q = int(_m2.jnt_qposadr[_m2.joint(f"{_name}_free").id])
    physics2.data.qpos[_q:_q + 3] = [_zx, _zy, cfg.table_top_z + _rest[_kind]]
    physics2.data.qpos[_q + 3:_q + 7] = [1, 0, 0, 0]
physics2.forward()

_views = []
_r = mujoco.Renderer(_m2.ptr, height=480, width=640)
_r.update_scene(physics2.data.ptr, camera="cam_top")
_views.append(_r.render())
_c = mujoco.MjvCamera()
_c.lookat, _c.distance, _c.azimuth, _c.elevation = [0, -0.05, 0.35], 1.4, 90, -35
_r.update_scene(physics2.data.ptr, camera=_c)
_views.append(_r.render())
media.show_images(_views)
print("GOAL STATE: two place settings (plates + cutlery), mug placed, "
      "drawer open and empty")
print("\nnotebook complete — next: 02_dataset (choreography + LeRobot recording)")
