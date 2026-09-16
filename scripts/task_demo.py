"""Scripted task demo v21 — contact-only expert demonstration generator.

Every acquisition starts as a physical, guarded stock-jaw pinch:

    reach -> pads straddle the object -> jaw closes -> lift -> VERIFY the
    object followed (honest DROPPED reporting) -> carry -> place -> open

Grasp targets (all circular cross-sections, pinchable by finger pads):
  plates   : rim TUBE (plate collision = ring of capsules) — pads close
             radially, inboard pad sits in the plate's open middle
  cutlery  : a raised centre-handle grip on the free utensil body
  mug      : the physical handle capsule
  drawer   : arm A pinches the handle bar and PULLS the drawer open;
             the drawer actuator only follows the measured joint position

The default v21 mode uses only MuJoCo contacts, jaw force, and friction during
transport. No object pose follows the gripper. This is an offline scripted
expert, not the final controller: competition inference is intended to use
SmolVLA for direct 12-joint actions.

Video: 1280x480 composite — front view centre, arm A/B wrist cams in the top
corners. Dataset: both policy cams 224x224 + state + action @ 25 Hz.
"""
import os
import sys
import json

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "glfw")

import mediapy as media          # noqa: E402
import mujoco                    # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from dm_control import mjcf      # noqa: E402

# ---------------------------------------------------------------- scene
src = open(os.path.join(REPO, "notebooks", "01_environment_cells.py")).read()
body = src.split("# CELL 8 of 13")[0]
body = "# CELL 2 of 13" + body.split("# CELL 2 of 13")[1]
body = body.replace("SO-ARM100/Simulation/SO101/scene.xml",
                    os.path.join(REPO, "third_party", "SO-ARM100",
                                 "Simulation", "SO101", "scene.xml"))
ns = {"os": os, "sys": sys}
exec(compile(body, "<cells>", "exec"), ns)

cfg, build_arena, lookat_xyaxes = ns["cfg"], ns["build_arena"], ns["lookat_xyaxes"]
# The arm can physically complete a 13 cm pull.  The existing deterministic
# tray ride adds 2 cm of forward content slip during the pull, so final object
# poses match the 15 cm exposure used by the successful grasp probes.
cfg.drawer_travel = 0.13
cfg.table_half = (0.35, 0.17)
cfg.arm_a_pos = (-0.19, -0.025)
cfg.arm_b_pos = (0.19, -0.025)
cfg.arm_yaw_deg = -90.0
cfg.start_xy["mug"] = (0.380, 0.020)            # reachable rear-right body pickup

# Bulk-collection controls.  Seed zero and randomization zero reproduce the
# validated v21 rollout.  Kaggle workers set both through environment variables
# so four independent processes never edit this source file.
EPISODE_SEED = int(os.environ.get("DEMO_EPISODE_SEED", "0"))
GLOBAL_EPISODE_ID = int(os.environ.get("DEMO_GLOBAL_EPISODE_ID", "0"))
RANDOMIZE = os.environ.get("DEMO_RANDOMIZE", "0") == "1"
RNG = np.random.default_rng(EPISODE_SEED)
PROMPTS = (
    "Open the drawer and set the table for two with both plates, forks, spoons, and the mug.",
    "Pull open the top drawer, arrange two place settings, and put the mug beside them.",
    "Set a dinner table for two: place the plates, arrange the cutlery, and add the mug.",
    "Please open the drawer and prepare both dinner settings with the utensils and mug.",
    "Make two complete place settings after opening the drawer, then position the mug.",
    "Arrange the dinner table for two people using the plates, forks, spoons, and mug.",
    "Open the drawer, set down both plates, place each utensil beside them, and move the mug.",
    "Prepare the table for two diners and leave the mug upright next to the settings.",
    "Pull the drawer open and neatly organize the two plates, four utensils, and mug.",
    "Complete the two-person dinner setup, including opening the drawer and placing the mug.",
)
cfg.instruction = os.environ.get(
    "DEMO_INSTRUCTION", PROMPTS[EPISODE_SEED % len(PROMPTS)])
arena, spec = build_arena(cfg)

if RANDOMIZE:
    # Millimetre-scale pose diversity stays within the contact-only expert's
    # measured reach margin.  The expert re-reads every live object pose.
    for _name in ("plate_l", "plate_r"):
        _body = arena.find("body", _name)
        _p = np.asarray(_body.pos, dtype=float)
        _p[:2] += RNG.uniform(-0.0025, 0.0025, 2)
        _body.pos = _p.tolist()
    for _name in ("fork_l", "spoon_l", "fork_r", "spoon_r", "mug"):
        _body = arena.find("body", _name)
        _p = np.asarray(_body.pos, dtype=float)
        _p[:2] += RNG.uniform(-0.004, 0.004, 2)
        _body.pos = _p.tolist()

    # Safe visual-domain randomization for VLA robustness.  Collision shapes,
    # grasp geometry, arm mounts, and target zones remain the validated v21
    # values.  Every random value is recoverable from EPISODE_SEED.
    _top = arena.find("light", "top_light")
    _fill = arena.find("light", "fill_light")
    _top.diffuse = np.clip(RNG.uniform(0.68, 0.98, 3), 0, 1).tolist()
    _fill.diffuse = np.clip(RNG.uniform(0.22, 0.48, 3), 0, 1).tolist()
    _palette = {
        "plate": np.array([0.92, 0.92, 0.92]),
        "fork": np.array([0.58, 0.60, 0.62]),
        "spoon": np.array([0.58, 0.60, 0.62]),
        "mug": np.array([0.62, 0.67, 0.82]),
    }
    for _name in ("plate_l", "plate_r", "fork_l", "spoon_l", "fork_r", "spoon_r", "mug"):
        _visual = arena.find("geom", f"{_name}_visual")
        if _visual is not None:
            _rgb = np.clip(_palette[_name.split("_")[0]] + RNG.uniform(-0.12, 0.12, 3), 0.15, 1.0)
            _visual.rgba = [*_rgb.tolist(), 1.0]

# Proud pull bar and its two rigid standoffs are all attached to the sliding
# drawer body in build_arena.  Keep the proven reachable local bar position.
arena.find("geom", "drawer_handle").pos = [-0.16, -0.240, 0.030]

arena.worldbody.add("camera", name="cam_demo",
                    pos=[0.85, -0.70, 0.92],
                    xyaxes=lookat_xyaxes([0.85, -0.70, 0.92],
                                         [0.0, -0.05, 0.33]), fovy=50)
physics = mjcf.Physics.from_mjcf_model(arena)
physics.forward()

model, data = physics.model, physics.data

# ---------------------------------------------------------------- binding
def resolve(kind, short):
    counts = {"joint": model.njnt, "actuator": model.nu,
              "body": model.nbody, "geom": model.ngeom}
    acc = getattr(model, kind)
    hits = [(acc(i).name, i) for i in range(counts[kind])
            if acc(i).name == short or acc(i).name.endswith("/" + short)]
    assert len(hits) == 1, (short, hits)
    return hits[0][1]

# Hold all task objects only during initial settling.  The utensils are now
# staged flat on the tabletop; only the plates will ride with the drawer.
_HOLD = {}
for _n in ("plate_l", "plate_r", "fork_l", "spoon_l", "fork_r", "spoon_r", "mug"):
    # Prevent scattered props from kicking the arms during compilation-time
    # settling. Table support is restored through bit 8 below.
    _bid = resolve("body", _n)
    for _gid in range(model.ngeom):
        if int(model.geom_bodyid[_gid]) == _bid:
            model.geom_contype[_gid] = 0
            model.geom_conaffinity[_gid] = 8
    _q = int(model.jnt_qposadr[resolve("joint", f"{_n}_free")])
    _d = int(model.jnt_dofadr[resolve("joint", f"{_n}_free")])
    _HOLD[_n] = (_q, _d, data.qpos[_q:_q + 7].copy())
for _ in range(300):
    physics.step()
    for _q, _d, _p0 in _HOLD.values():
        data.qpos[_q:_q + 7] = _p0
        data.qvel[_d:_d + 6] = 0
physics.forward()

class Arm:
    def __init__(self, name):
        self.name = name
        jnames = spec.arm_joint_names[name]
        self.jaw_idx = next((i for i, jn in enumerate(jnames)
                             if "jaw" in jn.lower() or "gripper" in jn.lower()),
                            len(jnames) - 1)
        self.roll_idx = next((i for i, jn in enumerate(jnames)
                              if "roll" in jn.lower()), None)
        self.ik_idx = [i for i in range(len(jnames))
                       if i != self.jaw_idx and i != self.roll_idx]
        self.act, self.qadr, self.vadr = [], [], []
        for short in spec.arm_actuators[name]:
            aid = resolve("actuator", short)
            jid = int(model.actuator_trnid[aid, 0])
            self.act.append(aid)
            self.qadr.append(int(model.jnt_qposadr[jid]))
            self.vadr.append(int(model.jnt_dofadr[jid]))
        self.ctrl_lo = np.array([model.actuator_ctrlrange[a][0] for a in self.act])
        self.ctrl_hi = np.array([model.actuator_ctrlrange[a][1] for a in self.act])
        self.home_ctrl = np.clip(0.0, self.ctrl_lo, self.ctrl_hi)
        self.grip_bid = resolve("body", f"{name}/gripper")
        self.wrist_bid = resolve("body", f"{name}/wrist")
        self.static_pad_gid = resolve("geom", f"{name}/static_finger_contact_pad")
        self.moving_pad_gid = resolve("geom", f"{name}/moving_finger_contact_pad")
        self.static_pad_base = model.geom_pos[self.static_pad_gid].copy()
        self.moving_pad_base = model.geom_pos[self.moving_pad_gid].copy()
        print(f"{name}: jaw={self.jaw_idx} roll={self.roll_idx} "
              f"jaw_range={np.round([self.ctrl_lo[self.jaw_idx], self.ctrl_hi[self.jaw_idx]], 3)}")

    def qpos(self):
        return np.array([data.qpos[q] for q in self.qadr])

    def set_ctrl(self, ctrl):
        for a, c in zip(self.act, ctrl):
            data.ctrl[a] = c

    def set_jaw(self, value):
        data.ctrl[self.act[self.jaw_idx]] = float(value)

    def set_roll(self, value):
        if self.roll_idx is not None:
            data.ctrl[self.act[self.roll_idx]] = float(value)

    def gripper_pos(self):
        return data.xpos[self.grip_bid].copy()

    # the controlled point: between the finger pads (where pinches happen)
    def point_pos(self, point_local):
        return (self.gripper_pos()
                + data.xmat[self.grip_bid].reshape(3, 3) @ point_local)

    def pad_pos(self):
        # Deep fingertip point proven reachable at the drawer/table edge.
        return self.point_pos(PINCH_LOCAL)

    def approach(self):
        g, w = data.xpos[self.grip_bid], data.xpos[self.wrist_bid]
        v = g - w
        return v / np.linalg.norm(v)

    def closing_dir(self):
        """Run-28 calibrated pad-closing line (gripper local x)."""
        return data.xmat[self.grip_bid].reshape(3, 3)[:, 0]


ARMS = {"arm_a": Arm("arm_a"), "arm_b": Arm("arm_b")}
A, B = ARMS["arm_a"], ARMS["arm_b"]
print("closing axes at home:",
      {a.name: (np.round(a.closing_dir(), 3).tolist(),
                np.round(a.approach(), 3).tolist()) for a in (A, B)})
DRAWER_ACT = resolve("actuator", "drawer_act")
DRAWER_QADR = int(model.jnt_qposadr[resolve("joint", "drawer_slide")])

PINCH_LOCAL = np.array([0.012, 0.0, -0.098])   # deep tip aim (run-25 reaches) + contact-zone x
GRASPFRAME_LOCAL = np.array([0.012, 0.0, -0.065])

def task_point_local(kind):
    """Use the measured fingertip TCP for every physical pinch."""
    # The higher graspframe made the mug appear centred in Cartesian logs but
    # left the actual fingertip pads above/beside the cup.  The deep TCP is the
    # contact location validated for the drawer rim and tabletop handles.
    return PINCH_LOCAL
# Measured SO-101 asymmetric fingertip midpoint as the moving jaw closes.
# Keeping this point over a narrow handle prevents the ~18 mm sideways sweep
# that pushed forks/spoons out of the clamp in run 28.  Z stays at the proven
# deep-tip aim; raising it to the geometric center regresses drawer reaches.
GRASP_MID_Q = np.array([-0.1745, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                        0.6, 0.8, 1.0, 1.2])
GRASP_MID_X = np.array([-0.00573, 0.00106, 0.00408, 0.00815, 0.01221,
                        0.01621, 0.02011, 0.02386, 0.03076, 0.03668,
                        0.04135])
def closing_pinch_local(jaw_q, z_local=PINCH_LOCAL[2]):
    return np.array([np.interp(jaw_q, GRASP_MID_Q, GRASP_MID_X),
                     0.0, z_local])


# approach directions: TILTED-from-front for drawer contents (a vertical
# descent puts the WRIST through the tabletop edge — stalls at y=-0.175;
# tilted ~40 deg keeps the wrist in front of the table, human-style)
_v = np.array([0.0, -0.34, -0.94])
DES_TILT = _v / np.linalg.norm(_v)
DES_DOWN = np.array([0.0, 0.0, -1.0])
JAW_OPEN = 0.45                                   # proven plate/bar approach opening
JAW_CLOSE = -0.1745                               # clamp: servo stalls on
JAW_GAP = {k: JAW_CLOSE for k in                  # contact = grip force
           ("plate", "spoon", "fork", "mug", "bar")}
JAW_OPEN_BY_KIND = {
    "plate": 0.45, "fork": 0.45, "spoon": 0.45,
    "mug": 0.45, "bar": 0.80,
}
COMPENSATE_ASYMMETRIC_CLOSE = False

# Keep the mug upright and rotate its physical handle toward arm B's reachable
# front-side grasp line. This changes only the free object's spawn pose.
_q = int(model.jnt_qposadr[resolve("joint", "mug_free")])
data.qpos[_q + 3:_q + 7] = [np.cos(np.pi / 4), 0.0, 0.0,
                            -np.sin(np.pi / 4)]
_HOLD["mug"] = (_HOLD["mug"][0], _HOLD["mug"][1],
                data.qpos[_q:_q + 7].copy())
physics.forward()
for _ in range(100):
    physics.step()
    for _q2, _d2, _p0 in _HOLD.values():
        data.qpos[_q2:_q2 + 7] = _p0
        data.qvel[_d2:_d2 + 6] = 0
physics.forward()

# Contact layers: keep the proven arm masks untouched. The tabletop advertises
# support bit 8; inactive task objects accept only bit 8, then restore normal
# bit-1 contact during their own manipulation phase.
_table_gid = resolve("geom", "tabletop")
model.geom_contype[_table_gid] = 8
model.geom_conaffinity[_table_gid] = 1
_riser_gid = resolve("geom", "tray_riser")
model.geom_contype[_riser_gid] = 8
model.geom_conaffinity[_riser_gid] = 1

def set_arm_object_contact(name, enabled):
    bid = resolve("body", name)
    kind = name.split("_")[0]
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == bid:
            gname = model.geom(gid).name.rsplit("/", 1)[-1]
            is_grasp_proxy = (
                (kind in ("fork", "spoon") and gname == f"{name}_handle_grasp")
                or (kind == "mug" and gname == f"{name}_handle_grasp"))
            proxy_only = kind in ("fork", "spoon", "mug")
            arm_active = enabled and (is_grasp_proxy or not proxy_only)
            model.geom_contype[gid] = 1 if arm_active else 0
            # Non-grasp visual/hull geoms keep table support through bit 8,
            # but cannot snag the gripper through bit 1.
            model.geom_conaffinity[gid] = 9 if arm_active else 8

for _n in ("plate_l", "plate_r", "fork_l", "spoon_l",
           "fork_r", "spoon_r", "mug"):
    set_arm_object_contact(_n, False)

# ---------------------------------------------------------------- IK (pad point)
_JACP = np.zeros((3, model.nv))
_JACR = np.zeros((3, model.nv))

def ik_step(arm, target_pos, desired_approach, kp=14.0, kw=6.0, damp=0.05,
            max_dq=0.06, point_local=None):
    if point_local is None:
        pos = arm.pad_pos()
    else:
        pos = (arm.gripper_pos()
               + data.xmat[arm.grip_bid].reshape(3, 3) @ point_local)
    approach = arm.approach()
    mujoco.mj_jac(model.ptr, data.ptr, _JACP, _JACR,
                  np.ascontiguousarray(pos), arm.grip_bid)
    dofs = [arm.vadr[i] for i in arm.ik_idx]
    if desired_approach is None:
        J = _JACP[:, dofs]
        v = kp * (target_pos - pos)
    else:
        J = np.vstack([_JACP[:, dofs], _JACR[:, dofs]])
        v = np.concatenate([kp * (target_pos - pos),
                            kw * np.cross(approach, desired_approach)])
    JJt = J @ J.T + (damp ** 2) * np.eye(J.shape[0])
    dq = np.clip(J.T @ np.linalg.solve(JJt, v), -max_dq, max_dq)
    q = arm.qpos()
    new = q.copy()
    for k, i in enumerate(arm.ik_idx):
        new[i] = np.clip(q[i] + dq[k], arm.ctrl_lo[i], arm.ctrl_hi[i])
    return new

def obj_pos(name):
    return data.xpos[resolve("body", name)].copy()

def physical_midpoint_local(arm):
    """Live midpoint of the two reference fingertip patches in grip frame."""
    midpoint = 0.5 * (data.geom_xpos[arm.static_pad_gid]
                      + data.geom_xpos[arm.moving_pad_gid])
    R = data.xmat[arm.grip_bid].reshape(3, 3)
    return R.T @ (midpoint - arm.gripper_pos())

def asymmetric_grasp_local(arm, object_half_width):
    """Open-jaw target that preloads an object toward the moving finger.

    SO-101 has one fixed and one moving finger.  Starting at the geometric
    midpoint lets the moving mesh bat a light object past the fixed mesh
    before opposition forms.  Starting one object radius inside the moving
    face gives it room to travel into the fixed face during a slow close.
    """
    R = data.xmat[arm.grip_bid].reshape(3, 3)
    static = R.T @ (data.geom_xpos[arm.static_pad_gid] - arm.gripper_pos())
    moving = R.T @ (data.geom_xpos[arm.moving_pad_gid] - arm.gripper_pos())
    line = moving - static
    line /= np.linalg.norm(line) + 1e-9
    return moving - line * max(float(object_half_width) - 0.003, 0.004)

def configure_tip_pads(arm, kind):
    """Set a thin physical contact face appropriate for the target geometry."""
    if kind == "mug":
        size = np.array([0.0025, 0.018, 0.012])
    elif kind in ("fork", "spoon"):
        size = np.array([0.0025, 0.018, 0.010])
    else:
        size = np.array([0.0025, 0.012, 0.012])
    model.geom_size[arm.static_pad_gid, :3] = size
    model.geom_size[arm.moving_pad_gid, :3] = size
    # The mesh surface shadows pads placed exactly at the CAD coordinates.
    # Project the two small patches toward one another along the *live* jaw
    # line, as a real rubber fingertip cap would protrude from each finger.
    # The pads remain rigidly attached geoms and exert only MuJoCo contacts.
    model.geom_pos[arm.static_pad_gid] = arm.static_pad_base
    model.geom_pos[arm.moving_pad_gid] = arm.moving_pad_base
    mujoco.mj_forward(model.ptr, data.ptr)
    s = data.geom_xpos[arm.static_pad_gid].copy()
    m = data.geom_xpos[arm.moving_pad_gid].copy()
    u = (m - s) / (np.linalg.norm(m - s) + 1e-9)
    protrusion = 0.005 if kind == "mug" else 0.003
    sb = int(model.geom_bodyid[arm.static_pad_gid])
    mb = int(model.geom_bodyid[arm.moving_pad_gid])
    Rs = data.xmat[sb].reshape(3, 3)
    Rm = data.xmat[mb].reshape(3, 3)
    model.geom_pos[arm.static_pad_gid] = arm.static_pad_base + Rs.T @ (u * protrusion)
    model.geom_pos[arm.moving_pad_gid] = arm.moving_pad_base + Rm.T @ (-u * protrusion)
    mujoco.mj_forward(model.ptr, data.ptr)

def grasp_geometry_report(arm, name, tag):
    """Report the real object position relative to the two fingertip pads."""
    kind = name.split("_")[0]
    if kind not in ("fork", "spoon", "mug"):
        return
    geom_name = f"{name}_handle_grasp"
    centre = data.geom_xpos[resolve("geom", geom_name)].copy()
    static = data.geom_xpos[arm.static_pad_gid].copy()
    moving = data.geom_xpos[arm.moving_pad_gid].copy()
    span = moving - static
    length = np.linalg.norm(span)
    unit = span / (length + 1e-9)
    along = float(np.dot(centre - static, unit))
    nearest = static + np.clip(along, 0.0, length) * unit
    perp = float(np.linalg.norm(centre - nearest))
    print(f"    {tag} geometry: span={length*1000:.0f}mm "
          f"along={along*1000:.0f}mm perp={perp*1000:.0f}mm "
          f"obj={np.round(centre, 3)}")
    if tag == "open":
        for label, gid in (("static", arm.static_pad_gid),
                           ("moving", arm.moving_pad_gid)):
            axes = data.geom_xmat[gid].reshape(3, 3)
            dots = np.abs(axes.T @ unit)
            print(f"      {label} pad axis alignment={np.round(dots, 2)} "
                  f"size={np.round(model.geom_size[gid, :3]*1000, 1)}mm "
                  f"type={int(model.geom_type[gid])} "
                  f"mask={int(model.geom_contype[gid])}/{int(model.geom_conaffinity[gid])}")
        ogid = resolve("geom", geom_name)
        print(f"      object mask={int(model.geom_contype[ogid])}/"
              f"{int(model.geom_conaffinity[ogid])} type={int(model.geom_type[ogid])}")
    if tag == "closed":
        obj_bid = resolve("body", name)
        pairs = []
        for ci in range(data.ncon):
            g1 = int(data.contact[ci].geom1)
            g2 = int(data.contact[ci].geom2)
            b1 = int(model.geom_bodyid[g1])
            b2 = int(model.geom_bodyid[g2])
            if b1 == obj_bid or b2 == obj_bid:
                other = g2 if b1 == obj_bid else g1
                oname = model.geom(other).name
                if arm.name in oname:
                    pairs.append(oname.rsplit("/", 1)[-1])
        print(f"    closed contacts: {pairs}")


# Scripted-expert stabilization.  It captures the live transform only after a
# guarded open-jaw approach and then enforces that transform during transport;
# the arm must still execute every lift/carry waypoint and releases while the
# jaw opens.  This is explicitly a demonstration constraint, not evidence of
# a purely frictional grasp and not part of the intended SmolVLA controller.
DEMO_GRASP_STABILIZE = os.environ.get("DEMO_GRASP_STABILIZE", "0") == "1"
_STABLE_GRASPS = {}
_PREGRASP_HOLDS = {}
# The asymmetric mug handle can make the free cup walk during the long plate
# and cutlery phases even though no arm is touching it.  Hold its exact spawn
# pose until arm B's verified open-jaw approach begins; this is staging only,
# not transport or placement.
_mq = int(model.jnt_qposadr[resolve("joint", "mug_free")])
_md = int(model.jnt_dofadr[resolve("joint", "mug_free")])
_PREGRASP_HOLDS["mug"] = (_mq, _md, data.qpos[_mq:_mq + 7].copy())

def engage_stable_grasp(arm, name):
    if not DEMO_GRASP_STABILIZE:
        return
    kind = name.split("_")[0]
    Ro = data.xmat[resolve("body", name)].reshape(3, 3).copy()
    if kind in ("fork", "spoon"):
        # Canonical flat spawn attitude (180 degrees about world Z).
        Ro = np.diag([-1.0, -1.0, 1.0])
    elif kind in ("plate", "mug"):
        # Plates stay horizontal and the mug stays upright.
        Ro = np.eye(3)
    point_local = task_point_local(kind)
    pinch_world = arm.point_pos(point_local)
    _STABLE_GRASPS[arm.name] = {
        "name": name,
        # The motion controller drives this exact pinch point.  Preserve its
        # captured world-space displacement to the object centre so changing
        # wrist orientation cannot swing dinnerware away from the commanded
        # placement target.
        "point_local": point_local.copy(),
        "p_delta": obj_pos(name) - pinch_world,
        "R_world": Ro,
    }
    bid = resolve("body", name)
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == bid:
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0

def release_stable_grasp(arm):
    grasp = _STABLE_GRASPS.pop(arm.name, None)
    if grasp is not None:
        name = grasp["name"]
        # Release with no inherited kinematic velocity, then return to the
        # table-only mask.  The object settles under gravity but the opening
        # fingers and retreat cannot bat a correctly placed plate sideways.
        d = int(model.jnt_dofadr[resolve("joint", f"{name}_free")])
        data.qvel[d:d + 6] = 0.0
        set_arm_object_contact(name, False)
        if name.startswith(("fork_", "spoon_")):
            # The broad invisible sleeve is only a grasp aid.  Once released,
            # let the real convex utensil hulls—not the sleeve—touch the table.
            gid = resolve("geom", f"{name}_handle_grasp")
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0

def apply_stable_grasps():
    for arm_name, grasp in _STABLE_GRASPS.items():
        arm = ARMS[arm_name]
        name = grasp["name"]
        p = arm.point_pos(grasp["point_local"]) + grasp["p_delta"]
        # Dinnerware stays flat/upright while the wrist chooses a reachable
        # roll; this prevents thin handles intersecting the table on release.
        R = grasp["R_world"]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R.reshape(9)))
        q = int(model.jnt_qposadr[resolve("joint", f"{name}_free")])
        d = int(model.jnt_dofadr[resolve("joint", f"{name}_free")])
        data.qpos[q:q + 3] = p
        data.qpos[q + 3:q + 7] = quat
        d = int(model.jnt_dofadr[resolve("joint", f"{name}_free")])
        data.qvel[d:d + 6] = 0.0
    for name, (q, d, pose) in _PREGRASP_HOLDS.items():
        data.qpos[q:q + 7] = pose
        data.qvel[d:d + 6] = 0.0
    mujoco.mj_forward(model.ptr, data.ptr)


# A separate kinematic data buffer lets us solve bounded joint targets without
# disturbing live physics. Multi-starts escape the exact-home singularity that
# made reachable tabletop points appear impossible to the online DLS loop.
_IK_DATA = mujoco.MjData(model.ptr)
_IK_RNG = np.random.default_rng(15)
for _probe_q in (JAW_CLOSE, JAW_OPEN):
    _IK_DATA.qpos[:] = data.qpos
    _IK_DATA.qpos[A.qadr[A.jaw_idx]] = _probe_q
    mujoco.mj_forward(model.ptr, _IK_DATA)
    _gR = _IK_DATA.xmat[A.grip_bid].reshape(3, 3)
    _gap_local = _gR.T @ (_IK_DATA.geom_xpos[A.moving_pad_gid]
                          - _IK_DATA.geom_xpos[A.static_pad_gid])
    print(f"jaw {_probe_q:+.3f} contact-pad delta local:", np.round(_gap_local, 4))

def solve_joint_target(arm, target, point_local=None, desired_approach=None,
                       desired_closing=None, fixed_roll=None):
    point_local = PINCH_LOCAL if point_local is None else point_local
    indices = [i for i in range(len(arm.qadr))
               if i != arm.jaw_idx and not (fixed_roll is not None
                                             and i == arm.roll_idx)]
    qadr = [arm.qadr[i] for i in indices]
    lo = arm.ctrl_lo[indices]
    hi = arm.ctrl_hi[indices]
    live = arm.qpos()[indices]
    base_qpos = data.qpos.copy()
    if fixed_roll is not None and arm.roll_idx is not None:
        base_qpos[arm.qadr[arm.roll_idx]] = float(fixed_roll)

    def residual(q):
        _IK_DATA.qpos[:] = base_qpos
        for adr, value in zip(qadr, q):
            _IK_DATA.qpos[adr] = value
        mujoco.mj_forward(model.ptr, _IK_DATA)
        R = _IK_DATA.xmat[arm.grip_bid].reshape(3, 3)
        p = _IK_DATA.xpos[arm.grip_bid] + R @ point_local
        out = [*(p - target)]
        if desired_approach is not None:
            g = _IK_DATA.xpos[arm.grip_bid]
            w = _IK_DATA.xpos[arm.wrist_bid]
            a = (g - w) / (np.linalg.norm(g - w) + 1e-9)
            out.extend(0.15 * np.cross(a, desired_approach))
        if desired_closing is not None:
            c = R[:, 0]
            d = np.asarray(desired_closing, float)
            d /= np.linalg.norm(d) + 1e-9
            if np.dot(c, d) < 0:
                d = -d
            out.extend(0.15 * np.cross(c, d))
        return np.asarray(out)

    seeds = [live, (lo + hi) * 0.5]
    for _ in range(8):
        seeds.append(_IK_RNG.uniform(lo, hi))
    best_q, best_err = live, np.inf
    for seed in seeds:
        result = least_squares(residual, np.clip(seed, lo, hi), bounds=(lo, hi),
                               max_nfev=120, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        err = np.linalg.norm(residual(result.x)[:3])
        if err < best_err:
            best_q, best_err = result.x.copy(), err
    full = arm.qpos()
    for i, value in zip(indices, best_q):
        full[i] = value
    return full, best_err

# Fast reachability audit for scene-layout tuning.  This exits before any
# scripted phase and never mutates the normal demo path.
if os.environ.get("DEMO_PROBE_MUG_WORKSPACE", "0") == "1":
    print("arm_b mug-point workspace (millimetres of Cartesian residual):")
    for _y in (-0.06, -0.02, 0.02, 0.06, 0.10, 0.14):
        _row = []
        for _x in (0.22, 0.26, 0.30, 0.34, 0.38, 0.42):
            _, _e = solve_joint_target(
                B, np.array([_x, _y, cfg.table_top_z + 0.044]),
                point_local=task_point_local("mug"),
                desired_approach=DES_DOWN)
            _row.append(f"{_e*1000:5.1f}")
        print(f"  y={_y:+.2f}: " + " ".join(_row))
    raise SystemExit(0)

# ---------------------------------------------------------------- recording
NO_RENDER = os.environ.get("DEMO_NO_RENDER", "0") == "1"
TRAJECTORY_ONLY = os.environ.get("DEMO_TRAJECTORY_ONLY", "0") == "1"
SAVE_DEMO_VIDEO = (not NO_RENDER and
                   os.environ.get("DEMO_NO_DEMO_VIDEO", "0") != "1")
frames = []
db = {"img_front": [], "img_a": [], "img_b": [], "img_top": [],
      "state": [], "action": [], "qpos": []}
tick = [0]
VID_EVERY, DATA_EVERY = 5, 10

REN_FRONT = None if not SAVE_DEMO_VIDEO else mujoco.Renderer(model.ptr, height=480, width=640)
REN_W = {} if not SAVE_DEMO_VIDEO else {
    n: mujoco.Renderer(model.ptr, height=240, width=320)
    for n in ("arm_a", "arm_b")}
REN_OBS = {} if NO_RENDER else {
    "cam_front": mujoco.Renderer(model.ptr, height=224, width=224),
    "cam_arm_a": mujoco.Renderer(model.ptr, height=224, width=224),
    "cam_arm_b": mujoco.Renderer(model.ptr, height=224, width=224),
    "cam_top": mujoco.Renderer(model.ptr, height=224, width=224)}
WCAM = {}
for i in range(model.ncam):
    cn = model.camera(i).name
    for n in ("arm_a", "arm_b"):
        if cn.endswith("_wrist") and n in cn:
            WCAM[n] = i

def norm_action():
    out = []
    for arm in ARMS.values():
        c = np.array([data.ctrl[a] for a in arm.act])
        out.append(np.clip((c - arm.ctrl_lo) / (arm.ctrl_hi - arm.ctrl_lo)
                           * 2.0 - 1.0, -1.0, 1.0))
    return np.concatenate(out)

def record(every=VID_EVERY):
    apply_stable_grasps()
    tick[0] += 1
    if NO_RENDER and not TRAJECTORY_ONLY:
        return
    if not NO_RENDER and SAVE_DEMO_VIDEO and tick[0] % every == 0:
        REN_FRONT.update_scene(data.ptr, camera="cam_demo")
        f = np.zeros((480, 1280, 3), dtype=np.uint8)
        f[:, 320:960] = REN_FRONT.render()
        for i, n in enumerate(("arm_a", "arm_b")):
            REN_W[n].update_scene(data.ptr, camera=WCAM[n])
            f[0:240, 320 * (3 * i):320 * (3 * i + 1)] = REN_W[n].render()
        frames.append(f)
    if tick[0] % DATA_EVERY == 0:
        if not NO_RENDER:
            for cam, key in (("cam_front", "img_front"),
                             ("cam_arm_a", "img_a"),
                             ("cam_arm_b", "img_b"),
                             ("cam_top", "img_top")):
                REN_OBS[cam].update_scene(data.ptr, camera=cam)
                db[key].append(REN_OBS[cam].render()[..., :3])
        db["state"].append(np.concatenate([ARMS[n].qpos() for n in ("arm_a", "arm_b")]))
        db["action"].append(norm_action())
        db["qpos"].append(np.array(data.qpos, dtype=np.float32, copy=True))

def wrist_snap(arm, tag):
    if not SAVE_DEMO_VIDEO:
        return
    r = mujoco.Renderer(model.ptr, height=360, width=480)
    r.update_scene(data.ptr, camera=WCAM[arm.name])
    media.write_image(os.path.join(REPO, "outputs", f"wrist_{tag}.png"), r.render())

# ---------------------------------------------------------------- primitives
def move_pad(arm, target, other, des=None, tol=0.008, max_steps=500,
             jaw=JAW_OPEN, roll=None, max_dq=0.06, ik_kw=6.0,
             point_local=None, use_solver=False, desired_closing=None):
    point = arm.pad_pos if point_local is None else lambda: arm.point_pos(point_local)
    start = point()
    desired = None
    if not isinstance(des, str) and des is not None:
        desired = np.asarray(des, float)
    if use_solver:
        joint_target, solve_err = solve_joint_target(
            arm, np.asarray(target, float), point_local=point_local,
            desired_approach=desired, desired_closing=desired_closing,
            fixed_roll=roll)
    else:
        joint_target, solve_err = arm.qpos(), np.inf
    if use_solver and solve_err < 0.035:
        q0 = arm.qpos()
        ndrive = min(max_steps, 260)
        for i in range(ndrive):
            # cubic smoothstep prevents acceleration shocks to held objects
            u = (i + 1) / ndrive
            u = u * u * (3.0 - 2.0 * u)
            arm.set_ctrl(q0 + u * (joint_target - q0))
            arm.set_jaw(jaw)
            if roll is not None:
                arm.set_roll(roll)
            other.set_ctrl(other.home_ctrl)
            physics.step()
            record()
        if np.linalg.norm(point() - target) < tol:
            return True
    for _ in range(max_steps):
        if isinstance(des, str) and des == "toward":
            v = target - point()
            d = v / (np.linalg.norm(v) + 1e-9)
        elif isinstance(des, str) and des == "free":
            d = None
        else:
            d = des if des is not None else arm.approach()
        arm.set_ctrl(ik_step(arm, target, d, max_dq=max_dq, kw=ik_kw,
                             point_local=point_local))
        arm.set_jaw(jaw)
        if roll is not None:
            arm.set_roll(roll)
        other.set_ctrl(other.home_ctrl)
        physics.step()
        record()
        if np.linalg.norm(point() - target) < tol:
            return True
    moved = np.linalg.norm(point() - start)
    if moved < 0.03:
        print(f"  [STALL] {arm.name} moved only {moved*1000:.0f} mm toward "
              f"{np.round(target, 3)} — IK frozen")
    return False

def align_roll(arm, close_axis):
    """Exact wrist-roll target so the pads close along `close_axis`:
    measure the signed angle between the current closing direction and the
    axis, in the plane perpendicular to the approach, and rotate the roll
    by exactly that. (The old +/-90-deg heuristic could not align to an
    arbitrary rim angle — every plate close was misaligned unless lucky.)"""
    if arm.roll_idx is None or close_axis is None:
        return None
    a = arm.approach()
    c = arm.closing_dir()
    ch = c - (c @ a) * a
    dh = np.asarray(close_axis, float) - (np.asarray(close_axis, float) @ a) * a
    nch, ndh = np.linalg.norm(ch), np.linalg.norm(dh)
    if nch < 0.2 or ndh < 0.2:
        return None
    ch, dh = ch / nch, dh / ndh
    # A parallel-jaw closing direction is a line, not an arrow: +axis and
    # -axis describe the same pinch.  Choose the equivalent sign requiring
    # less than 90 degrees of wrist rotation, especially important for the
    # mirrored arm B near its roll limit.
    if arm.name == "arm_b" and np.dot(ch, dh) < 0.0:
        dh = -dh
    ang = np.arctan2(np.dot(np.cross(ch, dh), a), np.dot(ch, dh))
    return arm.qpos()[arm.roll_idx] + float(ang)

def pinch_grasp(arm, name, grasp_pos, close_axis, gap, des=DES_TILT,
                verify=True, direct=False):
    """Reach, straddle, close, lift, verify. Returns True if HELD.
    verify=False for grasps whose success is not 'did it rise' (drawer)."""
    kind = name.split("_")[0]
    configure_tip_pads(arm, kind)
    if kind == "plate" and name not in _PREGRASP_HOLDS:
        _hq = int(model.jnt_qposadr[resolve("joint", f"{name}_free")])
        _hd = int(model.jnt_dofadr[resolve("joint", f"{name}_free")])
        _PREGRASP_HOLDS[name] = (_hq, _hd, data.qpos[_hq:_hq + 7].copy())
    point_local = task_point_local(kind)
    point_pos = lambda: arm.point_pos(point_local)
    # Light utensils otherwise skate away from the moving SO-101 finger
    # before the fixed finger makes opposing contact.  A gentler physical
    # position servo preserves the same close command with less impact force.
    _jaw_aid = arm.act[arm.jaw_idx]
    _jaw_kp = 80.0 if kind in ("fork", "spoon", "mug") else 400.0
    model.actuator_gainprm[_jaw_aid, 0] = _jaw_kp
    model.actuator_biasprm[_jaw_aid, 1] = -_jaw_kp
    # Rear-table pickups require the shoulder to pass into a different IK
    # branch than the forward-facing drawer work.  Use the multi-start joint
    # solver only to acquire those objects; closing/lifting remain physical.
    use_solver = kind in ("fork", "spoon", "mug")
    managed_contact = name in ("plate_l", "plate_r", "fork_l", "spoon_l",
                               "fork_r", "spoon_r", "mug")
    if managed_contact:
        set_arm_object_contact(name, False)
    open_q = JAW_OPEN_BY_KIND.get(kind, JAW_OPEN)
    grasp_rel = np.asarray(grasp_pos, float) - obj_pos(name)
    o = ARMS["arm_b"] if arm.name == "arm_a" else ARMS["arm_a"]

    # HOME RESET FIRST: every probe-verified reach starts from the home
    # config — descending from the previous phase's leftover pose is what
    # stalled arm A (local minimum), not geometry
    _pre = arm.home_ctrl.copy()
    _pre[1] += 0.12
    _pre[2] += 0.18
    for _ in range(250):
        arm.set_ctrl(_pre)
        arm.set_jaw(open_q)
        o.set_ctrl(o.home_ctrl)
        physics.step()
        record()
    roll = align_roll(arm, close_axis) if close_axis is not None else None
    # For rear/tabletop acquisitions, solve wrist roll jointly with position
    # and approach.  A decoupled roll heuristic produced a 90-degree error on
    # the mug even though the TCP position was correct.
    solver_roll = (None if use_solver and close_axis is not None else roll)
    for _ in range(120):                     # settle the roll shift
        arm.set_ctrl(_pre)
        arm.set_jaw(open_q)
        if roll is not None:
            arm.set_roll(roll)
        o.set_ctrl(o.home_ctrl)
        physics.step()
        record()

    if not DEMO_GRASP_STABILIZE and kind in ("fork", "spoon", "mug"):
        # Use the measured stock-mesh pinch centre at the commanded aperture.
        # The auxiliary pad-centre transform is not invariant through the
        # asymmetric finger linkage and drifted several centimetres on close.
        point_local = closing_pinch_local(open_q)

    if direct:
        move_pad(arm, grasp_pos, o, des="free", tol=0.015,
                 max_steps=900, jaw=open_q, roll=roll,
                 point_local=point_local)
    else:
        # hover above the grasp point, then a VERTICAL descent (no
        # orientation fight), then push down to CONTACT — the pads must
        # land on the object before the jaw closes
        move_pad(arm, grasp_pos + np.array([0, 0, 0.07]), o, des="toward",
                 tol=0.025, max_steps=700, jaw=open_q, roll=solver_roll,
                 point_local=point_local, use_solver=use_solver,
                 desired_closing=close_axis if use_solver else None)
        if managed_contact and kind == "plate":
            set_arm_object_contact(name, True)
        move_pad(arm, grasp_pos + np.array([0, 0, 0.008]), o, des=DES_DOWN,
                 tol=0.012, max_steps=350, jaw=open_q, roll=solver_roll,
                 point_local=point_local, use_solver=use_solver,
                 desired_closing=close_axis if use_solver else None)
        if kind == "plate":
            for _ in range(120):               # push rim jaws to contact
                arm.set_ctrl(ik_step(arm, grasp_pos - np.array([0, 0, 0.012]),
                                     DES_DOWN, point_local=point_local))
                arm.set_jaw(open_q)
                if roll is not None:
                    arm.set_roll(roll)
                o.set_ctrl(o.home_ctrl)
                physics.step()
                record()
        # Finish at the requested pinch point after the contact push.  The
        # final correction was present in the successful run-28 plate recipe.
        if np.linalg.norm(point_pos() - grasp_pos) > 0.008:
            move_pad(arm, grasp_pos, o, des=DES_DOWN, tol=0.008,
                     max_steps=300, jaw=open_q, roll=solver_roll,
                     point_local=point_local, use_solver=use_solver,
                     desired_closing=close_axis if use_solver else None)

    if kind in ("fork", "spoon") and close_axis is not None and not use_solver:
        roll = align_roll(arm, close_axis)
        for _ in range(100):
            arm.set_ctrl(arm.qpos())
            arm.set_jaw(open_q)
            if roll is not None:
                arm.set_roll(roll)
            o.set_ctrl(o.home_ctrl)
            physics.step()
            record()
        move_pad(arm, grasp_pos, o, des=DES_DOWN, tol=0.010,
                 max_steps=300, jaw=open_q, roll=roll,
                 point_local=point_local)

    if use_solver and arm.roll_idx is not None:
        # Preserve the coupled orientation found by the full solve while the
        # asymmetric jaw closes and throughout the initial lift.
        roll = arm.qpos()[arm.roll_idx]

    err = np.linalg.norm(point_pos() - grasp_pos)
    print(f"  reach {name}: at={np.round(point_pos(), 3)} "
          f"target={np.round(grasp_pos, 3)} err={err*1000:.0f} mm"
          + f" axis={np.round(arm.closing_dir(), 2)}"
          + f" approach={np.round(arm.approach(), 2)}")
    reach_limit = (0.035 if not verify else
                   (0.040 if DEMO_GRASP_STABILIZE and kind in ("fork", "spoon")
                    else 0.030 if kind in ("fork", "spoon", "mug")
                    else 0.015))
    if err > reach_limit:                      # never close beside an object
        print(f"  grasp {name}: SKIPPED (err {err*1000:.0f} mm > "
              f"{reach_limit*1000:.0f} mm)")
        arm.set_jaw(open_q)
        return False

    # Tabletop props are approached with arm/object contacts disabled so an
    # open finger cannot sweep a 25 g handle or asymmetric mug away.  Contact
    # begins only after the measured fingertip midpoint passes the reach guard.
    if managed_contact and kind in ("fork", "spoon", "mug"):
        set_arm_object_contact(name, True)
        for _ in range(12):
            arm.set_ctrl(arm.qpos())
            arm.set_jaw(open_q)
            o.set_ctrl(o.home_ctrl)
            physics.step()
            record()

    grasp_geometry_report(arm, name, "open")

    # Once the open jaws have reached and straddled the verified grasp point,
    # activate the demonstration grasp constraint before closing impact can
    # launch a thin rim/handle. Nothing follows during approach.
    # Staging holds exist only to stop untouched props drifting during long
    # preceding phases.  They must be removed for *every* verified grasp.
    # Previously this pop lived inside the stabilization branch, so setting
    # DEMO_GRASP_STABILIZE=0 left plates and the mug pinned to their spawn
    # poses and made a genuine lift mathematically impossible.
    if verify:
        _PREGRASP_HOLDS.pop(name, None)
    if verify and DEMO_GRASP_STABILIZE:
        engage_stable_grasp(arm, name)

    # Close slowly while keeping the measured, jaw-dependent finger midpoint
    # on the live grasp point.  This cancels the asymmetric SO-101 finger
    # sweep highlighted in the reference grasping work.
    z0 = obj_pos(name)[2]
    # Give the arm enough time to cancel the SO-101 moving finger's lateral
    # sweep.  With a 250-step ramp the first contact displaced light handles
    # 30+ mm before the fixed finger arrived.
    close_steps = 800 if kind in ("fork", "spoon", "mug") else 80
    close_arm_ctrl = arm.qpos().copy()
    for t in range(close_steps):
        jaw_cmd = open_q + (gap - open_q) * (t + 1) / close_steps
        # The jaw-dependent midpoint compensation is useful for a purely
        # frictional pinch, but it must not run while the demonstration grasp
        # constraint is active: both controllers would move the same object
        # from different reference points during the close and can create a
        # large, bogus carry offset.  With stabilization enabled, keep the
        # wrist fixed and let only the physical jaw actuator close.
        if (COMPENSATE_ASYMMETRIC_CLOSE
                and kind in ("fork", "spoon", "mug")
                and not DEMO_GRASP_STABILIZE):
            # Keep the intended pinch fixed in world space.  Chasing the live
            # pose of a light object after the moving finger nudges it turns a
            # millimetre disturbance into an unstable pursuit.
            live_target = np.asarray(grasp_pos, float)
            local = closing_pinch_local(jaw_cmd)
            arm.set_ctrl(ik_step(arm, live_target, DES_DOWN,
                                 point_local=local, max_dq=0.006))
        else:
            # Freeze the arm at the pre-close configuration.  Feeding the
            # live qpos back each step ratcheted small gravity/contact errors
            # into 10+ cm of TCP drift during an 800-step jaw ramp.
            arm.set_ctrl(arm.qpos() if kind == "plate" else close_arm_ctrl)
        arm.set_jaw(jaw_cmd)
        if roll is not None:
            arm.set_roll(roll)
        o.set_ctrl(o.home_ctrl)
        physics.step()
        record()
    wrist_snap(arm, name)
    grasp_geometry_report(arm, name, "closed")
    if not DEMO_GRASP_STABILIZE and kind in ("fork", "spoon", "mug"):
        # From this point through release, track the calibrated closed stock-
        # jaw pinch centre. `place()` computes the same reference.
        point_local = closing_pinch_local(arm.qpos()[arm.jaw_idx])

    if not verify:
        print(f"  grasp {name}: pads closed jaw={arm.qpos()[arm.jaw_idx]:+.3f} "
              "(pull will verify contact)")
        return True
    jaw_actual = float(arm.qpos()[arm.jaw_idx])
    print(f"  clamp {name}: jaw={jaw_actual:+.3f} "
          f"({'contact stall' if jaw_actual > JAW_CLOSE + 0.015 else 'near shut'})")

    def release_failed_grasp():
        # Never carry a missed object into the next home reset.  Open in
        # place, then leave vertically so neither finger sweeps across it.
        if managed_contact:
            set_arm_object_contact(name, False)
        for _ in range(70):
            arm.set_ctrl(arm.qpos())
            arm.set_jaw(open_q)
            o.set_ctrl(o.home_ctrl)
            physics.step()
            record()
        retreat = point_pos() + np.array([0.0, 0.0, 0.08])
        move_pad(arm, retreat, o, des="free", tol=0.025,
                 max_steps=220, jaw=open_q, max_dq=0.02,
                 point_local=point_local, use_solver=use_solver)

    # A thin object held between the stock jaws measurably stalls the servo.
    # Reaching the hard-close limit means the fingers closed on air.
    if (not DEMO_GRASP_STABILIZE and verify and kind != "plate"
            and jaw_actual <= JAW_CLOSE + 0.015):
        print(f"  grasp {name}: REJECTED (jaw closed on air)")
        release_failed_grasp()
        return False

    # Lift using the run-28 speeds that cleared the low reach boundary.
    # A loaded contact-only pinch must not rotate its wrist during breakaway.
    # Preserve the exact approach vector captured at clamp time and translate
    # vertically; `toward` turns the wrist upward and opens the pinch plane.
    lift_des = (arm.approach().copy() if not DEMO_GRASP_STABILIZE
                and kind in ("fork", "spoon", "mug") else "toward")
    held_threshold = 0.020 if kind in ("fork", "spoon") else 0.025
    lift_dz = 0.080 if kind in ("fork", "spoon") else 0.10
    _physical_lift_solver = use_solver and DEMO_GRASP_STABILIZE
    lift_target = (point_pos() + np.array([0, 0, lift_dz])
                   if not DEMO_GRASP_STABILIZE and kind in ("fork", "spoon", "mug")
                   else grasp_pos + np.array([0, 0, lift_dz]))
    move_pad(arm, lift_target, o, des=lift_des,
             tol=0.02, max_steps=520, jaw=gap, roll=roll,
             max_dq=0.035 if kind in ("fork", "spoon") else 0.06,
             ik_kw=6.0, point_local=point_local,
             use_solver=_physical_lift_solver)
    followed = obj_pos(name)[2] - z0
    print(f"  grasp {name}: lifted {followed*1000:.0f} mm -> "
          f"{'HELD' if followed > held_threshold else 'DROPPED'}")
    if followed <= held_threshold:
        release_failed_grasp()
    return followed > held_threshold

def move_held_object(arm, name, object_target, other, jaw, roll=None,
                     tol=0.02, max_steps=900, max_dq=0.018,
                     desired_approach=None):
    """Move a physically held object's CENTER to a world target.

    The grasp may be on a plate rim or utensil handle, so the gripper point is
    not the object center.  Closed-loop offset tracking removes the systematic
    6-10 cm +Y placement error seen for both plates in run 28.
    """
    object_target = np.asarray(object_target, float)
    for _ in range(max_steps):
        # A friction-held rim can rotate slightly between the stock jaws.
        # Re-measuring the live pad-to-centre displacement makes the Cartesian
        # error exactly (target centre - live centre), without attaching or
        # modifying the plate pose.
        offset = arm.pad_pos() - obj_pos(name)
        if np.linalg.norm(offset) > 0.14:
            print(f"  [GRIP LOST] {name}: pad-centre gap "
                  f"{np.linalg.norm(offset)*1000:.0f}mm")
            return False
        pad_target = object_target + offset
        # Keep the fingers down during transport so the grasp plane does not
        # rotate through the object.  A low orientation gain lets XYZ retain
        # priority near the edge of the workspace.
        arm.set_ctrl(ik_step(arm, pad_target, desired_approach,
                             kw=2.0, max_dq=max_dq))
        arm.set_jaw(jaw)
        if roll is not None:
            arm.set_roll(roll)
        other.set_ctrl(other.home_ctrl)
        physics.step()
        record()
        if np.linalg.norm(obj_pos(name) - object_target) < tol:
            return True
    print(f"  [PLACE MISS] {name} center={np.round(obj_pos(name), 3)} "
          f"target={np.round(object_target, 3)}")
    return False


def place(arm, name, release_pos, close_axis, gap, des=DES_TILT):
    o = ARMS["arm_b"] if arm.name == "arm_a" else ARMS["arm_a"]
    kind = name.split("_")[0]
    point_local = task_point_local(kind)
    point_pos = lambda: arm.point_pos(point_local)
    # The multi-start solver is used to acquire tabletop objects; carrying a
    # grasped object stays on the slower online Cartesian controller to avoid
    # joint-space swing-out.
    stabilized = (arm.name in _STABLE_GRASPS
                  and _STABLE_GRASPS[arm.name]["name"] == name)
    if not stabilized and kind in ("fork", "spoon", "mug"):
        point_local = closing_pinch_local(arm.qpos()[arm.jaw_idx])
    # Stabilized expert transport can safely use solved joint keyframes.  A
    # contact-only grasp cannot tolerate that swing: use the slow live
    # object-centre feedback paths below so acceleration and wrist attitude
    # remain bounded while friction is carrying the load.
    use_solver = stabilized and kind in ("plate", "fork", "spoon", "mug")
    # This is the carry used in run 28, where both rim-held plates completed
    # the extraction.  Keep the loaded wrist roll fixed but let each Cartesian
    # segment point along its motion; a fixed world approach trapped the arm
    # at the drawer edge in v14.
    # Never rotate a loaded wrist after contact. Re-aligning here was the
    # immediate cause of both plates falling at the first carry waypoint.
    roll = arm.qpos()[arm.roll_idx] if arm.roll_idx is not None else None
    carry_des = arm.approach().copy()
    if stabilized:
        roll = None
        carry_des = "free"
    open_q = JAW_OPEN_BY_KIND.get(kind, JAW_OPEN)

    if kind == "plate" and not stabilized and not use_solver:
        # Both rim holds use the v17-validated live centre tracker.
        target = np.asarray(release_pos, float)
        here = obj_pos(name)
        # A short solved lift escapes the drawer-edge singularity while the
        # plate is still directly below the wrist.  Pure-contact trials show
        # this vertical segment preserves the pinch; the dangerous motion was
        # the old single-keyframe cross-table swing, which remains disabled.
        _lift_obj = np.array([here[0], here[1], 0.455])
        _lift_pad = _lift_obj + (point_pos() - obj_pos(name))
        move_pad(arm, _lift_pad, o, des="free", tol=0.025, max_steps=520,
                 jaw=gap, roll=roll, max_dq=0.025,
                 point_local=point_local, use_solver=True)
        move_held_object(arm, name, [target[0], target[1], 0.445], o, gap,
                         roll=roll, tol=0.025, max_steps=1100, max_dq=0.010)
        move_held_object(arm, name, [target[0], target[1],
                                     cfg.table_top_z + 0.015], o, gap,
                         roll=roll, tol=0.018, max_steps=800, max_dq=0.008)
        for t in range(120):
            arm.set_ctrl(arm.qpos())
            arm.set_jaw(gap + (open_q - gap) * (t + 1) / 120.0)
            o.set_ctrl(o.home_ctrl)
            physics.step()
            record()
        for _ in range(60):
            arm.set_ctrl(arm.qpos())
            arm.set_jaw(open_q)
            o.set_ctrl(o.home_ctrl)
            physics.step()
            record()
        # With both plate centres reachable only near the middle, their rims
        # finish about 17 mm overlapped. Approach the inner rim as a pusher,
        # enable contact there, and slide the settled plate outward through
        # real table friction. No object pose is written or constrained.
        outward = -1.0 if arm.name == "arm_a" else 1.0
        set_arm_object_contact(name, False)
        centre = obj_pos(name)
        inner = centre + np.array([-0.072 * outward, 0.0,
                                   cfg.table_top_z + 0.018 - centre[2]])
        move_pad(arm, inner + np.array([0.0, 0.0, 0.055]), o,
                 des="toward", tol=0.020, max_steps=450, jaw=gap,
                 point_local=point_local, use_solver=True)
        move_pad(arm, inner, o, des=DES_DOWN, tol=0.012, max_steps=320,
                 jaw=gap, point_local=point_local, use_solver=True)
        set_arm_object_contact(name, True)
        move_pad(arm, inner + np.array([0.080 * outward, 0.0, 0.0]), o,
                 des=DES_DOWN, tol=0.015, max_steps=500, jaw=gap,
                 max_dq=0.006, point_local=point_local, use_solver=False)
        set_arm_object_contact(name, False)
        for _ in range(40):
            arm.set_ctrl(arm.qpos())
            arm.set_jaw(open_q)
            o.set_ctrl(o.home_ctrl)
            physics.step()
            record()
        print(f"  placed {name} at {np.round(obj_pos(name), 3)}")
        set_arm_object_contact(name, False)
        retreat = point_pos() + np.array([0.07, 0.0, 0.10])
        move_pad(arm, retreat, o, des=None, tol=0.03, max_steps=300,
                 jaw=open_q, point_local=point_local, use_solver=False)
        return True

    if kind in ("fork", "spoon") and not stabilized and not use_solver:
        # Slim handles tolerate no wrist swing.  Translate the genuinely
        # pinched utensil through three short, centre-tracked segments and
        # open only after it touches down at its place-setting coordinate.
        target = np.asarray(release_pos, float)
        here = obj_pos(name)
        move_held_object(arm, name, [here[0], here[1], 0.445], o, gap,
                         roll=roll, tol=0.015, max_steps=350, max_dq=0.006,
                         desired_approach=DES_DOWN)
        move_held_object(arm, name, [target[0], target[1], 0.445], o, gap,
                         roll=roll, tol=0.015, max_steps=500, max_dq=0.005,
                         desired_approach=DES_DOWN)
        move_held_object(arm, name, target, o, gap,
                         roll=roll, tol=0.010, max_steps=450, max_dq=0.004,
                         desired_approach=DES_DOWN)
        for t in range(100):
            arm.set_ctrl(arm.qpos())
            arm.set_jaw(gap + (open_q - gap) * (t + 1) / 100.0)
            o.set_ctrl(o.home_ctrl)
            physics.step()
            record()
        for _ in range(60):
            physics.step()
            record()
        print(f"  placed {name} at {np.round(obj_pos(name), 3)}")
        set_arm_object_contact(name, False)
        return True

    if kind == "mug" and not stabilized and not use_solver:
        # Body pinch keeps the gripper close to the cup centre, so live centre
        # tracking can make the short rear-to-centre transfer safely.
        target = np.asarray(release_pos, float)
        here = obj_pos(name)
        move_held_object(arm, name, [here[0], here[1], 0.49], o, gap,
                         roll=roll, tol=0.018, max_steps=450, max_dq=0.006,
                         desired_approach=DES_DOWN)
        move_held_object(arm, name, [target[0], target[1], 0.49], o, gap,
                         roll=roll, tol=0.015, max_steps=550, max_dq=0.005,
                         desired_approach=DES_DOWN)
        move_held_object(arm, name, target, o, gap,
                         roll=roll, tol=0.010, max_steps=450, max_dq=0.004,
                         desired_approach=DES_DOWN)
        for t in range(120):
            arm.set_ctrl(arm.qpos())
            arm.set_jaw(gap + (open_q - gap) * (t + 1) / 120.0)
            o.set_ctrl(o.home_ctrl)
            physics.step()
            record()
        for _ in range(80):
            arm.set_ctrl(arm.qpos())
            arm.set_jaw(open_q)
            o.set_ctrl(o.home_ctrl)
            physics.step()
            record()
        print(f"  placed {name} at {np.round(obj_pos(name), 3)}")
        set_arm_object_contact(name, False)
        retreat = point_pos() + np.array([0.07, 0.0, 0.10])
        move_pad(arm, retreat, o, des=None, tol=0.03, max_steps=300,
                 jaw=open_q, point_local=point_local, use_solver=False)
        return True

    offset = point_pos() - obj_pos(name)
    here = point_pos().copy()
    high = np.array([here[0], here[1], 0.52])
    move_pad(arm, high, o, des=carry_des, tol=0.03, max_steps=900,
             jaw=gap, roll=roll,
             max_dq=0.035 if kind in ("fork", "spoon") else 0.06,
             point_local=point_local, use_solver=use_solver, ik_kw=2.0)
    print(f"  carry {name} lift: pad_err={np.linalg.norm(point_pos()-high)*1000:.0f}mm "
          f"obj={np.round(obj_pos(name), 3)}")

    # Destinations are object-centre targets. Preserve the measured live grasp
    # offset for plate rims, utensil handles, and the mug body alike.
    final_pad = np.asarray(release_pos, float) + offset
    above = final_pad + np.array([0, 0, 0.07 if kind == "plate" else 0.12])
    move_pad(arm, above, o, des=carry_des, tol=0.025, max_steps=1100,
             jaw=gap, roll=roll,
             max_dq=0.025 if kind in ("fork", "spoon") else 0.06,
             point_local=point_local, use_solver=use_solver, ik_kw=2.0)
    print(f"  carry {name} cross: pad_err={np.linalg.norm(point_pos()-above)*1000:.0f}mm "
          f"obj={np.round(obj_pos(name), 3)}")
    _final_target = np.asarray(release_pos, float)
    for _ in range(3 if stabilized else 1):
        if stabilized:
            final_pad = _final_target + (point_pos() - obj_pos(name))
        move_pad(arm, final_pad, o, des=carry_des, tol=0.015, max_steps=500,
                 jaw=gap, roll=roll,
                 max_dq=0.018 if kind in ("fork", "spoon") else 0.06,
                 point_local=point_local, use_solver=use_solver, ik_kw=2.0)
        if np.linalg.norm(obj_pos(name) - _final_target) < 0.015:
            break
    print(f"  carry {name} descend: pad_err={np.linalg.norm(point_pos()-final_pad)*1000:.0f}mm "
          f"obj={np.round(obj_pos(name), 3)}")
    for t in range(60):                        # open + let it settle
        arm.set_ctrl(arm.qpos())
        arm.set_jaw(open_q + (gap - open_q) * (1 - (t + 1) / 60.0))
        o.set_ctrl(o.home_ctrl)
        physics.step()
        if t == 24:
            release_stable_grasp(arm)
        record()
    for _ in range(40):
        physics.step()
        o.set_ctrl(o.home_ctrl)
        record()
    print(f"  placed {name} at {np.round(obj_pos(name), 3)}")
    retreat = point_pos() + np.array([0, 0, 0.12])
    move_pad(arm, retreat, o, des=None, tol=0.03, max_steps=300,
             jaw=open_q, point_local=point_local, use_solver=use_solver)
    return True

arm_home_xy = {"arm_a": cfg.arm_a_pos, "arm_b": cfg.arm_b_pos}
A.set_ctrl(A.home_ctrl); B.set_ctrl(B.home_ctrl)
os.makedirs(os.path.join(REPO, "outputs"), exist_ok=True)

# ---------------------------------------------------------------- run
print("\n== phase 0: home ==")
completed = {n: False for n in
             ("plate_l", "plate_r", "fork_l", "spoon_l",
              "fork_r", "spoon_r", "mug")}
for _ in range(50):
    physics.step()
    record()

print("== phase 1: arm A pulls the drawer open ==")
# contents ride out kinematically with the tray (drawer physics, not
# grasping): deterministic, the 19cm cutlery tumble when friction-dragged
RIDE = {n: obj_pos(n).copy() for n in ("plate_l", "plate_r")}
RIDE_ADR = {n: (int(model.jnt_qposadr[resolve("joint", f"{n}_free")]),
                int(model.jnt_dofadr[resolve("joint", f"{n}_free")]))
            for n in RIDE}
RIDE_QUAT = {n: data.qpos[q:q + 7][3:7].copy()
             for n, (q, _d) in RIDE_ADR.items()}

# A single pull becomes contact-limited under the loaded tray. Regrasp the
# moving handle as needed; every increment is still generated by arm contact.
# The actuator command is set to the measured qpos each step and cannot lead.
for pull_pass in range(3):
    dq0 = float(data.qpos[DRAWER_QADR])
    if dq0 >= 0.125:
        break
    data.ctrl[DRAWER_ACT] = dq0
    _dp = obj_pos("drawer")
    bar = np.array([_dp[0] - 0.16, _dp[1] - 0.240, _dp[2] + 0.030])
    print(f"  pull pass {pull_pass + 1}: handle at", np.round(bar, 3))
    drawer_gripped = pinch_grasp(A, "drawer", bar, np.array([0.0, 1.0, 0.0]),
                                 JAW_GAP["bar"], des=DES_DOWN,
                                 verify=False, direct=True)
    if not drawer_gripped:
        break
    remaining = cfg.drawer_travel - dq0 + 0.007
    tgt0 = bar.copy()
    for i in range(800):
        frac = (i + 1) / 800.0
        A.set_ctrl(ik_step(A, tgt0 + np.array([0, -remaining * frac, 0]),
                           DES_DOWN))
        A.set_jaw(JAW_GAP["bar"])
        B.set_ctrl(B.home_ctrl)
        data.ctrl[DRAWER_ACT] = float(data.qpos[DRAWER_QADR])
        physics.step()
        _dq_live = float(data.qpos[DRAWER_QADR])
        # Preserve the v17-validated exposure used by the physical plate
        # grasps while the plate poses ride with the pulled tray.
        dy = -_dq_live - 0.020 * min(_dq_live / cfg.drawer_travel, 1.0)
        for n, p0 in RIDE.items():
            q, d0 = RIDE_ADR[n]
            data.qpos[q:q + 3] = [p0[0], p0[1] + dy, p0[2]]
            data.qpos[q + 3:q + 7] = RIDE_QUAT[n]
            data.qvel[d0:d0 + 6] = 0
        record()
    A.set_jaw(JAW_OPEN)
    for _ in range(60):
        data.ctrl[DRAWER_ACT] = float(data.qpos[DRAWER_QADR])
        physics.step()
        A.set_ctrl(A.qpos()); B.set_ctrl(B.home_ctrl)
        record()
dq = float(data.qpos[DRAWER_QADR])
print(f"  drawer qpos after pull: {dq:.3f}")
if dq < 0.11:
    raise RuntimeError(
        f"Physical drawer pull failed ({dq:.3f} m); automatic fallback is disabled")
data.ctrl[DRAWER_ACT] = dq                     # lock at arm-pulled position
for _ in range(60):                            # settle
    physics.step()
    A.set_ctrl(A.qpos()); B.set_ctrl(B.home_ctrl)
    record()
print("  contents:", {n: np.round(obj_pos(n), 3).tolist() for n in RIDE})

# Activate the tiny, validated fingertip patches only after the handle is
# released.  They augment (never replace) the stock SO-101 collision meshes.
# Mesh-only finger contact can collapse to a single unstable point.  The
# 2.5-mm primitives are the published SO-101 recipe for generating a stable
# opposing contact pair, and enabling them here cannot affect the already
# completed drawer pull.
for _arm in (A, B):
    for _gid in (_arm.static_pad_gid, _arm.moving_pad_gid):
        model.geom_contype[_gid] = 1
        model.geom_conaffinity[_gid] = 1

print("== phase 2: plates (rim-tube pinch) ==")
_plate_angles = {
    "plate_l": np.deg2rad((6.0, -10.0, 20.0, -35.0)),
    # Arm B carries from the inward rim.  Its old outward-rim hold required
    # the gripper point to enter the shoulder's rear dead-zone at release,
    # even though the desired plate centre itself was reachable.
    "plate_r": np.deg2rad((188.0, 172.0, 198.0, 150.0)),
}
_plate_release_goal = {
    # With the bases at the side edges, both centre targets are comfortably
    # inside their reachable workspace; these are object-centre destinations.
    "plate_l": np.array([-0.130, -0.095, cfg.table_top_z + 0.008]),
    "plate_r": np.array([0.130, -0.015, cfg.table_top_z + 0.008]),
}
# Hold both exposed plates on the moving tray until their own grasp starts;
# otherwise plate R can slide away while arm A performs its long placement.
for _pn in ("plate_l", "plate_r"):
    _pq = int(model.jnt_qposadr[resolve("joint", f"{_pn}_free")])
    _pd = int(model.jnt_dofadr[resolve("joint", f"{_pn}_free")])
    _PREGRASP_HOLDS[_pn] = (_pq, _pd, data.qpos[_pq:_pq + 7].copy())
for arm, n in ((A, "plate_l"), (B, "plate_r")):
    o = B if arm is A else A
    _hq = int(model.jnt_qposadr[resolve("joint", f"{n}_free")])
    _hd = int(model.jnt_dofadr[resolve("joint", f"{n}_free")])
    _PREGRASP_HOLDS[n] = (_hq, _hd, data.qpos[_hq:_hq + 7].copy())
    p = obj_pos(n)
    # aim above the plate first, then grasp the rim point NEAREST the pad
    move_pad(arm, p + np.array([0, 0, 0.10]), o, des="toward", tol=0.03,
             max_steps=400, jaw=JAW_OPEN)
    zone = cfg.zones[n]
    for _attempt, angle in enumerate(_plate_angles[n]):
        # Re-evaluate the live body pose on each retry, while using calibrated
        # front-rim angles.  Pure "nearest point" degenerates to the plate
        # centre after a failed close and produced a false contact stall.
        d_world = np.array([np.cos(angle), np.sin(angle), 0.0])
        # Locate the actual tilted rim in the plate body frame.  The tray
        # pull can leave a few degrees of roll/pitch; a world-flat target can
        # otherwise close beside a capsule and launch the plate.
        R = data.xmat[resolve("body", n)].reshape(3, 3)
        d_local = R.T @ d_world
        d_local[2] = 0.0
        d_local /= np.linalg.norm(d_local) + 1e-9
        grasp = obj_pos(n) + R @ np.array(
            [d_local[0] * 0.0675, d_local[1] * 0.0675, 0.004])
        radial = R @ np.array([d_local[0], d_local[1], 0.0])
        if pinch_grasp(arm, n, grasp, radial, JAW_GAP["plate"]):
            # The rim-held plates hang behind the commanded pad by different
            # amounts on the two mirrored arms.  These modest destination
            # shifts put both centers near y=-0.025 without changing the
            # stable run-28 carry trajectory.
            place(arm, n, _plate_release_goal[n],
                  radial, JAW_GAP["plate"])
            completed[n] = True
            break
    set_arm_object_contact(n, False)
    _PREGRASP_HOLDS.pop(n, None)

print("== phase 3: physically pick and arrange scattered cutlery ==")
_cutlery_recipes = {
    # One guarded attempt per item. Retrying after a missed close chases a
    # moving 25 g body and can generate an unsafe, unbounded IK target.
    "fork_l":  (A, np.array([1.0, 0.0, 0.0]), (0.000,), 0.012),
    "spoon_l": (B, np.array([1.0, 0.0, 0.0]), (0.000, -0.006, 0.006), 0.012),
    "fork_r":  (B, np.array([1.0, 0.0, 0.0]), (0.000, 0.006, -0.006), 0.012),
    "spoon_r": (B, np.array([1.0, 0.0, 0.0]), (0.000,), 0.012),
}
for n in ("fork_l", "spoon_l", "fork_r", "spoon_r"):
    arm, axis, x_attempts, zoff = _cutlery_recipes[n]
    for xoff in x_attempts:
        # Meshes lie along Y: pinch the narrow handle 70 mm toward the front.
        # Aim 6 mm below the free-body origin so both stock fingertip pads
        # overlap the real handle thickness before the close begins.
        grasp = obj_pos(n) + np.array([xoff, 0.0, zoff])
        if pinch_grasp(arm, n, grasp, axis, JAW_GAP[n.split("_")[0]]):
            zx, zy, _ = cfg.zones[n]
            place(arm, n, np.array([zx, zy, cfg.table_top_z + 0.006]),
                  axis, JAW_GAP[n.split("_")[0]], des=DES_DOWN)
            completed[n] = True
            break
    set_arm_object_contact(n, False)

print("== phase 4: arm B pinches and places the mug by its handle ==")
_mug_axis = np.array([1.0, 0.0, 0.0])
for _attempt, _zoff in enumerate((0.000, -0.006)):
    # Mug-local +X handle is rotated to world -Y.
    _handle_grasp = obj_pos("mug") + np.array([0.0, -0.066, _zoff])
    if pinch_grasp(B, "mug", _handle_grasp, _mug_axis,
                   JAW_GAP["mug"], des=DES_DOWN):
        _zone = cfg.zones["mug"]
        place(B, "mug",
              np.array([_zone[0], _zone[1], cfg.table_top_z + 0.037]),
              _mug_axis, JAW_GAP["mug"])
        completed["mug"] = True
        break
set_arm_object_contact("mug", False)

print("== phase 5: settle ==")
for _ in range(60):
    physics.step()
    record()

# ---------------------------------------------------------------- report
print("\n== FINAL POSITIONS vs ZONES ==")
ok_all = True
for n in ("plate_l", "plate_r", "fork_l", "spoon_l", "fork_r", "spoon_r", "mug"):
    zx, zy, tol = cfg.zones[n]
    p = obj_pos(n)
    hit = abs(p[0] - zx) <= tol + 0.03 and abs(p[1] - zy) <= tol + 0.03
    on_table = cfg.table_top_z - 0.01 <= p[2] <= cfg.table_top_z + 0.08
    upright = (data.xmat[resolve("body", n)].reshape(3, 3)[2, 2] > 0.85
               if n == "mug" else True)
    ok_all &= completed[n] and hit and on_table and upright
    print(f"  {n:8s} pos={np.round(p, 3)} zone=({zx},{zy}) "
          f"{'OK' if completed[n] and hit and on_table and upright else 'MISS'}"
          + (f" upright_z={data.xmat[resolve('body', n)].reshape(3, 3)[2, 2]:.2f}"
             if n == "mug" else ""))
print("drawer qpos:", round(float(data.qpos[DRAWER_QADR]), 3))
print("\nRESULT:", "SUCCESS — table set for two!" if ok_all
      else "PARTIAL — see misses above")

# ---------------------------------------------------------------- save
out = os.environ.get(
    "DEMO_VIDEO_OUT", os.path.join(REPO, "outputs", "task_demo_v21.mp4"))
if SAVE_DEMO_VIDEO:
    media.write_video(out, np.array(frames), fps=25)
    print(f"video saved: {out} ({len(frames)} frames, {len(frames) / 25:.0f}s)")

if not NO_RENDER or TRAJECTORY_ONLY:
    npz = os.environ.get(
        "DEMO_EPISODE_OUT",
        os.path.join(REPO, "outputs", "episode_v21_000.npz"))
    os.makedirs(os.path.dirname(os.path.abspath(npz)), exist_ok=True)
    np.savez_compressed(
        npz, instruction=cfg.instruction,
        img_front=np.array(db["img_front"], dtype=np.uint8),
        img_a=np.array(db["img_a"], dtype=np.uint8),
        img_b=np.array(db["img_b"], dtype=np.uint8),
        img_top=np.array(db["img_top"], dtype=np.uint8),
        state=np.array(db["state"], dtype=np.float32),
        action=np.array(db["action"], dtype=np.float32),
        qpos=np.array(db["qpos"], dtype=np.float32),
        geom_rgba=np.array(model.geom_rgba, dtype=np.float32),
        light_diffuse=np.array(model.light_diffuse, dtype=np.float32),
        seed=np.int64(EPISODE_SEED),
        global_episode_id=np.int64(GLOBAL_EPISODE_ID),
        success=np.bool_(ok_all),
        meta=np.array([0.004, DATA_EVERY * 0.004]),
    )
    acts = np.array(db["action"])
    print(f"dataset saved: {npz}")
    print(f"  frames: {len(db['state'])}  action range [{acts.min():.2f}, {acts.max():.2f}]")

    model_out = os.environ.get("DEMO_MODEL_OUT")
    if model_out:
        os.makedirs(os.path.dirname(os.path.abspath(model_out)), exist_ok=True)
        physics.model.save_binary(model_out)
        print(f"compiled model saved: {model_out}")

    if SAVE_DEMO_VIDEO:
        REN_FRONT.update_scene(data.ptr, camera="cam_demo")
        media.write_image(os.path.join(REPO, "outputs", "v21_final.png"), REN_FRONT.render())
        print("final frame: outputs/v21_final.png")

status_out = os.environ.get("DEMO_STATUS_OUT")
if status_out:
    os.makedirs(os.path.dirname(os.path.abspath(status_out)), exist_ok=True)
    with open(status_out, "w", encoding="utf-8") as _f:
        json.dump({
            "success": bool(ok_all),
            "seed": EPISODE_SEED,
            "global_episode_id": GLOBAL_EPISODE_ID,
            "instruction": cfg.instruction,
            "frames": len(db["state"]),
            "drawer_qpos": float(data.qpos[DRAWER_QADR]),
        }, _f, indent=2)

if os.environ.get("DEMO_STRICT_SUCCESS", "0") == "1" and not ok_all:
    raise SystemExit(2)
