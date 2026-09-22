"""
task_profile.py
================
Part B: task generalisation.

Captures what differs between sustained-precision tasks this pipeline's
method (posture-based strain scoring + goal-based artifact repositioning)
could in principle be applied to: neutral posture assumptions, expected
reach/workspace envelope, which controller DOF are relevant, and any
task-specific notes.

CONSERVATION is the DEFAULT profile and is descriptive only until a caller
explicitly asks to apply it (apply_profile_dof / apply_profile_envelope
below) -- nothing in the existing pipeline imports this module, so no
existing command's behaviour changes just because this file exists.
Applying CONSERVATION reproduces today's ControllerConfig.DOF and
RobotConfig.ENVELOPE_MIN/MAX exactly (see tests/test_all.py's regression
guard); it is a description OF the current defaults, not a new default.

MANUFACTURING_ASSEMBLY is the one additional example profile Part B asks
for, so the method can be shown applied outside conservation. Its geometry
is a PLACEHOLDER, exactly as unmeasured as conservation's own defaults were
before RIG_SETUP.md's "measure on the rig" step -- it must never be treated
as validated for a real manufacturing cell.

Wiring pattern: apply_profile_dof/apply_profile_envelope return a NEW
subclass of the given config (or the live ControllerConfig/RobotConfig if
none given), the same dynamic-subclass technique goal_controller.py already
uses for per-call config overrides (see _scaled_step_cfg). The base config
classes themselves are never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class TaskProfile:
    name: str
    # Documentation / synthetic-generator seeding only -- NOT used to skip
    # PSS_v2's per-session neutral calibration (pss_v2.calibrate_neutral),
    # which is what actually absorbs individual neutral posture at runtime.
    neutral_notes: str
    # (min_xyz, max_xyz) metres, robot base frame. A first-cut starting
    # point for RobotConfig.ENVELOPE_MIN/MAX -- NEVER a substitute for the
    # "measure the workspace envelope on the rig" step in README/RIG_SETUP.md.
    workspace_envelope_m: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    # Which controller DOF this task's fixture/workpiece geometry makes
    # sensible to command at all. Still further gated at runtime by
    # DOF_REQUIRES / the active camera layout (goal_controller.enabled_dof)
    # -- this is the task-level ceiling, not a guarantee any DOF fires.
    relevant_dof: Tuple[str, ...]
    notes: str = ""


CONSERVATION = TaskProfile(
    name="conservation",
    neutral_notes=(
        "Seated/standing bench work; a handheld artifact is held on the "
        "end effector at roughly chest-bench height and reoriented as well "
        "as repositioned (tilt and rotation both relieve real strain here, "
        "e.g. tilting to reduce neck flexion, rotating to counter head "
        "turn)."),
    # Identical to RobotConfig.ENVELOPE_MIN/MAX's shipped defaults.
    workspace_envelope_m=((-0.33, -0.22, 0.38), (-0.18, -0.08, 0.56)),
    # Identical to ControllerConfig.DOF's shipped default, same order.
    relevant_dof=("dz", "dy", "dtilt", "drot", "dx"),
    notes=(
        "DEFAULT profile. Deliberately reproduces existing behaviour "
        "exactly -- see test_conservation_profile_reproduces_existing_"
        "behaviour in tests/test_all.py. This is a DESCRIPTION of the "
        "current shipped defaults, not a new default value."),
)


MANUFACTURING_ASSEMBLY = TaskProfile(
    name="manufacturing_assembly",
    neutral_notes=(
        "Seated electronics rework / small-part assembly at a fixed bench "
        "jig. The workpiece is typically closer to the operator and lower "
        "than conservation's bench-height artifact, and is not tilted or "
        "spun to relieve operator strain the way a handheld conservation "
        "artifact is -- a jig gets moved closer/farther/up/down, not "
        "reoriented."),
    # PLACEHOLDER geometry -- not measured on any real cell. Smaller and
    # lower than conservation's envelope, reflecting a closer, lower
    # bench-jig working distance; must be re-measured before any live use,
    # exactly as conservation's own defaults required on the rig.
    workspace_envelope_m=((-0.25, -0.15, 0.25), (-0.10, -0.02, 0.40)),
    # No dtilt/drot: a fixed assembly jig is repositioned, not reoriented,
    # for this task -- see neutral_notes. This is the concrete "different
    # DOF set" Part B asks the method to demonstrate on a non-conservation
    # task.
    relevant_dof=("dz", "dy", "dx"),
    notes=(
        "Example non-conservation profile (Part B, task generalisation). "
        "PLACEHOLDER geometry: not measured on any real manufacturing "
        "cell. Exists to demonstrate the method applies outside "
        "conservation, not as a validated deployment profile."),
)


PROFILES: Dict[str, TaskProfile] = {
    CONSERVATION.name: CONSERVATION,
    MANUFACTURING_ASSEMBLY.name: MANUFACTURING_ASSEMBLY,
}


def get_profile(name: str = "conservation") -> TaskProfile:
    if name not in PROFILES:
        raise ValueError(f"unknown task profile {name!r}; choose from {sorted(PROFILES)}")
    return PROFILES[name]


# --------------------------------------------------------------------------
# Wiring: additive, opt-in, default-conservation. Nothing calls these unless
# a profile is explicitly selected -- ControllerConfig/RobotConfig are never
# mutated, only subclassed per call (same pattern as goal_controller's
# _scaled_step_cfg).
# --------------------------------------------------------------------------

def apply_profile_dof(profile: TaskProfile, base_cfg=None):
    """A ControllerConfig-shaped class whose DOF is restricted to
    profile.relevant_dof; every other attribute (STEP_LIMIT, GAINS,
    THRESHOLD, ...) is inherited unchanged from base_cfg (or the live
    goal_controller.ControllerConfig if base_cfg is None)."""
    if base_cfg is None:
        from goal_controller import ControllerConfig as base_cfg

    class _ProfiledController(base_cfg):
        DOF = list(profile.relevant_dof)

    _ProfiledController.__name__ = f"ControllerConfig[{profile.name}]"
    return _ProfiledController


def apply_profile_envelope(profile: TaskProfile, base_cfg=None):
    """A RobotConfig-shaped class whose workspace envelope is
    profile.workspace_envelope_m; every other attribute is inherited
    unchanged from base_cfg (or the live robot_interface.RobotConfig)."""
    if base_cfg is None:
        from robot_interface import RobotConfig as base_cfg

    class _ProfiledRobot(base_cfg):
        ENVELOPE_MIN = profile.workspace_envelope_m[0]
        ENVELOPE_MAX = profile.workspace_envelope_m[1]

    _ProfiledRobot.__name__ = f"RobotConfig[{profile.name}]"
    return _ProfiledRobot


# --------------------------------------------------------------------------
# Self-test: no camera, no robot. Verifies conservation reproduces today's
# defaults exactly and manufacturing genuinely differs.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from goal_controller import ControllerConfig
    from robot_interface import RobotConfig

    print("task_profile.py self-test\n")
    ok = True

    cons_ctrl = apply_profile_dof(CONSERVATION)
    same_dof = list(cons_ctrl.DOF) == list(ControllerConfig.DOF)
    print(f"conservation DOF matches ControllerConfig.DOF exactly: {same_dof} "
         f"({cons_ctrl.DOF})")
    ok &= same_dof

    cons_robot = apply_profile_envelope(CONSERVATION)
    same_env = (tuple(cons_robot.ENVELOPE_MIN) == tuple(RobotConfig.ENVELOPE_MIN)
               and tuple(cons_robot.ENVELOPE_MAX) == tuple(RobotConfig.ENVELOPE_MAX))
    print(f"conservation envelope matches RobotConfig defaults exactly: {same_env}")
    ok &= same_env

    mfg_ctrl = apply_profile_dof(MANUFACTURING_ASSEMBLY)
    diff_dof = set(mfg_ctrl.DOF) != set(ControllerConfig.DOF)
    print(f"manufacturing DOF differs from conservation: {diff_dof} ({mfg_ctrl.DOF})")
    ok &= diff_dof
    ok &= ("dtilt" not in mfg_ctrl.DOF and "drot" not in mfg_ctrl.DOF)

    # base config itself must never be mutated by applying a profile
    untouched = list(ControllerConfig.DOF) == ["dz", "dy", "dtilt", "drot", "dx"]
    print(f"base ControllerConfig.DOF left untouched after applying profiles: {untouched}")
    ok &= untouched

    print(f"\nall task_profile checks passed: {bool(ok)}")
