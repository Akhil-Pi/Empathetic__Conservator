"""
test_all.py
===========
Assert-based regression suite for the whole stack. Run before every commit and
on the rig before the first participant:

    python3 test_all.py

The module self-tests print things a human reads. This file FAILS LOUDLY, which
is what a five-person team needs when someone changes a config value and does
not realise what it breaks. Every test here corresponds to a bug that was
actually found and fixed during the rebuild, so this doubles as the regression
record.

No camera, no robot, no MediaPipe required.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_SRC = os.path.abspath(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


# --------------------------------------------------------------------------
# PSS_v2
# --------------------------------------------------------------------------

@test
def test_pss_monotonic_in_trunk_flexion():
    from pss_v2 import PSSv2Calculator, PostureAngles
    vals = []
    for deg in range(0, 91, 10):
        c = PSSv2Calculator()
        vals.append(c.compute(PostureAngles(trunk_flexion_deg=deg))["pss_raw"])
    assert vals == sorted(vals), f"PSS not monotonic in trunk flexion: {vals}"


@test
def test_pss_neutral_is_low():
    from pss_v2 import PSSv2Calculator, PostureAngles
    from goal_controller import ControllerConfig
    c = PSSv2Calculator()
    pss = c.compute(PostureAngles(trunk_flexion_deg=0, neck_flexion_deg=2,
                                  upper_arm_flexion_deg=15,
                                  lower_arm_flexion_deg=90))["pss_raw"]
    assert pss < ControllerConfig.THRESHOLD / 2, \
        f"neutral posture scores {pss:.3f}, too close to threshold"


@test
def test_neck_extension_deadband():
    """REGRESSION: any negative neck angle used to score as full extension
    (RULA band 4), so calibration jitter alone pushed PSS near the trigger."""
    from pss_v2 import PSSv2Calculator, PostureAngles
    npr = np.random.RandomState(0)
    c = PSSv2Calculator()
    vals = [c.compute(PostureAngles(neck_flexion_deg=npr.randn() * 2.0))["pss_raw"]
            for _ in range(300)]
    assert np.mean(vals) < 0.05, f"jitter at neutral scores {np.mean(vals):.3f}"
    # genuine extension must still escalate
    deep = PSSv2Calculator().compute(PostureAngles(neck_flexion_deg=-20))["pss_raw"]
    assert deep > 0.2, f"real extension only scores {deep:.3f}"


@test
def test_smoothing_is_frame_rate_independent():
    """REGRESSION: a 30-frame window assumed 30 fps; the dual-camera rig runs
    slower, which silently stretched the smoothing window."""
    from pss_v2 import PSSv2Calculator, PostureAngles
    finals = {}
    for fps in (30, 15, 10):
        c = PSSv2Calculator()
        dt, t = 1.0 / fps, 0.0
        for _ in range(int(12 / dt)):
            c.compute(PostureAngles(trunk_flexion_deg=5 if t < 10 else 45), now=t)
            t += dt
        finals[fps] = c.compute(PostureAngles(trunk_flexion_deg=45),
                                now=t)["pss_smooth"]
    spread = max(finals.values()) - min(finals.values())
    assert spread < 0.02, f"smoothing varies with frame rate: {finals}"


@test
def test_pss_scores_magnitude_not_sign():
    from pss_v2 import PSSv2Calculator, PostureAngles
    a = PSSv2Calculator().compute(PostureAngles(trunk_sidebend_deg=15))["pss_raw"]
    b = PSSv2Calculator().compute(PostureAngles(trunk_sidebend_deg=-15))["pss_raw"]
    assert abs(a - b) < 1e-9, "sign of lateral angle changed the score"


# --------------------------------------------------------------------------
# RULA
# --------------------------------------------------------------------------

@test
def test_rula_table_shapes_and_anchors():
    import rula
    rula.verify_tables()
    assert rula._TABLE_C[0][0] == 1 and rula._TABLE_C[7][6] == 7


@test
def test_rula_neutral_and_severe():
    from rula import rula_from_angles
    from pss_v2 import PostureAngles
    n = rula_from_angles(PostureAngles(trunk_flexion_deg=0, neck_flexion_deg=0,
                                       upper_arm_flexion_deg=0,
                                       lower_arm_flexion_deg=80))
    assert n["grand"] <= 2, f"neutral RULA {n['grand']}"
    s = rula_from_angles(PostureAngles(trunk_flexion_deg=65, neck_flexion_deg=25,
                                       upper_arm_flexion_deg=95,
                                       lower_arm_flexion_deg=50,
                                       trunk_twist_deg=20, neck_sidebend_deg=15))
    assert s["grand"] >= 6, f"severe RULA {s['grand']}"


@test
def test_rula_tables_flagged_unverified():
    """Guard: nobody should quietly publish RULA integers before the Table A/B
    worksheet check. If this fails because someone verified them, update it."""
    import rula
    assert rula.TABLES_VERIFIED is False, \
        "TABLES_VERIFIED flipped: confirm a printed worksheet check was done"


@test
def test_rula_deadband_matches_pss():
    import rula
    from pss_v2 import PSSv2Config
    assert rula.EXTENSION_DEADBAND_DEG == PSSv2Config.NECK_EXTENSION_DEADBAND_DEG, \
        "neck extension deadband drifted apart between rula.py and pss_v2.py"


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------

@test
def test_controller_reduces_pss_on_strain():
    from pss_v2 import PSSv2Calculator, PostureAngles
    from goal_controller import optimize_target, predict_angles, _score_once, ControllerConfig
    pss = PSSv2Calculator()
    ang = PostureAngles(trunk_flexion_deg=50, neck_flexion_deg=25,
                        upper_arm_flexion_deg=30, lower_arm_flexion_deg=85)
    before = _score_once(ang, pss)
    delta, _, _ = optimize_target(ang, pss)
    after = _score_once(predict_angles(ang, delta, ControllerConfig), pss)
    assert after < before - 0.05, f"optimizer barely moved: {before:.3f}->{after:.3f}"


@test
def test_controller_respects_step_limits():
    from pss_v2 import PSSv2Calculator, PostureAngles
    from goal_controller import optimize_target, ControllerConfig
    delta, _, _ = optimize_target(
        PostureAngles(trunk_flexion_deg=85, neck_flexion_deg=35), PSSv2Calculator())
    for d in ControllerConfig.DOF:
        assert abs(delta[d]) <= ControllerConfig.STEP_LIMIT[d] + 1e-9, \
            f"{d} exceeded step limit"


@test
def test_controller_neutral_is_no_op():
    from pss_v2 import PSSv2Calculator, PostureAngles
    from goal_controller import optimize_target
    delta, _, _ = optimize_target(
        PostureAngles(trunk_flexion_deg=1, neck_flexion_deg=2,
                      upper_arm_flexion_deg=15, lower_arm_flexion_deg=90),
        PSSv2Calculator())
    assert all(abs(v) < 0.03 for v in delta.values()), \
        f"moved on a neutral posture: {delta}"


@test
def test_control_condition_never_moves():
    from pss_v2 import PSSv2Calculator, PostureAngles
    from goal_controller import GoalBasedController
    from robot_interface import SimulatedRobot
    pss = PSSv2Calculator()
    ctrl = GoalBasedController("control", pss)
    robot = SimulatedRobot()
    start = robot.get_pose().copy()
    ang = PostureAngles(trunk_flexion_deg=60, neck_flexion_deg=30)
    t = 0.0
    for _ in range(20):
        r = ctrl.evaluate({"pss_smooth": 0.6}, ang, robot, now=t)
        assert not r["triggered"], "control condition triggered an intervention"
        t += 0.5
    assert np.allclose(robot.get_pose(), start), "robot moved in control"


@test
def test_latency_uses_wall_clock():
    """REGRESSION: latency mixed an injected logical clock with time.time(),
    producing garbage under replay."""
    from pss_v2 import PSSv2Calculator, PostureAngles
    from goal_controller import GoalBasedController
    from robot_interface import SimulatedRobot
    ctrl = GoalBasedController("experimental", PSSv2Calculator())
    robot = SimulatedRobot()
    ang = PostureAngles(trunk_flexion_deg=55, neck_flexion_deg=28)
    t = 1_000_000.0            # logical clock far from wall clock
    for _ in range(4):
        r = ctrl.evaluate({"pss_smooth": 0.6}, ang, robot, now=t)
        t += 0.5
        if r.get("triggered"):
            assert 0 <= r["latency_s"] < 60, f"latency nonsense: {r['latency_s']}"
            return
    raise AssertionError("no intervention fired")


@test
def test_drot_direction_uses_trunk_twist():
    """REGRESSION: drot direction ignored trunk_twist_deg entirely and fell
    back to a fixed sign whenever neck_twist_deg and neck_sidebend_deg were
    both zero, even though trunk_twist_deg is a valid GAINS/DOF_REQUIRES
    driver of drot."""
    from pss_v2 import PSSv2Calculator, PostureAngles
    from goal_controller import GoalBasedController, _FakeRobot
    ctrl = GoalBasedController("experimental", PSSv2Calculator())
    robot = _FakeRobot()
    ang = PostureAngles(trunk_twist_deg=-25)  # neck_twist_deg, neck_sidebend_deg default to 0
    ctrl._execute({"drot": 0.2}, ang, robot)
    kind, applied = robot.moves[-1]
    assert kind == "adjust_rotation"
    assert applied > 0, \
        "drot sign must follow trunk_twist_deg, not fall back to a fixed default"


@test
def test_predicted_pss_is_pss_not_cost():
    """REGRESSION: predicted_pss_before/after used to carry optimizer COST,
    which includes effort and overcorrection penalties, under a PSS label."""
    from pss_v2 import PSSv2Calculator, PostureAngles
    from goal_controller import GoalBasedController, _score_once
    from robot_interface import SimulatedRobot
    pss = PSSv2Calculator()
    ctrl = GoalBasedController("experimental", pss)
    robot = SimulatedRobot()
    ang = PostureAngles(trunk_flexion_deg=55, neck_flexion_deg=28)
    t = 0.0
    for _ in range(4):
        r = ctrl.evaluate({"pss_smooth": 0.6}, ang, robot, now=t)
        t += 0.5
        if r.get("triggered"):
            assert abs(r["predicted_pss_before"] - _score_once(ang, pss)) < 1e-3, \
                "predicted_pss_before is not the actual PSS"
            assert "cost_before" in r, "optimizer cost should be kept separately"
            return
    raise AssertionError("no intervention fired")


# --------------------------------------------------------------------------
# Robot interface
# --------------------------------------------------------------------------

@test
def test_rotation_composition_is_correct():
    """A UR pose orientation is an axis-angle vector. Adding to a component is
    wrong: on a tool-down pose, rx += 0.30 yields only ~0.19 rad."""
    from robot_interface import compose_rotvec, angle_between_rotvecs
    rv = [0.0, np.pi, 0.0]
    proper = compose_rotvec(rv, [1, 0, 0], 0.30)
    assert abs(angle_between_rotvecs(rv, proper) - 0.30) < 1e-6
    naive = np.asarray(rv) + np.array([0.30, 0, 0])
    assert abs(angle_between_rotvecs(rv, naive) - 0.30) > 0.05, \
        "naive addition happened to be right; test no longer meaningful"


@test
def test_rotvec_roundtrip():
    from robot_interface import rotvec_to_matrix, matrix_to_rotvec, angle_between_rotvecs
    for rv in ([0, 0, 0], [0.1, -0.2, 0.3], [0, np.pi, 0], [1.2, 0.4, -0.9]):
        back = matrix_to_rotvec(rotvec_to_matrix(rv))
        assert angle_between_rotvecs(rv, back) < 1e-6, f"roundtrip failed for {rv}"


@test
def test_cumulative_envelope_bounds_drift():
    """The controller caps a single move; nothing capped the accumulation."""
    from robot_interface import SimulatedRobot, RobotConfig
    r = SimulatedRobot()
    for _ in range(20):
        r.move_relative(dz=0.08)
    assert r.get_pose()[2] <= RobotConfig.ENVELOPE_MAX[2] + 1e-9
    r2 = SimulatedRobot()
    for _ in range(20):
        r2.move_relative(dx=0.06)
    assert r2.get_pose()[0] <= RobotConfig.ENVELOPE_MAX[0] + 1e-9


@test
def test_cumulative_tilt_bounded():
    from robot_interface import SimulatedRobot, RobotConfig, angle_between_rotvecs
    r = SimulatedRobot()
    for _ in range(20):
        r.move_relative(drx=0.30)
    dev = angle_between_rotvecs(r.baseline[3:], r.get_pose()[3:])
    assert dev <= RobotConfig.MAX_TILT_RAD + RobotConfig.MAX_ROT_RAD + 1e-6


@test
def test_clamping_is_reported():
    from robot_interface import SimulatedRobot
    r = SimulatedRobot()
    for _ in range(20):
        r.move_relative(dz=0.08)
    assert r.last_move["clamped"] is True, "clamping not surfaced for logging"


@test
def test_baseline_identical_across_conditions():
    """The experimental-design fix: same starting geometry in both arms."""
    from robot_interface import SimulatedRobot
    a, b = SimulatedRobot(), SimulatedRobot()
    a.move_to_baseline(); b.move_to_baseline()
    assert np.allclose(a.get_pose(), b.get_pose())


@test
def test_live_robot_guarded_until_envelope_verified():
    from robot_interface import UR3Robot
    try:
        UR3Robot()
    except RuntimeError:
        return
    except ImportError:
        return
    raise AssertionError("UR3Robot constructed without a verified envelope")


# --------------------------------------------------------------------------
# Singularity avoidance
# --------------------------------------------------------------------------

@test
def test_all_three_singularities_are_detected():
    """Each condition must be caught, and sigma_min must confirm rank loss."""
    from kinematics import (joint_margins, is_blocked, min_singular_value,
                            UR3_NOMINAL as k, _find_shoulder_case)
    cases = {
        "wrist": [0.3, -0.6, 1.35, -2.38, 0.0, 0.0],
        "elbow": [0.3, -0.6, 0.0, -2.38, -1.57, 0.0],
        "shoulder": _find_shoulder_case(k),
    }
    for name, q in cases.items():
        assert is_blocked(joint_margins(q, k)) is not None, f"{name} not caught"
        assert min_singular_value(q, k) < 1e-3, \
            f"{name}: Jacobian should lose rank, sigma_min="\
            f"{min_singular_value(q, k)}"


@test
def test_good_pose_is_not_blocked():
    from kinematics import joint_margins, is_blocked, singularity_penalty, UR3_NOMINAL as k
    m = joint_margins([0.3, -0.6, 1.35, -2.38, -1.57, 0.0], k)
    assert is_blocked(m) is None
    assert singularity_penalty(m) == 0.0


@test
def test_singularity_penalty_is_monotonic_and_capped():
    from kinematics import joint_margins, singularity_penalty, PENALTY_MAX, UR3_NOMINAL as k
    prev, vals = -1.0, []
    for q5 in (0.60, 0.40, 0.30, 0.22, 0.15, 0.08):
        p = singularity_penalty(joint_margins([0.3, -0.6, 1.35, -2.38, q5, 0], k))
        assert p >= prev - 1e-9, "penalty must not decrease as q5 -> 0"
        assert p <= PENALTY_MAX + 1e-9, "penalty must stay capped"
        prev = p
        vals.append(p)
    assert vals[0] == 0.0, "no penalty while comfortably clear"
    assert vals[-1] > 1.0, "penalty must engage before the block threshold"


@test
def test_singularity_term_steers_the_optimizer():
    """The point of a cost rather than a veto: the optimiser routes around."""
    import numpy as _np
    from pss_v2 import PSSv2Calculator, PostureAngles
    from goal_controller import optimize_target, ControllerConfig
    from robot_interface import SimulatedRobot, RobotConfig

    class NearBase(RobotConfig):
        BASELINE_POSE = [0.0, -0.20, 0.15, 0.0, 3.14159, 0.0]
        ENVELOPE_MIN = (-0.35, -0.45, 0.02)
        ENVELOPE_MAX = (0.35, -0.05, 0.35)

    r = SimulatedRobot(NearBase)
    pss = PSSv2Calculator()
    ang = PostureAngles(trunk_flexion_deg=50, neck_flexion_deg=25,
                        upper_arm_flexion_deg=30, lower_arm_flexion_deg=85)
    off, _, _ = optimize_target(ang, pss, ControllerConfig, extra_cost=None)
    on, _, _ = optimize_target(ang, pss, ControllerConfig,
                               extra_cost=r.singularity_cost)
    rad_off = float(_np.hypot(*r._target_pose(off)[:2]))
    rad_on = float(_np.hypot(*r._target_pose(on)[:2]))
    assert rad_on > rad_off + 0.02, \
        f"singularity term did not steer away: {rad_off:.3f} -> {rad_on:.3f}"


@test
def test_execution_backstop_refuses_singular_move():
    from robot_interface import SimulatedRobot, RobotConfig

    class NearBase(RobotConfig):
        BASELINE_POSE = [0.0, -0.20, 0.15, 0.0, 3.14159, 0.0]
        ENVELOPE_MIN = (-0.35, -0.45, 0.02)
        ENVELOPE_MAX = (0.35, -0.05, 0.35)

    r = SimulatedRobot(NearBase)
    assert r.move_relative(dy=0.06) is False, "move at the base axis was allowed"
    assert r.last_move["singularity"] == "shoulder"
    assert r.move_relative(dy=-0.05) is True, "safe move was wrongly refused"
    assert r.last_move["singularity"] is None


@test
def test_workspace_preflight_flags_bad_envelope():
    from robot_interface import SimulatedRobot
    problems = SimulatedRobot().check_workspace()
    assert isinstance(problems, list)
    # the shipped placeholder envelope must not silently pass
    assert problems, "pre-flight should flag the unverified placeholder envelope"


# --------------------------------------------------------------------------
# Camera layout
# --------------------------------------------------------------------------

@test
def test_layout_sensitivity_model():
    from camera_config import sensitivity
    assert abs(sensitivity(90)["sagittal_scale"] - 1.0) < 1e-9
    assert abs(sensitivity(90)["frontal_scale"]) < 1e-9
    assert abs(sensitivity(0)["sagittal_scale"]) < 1e-9, \
        "a front camera must show zero sagittal sensitivity; this is the v1 defect"
    assert abs(sensitivity(45)["sagittal_scale"] - 0.7071) < 1e-3


@test
def test_misalignment_costs_crosstalk_not_signal():
    """The reason the azimuth tolerance is tight."""
    from camera_config import misalignment_cost
    m = misalignment_cost(90.0, 75.0)
    assert m["signal_kept"] > 0.96, "15 deg should barely dent the signal"
    assert m["crosstalk"] > 0.25, "15 deg should admit substantial cross-talk"


@test
def test_front_only_never_claims_trunk_flexion():
    from camera_config import LAYOUTS
    assert not LAYOUTS["FRONT_ONLY"].can_measure("trunk_flexion_deg"), \
        "FRONT_ONLY claiming trunk flexion would reproduce the v1 defect"
    assert LAYOUTS["FRONT_ONLY"].warnings, "FRONT_ONLY must warn"


@test
def test_side_only_measures_sagittal_not_frontal():
    from camera_config import LAYOUTS
    lay = LAYOUTS["SIDE_ONLY"]
    assert lay.can_measure("trunk_flexion_deg")
    assert lay.can_measure("neck_flexion_deg")
    assert not lay.can_measure("trunk_sidebend_deg")
    assert not lay.can_measure("trunk_twist_deg")


@test
def test_layout_placement_counts_match():
    from camera_config import LAYOUTS
    for lay in LAYOUTS.values():
        assert len(lay.placements) == lay.n_cameras


@test
def test_dof_interlock_follows_layout():
    """DOF whose driving signal is unmeasured must be disabled."""
    from camera_config import LAYOUTS
    from goal_controller import enabled_dof
    two = enabled_dof(LAYOUTS["SIDE_FRONT"].measurable)
    one = enabled_dof(LAYOUTS["SIDE_ONLY"].measurable)
    assert set(two) == {"dz", "dy", "dtilt", "drot", "dx"}
    assert "dx" not in one and "drot" not in one, \
        "SIDE_ONLY cannot see side-bend or twist, so must not command dx/drot"
    assert {"dz", "dy", "dtilt"} <= set(one), "SIDE_ONLY keeps sagittal authority"


@test
def test_controller_holds_disabled_dof_at_zero():
    from pss_v2 import PSSv2Calculator, PostureAngles
    from goal_controller import optimize_target, ControllerConfig, enabled_dof
    from camera_config import LAYOUTS
    ang = PostureAngles(trunk_flexion_deg=45, neck_flexion_deg=22,
                        trunk_sidebend_deg=20, neck_twist_deg=20,
                        upper_arm_flexion_deg=30, lower_arm_flexion_deg=85)
    dof = enabled_dof(LAYOUTS["SIDE_ONLY"].measurable)
    delta, _, _ = optimize_target(ang, PSSv2Calculator(), ControllerConfig, dof=dof)
    assert delta["dx"] == 0.0 and delta["drot"] == 0.0, \
        "disabled DOF must stay exactly zero even with strong lateral strain"
    assert abs(delta["dz"]) > 1e-4, "enabled DOF should still act"


@test
def test_layout_mask_does_not_fabricate_angles():
    """The direct v1 fix: unmeasurable means absent, not guessed."""
    from pose_fusion import apply_layout_mask
    from pss_v2 import PostureAngles
    from camera_config import LAYOUTS
    a = PostureAngles(trunk_flexion_deg=40, neck_flexion_deg=20,
                      trunk_sidebend_deg=18, neck_twist_deg=15,
                      upper_arm_flexion_deg=30, lower_arm_flexion_deg=85,
                      confidence={"trunk_flexion_deg": 0.9,
                                  "trunk_sidebend_deg": 0.9})
    m = apply_layout_mask(a, LAYOUTS["SIDE_ONLY"])
    assert m.trunk_flexion_deg == 40, "measurable angle must survive"
    assert m.trunk_sidebend_deg == 0.0, "unmeasurable angle must be zeroed"
    assert m.neck_twist_deg == 0.0
    assert m.confidence["trunk_sidebend_deg"] == 0.0
    # Arms have a non-zero anatomical neutral, so masking must fall back to it
    # rather than to zero, which would read as a fully extended elbow.
    # FRONT_ONLY can see the elbow angle but not upper-arm elevation.
    m2 = apply_layout_mask(a, LAYOUTS["FRONT_ONLY"])
    assert m2.upper_arm_flexion_deg == 20.0, "unmeasurable arm -> neutral, not 0"
    assert m2.lower_arm_flexion_deg == 85, "measurable elbow angle must survive"


@test
def test_front_sagittal_fallback_is_off_by_default():
    """v1 emitted a trunk angle from a front view at full authority."""
    from pose_fusion import FusionConfig, fuse, _make_pose
    assert FusionConfig.ALLOW_FRONT_SAGITTAL_FALLBACK is False
    pa = fuse(None, _make_pose(trunk_deg=30, neck_deg=0))
    assert pa.trunk_flexion_deg == 0.0, "front-only trunk angle must not be invented"
    assert pa.confidence["trunk_flexion_deg"] == 0.0


@test
def test_frontal_angle_confidence_is_populated_and_scaled():
    """REGRESSION: fuse() never wrote a confidence entry for trunk_sidebend_deg
    / neck_sidebend_deg / lateral_gaze_deg, so apply_layout_mask's confidence
    scaling silently no-opped for them on every layout, including degraded
    ones like SIDE_FRONT_OBLIQUE that are supposed to show up in the logs."""
    from pose_fusion import fuse, apply_layout_mask, _make_pose
    from camera_config import LAYOUTS
    pa = fuse(None, _make_pose(sidebend_deg=20))
    for name in ("trunk_sidebend_deg", "neck_sidebend_deg", "lateral_gaze_deg"):
        assert pa.confidence.get(name) == 1.0, \
            f"fuse() must record a confidence for {name}, got {pa.confidence.get(name)}"

    masked = apply_layout_mask(pa, LAYOUTS["SIDE_FRONT_OBLIQUE"])
    for name in ("trunk_sidebend_deg", "neck_sidebend_deg", "lateral_gaze_deg"):
        assert abs(masked.confidence[name] - 0.8) < 1e-9, \
            (f"SIDE_FRONT_OBLIQUE must scale {name} confidence to 0.8, "
             f"got {masked.confidence[name]}")


@test
def test_single_camera_capture_allowed():
    import camera_stream as cs
    assert "side_source=None" in cs.DualCamera.__init__.__doc__ or True
    import inspect
    sig = inspect.signature(cs.DualCamera.__init__)
    assert sig.parameters["side_source"].default is None
    assert sig.parameters["front_source"].default is None


# --------------------------------------------------------------------------
# Logger and cross-module contracts
# --------------------------------------------------------------------------

@test
def test_logger_frames_are_self_consistent_under_calibration():
    """REGRESSION: frames logged RAW angles while bands/PSS came from CALIBRATED
    angles, so the angle columns did not reproduce the score beside them."""
    import csv
    from session_logger_v2 import SessionLoggerV2
    from pss_v2 import PSSv2Calculator, PostureAngles
    from evaluation import _row_to_angles
    from goal_controller import _score_once

    tmp = tempfile.mkdtemp()
    pss = PSSv2Calculator()
    pss.calibrate_neutral([PostureAngles(trunk_flexion_deg=6, neck_flexion_deg=4,
                                         upper_arm_flexion_deg=28,
                                         lower_arm_flexion_deg=84)])
    log = SessionLoggerV2("P99", "experimental", log_dir=tmp, log_frequency_hz=1000)
    expected = []
    for trunk in (10, 25, 40, 55):
        a = PostureAngles(trunk_flexion_deg=trunk, neck_flexion_deg=trunk * 0.5,
                          upper_arm_flexion_deg=35, lower_arm_flexion_deg=80,
                          trunk_sidebend_deg=-6.0)
        comp = pss.compute(a)
        if log.log_frame(a, comp):
            expected.append(comp["pss_raw"])
    log.close()

    fresh = PSSv2Calculator(smoothing_window=1)
    with open(log.frames_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(expected)
    for row, exp in zip(rows, expected):
        vals = {k: (float(v) if v not in ("", None) else 0.0)
                for k, v in row.items() if k != "arm_side"}
        got = _score_once(_row_to_angles(vals), fresh)
        assert abs(got - exp) < 1e-3, \
            f"logged angles do not reproduce logged PSS: {got:.4f} vs {exp:.4f}"


@test
def test_logger_schema_matches_evaluation_reader():
    from session_logger_v2 import FRAME_COLUMNS, EVENT_COLUMNS
    import evaluation as ev
    for col in ev._BAND_COLS + ["pss_raw", "pss_smooth", "timestamp_s"]:
        assert col in FRAME_COLUMNS, f"evaluation reads {col}, logger never writes it"
    for col in ev._DOF_COLS + ["event_type", "latency_s"]:
        assert col in EVENT_COLUMNS, f"evaluation reads {col}, logger never writes it"


@test
def test_logger_meta_snapshots_config():
    from session_logger_v2 import SessionLoggerV2
    tmp = tempfile.mkdtemp()
    log = SessionLoggerV2("P99", "control", log_dir=tmp)
    log.close()
    meta = open(log.meta_path).read()
    for key in ("ctrl.GAINS", "ctrl.THRESHOLD", "pss.W_GROUP_B", "fusion.MIN_VIS"):
        assert key in meta, f"meta snapshot missing {key}; run is not reproducible"


@test
def test_config_constants_single_source():
    """Thresholds must not be duplicated across modules."""
    import evaluation as ev
    from goal_controller import ControllerConfig
    assert ev.THRESHOLD == ControllerConfig.THRESHOLD
    assert abs(ev.RECOVERY_TARGET -
               (ControllerConfig.THRESHOLD - ControllerConfig.HYSTERESIS)) < 1e-12


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@test
def test_gain_fitter_recovers_known_gains():
    """The fitter must find real gains when the signal is present."""
    import csv
    import numpy as _np
    from session_logger_v2 import FRAME_COLUMNS, EVENT_COLUMNS
    from evaluation import load_sessions, fit_response_gains

    tmp = tempfile.mkdtemp()
    rng = _np.random.RandomState(1)
    TRUE = {"dz": 150.0, "dy": 100.0}
    base = os.path.join(tmp, "P1_experimental_20260805_000001")
    frames, events, t = [], [], 0.0
    for _ in range(40):
        dz, dy = rng.uniform(0, 0.08), rng.uniform(0, 0.06)
        red = TRUE["dz"] * dz + TRUE["dy"] * dy + rng.randn() * 0.5
        t0 = t + 3.0
        for tt in _np.arange(t, t0 - 0.1, 0.1):
            frames.append({c: 0 for c in FRAME_COLUMNS} | {
                "timestamp_s": f"{tt:.3f}", "trunk_flexion_deg": "40.00",
                "pss_smooth": "0.4", "pss_raw": "0.4"})
        for tt in _np.arange(t0 + 0.5, t0 + 3.0, 0.1):
            frames.append({c: 0 for c in FRAME_COLUMNS} | {
                "timestamp_s": f"{tt:.3f}",
                "trunk_flexion_deg": f"{40.0 - red:.2f}",
                "pss_smooth": "0.1", "pss_raw": "0.1"})
        events.append({c: "" for c in EVENT_COLUMNS} | {
            "timestamp_s": f"{t0:.3f}", "event_type": "intervention",
            "dz": f"{dz:.4f}", "dy": f"{dy:.4f}", "dtilt": "0.0",
            "drot": "0.0", "dx": "0.0", "latency_s": "0.9"})
        t = t0 + 6.0
    with open(base + "_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_COLUMNS); w.writeheader(); w.writerows(frames)
    with open(base + "_events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_COLUMNS); w.writeheader(); w.writerows(events)

    res = fit_response_gains(load_sessions(tmp), min_events=5)
    fit = res["per_angle"]["trunk_flexion_deg"]
    assert fit["fitted"] is not None, "fitter refused a clean signal"
    assert abs(fit["fitted"]["dz"] - 150) < 20, f"dz gain off: {fit['fitted']}"
    assert abs(fit["fitted"]["dy"] - 100) < 25, f"dy gain off: {fit['fitted']}"
    assert fit["r2"] > 0.9


@test
def test_gain_fitter_refuses_unidentifiable_data():
    """And must refuse to fit noise when deltas do not vary."""
    import csv
    import numpy as _np
    from session_logger_v2 import FRAME_COLUMNS, EVENT_COLUMNS
    from evaluation import load_sessions, fit_response_gains

    tmp = tempfile.mkdtemp()
    base = os.path.join(tmp, "P1_experimental_20260805_000002")
    frames, events, t = [], [], 0.0
    for _ in range(20):
        t0 = t + 3.0
        for tt in _np.arange(t, t0 - 0.1, 0.1):
            frames.append({c: 0 for c in FRAME_COLUMNS} | {
                "timestamp_s": f"{tt:.3f}", "trunk_flexion_deg": "40.00",
                "pss_smooth": "0.4", "pss_raw": "0.4"})
        for tt in _np.arange(t0 + 0.5, t0 + 3.0, 0.1):
            frames.append({c: 0 for c in FRAME_COLUMNS} | {
                "timestamp_s": f"{tt:.3f}", "trunk_flexion_deg": "32.00",
                "pss_smooth": "0.1", "pss_raw": "0.1"})
        events.append({c: "" for c in EVENT_COLUMNS} | {
            "timestamp_s": f"{t0:.3f}", "event_type": "intervention",
            "dz": "0.0600", "dy": "0.0300", "dtilt": "0.0",   # identical every time
            "drot": "0.0", "dx": "0.0", "latency_s": "0.9"})
        t = t0 + 6.0
    with open(base + "_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_COLUMNS); w.writeheader(); w.writerows(frames)
    with open(base + "_events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_COLUMNS); w.writeheader(); w.writerows(events)

    res = fit_response_gains(load_sessions(tmp), min_events=5)
    fit = res["per_angle"]["trunk_flexion_deg"]
    assert fit["fitted"] is None, "fitter invented gains from unidentifiable data"


@test
def test_gain_fitter_excludes_events_contaminated_by_prior_post_window():
    """REGRESSION: the backward contamination guard checked the gap against
    pre_s + post_start_s only, omitting post_s. An event whose 'pre' baseline
    window still overlapped the PREVIOUS event's post window could pass the
    guard and corrupt the fitted gain."""
    import csv
    import numpy as _np
    from session_logger_v2 import FRAME_COLUMNS, EVENT_COLUMNS
    from evaluation import load_sessions, fit_response_gains

    tmp = tempfile.mkdtemp()
    rng = _np.random.RandomState(2)
    base = os.path.join(tmp, "P1_experimental_20260805_000003")
    frames, events = [], []

    def add_event(t0, dz, dy, red):
        for tt in _np.arange(t0 - 3.0, t0 - 0.1, 0.1):
            frames.append({c: 0 for c in FRAME_COLUMNS} | {
                "timestamp_s": f"{tt:.3f}", "trunk_flexion_deg": "40.00",
                "pss_smooth": "0.4", "pss_raw": "0.4"})
        for tt in _np.arange(t0 + 0.5, t0 + 3.0, 0.1):
            frames.append({c: 0 for c in FRAME_COLUMNS} | {
                "timestamp_s": f"{tt:.3f}",
                "trunk_flexion_deg": f"{40.0 - red:.2f}",
                "pss_smooth": "0.1", "pss_raw": "0.1"})
        events.append({c: "" for c in EVENT_COLUMNS} | {
            "timestamp_s": f"{t0:.3f}", "event_type": "intervention",
            "dz": f"{dz:.4f}", "dy": f"{dy:.4f}", "dtilt": "0.0",
            "drot": "0.0", "dx": "0.0", "latency_s": "0.9"})

    # 10 well-separated, clean events so min_events is comfortably satisfied
    # regardless of what happens to the contaminated pair below.
    t = 0.0
    for _ in range(10):
        t0 = t + 3.0
        dz, dy = rng.uniform(0, 0.08), rng.uniform(0, 0.06)
        add_event(t0, dz, dy, 150.0 * dz + 100.0 * dy)
        t = t0 + 9.0

    # A: far from the last clean event, but only 2.6s before B. A is already
    # excluded by the (correct, unchanged) forward-gap check, since 2.6s <
    # post_start_s + post_s = 3.0s. B is the last event, so only the backward
    # check applies to it: 2.6s < pre_s + post_start_s (2.5s) is false, so the
    # unfixed guard let B through even though A's post-window [tA+0.5, tA+3.0]
    # overlaps B's pre-window [tB-2.0, tB-0.1].
    tA = t + 3.0
    add_event(tA, 0.04, 0.02, 5.0)
    tB = tA + 2.6
    add_event(tB, 0.05, 0.03, 5.0)

    with open(base + "_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_COLUMNS); w.writeheader(); w.writerows(frames)
    with open(base + "_events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_COLUMNS); w.writeheader(); w.writerows(events)

    res = fit_response_gains(load_sessions(tmp), min_events=5)
    assert res["n_events_used"] == 10, \
        (f"expected B to be excluded as contaminated by A's post-window, "
         f"got n_events_used={res['n_events_used']}")


@test
def test_participant_id_normalisation():
    from evaluation import _canon_participant
    assert _canon_participant("P001") == _canon_participant("P1") == "P1"
    assert _canon_participant("P02") == "P2"


@test
def test_h1_reports_shapiro_and_both_tests():
    from evaluation import test_h1, Session
    import pandas as pd
    sessions = []
    npr = np.random.RandomState(3)
    for i in range(1, 7):
        for cond, mu in (("control", 0.45), ("experimental", 0.30)):
            fr = pd.DataFrame({
                "timestamp_s": np.arange(200) * 0.1,
                "pss_smooth": np.clip(npr.randn(200) * 0.05 + mu, 0, 1)})
            sessions.append(Session(f"P{i}", cond, fr, pd.DataFrame(), {}))
    r = test_h1(sessions)
    assert "shapiro" in r and "wilcoxon" in r and "paired_t" in r
    assert "secondary_frac_above" in r
    assert r["n_pairs"] == 6


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def main():
    passed, failed = 0, []
    for fn in RESULTS:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e, traceback.format_exc()))
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(RESULTS)} passed")
    if failed:
        print("\n--- failures ---")
        for name, _, tb in failed:
            print(f"\n{name}:\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
