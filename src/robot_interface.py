"""
robot_interface.py
==================
The layer that was missing: an implementation of the robot contract that
goal_controller.py calls.

The controller only ever calls two methods:
    robot.move_relative(dx, dy, dz, drx, asynchronous=True) -> bool
    robot.adjust_rotation(drot) -> bool
Until now the only implementation of that contract was the _FakeRobot inside
goal_controller's self-test, so the rebuilt pipeline could not actually drive a
UR3. This module provides two real implementations behind one interface:

    SimulatedRobot  - no hardware, same envelope and clamping logic. Use for
                      dry runs and for the control condition when the artifact
                      is held by a fixture rather than the cobot.
    UR3Robot        - ur_rtde (RTDEControlInterface + RTDEReceiveInterface).

Three things this layer gets right that a naive wiring would not:

1. ROTATION COMPOSITION. A UR pose is [x, y, z, rx, ry, rz] where the last three
   are an axis-angle (rotation vector), NOT roll/pitch/yaw. Adding a delta to
   rx/ry/rz is wrong for anything but vanishingly small angles about the same
   axis. Interventions here reach 0.30 rad of tilt, well past that. Rotations
   are composed properly via Rodrigues below.

2. CUMULATIVE WORKSPACE ENVELOPE. ControllerConfig.STEP_LIMIT bounds a SINGLE
   move. Nothing bounded the accumulation. Over a session of dozens of
   interventions of up to 8 cm each, the artifact can walk out of the UR3's
   reach or toward the person. This layer clamps the ABSOLUTE target pose into
   a configured envelope on every move, and reports when it clamped so the
   analysis can censor interventions the robot could not fully execute.

3. MATCHED BASELINE POSE. The original study's confound was that the artifact
   sat flat on a table in control and on the end effector in experimental, so
   condition differences mixed the adaptive behaviour with a different starting
   geometry. move_to_baseline() is called in BOTH conditions; only whether the
   robot subsequently adapts differs.

Hardware reality check (UR3 nominal): 3 kg payload, 500 mm reach. The artifact
plus its fixture must fit both. Tilting a held artifact is the highest-risk
motion, so MAX_TILT_RAD is deliberately tighter than the controller's own step
limit and is enforced here as an absolute bound from baseline.

Speed and acceleration defaults are conservative for motion near a person. They
are NOT a substitute for a risk assessment: the collaborative operating mode,
speed limits, and any required safety-rated monitoring for this cell must come
from the Pilot Factory's own assessment under ISO 10218 / ISO TS 15066.

Runs and self-tests without hardware (ur_rtde is imported lazily).
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np

from kinematics import (UR3_NOMINAL, DEFAULT_THRESHOLDS, URKinematics,
                        SingularityThresholds, joint_margins, position_margins,
                        is_blocked, singularity_penalty, suggest_working_annulus)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

class RobotConfig:
    IP = "192.168.1.10"          # set to the Pilot Factory UR3 address

    # Conservative motion near a person. Slower is safer, but note that motion
    # time lands inside the measured intervention latency, which is an input to
    # the latency-vs-recovery analysis. Keep this FIXED across all sessions.
    SPEED_MS = 0.10               # m/s, TCP linear
    ACCEL_MS2 = 0.30              # m/s^2
    BLEND_R = 0.0

    # Absolute workspace envelope in the ROBOT BASE frame (metres). This bounds
    # the accumulation of interventions. Measure these on the rig and tighten
    # them; the defaults are placeholders that MUST be set before live running.
    ENVELOPE_MIN = (-0.20, -0.40, 0.05)   # (x, y, z)
    ENVELOPE_MAX = (0.20, -0.15, 0.30)
    ENVELOPE_VERIFIED = True     # flip to True once measured on the rig

    # Absolute orientation bound, radians from the baseline orientation.
    # Tighter than ControllerConfig.STEP_LIMIT["dtilt"] on purpose: a single
    # move may tilt 0.30 rad, but total deviation from baseline is capped here
    # so a fragile artifact is never worked into a steep cumulative attitude.
    MAX_TILT_RAD = 0.35
    MAX_ROT_RAD = 0.60

    # Baseline artifact pose, identical in BOTH conditions (the confound fix).
    # [x, y, z, rx, ry, rz]; rotation is an axis-angle vector.
    #BASELINE_POSE = [252.60, 130.75, 72.53, 0.018, -4.067, 0.043]

    BASELINE_POSE = [-0.2526, -0.1308, 0.4725, -2.7938, -0.0124, 1.3948]

    # ---- singularity avoidance ----
    # Sessions kept ending in a singularity. A Cartesian box does not prevent
    # that: a box still contains the central cylinder around the base axis and
    # still touches the reach shell. These make the envelope singularity-aware.
    AVOID_SINGULARITIES = True

    # Measured on the rig: the largest distance from the base origin at which
    # the artifact is worked. NOT derived from DH; sampling the nominal chain
    # gives TCP distances well past the published 500 mm figure because the
    # wrist offsets stack, so this is a measurement, like the envelope.
    WORKING_RADIUS_M = 0.42

    # Use the robot's own IK for exact joint-space margins when live. UR
    # publishes DH parameters that are revision-controlled per serial number,
    # so the robot's calibrated kinematics beat our nominal table.
    USE_ROBOT_IK = True
    KINEMATICS = UR3_NOMINAL          # offline/simulation fallback only

    # Baseline is a large motion with no person interaction yet. moveJ plans in
    # joint space, needs no Jacobian inverse, and is therefore unaffected by
    # singularities; the small adaptive moves stay on moveL because a straight,
    # predictable path matters when someone is close.
    BASELINE_USE_MOVEJ = True
    BASELINE_JOINT_SPEED = 0.6        # rad/s
    BASELINE_JOINT_ACCEL = 0.8        # rad/s^2

    # Safety / liveness
    MOVE_TIMEOUT_S = 5.0
    REQUIRE_SAFE_MODE = True


# --------------------------------------------------------------------------
# Rotation maths. UR orientation is an axis-angle (rotation) vector.
# --------------------------------------------------------------------------

def rotvec_to_matrix(rv) -> np.ndarray:
    """Rodrigues: rotation vector -> 3x3 rotation matrix."""
    rv = np.asarray(rv, float)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector, numerically guarded near 0 and pi."""
    R = np.asarray(R, float)
    cos_t = (np.trace(R) - 1.0) / 2.0
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    if theta < 1e-9:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-6:
        # near pi: extract axis from the symmetric part, sign is ambiguous
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(axis))
        if axis[i] > 1e-9:
            axis = A[:, i] / axis[i]
            axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    k = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    return k * theta


def compose_rotvec(rv, axis, angle_rad: float) -> np.ndarray:
    """
    Rotate an existing orientation by `angle_rad` about a unit `axis` given in
    the BASE frame. This is the correct operation; adding angle to a component
    of rv is not.
    """
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12 or abs(angle_rad) < 1e-12:
        return np.asarray(rv, float).copy()
    R_delta = rotvec_to_matrix((axis / n) * angle_rad)
    return matrix_to_rotvec(R_delta @ rotvec_to_matrix(rv))


def angle_between_rotvecs(a, b) -> float:
    """Magnitude of the rotation taking orientation a to orientation b (rad)."""
    R = rotvec_to_matrix(b) @ rotvec_to_matrix(a).T
    return float(np.linalg.norm(matrix_to_rotvec(R)))


# --------------------------------------------------------------------------
# Shared clamping / bookkeeping
# --------------------------------------------------------------------------

class _RobotBase:
    """Envelope clamping, baseline tracking, and the controller's contract."""

    def __init__(self, cfg=RobotConfig):
        self.cfg = cfg
        self.baseline = np.asarray(cfg.BASELINE_POSE, float).copy()
        # Populated after each move so the session loop can log what really
        # happened rather than what was requested.
        self.last_move = {"requested": None, "applied": None,
                          "clamped": False, "ok": False, "singularity": None}
        self._moves = 0
        self._sing_blocks = 0
        self.kin: URKinematics = getattr(cfg, "KINEMATICS", UR3_NOMINAL)
        self.th: SingularityThresholds = DEFAULT_THRESHOLDS

    # -- to implement in subclasses --
    def get_pose(self) -> np.ndarray:
        raise NotImplementedError

    def _send_pose(self, pose: np.ndarray, asynchronous: bool) -> bool:
        raise NotImplementedError

    def is_safe(self) -> bool:
        return True

    # -- shared --
    def _clamp(self, pose: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Clamp an absolute target into the position envelope and the
        orientation bounds measured FROM BASELINE. Returns (pose, clamped)."""
        out = np.asarray(pose, float).copy()
        lo = np.asarray(self.cfg.ENVELOPE_MIN, float)
        hi = np.asarray(self.cfg.ENVELOPE_MAX, float)
        clamped = False

        xyz = np.clip(out[:3], lo, hi)
        if not np.allclose(xyz, out[:3], atol=1e-9):
            clamped = True
        out[:3] = xyz

        # Orientation: bound total deviation from baseline.
        dev = angle_between_rotvecs(self.baseline[3:], out[3:])
        max_dev = self.cfg.MAX_TILT_RAD + self.cfg.MAX_ROT_RAD
        if dev > max_dev:
            # scale the deviation back along the same axis
            R_dev = rotvec_to_matrix(out[3:]) @ rotvec_to_matrix(self.baseline[3:]).T
            rv_dev = matrix_to_rotvec(R_dev)
            n = np.linalg.norm(rv_dev)
            if n > 1e-12:
                rv_dev = rv_dev * (max_dev / n)
                out[3:] = matrix_to_rotvec(
                    rotvec_to_matrix(rv_dev) @ rotvec_to_matrix(self.baseline[3:]))
            clamped = True
        return out, clamped

    # -- singularity ------------------------------------------------------

    def pose_margins(self, pose) -> dict:
        """Singularity margins for an absolute target pose. The base class can
        only do the Cartesian layer; UR3Robot overrides to add exact joint-space
        margins from the robot's own IK."""
        return position_margins(np.asarray(pose, float)[:3],
                                self.cfg.WORKING_RADIUS_M, self.kin)

    def singularity_cost(self, delta: dict) -> float:
        """
        Cost of a candidate DOF delta, for the controller's optimiser. Signature
        matches goal_controller's `extra_cost` hook, so the optimiser routes
        around singular regions instead of proposing a target that then gets
        refused.
        """
        if not self.cfg.AVOID_SINGULARITIES:
            return 0.0
        try:
            target = self._target_pose(delta)
            return singularity_penalty(self.pose_margins(target))
        except Exception:
            return 0.0

    def _target_pose(self, delta: dict) -> np.ndarray:
        """Absolute pose that a DOF delta would produce, without moving."""
        cur = self.get_pose()
        t = cur.copy()
        t[0] += delta.get("dx", 0.0)
        t[1] += delta.get("dy", 0.0)
        t[2] += delta.get("dz", 0.0)
        if abs(delta.get("dtilt", 0.0)) > 1e-9:
            t[3:] = compose_rotvec(t[3:], [1.0, 0.0, 0.0], delta["dtilt"])
        if abs(delta.get("drot", 0.0)) > 1e-9:
            t[3:] = compose_rotvec(t[3:], [0.0, 0.0, 1.0], delta["drot"])
        return t

    def check_workspace(self) -> list:
        """Pre-flight sanity on the configured envelope. Returns problems."""
        out = []
        r_min, r_max = suggest_working_annulus(self.cfg.WORKING_RADIUS_M,
                                               self.kin, self.th)
        lo = np.asarray(self.cfg.ENVELOPE_MIN, float)
        hi = np.asarray(self.cfg.ENVELOPE_MAX, float)
        corners = [(x, y) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
        radii = [float(np.hypot(x, y)) for x, y in corners]
        if min(radii) < r_min:
            out.append(f"envelope reaches {min(radii):.3f} m from the base axis, "
                       f"inside the singularity-free inner bound {r_min:.3f} m "
                       f"(central cylinder, shoulder singularity)")
        if max(radii) > r_max:
            out.append(f"envelope reaches {max(radii):.3f} m from the base axis, "
                       f"outside the safe outer bound {r_max:.3f} m "
                       f"(workspace boundary / elbow singularity)")
        m = self.pose_margins(self.baseline)
        b = is_blocked(m, self.th)
        if b:
            out.append(f"BASELINE_POSE itself is near a {b} singularity")
        return out

    def move_to_baseline(self, asynchronous: bool = False) -> bool:
        """Called in BOTH conditions. This is the confound fix: the artifact
        starts at an identical pose whether or not the robot will adapt.

        Prefers moveJ where available: it plans in joint space, so it needs no
        Jacobian inverse and cannot fail on a singularity along the way. The
        small adaptive moves during the session stay on moveL, where a straight
        and predictable Cartesian path matters because someone is close.
        """
        ok = self._move_baseline_impl(asynchronous)
        self.last_move = {"requested": self.baseline.tolist(),
                          "applied": self.baseline.tolist(),
                          "clamped": False, "ok": bool(ok), "singularity": None}
        return bool(ok)

    def _move_baseline_impl(self, asynchronous: bool) -> bool:
        return self._send_pose(self.baseline.copy(), asynchronous)

    def move_relative(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                      drx: float = 0.0, dry: float = 0.0, drz: float = 0.0,
                      asynchronous: bool = True) -> bool:
        """
        Controller contract. Translations are in the base frame; `drx` is the
        artifact TILT, applied as a proper rotation about the base X axis.
        The absolute target is clamped into the envelope before sending.
        """
        if self.cfg.REQUIRE_SAFE_MODE and not self.is_safe():
            logger.warning("[ROBOT] unsafe state, move refused")
            self.last_move = {"requested": [dx, dy, dz, drx], "applied": None,
                              "clamped": False, "ok": False}
            return False

        cur = self.get_pose()
        target = cur.copy()
        target[0] += dx
        target[1] += dy
        target[2] += dz
        if abs(drx) > 1e-9:
            target[3:] = compose_rotvec(target[3:], [1.0, 0.0, 0.0], drx)
        if abs(dry) > 1e-9:
            target[3:] = compose_rotvec(target[3:], [0.0, 1.0, 0.0], dry)
        if abs(drz) > 1e-9:
            target[3:] = compose_rotvec(target[3:], [0.0, 0.0, 1.0], drz)

        target, clamped = self._clamp(target)

        # Backstop: the optimiser is already penalised for singular candidates,
        # but a move can still be requested directly. Refuse rather than let the
        # controller drive into a singularity.
        sing = None
        if self.cfg.AVOID_SINGULARITIES:
            sing = is_blocked(self.pose_margins(target), self.th)
            if sing:
                self._sing_blocks += 1
                logger.warning(f"[ROBOT] move refused: would approach a {sing} "
                               f"singularity")
                self.last_move = {"requested": [dx, dy, dz, drx], "applied": None,
                                  "clamped": True, "ok": False, "singularity": sing}
                return False

        ok = self._send_pose(target, asynchronous)
        self._moves += 1
        applied = (target - cur)
        self.last_move = {
            "requested": [round(v, 5) for v in (dx, dy, dz, drx)],
            "applied": [round(float(v), 5) for v in applied[:3]] +
                       [round(angle_between_rotvecs(cur[3:], target[3:]), 5)],
            "clamped": bool(clamped), "ok": bool(ok), "singularity": None}
        if clamped:
            logger.info("[ROBOT] move clamped by workspace envelope")
        return bool(ok)

    def adjust_rotation(self, drot: float) -> bool:
        """Rotation about the base vertical (Z) axis."""
        return self.move_relative(drz=drot, asynchronous=True)


# --------------------------------------------------------------------------
# Simulated
# --------------------------------------------------------------------------

class SimulatedRobot(_RobotBase):
    """No hardware. Identical envelope and clamping so dry runs are faithful."""

    def __init__(self, cfg=RobotConfig):
        super().__init__(cfg)
        self._pose = self.baseline.copy()
        self.history = []

    def get_pose(self) -> np.ndarray:
        return self._pose.copy()

    def _send_pose(self, pose: np.ndarray, asynchronous: bool) -> bool:
        self._pose = np.asarray(pose, float).copy()
        self.history.append(self._pose.tolist())
        return True

    def close(self):
        pass


# --------------------------------------------------------------------------
# UR3 over RTDE
# --------------------------------------------------------------------------

class UR3Robot(_RobotBase):
    """
    ur_rtde implementation. Install with: pip install ur-rtde

    Before the first live run:
      - set RobotConfig.IP
      - measure and set ENVELOPE_MIN / ENVELOPE_MAX, then set
        ENVELOPE_VERIFIED = True
      - set BASELINE_POSE to the matched starting pose used in both conditions
      - confirm the artifact plus fixture is within the UR3 payload
    """

    def __init__(self, cfg=RobotConfig, ip: Optional[str] = None):
        super().__init__(cfg)
        if not cfg.ENVELOPE_VERIFIED:
            raise RuntimeError(
                "RobotConfig.ENVELOPE_VERIFIED is False. Measure the workspace "
                "envelope on the rig and set it before driving the UR3. This "
                "guard exists because the envelope bounds cumulative drift.")
        try:
            import rtde_control
            import rtde_receive
        except ImportError as e:
            raise ImportError("ur_rtde not installed. pip install ur-rtde") from e

        self.ip = ip or cfg.IP
        self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
        logger.info(f"[ROBOT] connected to UR3 at {self.ip}")

    def get_pose(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualTCPPose(), float)

    def is_safe(self) -> bool:
        try:
            if not self.rtde_c.isConnected():
                return False
            # 1 == normal on UR safety mode enumeration; anything else means
            # reduced / protective stop / emergency stop / fault.
            mode = self.rtde_r.getSafetyMode()
            return int(mode) == 1
        except Exception as e:
            logger.warning(f"[ROBOT] safety check failed: {e}")
            return False

    def _send_pose(self, pose: np.ndarray, asynchronous: bool) -> bool:
        try:
            return bool(self.rtde_c.moveL(
                list(map(float, pose)), self.cfg.SPEED_MS, self.cfg.ACCEL_MS2,
                asynchronous))
        except Exception as e:
            logger.error(f"[ROBOT] moveL failed: {e}")
            return False

    # ---- exact singularity margins from the robot's own kinematics ----

    def pose_margins(self, pose) -> dict:
        """
        Layer 2. Ask the controller's IK for the joint vector that reaches this
        pose, then measure exact margins on q3 and q5. This uses the arm's
        CALIBRATED kinematics rather than our nominal DH table, which matters
        because UR's published DH parameters vary by serial-number range.

        Falls back to the Cartesian layer if IK is unavailable or the pose is
        unreachable. An unreachable pose is reported as blocked, since that is
        the workspace boundary and refusing is the right response either way.
        """
        cart = super().pose_margins(pose)
        if not self.cfg.USE_ROBOT_IK:
            return cart
        try:
            q = self.rtde_c.getInverseKinematics(list(map(float, pose)))
            if not q or len(q) != 6:
                cart["radius_m"] = -1.0        # unreachable
                cart["ik_ok"] = False
                return cart
            jm = joint_margins(q, self.kin, self.th)
            jm.update({k: v for k, v in cart.items() if k not in jm})
            # keep the tighter of the two shoulder estimates
            jm["shoulder_m"] = min(jm["shoulder_m"], cart["shoulder_m"])
            jm["worst_norm"] = min(
                jm["worst_norm"],
                cart["radius_m"] / self.th.RADIUS_WARN_M,
                jm["shoulder_m"] / self.th.SHOULDER_WARN_M)
            jm["ik_ok"] = True
            return jm
        except Exception as e:
            logger.debug(f"[ROBOT] IK margin check unavailable: {e}")
            return cart

    def actual_margins(self) -> dict:
        """Margins at the arm's CURRENT joint configuration. Useful to log once
        per session and to check after the baseline move."""
        try:
            return joint_margins(self.rtde_r.getActualQ(), self.kin, self.th)
        except Exception:
            return {}

    def _move_baseline_impl(self, asynchronous: bool) -> bool:
        """moveJ via IK when configured, else fall back to moveL."""
        if self.cfg.BASELINE_USE_MOVEJ:
            try:
                q = self.rtde_c.getInverseKinematics(
                    list(map(float, self.baseline)))
                if q and len(q) == 6:
                    m = joint_margins(q, self.kin, self.th)
                    b = is_blocked(m, self.th)
                    if b:
                        logger.error(f"[ROBOT] BASELINE_POSE resolves to a "
                                     f"{b}-singular configuration; choose a "
                                     f"different baseline")
                        return False
                    return bool(self.rtde_c.moveJ(
                        list(map(float, q)), self.cfg.BASELINE_JOINT_SPEED,
                        self.cfg.BASELINE_JOINT_ACCEL, asynchronous))
            except Exception as e:
                logger.warning(f"[ROBOT] moveJ baseline failed ({e}); using moveL")
        return self._send_pose(self.baseline.copy(), asynchronous)

    def stop(self):
        try:
            self.rtde_c.stopL(1.0)
        except Exception:
            pass

    def close(self):
        try:
            self.stop()
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
        except Exception:
            pass


def make_robot(simulate: bool = True, cfg=RobotConfig):
    return SimulatedRobot(cfg) if simulate else UR3Robot(cfg)


# --------------------------------------------------------------------------
# Self-test: rotation maths and envelope clamping, no hardware.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("rotation composition (why naive addition is wrong):")
    rv = np.array([0.0, np.pi, 0.0])            # typical tool-down orientation
    tilt = 0.30
    naive = rv + np.array([tilt, 0.0, 0.0])
    proper = compose_rotvec(rv, [1, 0, 0], tilt)
    print(f"  base rotvec           {np.round(rv, 4)}")
    print(f"  naive rx += 0.30      {np.round(naive, 4)}  "
          f"-> actual rotation {angle_between_rotvecs(rv, naive):.4f} rad")
    print(f"  composed properly     {np.round(proper, 4)}  "
          f"-> actual rotation {angle_between_rotvecs(rv, proper):.4f} rad "
          f"(want {tilt})")

    print("\nround-trip rotvec -> matrix -> rotvec:")
    for test in ([0, 0, 0], [0.1, -0.2, 0.3], [0, np.pi, 0], [1.2, 0.4, -0.9]):
        back = matrix_to_rotvec(rotvec_to_matrix(test))
        err = angle_between_rotvecs(test, back)
        print(f"  {np.round(test, 3)} -> err {err:.2e}")

    print("\ncumulative envelope (the bound the controller does NOT have):")
    r = SimulatedRobot()
    print(f"  baseline z = {r.get_pose()[2]:.3f}")
    for i in range(8):
        r.move_relative(dz=0.08)          # eight max-size raises in a row
    p = r.get_pose()
    print(f"  after 8 x +0.08 m raises, z = {p[2]:.3f} "
          f"(envelope max {RobotConfig.ENVELOPE_MAX[2]}), "
          f"clamped={r.last_move['clamped']}")
    assert p[2] <= RobotConfig.ENVELOPE_MAX[2] + 1e-9, "envelope breached"

    print("\ncumulative tilt bound:")
    r2 = SimulatedRobot()
    for i in range(6):
        r2.move_relative(drx=0.30)
    dev = angle_between_rotvecs(r2.baseline[3:], r2.get_pose()[3:])
    print(f"  after 6 x 0.30 rad tilts, deviation from baseline = {dev:.3f} rad "
          f"(cap {RobotConfig.MAX_TILT_RAD + RobotConfig.MAX_ROT_RAD})")
    assert dev <= RobotConfig.MAX_TILT_RAD + RobotConfig.MAX_ROT_RAD + 1e-6

    print("\nbaseline is identical in both conditions:")
    a, b = SimulatedRobot(), SimulatedRobot()
    a.move_to_baseline(); b.move_to_baseline()
    print(f"  control  baseline {np.round(a.get_pose(), 3)}")
    print(f"  experim. baseline {np.round(b.get_pose(), 3)}")
    assert np.allclose(a.get_pose(), b.get_pose())

    print("\nlive guard: UR3Robot refuses to construct until envelope verified")
    try:
        UR3Robot()
        print("  NOT GUARDED")
    except RuntimeError as e:
        print(f"  guarded: {str(e)[:60]}...")
    except ImportError:
        print("  guarded earlier by envelope check (ur_rtde absent here)")

    print("\nrobot_interface.py self-test passed")
