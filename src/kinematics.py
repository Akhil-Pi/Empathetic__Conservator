"""
kinematics.py
=============
UR forward kinematics, Jacobian, and singularity margins. Pure maths, no robot
and no ur_rtde needed, so it is unit-testable offline.

Why this module exists
----------------------
Sessions kept ending in a singularity. That is not bad luck; it follows from how
v1 and the early v2 controller moved. Both issue repeated relative nudges with
no bound on where the arm ends up, so over a session the tool walks toward the
workspace boundary (elbow singularity) or drifts over the base axis (shoulder
singularity). The Cartesian envelope added earlier bounds a BOX, and a box in
Cartesian space still contains singular regions. This module lets the envelope
be singularity-aware instead.

The three conditions, for a UR arm specifically
-----------------------------------------------
Universal Robots' own guidance describes joints 2, 3 and 4 (shoulder, elbow,
wrist 1) as rotating in a common plane, with trouble when wrist 2 aligns to that
same plane.

  WRIST     q5 -> 0 or +/-pi. Axes of joints 4 and 6 become coincident, one DOF
            is lost, and joints 4 and 6 are commanded to spin to compensate.
            This is the one most often hit in practice, and a flat tray held
            with the tool pointing straight down sits close to it.

  ELBOW     q3 -> 0. Upper arm and forearm become collinear; the arm is at the
            edge of its reach and the elbow branch flips between up and down.

  SHOULDER  The wrist centre approaches the axis of joint 1. UR's practical
            advice is to lay the task out so it is never necessary to work in
            or near that central cylinder.

Two-layer strategy used by robot_interface.py
---------------------------------------------
  Layer 1, geometric and always on: keep the TARGET POSITION out of the central
  cylinder and off the reach shell. Needs only nominal geometry, works offline,
  and is what makes the workspace envelope singularity-aware.

  Layer 2, joint space and live only: ask the robot's own IK for the joint
  vector that reaches the target, then measure margins on q3 and q5. This uses
  the robot's CALIBRATED kinematics rather than the nominal numbers below, which
  matters because UR publishes DH parameters that are revision-controlled per
  serial number, so nominal values are an approximation of any individual arm.

Accordingly the constants below are marked nominal and are used for offline
checks and simulation only. Anything driving a real arm should prefer
RTDEControlInterface.getInverseKinematics / getForwardKinematics.

Note also that the UR wrist is NOT spherical: the d5 offset means the three
wrist axes do not meet at a point, so shortcuts that assume a spherical wrist do
not apply here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Nominal geometry. NOMINAL, not calibrated. See module docstring.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class URKinematics:
    """Standard DH parameters. Defaults are the widely published nominal UR3
    values; UR3e differs slightly. Verify against the DH sheet for your arm's
    serial-number range, or prefer the robot's own FK/IK over these."""
    d1: float = 0.1519
    a2: float = -0.24365
    a3: float = -0.21325
    d4: float = 0.11235
    d5: float = 0.08535
    d6: float = 0.0819
    name: str = "UR3 (nominal)"
    calibrated: bool = False
    # Published spec reach, used only for sanity messages. Do NOT derive a
    # working radius from DH: sampling the nominal chain gives TCP distances
    # well beyond the published figure because the wrist offsets stack in
    # extended configurations. The working radius is a MEASURED value, set in
    # RobotConfig and verified on the rig.
    stated_reach: float = 0.500

    @property
    def planar_reach(self) -> float:
        """Two-link planar reach, shoulder to wrist-1. Not a TCP reach bound."""
        return abs(self.a2) + abs(self.a3)

    @property
    def shoulder_cylinder_r(self) -> float:
        """Radius of the central cylinder about the joint-1 axis that the wrist
        centre cannot enter. Set by the lateral offsets of the wrist chain."""
        return abs(self.d4)


UR3_NOMINAL = URKinematics()
UR3E_NOMINAL = URKinematics(d1=0.15185, a2=-0.24355, a3=-0.2132,
                            d4=0.13105, d5=0.08535, d6=0.0921,
                            name="UR3e (nominal)", stated_reach=0.500)


def _dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def dh_table(k: URKinematics):
    """(d, a, alpha) per joint, standard DH for UR arms."""
    return [
        (k.d1, 0.0, np.pi / 2),
        (0.0, k.a2, 0.0),
        (0.0, k.a3, 0.0),
        (k.d4, 0.0, np.pi / 2),
        (k.d5, 0.0, -np.pi / 2),
        (k.d6, 0.0, 0.0),
    ]


def forward_kinematics(q: Sequence[float], k: URKinematics = UR3_NOMINAL):
    """Return (T_0_6, list of cumulative frames T_0_i for i = 0..6)."""
    T = np.eye(4)
    frames = [T.copy()]
    for qi, (d, a, alpha) in zip(q, dh_table(k)):
        T = T @ _dh_transform(float(qi), d, a, alpha)
        frames.append(T.copy())
    return T, frames


def wrist_centre(q: Sequence[float], k: URKinematics = UR3_NOMINAL) -> np.ndarray:
    """Origin of frame 5, the practical 'wrist centre' for a UR arm. The wrist
    is not spherical, so this is a representative point, not an exact
    intersection of the three wrist axes."""
    _, frames = forward_kinematics(q, k)
    return frames[5][:3, 3]


def geometric_jacobian(q: Sequence[float], k: URKinematics = UR3_NOMINAL) -> np.ndarray:
    """6x6 geometric Jacobian in the base frame. For revolute joint i:
    Jv_i = z_{i-1} x (p_e - p_{i-1}), Jw_i = z_{i-1}."""
    T, frames = forward_kinematics(q, k)
    p_e = T[:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z = frames[i][:3, 2]
        p = frames[i][:3, 3]
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return J


def min_singular_value(q: Sequence[float], k: URKinematics = UR3_NOMINAL) -> float:
    """Smallest singular value of the Jacobian. Useful as a cross-check only:
    the Jacobian mixes metres and radians, so its absolute scale has no clean
    physical meaning. The geometric margins below are unit-free and are what the
    interlock actually uses."""
    return float(np.linalg.svd(geometric_jacobian(q, k), compute_uv=False)[-1])


# --------------------------------------------------------------------------
# Singularity margins. Unit-free, interpretable, and what the interlock uses.
# --------------------------------------------------------------------------

@dataclass
class SingularityThresholds:
    """Distance from each singular condition at which we start caring.
    WARN is where the move gets penalised; BLOCK is where it is refused."""
    WRIST_WARN_RAD: float = 0.35      # ~20 deg of q5 away from 0 or +/-pi
    WRIST_BLOCK_RAD: float = 0.17     # ~10 deg
    ELBOW_WARN_RAD: float = 0.30      # ~17 deg of q3 away from 0
    ELBOW_BLOCK_RAD: float = 0.15     # ~9 deg
    # Radial clearance of the wrist centre outside the central cylinder.
    SHOULDER_WARN_M: float = 0.08
    SHOULDER_BLOCK_M: float = 0.04
    # Cartesian only: clearance inside the MEASURED working radius. Elbow
    # singularity itself is captured exactly in joint space by ELBOW_*, so this
    # is a workspace-boundary guard rather than a second elbow test.
    RADIUS_WARN_M: float = 0.07
    RADIUS_BLOCK_M: float = 0.03


DEFAULT_THRESHOLDS = SingularityThresholds()

# Ceiling on the singularity cost term. Large enough to dominate any realistic
# PSS improvement (PSS is in [0, 1]), small enough not to wreck the cost scale.
PENALTY_MAX = 10.0


def _wrap_to_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def joint_margins(q: Sequence[float], k: URKinematics = UR3_NOMINAL,
                  th: SingularityThresholds = DEFAULT_THRESHOLDS) -> Dict[str, float]:
    """
    Margins from the three singular conditions, given a joint vector.

    wrist_rad    : how far q5 is from the nearest of 0, +pi, -pi
    elbow_rad    : how far q3 is from 0
    shoulder_m   : radial clearance of the wrist centre outside the central cylinder
    worst_norm   : the tightest of the three, normalised so 1.0 = at the WARN
                   threshold and 0.0 = exactly singular. Below 0 means past BLOCK.

    These three ARE the singularity set in joint space; each is exact and
    unit-free. Workspace-boundary distance is a separate Cartesian concern,
    handled by position_margins against a measured working radius.
    """
    q = list(map(float, q))
    q5 = _wrap_to_pi(q[4])
    wrist = min(abs(q5), abs(abs(q5) - np.pi))
    elbow = abs(_wrap_to_pi(q[2]))

    wc = wrist_centre(q, k)
    radial = float(np.hypot(wc[0], wc[1]))
    shoulder = radial - k.shoulder_cylinder_r

    norms = [
        wrist / th.WRIST_WARN_RAD,
        elbow / th.ELBOW_WARN_RAD,
        shoulder / th.SHOULDER_WARN_M,
    ]
    return {
        "wrist_rad": wrist,
        "elbow_rad": elbow,
        "shoulder_m": shoulder,
        "worst_norm": float(min(norms)),
    }


def is_blocked(m: Dict[str, float],
               th: SingularityThresholds = DEFAULT_THRESHOLDS) -> Optional[str]:
    """Name of the violated condition, or None. Accepts either a joint_margins
    dict or a position_margins dict; absent terms are simply not tested."""
    if m.get("wrist_rad", np.inf) < th.WRIST_BLOCK_RAD:
        return "wrist"
    if m.get("elbow_rad", np.inf) < th.ELBOW_BLOCK_RAD:
        return "elbow"
    if m.get("shoulder_m", np.inf) < th.SHOULDER_BLOCK_M:
        return "shoulder"
    if m.get("radius_m", np.inf) < th.RADIUS_BLOCK_M:
        return "radius"
    return None


def singularity_penalty(m: Dict[str, float]) -> float:
    """
    Smooth cost added to the controller's objective for approaching a
    singularity. Zero once every margin is at or beyond its WARN threshold, and
    rising steeply as any margin closes. Being a cost rather than a veto lets
    the optimiser route around singular regions instead of picking a target and
    then having the move refused.
    """
    w = m.get("worst_norm")
    if w is None:
        norms = []
        if "shoulder_m" in m:
            norms.append(m["shoulder_m"] / DEFAULT_THRESHOLDS.SHOULDER_WARN_M)
        if "radius_m" in m:
            norms.append(m["radius_m"] / DEFAULT_THRESHOLDS.RADIUS_WARN_M)
        w = min(norms) if norms else 1.0
    if w >= 1.0:
        return 0.0
    if w <= 0.0:
        return PENALTY_MAX
    # Capped so a near-singular candidate is strongly avoided without the cost
    # scale exploding and swamping the PSS term the optimiser is really minimising.
    return float(min(PENALTY_MAX, (1.0 / max(w, 1e-3) - 1.0) ** 2))


# --------------------------------------------------------------------------
# Position-only checks. No joint vector needed, so these work on a target pose
# before any IK, and are what makes the Cartesian envelope singularity-aware.
# --------------------------------------------------------------------------

def position_margins(xyz: Sequence[float], working_radius_m: float,
                     k: URKinematics = UR3_NOMINAL) -> Dict[str, float]:
    """
    Position-only margins for a TCP target, usable before any IK. This is what
    makes the Cartesian envelope singularity-aware rather than just a box.

    `working_radius_m` is a MEASURED value from the rig (RobotConfig), not
    derived from DH. Uses the TCP rather than the wrist centre, so the shoulder
    term is approximate; the thresholds carry margin and the live joint-space
    check is the exact one.
    """
    p = np.asarray(xyz, float)[:3]
    radial = float(np.hypot(p[0], p[1]))
    dist = float(np.linalg.norm(p - np.array([0.0, 0.0, k.d1])))
    return {
        "shoulder_m": radial - k.shoulder_cylinder_r,
        "radius_m": working_radius_m - dist,
        "radial_m": radial,
        "dist_from_shoulder_m": dist,
    }


def suggest_working_annulus(working_radius_m: float,
                            k: URKinematics = UR3_NOMINAL,
                            th: SingularityThresholds = DEFAULT_THRESHOLDS):
    """
    The singularity-free band to lay the task out in: keep the artifact between
    these distances from the base axis. Inner bound is the central cylinder plus
    margin (shoulder singularity); outer bound is the measured working radius
    less margin. Metres.
    """
    return (k.shoulder_cylinder_r + th.SHOULDER_WARN_M,
            working_radius_m - th.RADIUS_WARN_M)


def _find_shoulder_case(k: URKinematics = UR3_NOMINAL):
    """Search the nominal chain for a pose that is shoulder-singular while the
    elbow and wrist stay clear of their own singular values."""
    best = None
    for q2 in np.linspace(-2.6, -0.2, 40):
        for q3 in np.linspace(0.6, 2.4, 40):
            for q4 in np.linspace(-3.0, 1.0, 60):
                q = [0.0, q2, q3, q4, -1.4, 0.0]
                wc = wrist_centre(q, k)
                radial = float(np.hypot(wc[0], wc[1]))
                if best is None or radial < best[0]:
                    best = (radial, q)
    return best[1]


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    k = UR3_NOMINAL
    WORKING_R = 0.45   # stand-in for the MEASURED RobotConfig value
    print(f"{k.name}: planar reach {k.planar_reach:.3f} m, "
          f"published reach {k.stated_reach:.3f} m, "
          f"central cylinder r = {k.shoulder_cylinder_r:.3f} m")
    r_min, r_max = suggest_working_annulus(WORKING_R, k)
    print(f"singularity-free annulus for a {WORKING_R:.2f} m working radius: "
          f"{r_min:.3f} m to {r_max:.3f} m from the base axis\n")

    # A pose whose wrist centre sits on the joint-1 axis, with the elbow and
    # wrist deliberately kept away from their own singular values so the
    # shoulder condition is isolated. Found by search over the nominal chain.
    SHOULDER_CASE = _find_shoulder_case(k)

    good = [0.3, -0.6, 1.35, -2.38, -1.57, 0.0]
    m = joint_margins(good, k)
    print("representative good pose q =", [round(v, 2) for v in good])
    print(f"  wrist {m['wrist_rad']:.3f} rad, elbow {m['elbow_rad']:.3f} rad, "
          f"shoulder {m['shoulder_m']:.3f} m")
    print(f"  worst_norm {m['worst_norm']:.2f}, blocked: {is_blocked(m)}, "
          f"penalty {singularity_penalty(m):.3f}")
    print(f"  min singular value (cross-check) {min_singular_value(good, k):.4f}")
    assert is_blocked(m) is None and singularity_penalty(m) == 0.0

    print("\neach singular condition is detected, and sigma_min confirms rank loss:")
    cases = {
        "wrist  (q5 = 0)":      [0.3, -0.6, 1.35, -2.38, 0.0, 0.0],
        "elbow  (q3 = 0)":      [0.3, -0.6, 0.0, -2.38, -1.57, 0.0],
        "shoulder (wrist over base axis)": SHOULDER_CASE,
    }
    for name, q in cases.items():
        mm = joint_margins(q, k)
        print(f"  {name:<22} blocked={str(is_blocked(mm)):<9} "
              f"penalty={singularity_penalty(mm):6.2f} "
              f"sigma_min={min_singular_value(q, k):.4f}")
        assert is_blocked(mm) is not None, f"{name} not caught"

    print("\npenalty rises monotonically as the wrist closes on q5 = 0:")
    prev, mono = -1.0, True
    for q5 in (0.60, 0.40, 0.30, 0.22, 0.15, 0.08):
        p = singularity_penalty(joint_margins([0.3, -0.6, 1.35, -2.38, q5, 0], k))
        mono &= p >= prev - 1e-9
        print(f"  q5={q5:.2f} rad -> penalty {p:7.3f}")
        prev = p
    print(f"monotonic: {mono}")
    assert mono and prev > 0.0, "penalty must engage before the block threshold"

    print("\nCartesian pre-IK check (no joint vector needed):")
    for label, xyz in {"over the base (shoulder)": (0.02, 0.01, 0.30),
                       "good working position":    (0.0, -0.30, 0.15),
                       "beyond working radius":    (0.0, -0.52, 0.20)}.items():
        pm = position_margins(xyz, WORKING_R, k)
        print(f"  {label:<26} shoulder {pm['shoulder_m']:+.3f} m, "
              f"radius {pm['radius_m']:+.3f} m -> {is_blocked(pm)}")

    print("\nFK sanity: joint-1 rotation moves the TCP on a circle about the base")
    r = [np.hypot(*forward_kinematics([a, -0.6, 1.35, -2.38, -1.57, 0], k)[0][:2, 3])
         for a in (0.0, 0.7, 1.4, 2.1)]
    assert max(r) - min(r) < 1e-9
    print(f"  radial distance constant at {r[0]:.4f} m across q1 sweep")
    print("\nkinematics.py self-test passed")
