"""
camera_config.py
================
How many cameras there are, where they sit relative to the cobot, and what that
means for which angles can honestly be measured.

Why this is a config file and not a constant
--------------------------------------------
v1 ran one front/overhead-left camera and computed a trunk angle from it. The
trunk term ended up contributing about 2 percent of the strain score, because
forward lean happens almost entirely ALONG that camera's optical axis. The
signal was not noisy, it was absent, and nothing downstream knew that. The
system reported a trunk angle with the same apparent authority as a real one.

So placement is not a deployment detail, it decides which claims are available.
This module makes that explicit: each layout declares what it can and cannot
measure, and the rest of the pipeline is required to respect it. Unmeasurable
quantities are returned as zero with zero confidence, never fabricated, and the
controller disables the degrees of freedom whose driving signal is missing.

The geometry, first order
-------------------------
Put the person in their own frame: the sagittal plane splits left from right
(forward lean happens here), the frontal plane splits front from back (side-bend
happens here), the transverse plane is horizontal (twist happens here).

Let phi be the camera azimuth measured from the person's forward direction, so
phi = 0 is directly in front of the face and phi = 90 is a pure side view.
A rotation in a given plane is resolved well when it happens across the image
and badly when it happens along the optical axis. To first order:

    sagittal sensitivity  proportional to  sin(phi)
    frontal  sensitivity  proportional to  cos(phi)

Both fall out of the same projection, and they trade off against each other:

    phi = 90 (side)     sagittal 1.00, frontal 0.00
    phi = 75            sagittal 0.97, frontal 0.26
    phi = 45 (oblique)  sagittal 0.71, frontal 0.71
    phi = 0  (front)    sagittal 0.00, frontal 1.00   <- the v1 failure

The important consequence is about ALIGNMENT TOLERANCE, and it is not what most
people expect. A side camera misaligned by 15 degrees loses only about 3 percent
of the sagittal signal (cos 15 = 0.966) but picks up about 26 percent cross-talk
from side-bend (sin 15 = 0.259). The tolerance is set by cross-talk, not by
signal loss, which is why the azimuth spec below is tight even though a few
degrees "obviously" would not matter.

Second consequence, and the reason height and distance are specified loosely:
PSS_v2 calibrates a per-person neutral offset at the start of every session. A
CONSTANT bias, which is what a slightly wrong mounting height mostly produces,
is absorbed by that calibration. A SCALE error and CROSS-TALK are not absorbed,
because they depend on the posture being measured. So azimuth alignment matters
much more than getting the height or distance exactly right.

These are first-order geometric estimates for a projective view. MediaPipe world
landmarks are a monocular 3D lift rather than a raw projection, so the true
behaviour is milder than pure projection but follows the same trend, and the
model is trained mostly on roughly level viewpoints, which is why the elevation
guidance below still applies. Treat the numbers as design guidance, not
calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# What a layout can measure
# --------------------------------------------------------------------------

# Angle names as used by PostureAngles, grouped by the plane they live in.
SAGITTAL_ANGLES = ("trunk_flexion_deg", "neck_flexion_deg",
                   "upper_arm_flexion_deg", "lower_arm_flexion_deg")
FRONTAL_ANGLES = ("trunk_sidebend_deg", "neck_sidebend_deg", "lateral_gaze_deg")
TRANSVERSE_ANGLES = ("trunk_twist_deg", "neck_twist_deg")


@dataclass(frozen=True)
class CameraPlacement:
    """One physical camera. Azimuth is measured from the person's forward
    direction, positive toward the person's left. Height is relative to the
    person's trunk mid-height (sternum), positive up."""
    role: str                      # "side" | "front" | "oblique"
    azimuth_deg: float
    azimuth_tolerance_deg: float
    height_offset_m: float
    height_tolerance_m: float
    distance_m: Tuple[float, float]
    notes: str = ""

    def sagittal_scale(self) -> float:
        return abs(math.sin(math.radians(self.azimuth_deg)))

    def frontal_scale(self) -> float:
        return abs(math.cos(math.radians(self.azimuth_deg)))


@dataclass(frozen=True)
class CameraLayout:
    name: str
    n_cameras: int
    placements: Tuple[CameraPlacement, ...]
    measurable: Tuple[str, ...]          # angle names this layout can measure
    confidence_scale: Dict[str, float]   # per-angle nominal confidence multiplier
    recommended: bool
    rationale: str
    limitations: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def can_measure(self, angle: str) -> bool:
        return angle in self.measurable

    def scale_for(self, angle: str) -> float:
        return self.confidence_scale.get(angle, 1.0 if self.can_measure(angle) else 0.0)


# --------------------------------------------------------------------------
# Placement relative to the cobot
# --------------------------------------------------------------------------
#
# Constraints that apply to EVERY layout, in priority order:
#
#  1. The hips must be visible. Trunk angle is the vector from hip midpoint to
#     shoulder midpoint. If the bench edge occludes the hips there is no trunk
#     angle at all, regardless of azimuth. This is the single most common
#     practical failure for a side camera set at desk height, and it fails
#     silently because MediaPipe will still emit a low-visibility landmark.
#
#  2. The camera must not sit inside the cobot's working volume. Both a
#     collision risk and a safety-assessment problem.
#
#  3. The cobot arm and the artifact must not cross the line of sight to the
#     torso. The UR3 holds the artifact between the person and the bench, so a
#     camera on the same side as the arm's dominant sweep gets intermittent
#     occlusion exactly during interventions, which is the worst possible time.
#     Place the side camera on the side OPPOSITE the robot base where possible,
#     or raise it enough to look over the arm.
#
#  4. Distance 1.5 to 2.5 m at 720p. Closer clips the torso and adds radial
#     distortion near the frame edge; further shrinks the subject and the
#     landmark noise starts to matter relative to the angles being measured.
#
#  5. Keep the camera within about 20 degrees of the horizontal plane through
#     the trunk. An elevated view foreshortens the vertical component of the
#     trunk vector; calibration absorbs the constant part but not the scale.

CAM_SIDE = CameraPlacement(
    role="side", azimuth_deg=90.0, azimuth_tolerance_deg=15.0,
    height_offset_m=0.0, height_tolerance_m=0.20, distance_m=(1.5, 2.5),
    notes=("Perpendicular to the sagittal plane, at trunk mid-height. Mount on "
           "the side opposite the robot base so the arm does not cross the line "
           "of sight to the torso. Check the hips are visible above the bench "
           "edge; if not, move the camera forward of the coronal plane or up."))

CAM_FRONT = CameraPlacement(
    role="front", azimuth_deg=0.0, azimuth_tolerance_deg=20.0,
    height_offset_m=0.10, height_tolerance_m=0.25, distance_m=(1.5, 2.5),
    notes=("Facing the person, slightly above trunk mid-height so the head and "
           "shoulder line are clear of the artifact. Must not look straight "
           "down the robot's approach path."))

CAM_OBLIQUE = CameraPlacement(
    role="oblique", azimuth_deg=45.0, azimuth_tolerance_deg=10.0,
    height_offset_m=0.05, height_tolerance_m=0.20, distance_m=(1.5, 2.5),
    notes=("Compromise position for rigs that cannot fit a true side view. "
           "Resolves both planes at about 71 percent with heavy cross-talk."))

CAM_FRONT_OBLIQUE = CameraPlacement(
    role="front", azimuth_deg=35.0, azimuth_tolerance_deg=15.0,
    height_offset_m=0.10, height_tolerance_m=0.25, distance_m=(1.5, 2.5),
    notes=("Fallback frontal position when the true front is blocked by the "
           "bench or the robot column."))


# --------------------------------------------------------------------------
# The layout registry
# --------------------------------------------------------------------------

LAYOUTS: Dict[str, CameraLayout] = {

    "SIDE_FRONT": CameraLayout(
        name="SIDE_FRONT", n_cameras=2,
        placements=(CAM_SIDE, CAM_FRONT),
        measurable=SAGITTAL_ANGLES + FRONTAL_ANGLES + TRANSVERSE_ANGLES,
        confidence_scale={a: 1.0 for a in SAGITTAL_ANGLES + FRONTAL_ANGLES}
                         | {a: 0.6 for a in TRANSVERSE_ANGLES},
        recommended=True,
        rationale=(
            "The design default and the best two-camera arrangement. The two "
            "views are orthogonal, so each plane of motion is in-plane for "
            "exactly one camera and cross-talk between them is near zero. Every "
            "angle PSS_v2 scores has a camera that resolves it well, so "
            "best-view fusion has an unambiguous winner per joint."),
        limitations=(
            "Twist is still the weakest measurement. It is inferred from the "
            "projected shoulder line against the hip line rather than measured "
            "directly, hence the reduced confidence.",)),

    "SIDE_ONLY": CameraLayout(
        name="SIDE_ONLY", n_cameras=1,
        placements=(CAM_SIDE,),
        measurable=SAGITTAL_ANGLES,
        confidence_scale={a: 1.0 for a in SAGITTAL_ANGLES},
        recommended=True,
        rationale=(
            "The correct single-camera choice. Trunk and neck flexion carry 70 "
            "percent of the PSS_v2 weight and both are sagittal, so a side view "
            "buys the most measurable strain per camera. Arm elevation and "
            "elbow angle also come through."),
        limitations=(
            "Side-bend, twist and lateral gaze are unmeasurable. Their PSS_v2 "
            "adders are held at zero rather than guessed, so a SIDE_ONLY PSS is "
            "a REDUCED score and is not numerically comparable to a SIDE_FRONT "
            "PSS. Do not pool the two in one analysis.",
            "The controller loses lateral (dx) and rotation (drot) authority, "
            "because the signals that would justify those moves are not "
            "observed. It retains height, depth and tilt, which is where most "
            "of the achievable relief is anyway.",)),

    "OBLIQUE_ONLY": CameraLayout(
        name="OBLIQUE_ONLY", n_cameras=1,
        placements=(CAM_OBLIQUE,),
        measurable=SAGITTAL_ANGLES + ("trunk_sidebend_deg", "neck_sidebend_deg",
                                      "lateral_gaze_deg"),
        confidence_scale={a: 0.5 for a in SAGITTAL_ANGLES}
                         | {a: 0.5 for a in FRONTAL_ANGLES},
        recommended=False,
        rationale=(
            "Last-resort single-camera position for a rig that physically "
            "cannot take a side view. Resolves both planes partially, at about "
            "71 percent each geometrically, but every angle carries cross-talk "
            "from the other plane."),
        limitations=(
            "Sagittal and frontal readings are mutually contaminated, and "
            "calibration cannot remove cross-talk because it is "
            "posture-dependent rather than a constant offset.",),
        warnings=(
            "OBLIQUE_ONLY produces angles that are mixtures of two planes. "
            "Report it as an exploratory configuration; do not use it for the "
            "RULA validation.",)),

    "FRONT_ONLY": CameraLayout(
        name="FRONT_ONLY", n_cameras=1,
        placements=(CAM_FRONT,),
        measurable=("trunk_sidebend_deg", "neck_sidebend_deg", "lateral_gaze_deg",
                    "lower_arm_flexion_deg"),
        confidence_scale={"trunk_sidebend_deg": 1.0, "neck_sidebend_deg": 1.0,
                          "lateral_gaze_deg": 1.0, "lower_arm_flexion_deg": 0.5},
        recommended=False,
        rationale=(
            "This is the v1 configuration, kept only so the config can express "
            "it and warn. Forward lean happens along this camera's optical "
            "axis, which is exactly why the v1 trunk term collapsed to about 2 "
            "percent of the score."),
        limitations=(
            "Trunk and neck flexion are not measurable. That removes about 70 "
            "percent of the PSS_v2 weight, so the score is not meaningful for "
            "this application.",),
        warnings=(
            "FRONT_ONLY cannot measure the strain this project is about. It "
            "reproduces the v1 defect. Use SIDE_ONLY if only one camera is "
            "available.",)),

    "SIDE_FRONT_OBLIQUE": CameraLayout(
        name="SIDE_FRONT_OBLIQUE", n_cameras=2,
        placements=(CAM_SIDE, CAM_FRONT_OBLIQUE),
        measurable=SAGITTAL_ANGLES + FRONTAL_ANGLES + TRANSVERSE_ANGLES,
        confidence_scale={a: 1.0 for a in SAGITTAL_ANGLES}
                         | {a: 0.8 for a in FRONTAL_ANGLES}
                         | {a: 0.5 for a in TRANSVERSE_ANGLES},
        recommended=True,
        rationale=(
            "Two-camera fallback when the true frontal position is blocked by "
            "the bench or the robot column. The side camera is unchanged, so "
            "the sagittal angles that dominate PSS_v2 are unaffected; only the "
            "frontal terms are degraded, by about cos(35) = 0.82."),
        limitations=(
            "Side-bend and twist carry reduced confidence. Note the frontal "
            "camera azimuth in the session metadata so it can be reported.",)),
}


# --------------------------------------------------------------------------
# Active configuration
# --------------------------------------------------------------------------

class CameraConfig:
    """Set LAYOUT to one of the keys in LAYOUTS. Everything downstream reads
    the resulting layout and respects what it says is measurable."""

    LAYOUT = "SIDE_FRONT"

    # Physical camera indices or device paths, by role. Set at wiring time.
    # Unused roles stay None. On Linux prefer /dev/v4l/by-id/... paths, which
    # survive a replug; integer indices do not.
    SIDE_SOURCE = 0
    FRONT_SOURCE = 1
    OBLIQUE_SOURCE = None

    # As-built geometry, filled in during rig setup and recorded in the session
    # metadata so a reviewer can see what was actually deployed rather than what
    # was intended. Azimuths in degrees from the person's forward direction.
    AS_BUILT_SIDE_AZIMUTH_DEG: Optional[float] = None
    AS_BUILT_FRONT_AZIMUTH_DEG: Optional[float] = None
    AS_BUILT_VERIFIED = False

    # Which side of the person the robot base sits on, "left" or "right".
    # Used only to warn if the side camera is placed into the arm's sweep.
    ROBOT_BASE_SIDE = "right"

    @classmethod
    def layout(cls) -> CameraLayout:
        if cls.LAYOUT not in LAYOUTS:
            raise ValueError(
                f"unknown CameraConfig.LAYOUT {cls.LAYOUT!r}; "
                f"choose one of {sorted(LAYOUTS)}")
        return LAYOUTS[cls.LAYOUT]

    @classmethod
    def n_cameras(cls) -> int:
        return cls.layout().n_cameras

    @classmethod
    def measurable(cls) -> Tuple[str, ...]:
        return cls.layout().measurable

    @classmethod
    def sources(cls) -> Dict[str, object]:
        """Role -> source, for the roles this layout actually uses."""
        by_role = {"side": cls.SIDE_SOURCE, "front": cls.FRONT_SOURCE,
                   "oblique": cls.OBLIQUE_SOURCE}
        return {p.role: by_role.get(p.role) for p in cls.layout().placements}


# --------------------------------------------------------------------------
# Sensitivity model and validation
# --------------------------------------------------------------------------

def sensitivity(azimuth_deg: float) -> Dict[str, float]:
    """
    First-order resolution of each motion plane for a camera at this azimuth,
    plus the cross-talk each reading picks up from the other plane.
    """
    r = math.radians(azimuth_deg)
    s, c = abs(math.sin(r)), abs(math.cos(r))
    return {"sagittal_scale": s, "frontal_scale": c,
            "sagittal_crosstalk": c, "frontal_crosstalk": s}


def misalignment_cost(nominal_deg: float, actual_deg: float) -> Dict[str, float]:
    """
    What a placement error actually costs. Reported as the fraction of signal
    kept and the fraction of the other plane that leaks in, because those two
    behave very differently and only the second one is not absorbed by the
    per-session neutral calibration.
    """
    err = abs(actual_deg - nominal_deg)
    return {"error_deg": err,
            "signal_kept": math.cos(math.radians(err)),
            "crosstalk": abs(math.sin(math.radians(err)))}


def validate(cfg=CameraConfig) -> List[str]:
    """Problems worth telling the operator about before a session starts."""
    out: List[str] = []
    layout = cfg.layout()

    out.extend(layout.warnings)
    if not layout.recommended:
        out.append(f"layout {layout.name} is not recommended: {layout.rationale}")

    for role, src in cfg.sources().items():
        if src is None:
            out.append(f"layout {layout.name} needs a {role} camera but "
                       f"{role.upper()}_SOURCE is None")

    if not cfg.AS_BUILT_VERIFIED:
        out.append("CameraConfig.AS_BUILT_VERIFIED is False: measure the actual "
                   "camera azimuths and record them before collecting data")
    else:
        pairs = [(CAM_SIDE, cfg.AS_BUILT_SIDE_AZIMUTH_DEG, "side"),
                 (CAM_FRONT, cfg.AS_BUILT_FRONT_AZIMUTH_DEG, "front")]
        roles = {p.role for p in layout.placements}
        for nominal, actual, role in pairs:
            if role not in roles or actual is None:
                continue
            m = misalignment_cost(nominal.azimuth_deg, actual)
            if m["error_deg"] > nominal.azimuth_tolerance_deg:
                out.append(
                    f"{role} camera is {m['error_deg']:.0f} deg off nominal "
                    f"(tolerance {nominal.azimuth_tolerance_deg:.0f}): keeps "
                    f"{m['signal_kept']*100:.0f}% of signal but admits "
                    f"{m['crosstalk']*100:.0f}% cross-talk from the other plane")

    if layout.n_cameras == 1:
        out.append(f"single-camera mode ({layout.name}): PSS is a REDUCED score "
                   f"and is not comparable to a two-camera PSS; do not pool them")
    return out


def describe(cfg=CameraConfig) -> str:
    """Human-readable summary, written into the session metadata."""
    layout = cfg.layout()
    lines = [f"layout: {layout.name} ({layout.n_cameras} camera"
             f"{'s' if layout.n_cameras > 1 else ''}, "
             f"{'recommended' if layout.recommended else 'NOT recommended'})"]
    for p in layout.placements:
        s = sensitivity(p.azimuth_deg)
        lines.append(
            f"  {p.role:<8} azimuth {p.azimuth_deg:+.0f} +/-{p.azimuth_tolerance_deg:.0f} deg, "
            f"height {p.height_offset_m:+.2f} m, distance {p.distance_m[0]}-{p.distance_m[1]} m "
            f"| sagittal {s['sagittal_scale']:.2f}, frontal {s['frontal_scale']:.2f}")
    lines.append(f"  measurable: {', '.join(layout.measurable)}")
    unmeasurable = [a for a in SAGITTAL_ANGLES + FRONTAL_ANGLES + TRANSVERSE_ANGLES
                    if a not in layout.measurable]
    if unmeasurable:
        lines.append(f"  NOT measurable (held at zero): {', '.join(unmeasurable)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("sensitivity against camera azimuth (0 = front, 90 = side):\n")
    print(f"  {'azimuth':>8} {'sagittal':>9} {'frontal':>8}   note")
    for phi in (0, 15, 30, 45, 60, 75, 90):
        s = sensitivity(phi)
        note = ""
        if phi == 0:
            note = "the v1 configuration: trunk flexion unmeasurable"
        elif phi == 45:
            note = "oblique compromise, both planes contaminated"
        elif phi == 90:
            note = "ideal for trunk and neck flexion"
        print(f"  {phi:>7}d {s['sagittal_scale']:>9.2f} {s['frontal_scale']:>8.2f}   {note}")

    print("\nwhy the azimuth tolerance is tight (side camera, nominal 90 deg):\n")
    print(f"  {'error':>6} {'signal kept':>12} {'cross-talk':>11}")
    for err in (0, 5, 10, 15, 20, 30):
        m = misalignment_cost(90.0, 90.0 - err)
        print(f"  {err:>5}d {m['signal_kept']*100:>11.0f}% {m['crosstalk']*100:>10.0f}%")
    print("  a 15 deg error costs 3% of signal but admits 26% cross-talk;")
    print("  calibration absorbs a constant offset, it does not absorb cross-talk")

    print("\n\nlayouts:\n")
    for name, lay in LAYOUTS.items():
        flag = "recommended" if lay.recommended else "NOT recommended"
        print(f"  {name:<20} {lay.n_cameras} cam  {flag}")
        for w in lay.warnings:
            print(f"      warning: {w}")

    print("\n\nactive configuration:\n")
    print(describe())
    print("\nvalidation messages:")
    for msg in validate():
        print(f"  - {msg}")

    # structural assertions
    assert LAYOUTS["SIDE_ONLY"].can_measure("trunk_flexion_deg")
    assert not LAYOUTS["SIDE_ONLY"].can_measure("trunk_sidebend_deg")
    assert not LAYOUTS["FRONT_ONLY"].can_measure("trunk_flexion_deg"), \
        "FRONT_ONLY must not claim trunk flexion; that was the v1 defect"
    assert LAYOUTS["SIDE_FRONT"].can_measure("trunk_flexion_deg")
    assert LAYOUTS["SIDE_FRONT"].can_measure("trunk_sidebend_deg")
    for lay in LAYOUTS.values():
        assert len(lay.placements) == lay.n_cameras
    print("\ncamera_config.py self-test passed")
