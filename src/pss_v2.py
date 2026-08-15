"""
pss_v2.py
=========
Postural Strain Score, version 2.

This replaces the gaze-dominated PSS in pss_calculator.py. The old score was
roughly 56 percent head-turn plus 42 percent lateral head offset, with trunk
and lean contributing about 2 percent combined, because a single overhead-left
2D camera cannot resolve trunk flexion. PSS_v2 is built instead on the posture
angle criteria used by RULA (McAtamney and Corlett, 1993) and REBA (Hignett and
McAtamney, 2000).

Design principle: this module is source-agnostic. It consumes anatomical angles
(a PostureAngles record), not raw landmarks. Part 2 (dual-camera fusion) is
responsible for producing accurate angles; this module only scores them. That
separation is what makes the scoring unit-testable without a camera or a robot.

What is grounded in the standards:
  - The per-joint angle breakpoints are the published RULA thresholds.
  - Each joint angle maps to a continuous "band" value that tracks the RULA
    integer bands (e.g. trunk band moves smoothly from 1 at upright to 4 beyond
    60 degrees of flexion) rather than jumping in integer steps.

What is an explicit design choice (documented, not smuggled in as "the standard"):
  - The aggregation weights that combine the joint bands into one [0, 1] score.
    Neck and trunk (RULA Group B) are weighted above the arm (Group A), because
    sustained static neck and trunk flexion is the dominant MSD driver in
    fine-detail precision work (dentistry: Valachi and Valachi 2003; laparoscopic
    surgery: Berguer et al. 1999; conservation: Stoner and Rushfield 2012).
  - The full RULA Table A/B/C grand-score lookup is NOT implemented here. It is
    only needed to validate against expert RULA, which happens in Part 5 on the
    August data, where the tables will be sourced from a reference.

Author: M1 (Tech Lead), Empathetic Conservator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List, Dict
import time as _time
import numpy as np


# --------------------------------------------------------------------------
# Configuration. Tune here, not in the logic. Every breakpoint below is a
# published RULA angle threshold; the weights are the documented design choice.
# --------------------------------------------------------------------------

class PSSv2Config:
    # Smoothing: mean over the last SMOOTHING_TIME_S seconds of samples,
    # independent of camera frame rate. SMOOTHING_MAX_SAMPLES only bounds
    # memory. (v1 used a 30-frame window, which assumes 30 fps; the dual-camera
    # rig will run slower and would silently stretch the window.)
    SMOOTHING_TIME_S = 1.0
    SMOOTHING_MAX_SAMPLES = 300

    # RULA per-joint angle knots as (angle_deg, band_value). Piecewise linear
    # between knots, clamped outside. Bands mirror the RULA integer bands.
    # Trunk flexion: upright -> 1, 0-20 -> 2, 20-60 -> 3, >60 -> 4.
    TRUNK_KNOTS = [(0.0, 1.0), (20.0, 2.0), (60.0, 3.0), (85.0, 4.0)]
    # Genuine neck extension starts beyond this deadband. Within it, negative
    # flexion is treated as upright (calibration jitter, not extension).
    # KEEP IN SYNC with rula.EXTENSION_DEADBAND_DEG.
    NECK_EXTENSION_DEADBAND_DEG = 5.0

    # Neck flexion: 0-10 -> 2 boundary, 10-20 -> 3, >20 -> 4. Upright ~0 -> 1.
    NECK_KNOTS = [(0.0, 1.0), (10.0, 2.0), (20.0, 3.0), (35.0, 4.0)]
    # Upper-arm elevation (shoulder flexion): <20 -> 1, 20-45 -> 2, 45-90 -> 3, >90 -> 4.
    UPPER_ARM_KNOTS = [(0.0, 1.0), (20.0, 1.0), (45.0, 2.0), (90.0, 3.0), (160.0, 4.0)]
    # Lower-arm (elbow) flexion: comfortable in 60-100, penalty outside.
    LOWER_ARM_KNOTS = [(0.0, 2.0), (60.0, 1.0), (100.0, 1.0), (160.0, 2.0)]

    # Band ranges, used to normalise each joint to [0, 1] (band_min -> 0).
    TRUNK_BAND_RANGE = (1.0, 4.0)
    NECK_BAND_RANGE = (1.0, 4.0)
    UPPER_ARM_BAND_RANGE = (1.0, 4.0)
    LOWER_ARM_BAND_RANGE = (1.0, 2.0)

    # Twist / side-bend adders (RULA adds +1 to the joint band for each).
    # These come from the front camera in Part 2; default 0 until then.
    MAX_TWIST_ADDER = 1.0
    MAX_SIDEBEND_ADDER = 1.0
    # Lateral angle (deg) at which the adder saturates to its max.
    TWIST_SATURATION_DEG = 30.0
    SIDEBEND_SATURATION_DEG = 20.0

    # Within-group weights (each group sums to 1).
    W_TRUNK = 0.55
    W_NECK = 0.45
    W_UPPER_ARM = 0.70
    W_LOWER_ARM = 0.30

    # Between-group weights (sum to 1). Group B (neck+trunk) dominant.
    W_GROUP_B = 0.70
    W_GROUP_A = 0.30

    # Sustained-static load, RULA's muscle-use +1 for a posture held statically.
    # If enabled, a moderate posture that persists slowly accrues extra risk.
    ENABLE_STATIC_LOAD = False
    STATIC_LOAD_MAX = 0.10          # max fraction added to pss_raw
    STATIC_LOAD_RAMP_S = 60.0       # seconds of sustained load to reach the max
    STATIC_LOAD_MIN_BAND = 2.0      # only accrues when group-B band exceeds this


# --------------------------------------------------------------------------
# Canonical, source-agnostic posture input.
# --------------------------------------------------------------------------

@dataclass
class PostureAngles:
    """
    Anatomical angles for one frame, in degrees. Sagittal flexions are positive
    forward. Lateral quantities (sidebend, twist, gaze) are SIGNED: fusion emits
    the direction, the controller uses the sign to pick the counter-move
    direction, and scoring takes abs() internally so sign never affects PSS.
    Part 2 fills these from fused two-camera pose; the interim single-view
    extractor fills what it can and marks the rest low-confidence.
    """
    trunk_flexion_deg: float = 0.0
    neck_flexion_deg: float = 0.0
    upper_arm_flexion_deg: float = 20.0   # ~20 is a relaxed neutral for the arm
    lower_arm_flexion_deg: float = 90.0   # ~90 elbow is neutral

    trunk_sidebend_deg: float = 0.0
    trunk_twist_deg: float = 0.0
    neck_sidebend_deg: float = 0.0
    neck_twist_deg: float = 0.0

    # Optional continuity signal from the old pipeline, logged but not scored.
    lateral_gaze_deg: float = 0.0

    # Per-axis confidence in [0, 1]; lets fusion down-weight a poor view.
    confidence: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Ramp helper and per-joint band functions.
# --------------------------------------------------------------------------

def _piecewise(angle: float, knots) -> float:
    """Piecewise-linear interpolation through (angle, band) knots, clamped."""
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    if angle <= xs[0]:
        return ys[0]
    if angle >= xs[-1]:
        return ys[-1]
    return float(np.interp(angle, xs, ys))


def _lateral_adder(angle_deg: float, saturation_deg: float, max_add: float) -> float:
    """Map an unsigned lateral angle to a 0..max_add band adder."""
    if saturation_deg <= 0:
        return 0.0
    return max_add * float(np.clip(abs(angle_deg) / saturation_deg, 0.0, 1.0))


def trunk_band(a: PostureAngles, cfg=PSSv2Config) -> float:
    base = _piecewise(a.trunk_flexion_deg, cfg.TRUNK_KNOTS)
    base += _lateral_adder(a.trunk_sidebend_deg, cfg.SIDEBEND_SATURATION_DEG, cfg.MAX_SIDEBEND_ADDER)
    base += _lateral_adder(a.trunk_twist_deg, cfg.TWIST_SATURATION_DEG, cfg.MAX_TWIST_ADDER)
    return float(np.clip(base, cfg.TRUNK_BAND_RANGE[0], cfg.TRUNK_BAND_RANGE[1] + cfg.MAX_TWIST_ADDER + cfg.MAX_SIDEBEND_ADDER))


def neck_band(a: PostureAngles, cfg=PSSv2Config) -> float:
    # Neck in extension is RULA band 4, but a calibrated neutral jitters around
    # 0 deg, so a small deadband is required: without it, -1 deg of sensor noise
    # scores as "extension" and neutral rest lands near the intervention
    # threshold. Only genuine extension (beyond the deadband) escalates.
    n = a.neck_flexion_deg
    if n < -cfg.NECK_EXTENSION_DEADBAND_DEG:
        over = abs(n) - cfg.NECK_EXTENSION_DEADBAND_DEG
        base = _piecewise(min(35.0, 20.0 + over), cfg.NECK_KNOTS)
    else:
        base = _piecewise(max(0.0, n), cfg.NECK_KNOTS)
    base += _lateral_adder(a.neck_sidebend_deg, cfg.SIDEBEND_SATURATION_DEG, cfg.MAX_SIDEBEND_ADDER)
    base += _lateral_adder(a.neck_twist_deg, cfg.TWIST_SATURATION_DEG, cfg.MAX_TWIST_ADDER)
    return float(np.clip(base, cfg.NECK_BAND_RANGE[0], cfg.NECK_BAND_RANGE[1] + cfg.MAX_TWIST_ADDER + cfg.MAX_SIDEBEND_ADDER))


def upper_arm_band(a: PostureAngles, cfg=PSSv2Config) -> float:
    return _piecewise(a.upper_arm_flexion_deg, cfg.UPPER_ARM_KNOTS)


def lower_arm_band(a: PostureAngles, cfg=PSSv2Config) -> float:
    return _piecewise(a.lower_arm_flexion_deg, cfg.LOWER_ARM_KNOTS)


def _norm(value: float, band_range) -> float:
    lo, hi = band_range
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


# --------------------------------------------------------------------------
# The calculator.
# --------------------------------------------------------------------------

class PSSv2Calculator:
    def __init__(self, cfg=PSSv2Config, smoothing_window: Optional[int] = None):
        self.cfg = cfg
        # smoothing_window caps SAMPLES (smoothing_window=1 -> smooth == raw,
        # used by the controller's scratch scorer). The primary smoothing is
        # TIME-based (SMOOTHING_TIME_S), so behavior does not depend on camera
        # frame rate: two MediaPipe instances on the rig may run well below
        # 30 fps, and a frame-count window would silently stretch to 2 s+.
        self.window = smoothing_window or cfg.SMOOTHING_MAX_SAMPLES
        self.buffer = deque(maxlen=self.window)   # entries: (t, pss_raw)
        # Neutral offsets learned during calibration (subtracted from raw angles).
        self._neutral = {
            "trunk_flexion_deg": 0.0,
            "neck_flexion_deg": 0.0,
            "upper_arm_flexion_deg": 0.0,
            "lower_arm_flexion_deg": 0.0,
        }
        self._calibrated = False
        self._static_load_since: Optional[float] = None

    # ---- calibration -----------------------------------------------------

    def calibrate_neutral(self, samples: List[PostureAngles]) -> None:
        """
        Learn per-person measurement offsets from a neutral-posture capture.
        The RULA anchoring is anatomical (0 deg = upright), so calibration only
        removes systematic measurement bias (camera mounting, body proportions),
        it does not move the risk thresholds. Median is used, robust to twitches.
        """
        if not samples:
            self._calibrated = True
            return
        for key in self._neutral:
            vals = [getattr(s, key) for s in samples]
            self._neutral[key] = float(np.median(vals))
        # Arms have a non-zero neutral by definition; store offset relative to
        # the assumed neutral so calibration does not zero out a normal posture.
        self._neutral["upper_arm_flexion_deg"] -= 20.0
        self._neutral["lower_arm_flexion_deg"] -= 90.0
        self._calibrated = True

    def _apply_neutral(self, a: PostureAngles) -> PostureAngles:
        return PostureAngles(
            trunk_flexion_deg=a.trunk_flexion_deg - self._neutral["trunk_flexion_deg"],
            neck_flexion_deg=a.neck_flexion_deg - self._neutral["neck_flexion_deg"],
            upper_arm_flexion_deg=a.upper_arm_flexion_deg - self._neutral["upper_arm_flexion_deg"],
            lower_arm_flexion_deg=a.lower_arm_flexion_deg - self._neutral["lower_arm_flexion_deg"],
            trunk_sidebend_deg=a.trunk_sidebend_deg,
            trunk_twist_deg=a.trunk_twist_deg,
            neck_sidebend_deg=a.neck_sidebend_deg,
            neck_twist_deg=a.neck_twist_deg,
            lateral_gaze_deg=a.lateral_gaze_deg,
            confidence=a.confidence,
        )

    # ---- scoring ---------------------------------------------------------

    def compute(self, angles: PostureAngles, now: Optional[float] = None) -> dict:
        a = self._apply_neutral(angles) if self._calibrated else angles

        tb = trunk_band(a, self.cfg)
        nb = neck_band(a, self.cfg)
        ua = upper_arm_band(a, self.cfg)
        la = lower_arm_band(a, self.cfg)

        trunk_n = _norm(tb, self.cfg.TRUNK_BAND_RANGE)
        neck_n = _norm(nb, self.cfg.NECK_BAND_RANGE)
        uparm_n = _norm(ua, self.cfg.UPPER_ARM_BAND_RANGE)
        lowarm_n = _norm(la, self.cfg.LOWER_ARM_BAND_RANGE)

        group_b = self.cfg.W_TRUNK * trunk_n + self.cfg.W_NECK * neck_n
        group_a = self.cfg.W_UPPER_ARM * uparm_n + self.cfg.W_LOWER_ARM * lowarm_n

        pss_raw = self.cfg.W_GROUP_B * group_b + self.cfg.W_GROUP_A * group_a

        # Optional sustained-static accrual (RULA muscle-use +1 analogue).
        static_component = 0.0
        if self.cfg.ENABLE_STATIC_LOAD:
            static_component = self._update_static_load(group_b_band=(self.cfg.W_TRUNK * tb + self.cfg.W_NECK * nb), now=now)
            pss_raw = min(1.0, pss_raw + static_component)

        pss_raw = float(np.clip(pss_raw, 0.0, 1.0))
        t_now = now if now is not None else _time.time()
        self.buffer.append((t_now, pss_raw))
        recent = [v for (t, v) in self.buffer if (t_now - t) <= self.cfg.SMOOTHING_TIME_S]
        pss_smooth = float(np.mean(recent)) if recent else pss_raw

        return {
            # continuous RULA-style band values (comparable to expert RULA later)
            "trunk_band": tb,
            "neck_band": nb,
            "upper_arm_band": ua,
            "lower_arm_band": la,
            # group risks
            "group_B": group_b,
            "group_A": group_a,
            # angles echoed for logging / reconstruction
            "trunk_flexion_deg": a.trunk_flexion_deg,
            "neck_flexion_deg": a.neck_flexion_deg,
            "upper_arm_flexion_deg": a.upper_arm_flexion_deg,
            "lower_arm_flexion_deg": a.lower_arm_flexion_deg,
            "lateral_gaze_deg": a.lateral_gaze_deg,
            "static_load": static_component,
            # the score
            "pss_raw": pss_raw,
            "pss_smooth": pss_smooth,
        }

    def _update_static_load(self, group_b_band: float, now) -> float:
        now = now if now is not None else _time.time()
        if group_b_band < self.cfg.STATIC_LOAD_MIN_BAND:
            self._static_load_since = None
            return 0.0
        if self._static_load_since is None:
            self._static_load_since = now
            return 0.0
        held = now - self._static_load_since
        frac = float(np.clip(held / self.cfg.STATIC_LOAD_RAMP_S, 0.0, 1.0))
        return frac * self.cfg.STATIC_LOAD_MAX

    def reset(self) -> None:
        self.buffer.clear()
        self._static_load_since = None


# --------------------------------------------------------------------------
# Interim single-view angle extractor.
#
# This lets PSS_v2 run end-to-end today on the existing single-camera pipeline
# so the scoring can be smoke-tested. It is a stopgap: the front/overhead view
# still resolves trunk flexion poorly, so trunk confidence is marked low. Part 2
# replaces this with dual-camera fusion. It expects the landmark dict format
# from pose_detector.py: name -> (x, y, z, visibility), coordinates normalised.
# Using the z channel gives a rough sagittal component, better than pure 2D.
# --------------------------------------------------------------------------

def _angle_between(v1, v2) -> float:
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def extract_angles_from_landmarks(lm: dict) -> PostureAngles:
    if lm is None:
        return PostureAngles(confidence={"all": 0.0})

    def p(name):
        v = lm[name]
        return np.array([v[0], v[1], v[2]], float)

    sh_mid = (p("LEFT_SHOULDER") + p("RIGHT_SHOULDER")) / 2.0
    hip_mid = (p("LEFT_HIP") + p("RIGHT_HIP")) / 2.0
    ear_mid = (p("LEFT_EAR") + p("RIGHT_EAR")) / 2.0

    # Trunk: angle of the hip->shoulder vector from vertical, in the sagittal
    # (y, z) plane. Low confidence from a frontal view (weak z), flagged.
    trunk_vec = sh_mid - hip_mid
    trunk_sagittal = np.array([trunk_vec[1], trunk_vec[2]])   # (y, z)
    trunk_flex = _angle_between(trunk_sagittal, np.array([-1.0, 0.0]))

    # Neck: shoulder->ear vector from the trunk line, sagittal component.
    neck_vec = ear_mid - sh_mid
    neck_sagittal = np.array([neck_vec[1], neck_vec[2]])
    neck_flex = _angle_between(neck_sagittal, np.array([-1.0, 0.0]))

    # Upper arm: shoulder->elbow from the trunk line (use the more visible arm).
    def arm_flex(side):
        sh = p(f"{side}_SHOULDER"); el = p(f"{side}_ELBOW")
        return _angle_between(el - sh, hip_mid - sh_mid)

    def elbow_angle(side):
        sh = p(f"{side}_SHOULDER"); el = p(f"{side}_ELBOW"); wr = p(f"{side}_WRIST")
        return _angle_between(sh - el, wr - el)

    lvis = lm["LEFT_ELBOW"][3]; rvis = lm["RIGHT_ELBOW"][3]
    side = "LEFT" if lvis >= rvis else "RIGHT"
    upper = arm_flex(side)
    lower = elbow_angle(side)

    # Lateral cervical offset kept as a continuity (gaze) signal only.
    gaze_x = (ear_mid[0] - sh_mid[0])
    gaze_deg = float(np.degrees(np.arctan2(gaze_x, 1.0)))

    conf = {
        "trunk_flexion_deg": 0.3,   # single view: weak, flagged for fusion
        "neck_flexion_deg": 0.5,
        "upper_arm_flexion_deg": 0.6,
        "lower_arm_flexion_deg": 0.6,
    }
    return PostureAngles(
        trunk_flexion_deg=trunk_flex,
        neck_flexion_deg=neck_flex,
        upper_arm_flexion_deg=upper,
        lower_arm_flexion_deg=lower,
        lateral_gaze_deg=gaze_deg,
        confidence=conf,
    )


# --------------------------------------------------------------------------
# Self-test: synthetic postures with no camera. Verifies the scoring is
# monotonic and lands in sensible ranges against RULA intuition.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    calc = PSSv2Calculator()

    cases = {
        "neutral (upright, arm relaxed)":
            PostureAngles(trunk_flexion_deg=0, neck_flexion_deg=3,
                          upper_arm_flexion_deg=15, lower_arm_flexion_deg=90),
        "mild (slight flex)":
            PostureAngles(trunk_flexion_deg=15, neck_flexion_deg=12,
                          upper_arm_flexion_deg=40, lower_arm_flexion_deg=85),
        "moderate (bench lean)":
            PostureAngles(trunk_flexion_deg=35, neck_flexion_deg=22,
                          upper_arm_flexion_deg=60, lower_arm_flexion_deg=80),
        "severe (deep flex + twist)":
            PostureAngles(trunk_flexion_deg=65, neck_flexion_deg=30,
                          upper_arm_flexion_deg=100, lower_arm_flexion_deg=70,
                          trunk_twist_deg=25, neck_sidebend_deg=15),
    }

    print(f"{'case':<32} {'trunk_b':>7} {'neck_b':>7} {'arm_b':>6} "
          f"{'grpB':>6} {'grpA':>6} {'PSS_raw':>8}")
    print("-" * 82)
    for name, ang in cases.items():
        # fresh single-frame view (bypass smoothing history for a clean read)
        c = PSSv2Calculator()
        r = c.compute(ang)
        print(f"{name:<32} {r['trunk_band']:>7.2f} {r['neck_band']:>7.2f} "
              f"{r['upper_arm_band']:>6.2f} {r['group_B']:>6.2f} "
              f"{r['group_A']:>6.2f} {r['pss_raw']:>8.3f}")

    # Monotonicity check across a trunk-flexion sweep.
    print("\ntrunk flexion sweep (neck/arm held neutral):")
    prev = -1.0
    mono = True
    for deg in range(0, 91, 15):
        c = PSSv2Calculator()
        r = c.compute(PostureAngles(trunk_flexion_deg=deg, neck_flexion_deg=3,
                                    upper_arm_flexion_deg=15, lower_arm_flexion_deg=90))
        flag = "" if r["pss_raw"] >= prev - 1e-9 else "  <-- NON-MONOTONIC"
        if r["pss_raw"] < prev - 1e-9:
            mono = False
        print(f"  trunk={deg:>3} deg   PSS_raw={r['pss_raw']:.3f}{flag}")
        prev = r["pss_raw"]
    print(f"\nmonotonic in trunk flexion: {mono}")
