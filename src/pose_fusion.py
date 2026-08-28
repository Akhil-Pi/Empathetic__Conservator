"""
pose_fusion.py
==============
Part 2 of the Empathetic Conservator rebuild: two-camera pose fusion.

Produces the PostureAngles record that pss_v2.py consumes. This is what replaces
the interim single-view extractor and finally gives PSS_v2 a real trunk and neck
angle instead of a near-zero one.

Strategy (Approach A, monocular world-landmark best-view fusion):
  - Each camera runs its own MediaPipe PoseLandmarker and yields pose_world_landmarks,
    a metric 3D estimate (origin at the hip centre) lifted from that single view.
  - The SIDE camera resolves the sagittal plane (forward flexion) well, so trunk,
    neck and near-arm flexion are taken from it.
  - The FRONT camera resolves the frontal plane (left-right) well, so lateral
    side-bend, twist and the gaze continuity signal are taken from it.
  - Per-joint visibility gates each contribution; if the preferred view is poor,
    the other view is used as fallback.

Honesty note on axes: MediaPipe world-landmark axis signs are pinned down on the
real rig with one neutral-posture capture. The angle math below is written so a
sign flip only changes the reported direction, never the magnitude, and the
PSS_v2 neutral calibration absorbs any residual offset. The WORLD_* config flags
below are the two things to confirm on the first live run.

The pure angle functions have NO MediaPipe dependency, so the geometry is
unit-tested here with synthetic 3D poses (run `python pose_fusion.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import logging
logger = logging.getLogger(__name__)

from camera_config import (SAGITTAL_ANGLES, FRONTAL_ANGLES, TRANSVERSE_ANGLES,
                           CameraConfig)

# Anatomical neutral values used when an angle is not measurable. Zeroing an arm
# angle would read as a fully extended elbow, which is an extreme posture, so
# unmeasurable arm angles fall back to their neutral rather than to zero.
_NEUTRAL_DEFAULTS = {
    "trunk_flexion_deg": 0.0, "neck_flexion_deg": 0.0,
    "upper_arm_flexion_deg": 20.0, "lower_arm_flexion_deg": 90.0,
    "trunk_sidebend_deg": 0.0, "neck_sidebend_deg": 0.0,
    "trunk_twist_deg": 0.0, "neck_twist_deg": 0.0, "lateral_gaze_deg": 0.0,
}

# PostureAngles is the shared interface with pss_v2.py. Imported lazily so this
# module can be tested standalone if pss_v2 is not on the path.
try:
    from pss_v2 import PostureAngles
except Exception:  # pragma: no cover - fallback mirror for standalone testing
    @dataclass
    class PostureAngles:  # type: ignore
        trunk_flexion_deg: float = 0.0
        neck_flexion_deg: float = 0.0
        upper_arm_flexion_deg: float = 20.0
        lower_arm_flexion_deg: float = 90.0
        trunk_sidebend_deg: float = 0.0
        trunk_twist_deg: float = 0.0
        neck_sidebend_deg: float = 0.0
        neck_twist_deg: float = 0.0
        lateral_gaze_deg: float = 0.0
        confidence: Optional[Dict[str, float]] = None


# --------------------------------------------------------------------------
# Config: axis convention and view roles. Confirm the two WORLD_* flags on the
# first live capture; everything else is stable.
# --------------------------------------------------------------------------

class FusionConfig:
    # MediaPipe world-landmark axes (body-centric, metres). These are the
    # assumed conventions; verify on the rig with a neutral + deliberate-lean capture.
    WORLD_UP_AXIS = 1          # index of the vertical axis (x=0, y=1, z=2)
    WORLD_UP_SIGN = -1.0       # -1 if "up" is the negative direction on that axis
    WORLD_FWD_AXIS = 2         # sagittal (forward-back) axis, seen by the side cam
    WORLD_LAT_AXIS = 0         # frontal (left-right) axis, seen by the front cam

    # Minimum landmark visibility for a joint's contribution to be trusted.
    MIN_VIS = 0.5

    # Which physical camera plays which role. Set at wiring time.
    SIDE_CAM_ROLE = "side"
    FRONT_CAM_ROLE = "front"

    # Arm selection: "visible" (busier/clearer arm per frame), "left", "right".
    ARM_SELECTION = "visible"

    # Whether a frontal view may stand in for a missing sagittal view. OFF by
    # default: this is the v1 failure mode, where a front camera produced a
    # trunk angle that carried no real signal. Enable only for diagnostics.
    ALLOW_FRONT_SAGITTAL_FALLBACK = False


# Landmark indices we use (MediaPipe Pose 33-landmark model).
LM = {
    "NOSE": 0, "LEFT_EAR": 7, "RIGHT_EAR": 8,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
}


# --------------------------------------------------------------------------
# Pure geometry. World landmarks are dicts: name -> np.array([x, y, z, vis]).
# These functions know nothing about MediaPipe and are the unit-tested core.
# --------------------------------------------------------------------------

def _xyz(world: dict, name: str) -> np.ndarray:
    return np.asarray(world[name][:3], float)


def _vis(world: dict, name: str) -> float:
    return float(world[name][3])


def _midpoint(world: dict, a: str, b: str) -> np.ndarray:
    return (_xyz(world, a) + _xyz(world, b)) / 2.0


def sagittal_flexion(vec: np.ndarray, cfg=FusionConfig) -> float:
    """
    Forward-flexion angle of a body vector from vertical, in the sagittal
    (up, forward) plane. Sign-agnostic magnitude via atan2.
    """
    up = vec[cfg.WORLD_UP_AXIS] * cfg.WORLD_UP_SIGN
    fwd = vec[cfg.WORLD_FWD_AXIS]
    return float(np.degrees(np.arctan2(abs(fwd), abs(up)))) if up != 0 or fwd != 0 else 0.0


def frontal_deviation(vec: np.ndarray, cfg=FusionConfig) -> float:
    """
    Lateral (side-bend) angle of a body vector from vertical, frontal plane.
    SIGNED: the sign follows the lateral axis, so the controller can tell which
    way the person is leaning and shift the artifact to counter it. PSS_v2 takes
    the magnitude internally, so signed input does not affect scoring.

    The reference "vertical" length is the vector's projection onto the
    sagittal-vertical plane (up and forward/back), NOT just its raw up
    component. A segment (shoulder-hip, ear-shoulder) keeps roughly the same
    length in that plane whether the person bends forward or stands upright;
    only its split between up and forward changes. Dividing by the raw up
    component alone shrinks the denominator as someone leans forward, which
    inflates ANY nonzero lateral component into a large angle even with no
    real side-bend -- confirmed with a synthetic pose held at a constant 15
    deg true side-bend: the old formula reported 15.0 deg at trunk_deg=0 but
    57.1 deg at trunk_deg=80, purely from forward lean.
    """
    up = vec[cfg.WORLD_UP_AXIS]
    fwd = vec[cfg.WORLD_FWD_AXIS]
    lat = vec[cfg.WORLD_LAT_AXIS]
    vertical_ref = float(np.hypot(up, fwd))
    return float(np.degrees(np.arctan2(lat, vertical_ref))) if vertical_ref != 0 or lat != 0 else 0.0


def transverse_twist(upper_vec: np.ndarray, lower_vec: np.ndarray, cfg=FusionConfig) -> float:
    """
    Signed twist angle (deg) of `upper_vec` relative to `lower_vec`, measured
    in the transverse (horizontal) plane spanned by the lateral and
    forward-back axes. This is the "projected shoulder line against the hip
    line" measurement documented as the twist method in camera_config.py:
    trunk twist = shoulder line vs hip line, neck twist = ear line vs
    shoulder line. Zero means the two lines point the same way when viewed
    from above (no rotation about the vertical axis); it does NOT need the
    side camera, since twist about the vertical axis is invisible from the
    side and only shows up in the front view's horizontal-plane projection.
    """
    def _horiz(v: np.ndarray) -> np.ndarray:
        return np.array([v[cfg.WORLD_LAT_AXIS], v[cfg.WORLD_FWD_AXIS]])
    u, l = _horiz(upper_vec), _horiz(lower_vec)
    nu, nl = np.linalg.norm(u), np.linalg.norm(l)
    if nu < 1e-9 or nl < 1e-9:
        return 0.0
    cross = l[0] * u[1] - l[1] * u[0]
    dot = float(np.dot(l, u))
    return float(np.degrees(np.arctan2(cross, dot)))


def angle_at(joint: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Interior angle at `joint` formed by segments joint->a and joint->b."""
    u, v = a - joint, b - joint
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def arm_elevation(shoulder: np.ndarray, elbow: np.ndarray, hip: np.ndarray) -> float:
    """Upper-arm elevation: angle of shoulder->elbow from the trunk (hip->shoulder)."""
    trunk_dir = shoulder - hip
    arm_dir = elbow - shoulder
    nu, nv = np.linalg.norm(trunk_dir), np.linalg.norm(arm_dir)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    # elevation measured from hanging-down (opposite of trunk_dir)
    c = np.clip(np.dot(-trunk_dir, arm_dir) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


# --------------------------------------------------------------------------
# Fusion: two world-landmark dicts -> one PostureAngles.
# --------------------------------------------------------------------------

def _arm_side(world: dict, cfg=FusionConfig) -> str:
    if cfg.ARM_SELECTION in ("left", "right"):
        return cfg.ARM_SELECTION.upper()
    lv = _vis(world, "LEFT_ELBOW") + _vis(world, "LEFT_WRIST")
    rv = _vis(world, "RIGHT_ELBOW") + _vis(world, "RIGHT_WRIST")
    return "LEFT" if lv >= rv else "RIGHT"


def _trunk_vec(world: dict) -> np.ndarray:
    return _midpoint(world, "LEFT_SHOULDER", "RIGHT_SHOULDER") - _midpoint(world, "LEFT_HIP", "RIGHT_HIP")


def _neck_vec(world: dict) -> np.ndarray:
    return _midpoint(world, "LEFT_EAR", "RIGHT_EAR") - _midpoint(world, "LEFT_SHOULDER", "RIGHT_SHOULDER")


def _joint_ok(world: dict, names, cfg=FusionConfig) -> bool:
    return world is not None and all(_vis(world, n) >= cfg.MIN_VIS for n in names)


def fuse(side_world: Optional[dict], front_world: Optional[dict],
         cfg=FusionConfig) -> PostureAngles:
    """
    Combine the two views into one PostureAngles. Sagittal flexions come from the
    side view; lateral/twist and gaze from the front view. Each contribution
    falls back to the other view if its preferred view is missing or low-visibility.
    """
    conf: Dict[str, float] = {}

    # ---- trunk flexion (prefer side) ----
    trunk_flex, tconf = _pick_sagittal(side_world, front_world,
                                       _trunk_vec, ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"], cfg)
    conf["trunk_flexion_deg"] = tconf

    # ---- neck flexion (prefer side) ----
    neck_flex, nconf = _pick_sagittal(side_world, front_world,
                                      _neck_vec, ["LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER"], cfg)
    conf["neck_flexion_deg"] = nconf

    # ---- lateral / twist (prefer front) ----
    trunk_side = 0.0
    neck_side = 0.0
    trunk_twist = 0.0
    neck_twist = 0.0
    gaze_deg = 0.0
    trunk_side_needed = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
    neck_side_needed = ["LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
    if _joint_ok(front_world, trunk_side_needed, cfg):
        trunk_side = frontal_deviation(_trunk_vec(front_world), cfg)
        conf["trunk_sidebend_deg"] = min(_vis(front_world, n) for n in trunk_side_needed)
        # twist: shoulder line rotated against the hip line in the horizontal
        # plane. Invisible from the side camera by construction, so this is
        # front-view only with no side fallback (unlike flexion above).
        shoulder_line = _xyz(front_world, "RIGHT_SHOULDER") - _xyz(front_world, "LEFT_SHOULDER")
        hip_line = _xyz(front_world, "RIGHT_HIP") - _xyz(front_world, "LEFT_HIP")
        trunk_twist = transverse_twist(shoulder_line, hip_line, cfg)
        conf["trunk_twist_deg"] = conf["trunk_sidebend_deg"]
    if _joint_ok(front_world, neck_side_needed, cfg):
        neck_side = frontal_deviation(_neck_vec(front_world), cfg)
        neck_side_conf = min(_vis(front_world, n) for n in neck_side_needed)
        conf["neck_sidebend_deg"] = neck_side_conf
        # neck twist: ear line rotated against the shoulder line, same method
        # as trunk twist one level up the chain (head turned relative to torso).
        ear_line = _xyz(front_world, "RIGHT_EAR") - _xyz(front_world, "LEFT_EAR")
        shoulder_line = _xyz(front_world, "RIGHT_SHOULDER") - _xyz(front_world, "LEFT_SHOULDER")
        neck_twist = transverse_twist(ear_line, shoulder_line, cfg)
        conf["neck_twist_deg"] = neck_side_conf
        # gaze continuity (legacy field, not scored): ear-midpoint deviation
        # from shoulder-midpoint is the same "_neck_vec through vertical"
        # geometry as neck_sidebend_deg above, so reuse that value directly
        # instead of re-deriving it with a near-duplicate formula that could
        # drift out of sync.
        gaze_deg = neck_side
        conf["lateral_gaze_deg"] = neck_side_conf

    # ---- arm (prefer side; near arm is well seen from the side) ----
    upper_arm, lower_arm, aconf = 20.0, 90.0, 0.0
    arm_src = side_world if side_world is not None else front_world
    if arm_src is not None:
        s = _arm_side(arm_src, cfg)
        needed = [f"{s}_SHOULDER", f"{s}_ELBOW", f"{s}_WRIST"]
        if _joint_ok(arm_src, needed, cfg):
            sh = _xyz(arm_src, f"{s}_SHOULDER")
            el = _xyz(arm_src, f"{s}_ELBOW")
            wr = _xyz(arm_src, f"{s}_WRIST")
            hip = _midpoint(arm_src, "LEFT_HIP", "RIGHT_HIP")
            upper_arm = arm_elevation(sh, el, hip)
            lower_arm = angle_at(el, sh, wr)
            aconf = min(_vis(arm_src, n) for n in needed)
    conf["upper_arm_flexion_deg"] = aconf
    conf["lower_arm_flexion_deg"] = aconf

    return PostureAngles(
        trunk_flexion_deg=trunk_flex,
        neck_flexion_deg=neck_flex,
        upper_arm_flexion_deg=upper_arm,
        lower_arm_flexion_deg=lower_arm,
        trunk_sidebend_deg=trunk_side,
        neck_sidebend_deg=neck_side,
        trunk_twist_deg=trunk_twist,
        neck_twist_deg=neck_twist,
        lateral_gaze_deg=gaze_deg,
        confidence=conf,
    )


def apply_layout_mask(angles: "PostureAngles", layout=None) -> "PostureAngles":
    """
    Zero every angle the active camera layout cannot honestly measure, and set
    its confidence to 0.

    This is the direct fix for the v1 defect. v1 emitted a trunk angle from a
    front camera that could not resolve forward lean, and nothing downstream
    knew the number was meaningless. Here an unmeasurable quantity is reported
    as absent rather than guessed, so PSS_v2 adds nothing for it and the
    controller can see that it has no basis to act on it.

    Also scales the confidence of measurable angles by the layout's nominal
    factor, so an oblique or degraded placement is visible in the logs.
    """
    from dataclasses import replace as _replace
    if layout is None:
        from camera_config import CameraConfig
        layout = CameraConfig.layout()

    conf = dict(getattr(angles, "confidence", None) or {})
    updates = {}
    for name in (SAGITTAL_ANGLES + FRONTAL_ANGLES + TRANSVERSE_ANGLES):
        if not hasattr(angles, name):
            continue
        if layout.can_measure(name):
            if name in conf:
                conf[name] = float(conf[name]) * layout.scale_for(name)
        else:
            # arms have a non-zero anatomical neutral; zeroing them would read
            # as an extreme posture, so leave the dataclass default in place
            updates[name] = _NEUTRAL_DEFAULTS.get(name, 0.0)
            conf[name] = 0.0
    conf["_layout"] = layout.name
    return _replace(angles, confidence=conf, **updates)


def _pick_sagittal(side_world, front_world, vec_fn, needed, cfg) -> Tuple[float, float]:
    """Sagittal angle from the side view if trustworthy, else the front view."""
    if _joint_ok(side_world, needed, cfg):
        return sagittal_flexion(vec_fn(side_world), cfg), min(_vis(side_world, n) for n in needed)
    if _joint_ok(front_world, needed, cfg) and cfg.ALLOW_FRONT_SAGITTAL_FALLBACK:
        # A frontal view resolves sagittal motion badly: forward lean is along
        # its optical axis. v1 shipped exactly this number at full authority and
        # the trunk term collapsed to about 2 percent of the score. Kept behind
        # a config flag, defaulting OFF, and heavily discounted when enabled.
        return (sagittal_flexion(vec_fn(front_world), cfg),
                0.2 * min(_vis(front_world, n) for n in needed))
    return 0.0, 0.0


# Map from MediaPipe landmark index to the name keys used by the fusion geometry.
_IDX_TO_NAME = {v: k for k, v in LM.items()}


def _landmarks_to_dict(world_landmarks) -> dict:
    """Convert MediaPipe pose_world_landmarks into the dict format the fusion
    geometry expects: {name: np.array([x, y, z, visibility])}."""
    out = {}
    for idx, name in _IDX_TO_NAME.items():
        lm = world_landmarks[idx]
        out[name] = np.array([lm.x, lm.y, lm.z, lm.visibility])
    return out

# --------------------------------------------------------------------------
# MediaPipe wrapper. One instance per camera. Guarded so this file imports and
# self-tests without a camera. Extracts BOTH normalized and world landmarks.
# --------------------------------------------------------------------------

class PoseDetectorV2:
    def __init__(self, model_path: str = "pose_landmarker.task",
                 min_det: float = 0.6, min_track: float = 0.6):
        import os
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import RunningMode
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._mp = mp
        base = python.BaseOptions(model_asset_path=model_path)
        opts = vision.PoseLandmarkerOptions(
            base_options=base, running_mode=RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=min_det,
            min_pose_presence_confidence=min_track,
            min_tracking_confidence=min_track,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, frame, timestamp_ms: int):
        """Detect pose landmarks. Ensures timestamps are strictly increasing
        per detector instance, which MediaPipe requires for detect_for_video."""
        import cv2
        import mediapipe as mp
        if frame is None:
            return None
        # MediaPipe rejects non-increasing timestamps. On Windows, time.time()
        # resolution can produce duplicates at high frame rates.
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            result = self.landmarker.detect_for_video(image, timestamp_ms)
        except Exception as e:
            logger.debug(f"pose detection failed: {e}")
            return None
        if not result.pose_world_landmarks:
            return None
        world = _landmarks_to_dict(result.pose_world_landmarks[0])
        # attach normalized landmarks for optional skeleton drawing
        if result.pose_landmarks:
            world["_normalized"] = result.pose_landmarks[0]
        return world

    def close(self):
        self.landmarker.close()


# --------------------------------------------------------------------------
# Self-test: synthetic world poses, no camera, no MediaPipe. Verifies the
# geometry recovers known input angles.
# --------------------------------------------------------------------------

def _make_pose(trunk_deg=0.0, neck_deg=0.0, sidebend_deg=0.0,
              trunk_twist_deg=0.0, neck_twist_deg=0.0, cfg=FusionConfig) -> dict:
    """Build a synthetic world pose with known flexion/side-bend/twist. Twist
    rotates the shoulder line relative to the hip line (trunk_twist_deg) and
    the ear line relative to the shoulder line (neck_twist_deg) about the
    vertical axis, in the horizontal (lateral, forward) plane."""
    up = np.zeros(3); up[cfg.WORLD_UP_AXIS] = 1.0 / cfg.WORLD_UP_SIGN
    fwd = np.zeros(3); fwd[cfg.WORLD_FWD_AXIS] = 1.0
    lat = np.zeros(3); lat[cfg.WORLD_LAT_AXIS] = 1.0

    def _rotate_horiz(deg: float) -> np.ndarray:
        r = np.radians(deg)
        return np.cos(r) * lat + np.sin(r) * fwd

    hip_mid = np.zeros(3)
    tr = np.radians(trunk_deg)
    sb = np.radians(sidebend_deg)
    # shoulder above hips, leaned forward by trunk_deg and sideways by sidebend_deg
    sh_mid = hip_mid + 0.5 * (np.cos(tr) * np.cos(sb) * up
                              + np.sin(tr) * fwd
                              + np.sin(sb) * lat)
    nk = np.radians(neck_deg)
    ear_mid = sh_mid + 0.25 * (np.cos(nk) * up + np.sin(nk) * fwd)

    half = 0.20 * lat
    hip_half = half
    sh_half = 0.20 * _rotate_horiz(trunk_twist_deg)
    ear_half = 0.5 * 0.20 * _rotate_horiz(trunk_twist_deg + neck_twist_deg)
    world = {
        "LEFT_HIP": np.append(hip_mid - hip_half, 1.0),
        "RIGHT_HIP": np.append(hip_mid + hip_half, 1.0),
        "LEFT_SHOULDER": np.append(sh_mid - sh_half, 1.0),
        "RIGHT_SHOULDER": np.append(sh_mid + sh_half, 1.0),
        "LEFT_EAR": np.append(ear_mid - ear_half, 1.0),
        "RIGHT_EAR": np.append(ear_mid + ear_half, 1.0),
        "NOSE": np.append(ear_mid, 1.0),
        "LEFT_ELBOW": np.append(sh_mid - half + 0.3 * (-up) + 0.05 * fwd, 1.0),
        "RIGHT_ELBOW": np.append(sh_mid + half + 0.3 * (-up) + 0.05 * fwd, 1.0),
        "LEFT_WRIST": np.append(sh_mid - half + 0.55 * (-up) + 0.25 * fwd, 1.0),
        "RIGHT_WRIST": np.append(sh_mid + half + 0.55 * (-up) + 0.25 * fwd, 1.0),
    }
    return world


if __name__ == "__main__":
    print("fusion geometry self-test (synthetic world poses)\n")
    print(f"{'input trunk':>12} {'input neck':>11} {'-> fused trunk':>15} {'fused neck':>11}")
    print("-" * 52)
    ok = True
    for t, n in [(0, 0), (20, 10), (35, 20), (60, 25), (85, 30)]:
        side = _make_pose(trunk_deg=t, neck_deg=n)
        pa = fuse(side, None)
        # neck angle here is measured relative to vertical; the pose builds the
        # neck vector at neck_deg from vertical, so fused neck should track n.
        terr = abs(pa.trunk_flexion_deg - t)
        nerr = abs(pa.neck_flexion_deg - n)
        flag = "" if terr < 1.5 and nerr < 1.5 else "  <-- MISMATCH"
        if terr >= 1.5 or nerr >= 1.5:
            ok = False
        print(f"{t:>12} {n:>11} {pa.trunk_flexion_deg:>15.1f} {pa.neck_flexion_deg:>11.1f}{flag}")

    print("\nside-bend recovery (front view), trunk held upright:")
    for sb in [0, 10, 20]:
        front = _make_pose(trunk_deg=0, neck_deg=0, sidebend_deg=sb)
        pa = fuse(None, front)
        print(f"  input sidebend={sb:>3}  -> fused trunk_sidebend={pa.trunk_sidebend_deg:.1f}")

    print("\nfallback: side view missing, front-only trunk (flagged low-confidence):")
    front = _make_pose(trunk_deg=30, neck_deg=0)
    pa = fuse(None, front)
    print(f"  fused trunk={pa.trunk_flexion_deg:.1f}  confidence={pa.confidence['trunk_flexion_deg']:.2f}")

    print("\ntwist recovery (front view): trunk_twist_deg was never computed before "
          "this fix and always read 0 regardless of input -- this is what the "
          "goal_controller's drot DOF actually needed and never got.")
    twist_ok = True
    for tw in [0, 15, -25]:
        front = _make_pose(trunk_twist_deg=tw)
        pa = fuse(None, front)
        err = abs(pa.trunk_twist_deg - tw)
        if err >= 1.5:
            twist_ok = False
        print(f"  input trunk_twist={tw:>4}  -> fused={pa.trunk_twist_deg:6.1f}"
              f"{'  <-- MISMATCH' if err >= 1.5 else ''}")
    for tw in [0, 20]:
        front = _make_pose(neck_twist_deg=tw)
        pa = fuse(None, front)
        err = abs(pa.neck_twist_deg - tw)
        if err >= 1.5:
            twist_ok = False
        print(f"  input neck_twist ={tw:>4}  -> fused={pa.neck_twist_deg:6.1f}"
              f"{'  <-- MISMATCH' if err >= 1.5 else ''}")
    ok = ok and twist_ok
    print("\nall geometry checks passed" if ok else "\nSOME CHECKS FAILED")

    print(f"\nall sagittal recoveries within tolerance: {ok}")
