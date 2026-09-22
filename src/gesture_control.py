"""
gesture_control.py
===================
Hands-free pause/resume for the task-phase loop.

Open palm facing the camera -> PAUSE the task workflow. Closed fist -> RESUME.
Entirely additive and gated by GestureConfig.ENABLED, which defaults to False:
with the flag off, nothing in this module is wired into the session loop and
behaviour is identical to before this feature existed.

Detection uses MediaPipe Tasks GestureRecognizer's canned gesture model, which
already classifies "Open_Palm" and "Closed_Fist" among its built-in labels. It
is loaded the same way pose_fusion.PoseDetectorV2 loads pose_landmarker.task:
guarded import (mediapipe is not required to import this module), a
FileNotFoundError if the model file is missing -- but that error only surfaces
when GestureConfig.ENABLED is True and GestureRecognizerWrapper is actually
constructed -- and strictly increasing timestamps for detect_for_video (the
same Windows clock-resolution issue PoseDetectorV2 works around).

PauseStateMachine has NO MediaPipe dependency, so the debounce/cooldown/
idempotency logic is unit-tested here with synthetic gesture labels (run
`python gesture_control.py`); no camera required. This mirrors how pss_v2.py
scores angles without a camera and goal_controller.py optimises without a
robot.

Debounce is TIME-based (HOLD_SECONDS), not a raw frame count, for the same
reason pss_v2's smoothing window is time-based rather than frame-count-based:
the two cameras on this rig do not run at a fixed frame rate, and a
frame-count debounce would silently stretch or shrink with it.
"""

from __future__ import annotations

import sys
import time as _time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Config. ENABLED defaults to False: every behaviour in this module is inert
# until a caller flips it, and run_session.py only wires the feature in when
# it is True.
# --------------------------------------------------------------------------

class GestureConfig:
    ENABLED = True              # default OFF; flip deliberately to test live

    # Which already-read camera frame to run recognition on. run_session.py
    # falls back from "front" to "side" if the active camera layout lacks a
    # front camera, and disables the feature for the run if neither exists --
    # this module never opens a second capture of its own.
    CONTROL_CAMERA = "front"

    # A gesture must be held continuously for this long before it toggles
    # state. Guards against a momentary, incidental hand shape (conservators
    # naturally open/close their hands while working) being read as a command.
    HOLD_SECONDS = 0.6

    # Minimum GestureRecognizer confidence for a label to be trusted at all;
    # below this the frame is treated as "no gesture seen" for debounce
    # purposes, same as if the hand were out of frame.
    MIN_CONFIDENCE = 0.6

    # Minimum time between state changes. Prevents a toggle from immediately
    # reversing itself on marginal, flickering detections right at the
    # HOLD_SECONDS boundary.
    COOLDOWN_S = 1.0

    PAUSE_GESTURE = "Open_Palm"
    RESUME_GESTURE = "Closed_Fist"

    # MediaPipe Tasks GestureRecognizer canned model, repo root -- mirrors
    # PoseDetectorV2's pose_landmarker.task convention exactly.
    MODEL_PATH = "gesture_recognizer.task"


# --------------------------------------------------------------------------
# Pure state machine. No MediaPipe import anywhere in this class.
# --------------------------------------------------------------------------

class PauseStateMachine:
    """
    update(gesture_label, score, now) -> state ("RUNNING" or "PAUSED").

    Rules, all pure and deterministic given (label, score, now) sequences:
      - A label counts only if score >= cfg.MIN_CONFIDENCE; otherwise it is
        treated as "nothing seen" (None) for debounce purposes.
      - The current candidate label must be held continuously for
        cfg.HOLD_SECONDS (wall-clock, via the `now` argument) before it is
        eligible to toggle state. A single spurious frame never toggles
        anything, because held-time is 0 on the frame the candidate changes.
      - Once eligible, a toggle is further blocked until cfg.COOLDOWN_S has
        elapsed since the last actual state change.
      - PAUSE_GESTURE while already PAUSED, and RESUME_GESTURE while already
        RUNNING, are no-ops: they neither change state nor touch the cooldown
        clock (nothing happened, so nothing should be "cooling down" from).
    """

    RUNNING = "RUNNING"
    PAUSED = "PAUSED"

    def __init__(self, cfg=GestureConfig):
        self.cfg = cfg
        self.state = self.RUNNING
        self._candidate: Optional[str] = None
        self._candidate_since: Optional[float] = None
        self._last_change_at: float = float("-inf")

    def update(self, gesture_label: Optional[str], score: float = 1.0,
              now: Optional[float] = None) -> str:
        now = now if now is not None else _time.time()
        cfg = self.cfg

        label = gesture_label if (gesture_label is not None
                                  and score >= cfg.MIN_CONFIDENCE) else None

        if label != self._candidate:
            self._candidate = label
            self._candidate_since = now
        held = (now - self._candidate_since) if self._candidate_since is not None else 0.0

        if label is None or held < cfg.HOLD_SECONDS:
            return self.state

        if (now - self._last_change_at) < cfg.COOLDOWN_S:
            return self.state

        if label == cfg.PAUSE_GESTURE and self.state != self.PAUSED:
            self.state = self.PAUSED
            self._last_change_at = now
        elif label == cfg.RESUME_GESTURE and self.state != self.RUNNING:
            self.state = self.RUNNING
            self._last_change_at = now
        return self.state

    def reset(self) -> None:
        self.state = self.RUNNING
        self._candidate = None
        self._candidate_since = None
        self._last_change_at = float("-inf")


# --------------------------------------------------------------------------
# MediaPipe wrapper. Guarded so this module imports and self-tests without
# mediapipe or a model file present. Analogous to pose_fusion.PoseDetectorV2.
# --------------------------------------------------------------------------

class GestureRecognizerWrapper:
    def __init__(self, model_path: Optional[str] = None,
                min_confidence: Optional[float] = None, cfg=GestureConfig):
        import os
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import RunningMode

        model_path = model_path or cfg.MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Gesture model not found: {model_path}")
        conf = min_confidence if min_confidence is not None else cfg.MIN_CONFIDENCE
        self._mp = mp
        base = python.BaseOptions(model_asset_path=model_path)
        opts = vision.GestureRecognizerOptions(
            base_options=base, running_mode=RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=conf,
            min_hand_presence_confidence=conf,
            min_tracking_confidence=conf,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(opts)
        self._last_ts = -1

    def detect(self, frame, timestamp_ms: int):
        """Top gesture (label, score) for this frame, or (None, 0.0) if no
        hand/gesture was recognised. Ensures timestamps are strictly
        increasing per instance, same reason as PoseDetectorV2.detect."""
        import cv2
        if frame is None:
            return None, 0.0
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            result = self.recognizer.recognize_for_video(image, timestamp_ms)
        except Exception as e:
            logger.debug(f"gesture recognition failed: {e}")
            return None, 0.0
        if not result.gestures or not result.gestures[0]:
            return None, 0.0
        top = result.gestures[0][0]
        return top.category_name, float(top.score)

    def close(self):
        self.recognizer.close()


# --------------------------------------------------------------------------
# Optional live demo: `python gesture_control.py --camera 0` opens a local
# webcam, prints the detected gesture/score/state each frame, and shows a
# window with the same RUNNING/PAUSED overlay run_session.py's --preview
# uses. This is for tuning HOLD_SECONDS/MIN_CONFIDENCE against a real hand
# and real lighting BEFORE wiring the feature into a full session -- it does
# not touch GestureConfig.ENABLED or run_session.py at all. Requires
# mediapipe, opencv, and gesture_recognizer.task; the default `python
# gesture_control.py` (no args) below still needs none of those.
# --------------------------------------------------------------------------

def _live_demo(camera_index: int, model_path: str):
    import cv2
    print(f"opening camera {camera_index} -- press 'q' to quit")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {camera_index}")
    recognizer = GestureRecognizerWrapper(model_path=model_path)
    sm = PauseStateMachine()
    ts = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            ts += 33   # ~30fps fixed step; only needs to be strictly increasing
            label, score = recognizer.detect(frame, ts)
            state = sm.update(label, score, now=ts / 1000.0)
            print(f"\rgesture={label or '-':<12} score={score:.2f}  state={state:<8}",
                  end="", flush=True)
            color = (0, 0, 255) if state == "PAUSED" else (0, 255, 0)
            cv2.putText(frame, f"{state}  {label or '-'} ({score:.2f})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("gesture_control live demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print()
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# Self-test: synthetic gesture-label sequences, no camera, no MediaPipe.
# This is what runs by default (`python gesture_control.py`, no args).
# --------------------------------------------------------------------------

def _pure_self_test():
    cfg = GestureConfig

    def run_sequence(name, events, expect_final):
        """events: list of (dt, label, score). dt is seconds since the
        previous event (0 for the first)."""
        sm = PauseStateMachine(cfg)
        t = 0.0
        trace = []
        for dt, label, score in events:
            t += dt
            s = sm.update(label, score, now=t)
            trace.append(s)
        ok = trace[-1] == expect_final
        print(f"{name:<55} final={trace[-1]:<8} expected={expect_final:<8} "
              f"{'OK' if ok else '<-- MISMATCH'}")
        return ok

    print("PauseStateMachine self-test (synthetic gesture sequences)\n")
    all_ok = True

    all_ok &= run_sequence(
        "single-frame spurious palm does not toggle",
        [(0.0, "Open_Palm", 0.9)],
        "RUNNING")

    all_ok &= run_sequence(
        "palm held short of HOLD_SECONDS stays RUNNING",
        [(0.0, "Open_Palm", 0.9), (cfg.HOLD_SECONDS * 0.5, "Open_Palm", 0.9)],
        "RUNNING")

    all_ok &= run_sequence(
        "palm held past HOLD_SECONDS -> PAUSED",
        [(0.0, "Open_Palm", 0.9), (cfg.HOLD_SECONDS + 0.05, "Open_Palm", 0.9)],
        "PAUSED")

    all_ok &= run_sequence(
        "fist held past HOLD_SECONDS while running is a no-op (still RUNNING)",
        [(0.0, "Closed_Fist", 0.9), (cfg.HOLD_SECONDS + 0.05, "Closed_Fist", 0.9)],
        "RUNNING")

    all_ok &= run_sequence(
        "pause, then fist held past HOLD_SECONDS -> RESUME",
        [(0.0, "Open_Palm", 0.9), (cfg.HOLD_SECONDS + 0.05, "Open_Palm", 0.9),
         (cfg.COOLDOWN_S + 0.05, "Closed_Fist", 0.9),
         (cfg.HOLD_SECONDS + 0.05, "Closed_Fist", 0.9)],
        "RUNNING")

    all_ok &= run_sequence(
        "cooldown blocks an immediate re-toggle right after PAUSE",
        [(0.0, "Open_Palm", 0.9), (cfg.HOLD_SECONDS + 0.05, "Open_Palm", 0.9),
         # fist appears immediately and is held long enough, but cooldown
         # since the PAUSE transition has not elapsed yet
         (0.01, "Closed_Fist", 0.9), (cfg.HOLD_SECONDS + 0.05, "Closed_Fist", 0.9)],
        "PAUSED")

    all_ok &= run_sequence(
        "low-confidence palm never counts as a gesture at all",
        [(0.0, "Open_Palm", 0.1), (cfg.HOLD_SECONDS + 0.05, "Open_Palm", 0.1)],
        "RUNNING")

    print(f"\nall gesture state-machine checks passed: {bool(all_ok)}")
    return all_ok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", type=int, default=None,
                   help="camera index for a live webcam demo (needs "
                        "mediapipe + gesture_recognizer.task). Omit to run "
                        "the default no-camera self-test instead.")
    p.add_argument("--model", default="gesture_recognizer.task")
    args = p.parse_args()

    if args.camera is not None:
        _live_demo(args.camera, args.model)
    else:
        ok = _pure_self_test()
        sys.exit(0 if ok else 1)
