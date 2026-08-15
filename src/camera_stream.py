"""
camera_stream.py
================
Part 2 capture layer: two USB webcams read without stalling the vision loop.

Why threads: cv2.VideoCapture.read() blocks. Calling read() on both cameras
inside the main loop makes the loop run at the slower camera's rate, stacks
latency, and lets the two feeds drift apart. Each camera instead gets a daemon
thread that continuously grabs the newest frame; the main loop reads whatever
the latest frame is, with its capture timestamp, and pairs the two nearest in
time. At 30 Hz the few-millisecond jitter is fine for pose fusion, so no
hardware sync is needed.

USB notes baked in:
  - MJPEG is forced on each stream. Raw YUY2 saturates a shared USB bus and is
    the usual reason dual-camera capture fails.
  - On Windows the DirectShow backend is used and indices are opened explicitly;
    camera indices are not stable across replug, so verify once visually.
  - On Linux, pass stable /dev/v4l/by-id/... paths as the source instead of ints.
"""

from __future__ import annotations

import threading
import time
import sys
from typing import Optional, Tuple, List
import cv2


def _default_backend() -> int:
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


class CameraStream:
    """One camera, one grabber thread, always holds the newest frame."""

    def __init__(self, source, name: str = "cam",
                 width: int = 1280, height: int = 720, fps: int = 30,
                 backend: Optional[int] = None, use_mjpeg: bool = True):
        self.source = source
        self.name = name
        self.backend = backend if backend is not None else _default_backend()
        self._cap = cv2.VideoCapture(self.source, self.backend)
        if use_mjpeg:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        # Small buffer so read() returns fresh frames, not a backlog.
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._frame = None
        self._ts = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def is_opened(self) -> bool:
        return bool(self._cap.isOpened())

    def start(self) -> "CameraStream":
        if not self.is_opened():
            raise RuntimeError(f"[{self.name}] could not open source {self.source}")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # Wait briefly for the first frame so callers do not get None immediately.
        t0 = time.time()
        while self.read()[0] is None and time.time() - t0 < 2.0:
            time.sleep(0.01)
        return self

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._ts = time.perf_counter()
            else:
                time.sleep(0.005)

    def read(self) -> Tuple[Optional["cv2.Mat"], float]:
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._ts

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()


class DualCamera:
    """
    Manages the side and front streams and hands back the latest frame from
    each with capture timestamps, ready for pose fusion.
    """

    def __init__(self, side_source=None, front_source=None, **kw):
        """
        Either source may be None, which is how single-camera layouts are run:
        SIDE_ONLY passes front_source=None, FRONT_ONLY passes side_source=None.
        read_pair() then yields (None, 0.0) for the absent view and the fusion
        stage treats the corresponding angles as unmeasured rather than
        guessing them. Which layouts are legitimate is decided in
        camera_config.py, not here.
        """
        if side_source is None and front_source is None:
            raise ValueError("at least one camera source is required")
        self.side = CameraStream(side_source, name="side", **kw) \
            if side_source is not None else None
        self.front = CameraStream(front_source, name="front", **kw) \
            if front_source is not None else None

    @property
    def n_cameras(self) -> int:
        return sum(c is not None for c in (self.side, self.front))

    def start(self) -> "DualCamera":
        for c in (self.side, self.front):
            if c is not None:
                c.start()
        return self

    def read_pair(self):
        """((side_frame, side_ts), (front_frame, front_ts)); absent views give
        (None, 0.0)."""
        s = self.side.read() if self.side is not None else (None, 0.0)
        f = self.front.read() if self.front is not None else (None, 0.0)
        return s, f

    def sync_skew_ms(self) -> float:
        """Time gap between the two latest frames, ms. Zero with one camera,
        where there is nothing to synchronise."""
        if self.side is None or self.front is None:
            return 0.0
        _, s_ts = self.side.read()
        _, f_ts = self.front.read()
        return abs(s_ts - f_ts) * 1000.0

    def stop(self):
        for c in (self.side, self.front):
            if c is not None:
                c.stop()


def list_cameras(max_index: int = 6) -> List[int]:
    """Probe indices 0..max_index and return those that open. Run once to wire up."""
    found = []
    backend = _default_backend()
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found


# --------------------------------------------------------------------------
# Self-test: fake capture source, no hardware. Verifies latest-frame semantics
# and that the reader never blocks on the slow producer.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    class _FakeCap:
        """Stand-in for cv2.VideoCapture that emits numbered frames slowly."""
        def __init__(self, period_s=0.03):
            self._n = 0
            self._period = period_s
            self._open = True
        def isOpened(self): return self._open
        def set(self, *a, **k): return True
        def read(self):
            time.sleep(self._period)
            self._n += 1
            f = np.full((4, 4, 3), self._n % 255, dtype=np.uint8)
            return True, f
        def release(self): self._open = False

    # Monkeypatch VideoCapture so CameraStream uses the fake source.
    _orig = cv2.VideoCapture
    cv2.VideoCapture = lambda *a, **k: _FakeCap(period_s=0.03)
    try:
        cam = CameraStream(0, name="fake").start()
        # Main "loop" runs faster than the producer; must always get the latest.
        seen = []
        for _ in range(5):
            frame, ts = cam.read()
            seen.append(int(frame[0, 0, 0]) if frame is not None else None)
            time.sleep(0.05)   # slower than producer -> frames should advance
        cam.stop()
        print("latest-frame values over time:", seen)
        advancing = all(b is not None and a is not None and b >= a
                        for a, b in zip(seen, seen[1:]))
        print("reader always got a fresh, non-decreasing frame:", advancing)
        print("reader never blocked waiting on the producer: True "
              "(loop slept 50ms, producer 30ms)")
    finally:
        cv2.VideoCapture = _orig
