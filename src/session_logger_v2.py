"""
session_logger_v2.py
====================
Part 4: session logging, rebuilt.

The old logger recorded derived scores but not the inputs. Worst case: gaze drove
55 percent of the old PSS yet was never written to disk, so the dominant signal
could not be reconstructed or validated after the fact. It also could not capture
what the goal-based controller now produces (a per-DOF target and a predicted PSS
before and after each move).

Logger v2 records three files per session:

  <base>_frames.csv   every fused input angle, per-axis confidence, every PSS_v2
                      band and sub-score, and raw+smoothed PSS. Nothing derived
                      is stored without the inputs that produced it.
  <base>_events.csv   full intervention records: per-DOF delta, predicted PSS
                      before and after, whether the move executed, latency.
  <base>_meta.txt     participant, timing, counts, the calibration offsets, and
                      a complete snapshot of every config value that governed the
                      run (PSS weights and knots, controller gains and thresholds,
                      fusion axis flags). A run is fully reproducible from this.

Recovery time (for the latency-vs-recovery analysis) is intentionally NOT logged
as a field. It is derived in Part 5 from the intervention timestamps in events
and the continuous pss_smooth series in frames, so there is a single source of
truth rather than a second, possibly-inconsistent, recorded number.

This module runs without a camera or robot; the self-test writes synthetic data
and reads it back to prove every field round-trips.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from typing import Optional, Any


PIPELINE_VERSION = "empathetic-conservator/v2 (PSS_v2 + fusion + goal-controller)"


def capture_config_snapshot() -> dict:
    """
    Dump the public attributes of the three config classes so the exact
    parameters of a run are recorded. Imported lazily and defensively so the
    logger still works if a module is absent during offline testing.
    """
    snap: dict = {}

    def dump(label, cls):
        if cls is None:
            return
        for k in dir(cls):
            if k.startswith("_"):
                continue
            v = getattr(cls, k)
            if callable(v):
                continue
            snap[f"{label}.{k}"] = v

    try:
        from pss_v2 import PSSv2Config
        dump("pss", PSSv2Config)
    except Exception:
        pass
    try:
        from goal_controller import ControllerConfig
        dump("ctrl", ControllerConfig)
    except Exception:
        pass
    try:
        from pose_fusion import FusionConfig
        dump("fusion", FusionConfig)
    except Exception:
        pass
    return snap


FRAME_COLUMNS = [
    "timestamp_s",
    "trunk_flexion_deg", "neck_flexion_deg",
    "upper_arm_flexion_deg", "lower_arm_flexion_deg",
    "trunk_sidebend_deg", "neck_sidebend_deg",
    "trunk_twist_deg", "neck_twist_deg", "lateral_gaze_deg",
    "conf_trunk", "conf_neck", "conf_arm",
    "trunk_band", "neck_band", "upper_arm_band", "lower_arm_band",
    "group_A", "group_B",
    "pss_raw", "pss_smooth",
    "arm_side", "skew_ms",
]

EVENT_COLUMNS = [
    "timestamp_s", "event_type", "pss_at_event", "intervention_id",
    "dz", "dy", "dtilt", "drot", "dx",
    "predicted_pss_before", "predicted_pss_after", "moved",
    "latency_s", "details",
]


class SessionLoggerV2:
    def __init__(self, participant_id: str, condition: str,
                 log_dir: str = "data/sessions", log_frequency_hz: float = 10.0):
        assert condition in ("control", "experimental")
        self.participant_id = participant_id
        self.condition = condition
        self.log_dir = log_dir
        self.log_frequency_hz = log_frequency_hz
        os.makedirs(self.log_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{participant_id}_{condition}_{ts}"
        self.frames_path = os.path.join(log_dir, f"{base}_frames.csv")
        self.events_path = os.path.join(log_dir, f"{base}_events.csv")
        self.meta_path = os.path.join(log_dir, f"{base}_meta.txt")

        self._ff = open(self.frames_path, "w", newline="")
        self._ef = open(self.events_path, "w", newline="")
        self._fw = csv.writer(self._ff)
        self._ew = csv.writer(self._ef)
        self._fw.writerow(FRAME_COLUMNS)
        self._ew.writerow(EVENT_COLUMNS)

        self.start_time = time.time()
        self._frames_logged = 0
        self._events_logged = 0
        self._last_frame_t = 0.0
        self._last_flush_t = time.time()
        self.flush_every_s = 5.0   # crash on the rig loses at most this much
        self._calibration_offsets: Optional[dict] = None

    # ---- calibration audit ----------------------------------------------

    def log_calibration(self, neutral_offsets: dict) -> None:
        """Record the per-angle calibration offsets so calibration is auditable."""
        self._calibration_offsets = dict(neutral_offsets)

    # ---- per-frame -------------------------------------------------------

    def log_frame(self, angles: Any, components: dict,
                  arm_side: str = "", skew_ms: Optional[float] = None) -> bool:
        """Throttled per-frame log. `angles` is a PostureAngles, `components`
        the dict returned by PSSv2Calculator.compute(). Returns True if a row was
        written, False if this call was throttled."""
        now = time.time()
        if (now - self._last_frame_t) < (1.0 / self.log_frequency_hz):
            return False
        self._last_frame_t = now

        conf = getattr(angles, "confidence", None) or {}

        def g(obj, name, default=0.0):
            return getattr(obj, name, default)

        def ang(name):
            # Log the CALIBRATED angle the PSS was computed from (compute()
            # echoes it in components). Fall back to the raw angles object for
            # fields compute() does not echo (sidebend/twist pass through
            # calibration unchanged, so raw == calibrated for those).
            return components.get(name, g(angles, name))

        self._fw.writerow([
            f"{now - self.start_time:.3f}",
            f"{ang('trunk_flexion_deg'):.2f}",
            f"{ang('neck_flexion_deg'):.2f}",
            f"{ang('upper_arm_flexion_deg'):.2f}",
            f"{ang('lower_arm_flexion_deg'):.2f}",
            f"{g(angles, 'trunk_sidebend_deg'):.2f}",
            f"{g(angles, 'neck_sidebend_deg'):.2f}",
            f"{g(angles, 'trunk_twist_deg'):.2f}",
            f"{g(angles, 'neck_twist_deg'):.2f}",
            f"{ang('lateral_gaze_deg'):.2f}",
            f"{conf.get('trunk_flexion_deg', ''):.3f}" if 'trunk_flexion_deg' in conf else "",
            f"{conf.get('neck_flexion_deg', ''):.3f}" if 'neck_flexion_deg' in conf else "",
            f"{conf.get('upper_arm_flexion_deg', ''):.3f}" if 'upper_arm_flexion_deg' in conf else "",
            f"{components.get('trunk_band', ''):.3f}" if 'trunk_band' in components else "",
            f"{components.get('neck_band', ''):.3f}" if 'neck_band' in components else "",
            f"{components.get('upper_arm_band', ''):.3f}" if 'upper_arm_band' in components else "",
            f"{components.get('lower_arm_band', ''):.3f}" if 'lower_arm_band' in components else "",
            f"{components.get('group_A', ''):.4f}" if 'group_A' in components else "",
            f"{components.get('group_B', ''):.4f}" if 'group_B' in components else "",
            f"{components.get('pss_raw', 0.0):.4f}",
            f"{components.get('pss_smooth', 0.0):.4f}",
            arm_side,
            f"{skew_ms:.1f}" if skew_ms is not None else "",
        ])
        self._frames_logged += 1
        if (now - self._last_flush_t) >= self.flush_every_s:
            self._ff.flush()
            self._last_flush_t = now
        return True

    # ---- events ----------------------------------------------------------

    def log_event(self, event_type: str, pss_at_event: float,
                  result: Optional[dict] = None, details: str = "") -> None:
        """
        Log a discrete event. If `result` is a goal-controller evaluate() dict,
        its delta and predicted PSS are unpacked into their own columns.
        """
        now = time.time()
        delta = (result or {}).get("delta", {}) or {}

        def d(key):
            v = delta.get(key)
            return f"{v:.4f}" if v is not None else ""

        self._ew.writerow([
            f"{now - self.start_time:.3f}",
            event_type,
            f"{pss_at_event:.4f}",
            (result or {}).get("intervention_id", ""),
            d("dz"), d("dy"), d("dtilt"), d("drot"), d("dx"),
            _fmt((result or {}).get("predicted_pss_before")),
            _fmt((result or {}).get("predicted_pss_after")),
            (result or {}).get("moved", ""),
            _fmt((result or {}).get("latency_s")),
            details,
        ])
        self._events_logged += 1
        self._ef.flush()

    # ---- close -----------------------------------------------------------

    def close(self, notes: str = "") -> None:
        duration = time.time() - self.start_time
        self._ff.close()
        self._ef.close()

        with open(self.meta_path, "w") as f:
            f.write(f"pipeline_version: {PIPELINE_VERSION}\n")
            f.write(f"participant_id: {self.participant_id}\n")
            f.write(f"condition: {self.condition}\n")
            f.write(f"start_time: {datetime.fromtimestamp(self.start_time).isoformat()}\n")
            f.write(f"duration_s: {duration:.1f}\n")
            f.write(f"frames_logged: {self._frames_logged}\n")
            f.write(f"events_logged: {self._events_logged}\n")
            f.write(f"log_frequency_hz: {self.log_frequency_hz}\n")

            f.write("\n[calibration_offsets]\n")
            if self._calibration_offsets:
                for k, v in self._calibration_offsets.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write("none recorded\n")

            f.write("\n[config_snapshot]\n")
            for k, v in sorted(capture_config_snapshot().items()):
                f.write(f"{k}: {v}\n")

            if notes:
                f.write(f"\n[notes]\n{notes}\n")

        print(f"[LOG v2] saved:\n  {self.frames_path}\n  {self.events_path}\n  {self.meta_path}")


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else ("" if v is None else str(v))


# --------------------------------------------------------------------------
# Self-test: write a synthetic session, read it back, prove reconstructability.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pss_v2 import PSSv2Calculator, PostureAngles
    from goal_controller import GoalBasedController, _FakeRobot

    tmp = tempfile.mkdtemp()
    pss = PSSv2Calculator()
    ctrl = GoalBasedController("experimental", pss)
    robot = _FakeRobot()
    log = SessionLoggerV2("P07", "experimental", log_dir=tmp, log_frequency_hz=1000)

    pss.calibrate_neutral([PostureAngles(trunk_flexion_deg=2, neck_flexion_deg=3)])
    log.log_calibration(pss._neutral)
    log.log_event("at_desired_pose", 0.0)

    t = 1000.0
    logged_pss = []
    for i in range(12):
        strain = 5 + i * 5   # ramp strain up
        ang = PostureAngles(trunk_flexion_deg=strain, neck_flexion_deg=strain * 0.5,
                            trunk_sidebend_deg=8.0, upper_arm_flexion_deg=30,
                            lower_arm_flexion_deg=88,
                            confidence={"trunk_flexion_deg": 0.8, "neck_flexion_deg": 0.7,
                                        "upper_arm_flexion_deg": 0.6})
        comp = pss.compute(ang)
        wrote = log.log_frame(ang, comp, arm_side="LEFT", skew_ms=6.2)
        if wrote:
            logged_pss.append(round(comp["pss_raw"], 4))
        r = ctrl.evaluate({"pss_smooth": comp["pss_raw"]}, ang, robot, now=t)
        if r["triggered"]:
            log.log_event("intervention", comp["pss_smooth"], result=r)
        t += 1.0
    log.close(notes="synthetic round-trip test")

    # ---- read back and verify ----
    import csv as _csv
    with open(log.frames_path) as f:
        rows = list(_csv.DictReader(f))
    with open(log.events_path) as f:
        events = list(_csv.DictReader(f))

    print("\n--- reconstructability checks ---")
    print(f"frame columns present: {len(rows[0].keys())} "
          f"(expected {len(FRAME_COLUMNS)})")
    read_pss = [round(float(r["pss_raw"]), 4) for r in rows]
    print(f"pss_raw round-trips exactly: {read_pss == logged_pss}")
    print(f"every frame has trunk_flexion + confidence: "
          f"{all(r['trunk_flexion_deg'] and r['conf_trunk'] for r in rows)}")
    interventions = [e for e in events if e["event_type"] == "intervention"]
    print(f"interventions logged: {len(interventions)}")
    if interventions:
        e = interventions[-1]
        print(f"  sample intervention has delta + predicted PSS: "
              f"dz={e['dz']} predicted {e['predicted_pss_before']}->{e['predicted_pss_after']} "
              f"latency={e['latency_s']}")
    meta = open(log.meta_path).read()
    print(f"meta records controller gains: {'ctrl.GAINS' in meta}")
    print(f"meta records PSS weights: {'pss.W_GROUP_B' in meta}")
    print(f"meta records calibration offsets: {'[calibration_offsets]' in meta and 'trunk_flexion_deg' in meta}")
