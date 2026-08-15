"""
make_synthetic_sessions.py
==========================
Generates logger-v2-shaped sessions so the Part 5 pipeline runs end to end today.
The real August files (same schema) drop straight in and replace these.

The synthetic cohort is deliberately mixed so the analyses have something real to find:
  - trunk-dominated participants: strain is driven by trunk/neck flexion, which
    repositioning the artifact CAN relieve. Interventions recover well, so their
    experimental PSS drops below control.
  - arm-dominated participants: strain is driven by arm posture the task requires,
    which repositioning CANNOT relieve. Interventions fire but PSS barely falls.
    This is the residual that produces a null H1 across the pooled cohort and the
    "ergonomic paradox" the clustering is meant to explain.

Recovery time is coupled to intervention latency on purpose (longer latency ->
longer recovery) so the latency-vs-recovery analysis recovers a positive Spearman.

This is SYNTHETIC data for pipeline validation only. No real participants.
"""

from __future__ import annotations

import os as _os
import sys as _sys
_SRC = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

import csv
import os
import random
import numpy as np

from pss_v2 import PSSv2Calculator, PostureAngles
from session_logger_v2 import FRAME_COLUMNS, EVENT_COLUMNS, PIPELINE_VERSION
from goal_controller import ControllerConfig

DT = 0.1
DURATION_S = 120.0
THRESHOLD = ControllerConfig.THRESHOLD                          # 0.25
RECOVERY_TARGET = ControllerConfig.THRESHOLD - ControllerConfig.HYSTERESIS  # 0.17

TRUNK_DOMINATED = {"P1", "P2", "P3"}
ARM_DOMINATED = {"P4", "P5", "P6"}


def _frame_row(t, ang, comp, arm="LEFT", skew=6.0):
    return {
        "timestamp_s": f"{t:.3f}",
        "trunk_flexion_deg": f"{ang.trunk_flexion_deg:.2f}",
        "neck_flexion_deg": f"{ang.neck_flexion_deg:.2f}",
        "upper_arm_flexion_deg": f"{ang.upper_arm_flexion_deg:.2f}",
        "lower_arm_flexion_deg": f"{ang.lower_arm_flexion_deg:.2f}",
        "trunk_sidebend_deg": f"{ang.trunk_sidebend_deg:.2f}",
        "neck_sidebend_deg": f"{ang.neck_sidebend_deg:.2f}",
        "trunk_twist_deg": f"{ang.trunk_twist_deg:.2f}",
        "neck_twist_deg": f"{ang.neck_twist_deg:.2f}",
        "lateral_gaze_deg": "0.00",
        "conf_trunk": "0.80", "conf_neck": "0.75", "conf_arm": "0.65",
        "trunk_band": f"{comp['trunk_band']:.3f}",
        "neck_band": f"{comp['neck_band']:.3f}",
        "upper_arm_band": f"{comp['upper_arm_band']:.3f}",
        "lower_arm_band": f"{comp['lower_arm_band']:.3f}",
        "group_A": f"{comp['group_A']:.4f}",
        "group_B": f"{comp['group_B']:.4f}",
        "pss_raw": f"{comp['pss_raw']:.4f}",
        "pss_smooth": f"{comp['pss_smooth']:.4f}",
        "arm_side": arm, "skew_ms": f"{skew:.1f}",
    }


def _event_row(t, etype, pss, iid="", delta=None, pb=None, pa=None,
               moved="", latency=None, details=""):
    delta = delta or {}
    def d(k):
        return f"{delta[k]:.4f}" if k in delta else ""
    return {
        "timestamp_s": f"{t:.3f}", "event_type": etype, "pss_at_event": f"{pss:.4f}",
        "intervention_id": iid,
        "dz": d("dz"), "dy": d("dy"), "dtilt": d("dtilt"), "drot": d("drot"), "dx": d("dx"),
        "predicted_pss_before": f"{pb:.4f}" if pb is not None else "",
        "predicted_pss_after": f"{pa:.4f}" if pa is not None else "",
        "moved": moved,
        "latency_s": f"{latency:.4f}" if latency is not None else "",
        "details": details,
    }


def simulate_session(participant, condition, seed):
    rng = random.Random(seed)
    npr = np.random.RandomState(seed)
    pss_calc = PSSv2Calculator()
    trunk_dom = participant in TRUNK_DOMINATED

    frames, events = [], []
    n = int(DURATION_S / DT)

    trunk, drift = 14.0 + npr.randn() * 2, 0.0
    state = "building"          # building -> recovering -> hold -> building
    recovered_at, hold_until = None, None
    cooldown_until, above_since = 0.0, None
    intervention_i = 0

    for i in range(n):
        t = i * DT

        if trunk_dom:
            if state == "building":
                drift += 0.09
                trunk_target = 16.0 + drift + 5 * np.sin(t * 0.3)
            elif state == "recovering":       # strain persists until recovered_at
                trunk_target = 36.0 + 5 * np.sin(t * 0.5)
                if recovered_at is not None and t >= recovered_at:
                    state, hold_until = "hold", t + 2.5
            else:                              # hold: low posture, PSS recovers
                trunk_target = 4.5
                if t >= hold_until:
                    state, drift = "building", 0.0
            trunk += (trunk_target - trunk) * 0.2 + npr.randn() * 0.5
            trunk = float(np.clip(trunk, 3, 72))
            neck = float(np.clip(0.45 * trunk + npr.randn() * 1.2, 2, 40))
            upper, lower = 26 + npr.randn() * 2.5, 86 + npr.randn() * 2.5
            sideb = abs(npr.randn() * 3) + (0.5 if state == "hold" else 2)
        else:
            # arm-dominated: constant, task-required arm strain the cobot cannot relieve
            trunk = float(np.clip(8 + npr.randn() * 2, 3, 20))
            neck = float(np.clip(4 + npr.randn() * 1.5, 2, 15))
            upper, lower = 83 + npr.randn() * 3, 46 + npr.randn() * 3
            sideb = abs(npr.randn() * 3) + 3

        ang = PostureAngles(trunk_flexion_deg=trunk, neck_flexion_deg=neck,
                            upper_arm_flexion_deg=upper, lower_arm_flexion_deg=lower,
                            trunk_sidebend_deg=sideb, neck_sidebend_deg=sideb * 0.5,
                            trunk_twist_deg=abs(npr.randn() * 4))
        comp = pss_calc.compute(ang, now=t)
        frames.append(_frame_row(t, ang, comp))

        if condition != "experimental":
            continue
        pss = comp["pss_raw"]
        can_fire = (state == "building") if trunk_dom else True
        if pss > THRESHOLD and t >= cooldown_until and can_fire:
            if above_since is None:
                above_since = t
            elif (t - above_since) >= 0.75:
                latency = rng.uniform(0.6, 1.3)
                intervention_i += 1
                if trunk_dom:
                    desired_rec = 1.0 + 2.6 * latency + npr.randn() * 0.25
                    recovered_at, state = t + max(0.6, desired_rec), "recovering"
                    pa = pss - 0.20
                else:
                    pa = pss - 0.03           # arm strain: negligible predicted gain
                delta = {"dz": 0.06 + 0.02 * latency, "dy": 0.03, "dtilt": 0.15,
                         "drot": 0.0, "dx": -0.03}
                events.append(_event_row(
                    t + latency, "intervention", pss, iid=f"{participant}_{intervention_i}",
                    delta=delta, pb=pss, pa=pa, moved="True", latency=latency))
                cooldown_until = t + latency + 1.5
                above_since = None
        elif pss <= THRESHOLD:
            above_since = None

    return frames, events


def write_session(out_dir, participant, condition, frames, events, seed):
    ts = f"20260805_{seed:06d}"
    base = os.path.join(out_dir, f"{participant}_{condition}_{ts}")
    with open(base + "_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_COLUMNS); w.writeheader(); w.writerows(frames)
    with open(base + "_events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_COLUMNS); w.writeheader(); w.writerows(events)
    with open(base + "_meta.txt", "w") as f:
        f.write(f"pipeline_version: {PIPELINE_VERSION}\n")
        f.write(f"participant_id: {participant}\ncondition: {condition}\n")
        f.write(f"frames_logged: {len(frames)}\nevents_logged: {len(events)}\n")
        f.write("note: SYNTHETIC data for pipeline validation only\n")


def generate(out_dir="data/synthetic"):
    os.makedirs(out_dir, exist_ok=True)
    participants = [f"P{i}" for i in range(1, 7)]
    expert_rows = []
    pss_calc_dummy = PSSv2Calculator()
    from rula import rula_from_angles, RulaAssumptions
    from evaluation import _row_to_angles
    import pandas as pd

    seed = 100
    for p in participants:
        for cond in ("control", "experimental"):
            seed += 1
            frames, events = simulate_session(p, cond, seed)
            write_session(out_dir, p, cond, frames, events, seed)
            # synthetic expert RULA: auto p90 grand + small integer noise
            grands = []
            for fr in frames:
                ang = _row_to_angles({k: float(v) for k, v in fr.items()
                                      if k.endswith("_deg")})
                grands.append(rula_from_angles(ang, RulaAssumptions())["grand"])
            auto_p90 = int(round(np.percentile(grands, 90)))
            expert = int(np.clip(auto_p90 + np.random.RandomState(seed).randint(-1, 2), 1, 7))
            expert_rows.append({"participant": p, "condition": cond,
                                "expert_rula": expert})
    pd.DataFrame(expert_rows).to_csv(os.path.join(out_dir, "expert_rula.csv"), index=False)
    return out_dir


if __name__ == "__main__":
    out = generate()
    print(f"generated synthetic sessions in {out}\n")
    from evaluation import main
    res = main(out, expert_csv=os.path.join(out, "expert_rula.csv"), out_dir="eval_out")
    print(res["report"])
