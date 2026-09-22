"""
intervention_explainer.py
==========================
Part C: per-intervention explainability.

Turns one logged intervention event into a structured, human-readable
explanation: which signal triggered it, which DOF moved and by how much,
predicted PSS before/after, whether the move was clamped or refused (and
why), and which strain component it targeted versus which it structurally
cannot touch.

Works entirely from an EXISTING events.csv row -- no re-run, no camera, no
robot, no MediaPipe. run_session.py's `details` field (see its log_event
calls) already carries everything needed as a semi-structured string
("key=value key=value ..." where some values are Python dict/list reprs);
_parse_details() below recovers it by locating known key boundaries rather
than naively splitting on whitespace, since values like
`command={'dx': -0.0, 'dy': 0.0, ...}` contain spaces.

Synthetic sessions (tools/make_synthetic_sessions.py) log an empty
`details` field -- this module degrades honestly in that case: fields it
cannot recover are reported as "not recorded", never guessed. This mirrors
the project's standing rule (see pose_fusion.apply_layout_mask) that an
unmeasured quantity is reported as absent, not invented.

DOF -> strain-component targeting is derived from ControllerConfig.GAINS
itself (not a separate hard-coded table), so it can never drift out of sync
with what the controller actually does. Arm segments (upper_arm, lower_arm)
never appear in GAINS' coupling targets, which is exactly why they are
"structurally untouchable" here -- same fact evaluation.analyze_paradox's
_REDUCIBLE_SEGMENTS encodes for posture-mode clustering; this module derives
it independently, from the DOF actually moved in ONE event, rather than
importing evaluation's cluster-level constant, since the granularity differs
(one intervention vs. a posture mode).
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional

from goal_controller import ControllerConfig


# --------------------------------------------------------------------------
# details= string parsing
# --------------------------------------------------------------------------

# Fixed key order run_session.py's log_event(...) f-string always uses for
# an intervention event. Order matters here: we locate each key's position
# and slice the value as "everything up to the next known key", which is
# robust to values containing spaces/commas (dict and list reprs) without
# needing a full recursive-descent parser for them.
_DETAIL_KEYS = ["clamped", "applied", "ok", "singularity",
               "rot_applied", "rot_ok", "rot_clamped", "rot_singularity",
               "policy", "command", "angles"]


def _parse_details(details: str) -> Dict[str, str]:
    """Best-effort split of a details= string into {key: raw_value_str}.
    Returns {} for anything that doesn't look like this format (empty
    string, a synthetic session's blank details, a hand-written note) --
    callers must treat a missing key as "not recorded", not as False/None."""
    if not details or not isinstance(details, str):
        return {}
    positions = []
    for key in _DETAIL_KEYS:
        m = re.search(rf"(?<![\w.]){re.escape(key)}=", details)
        if m:
            positions.append((m.start(), key, m.end()))
    if not positions:
        return {}
    positions.sort()
    out = {}
    for i, (_, key, val_start) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(details)
        out[key] = details[val_start:end].strip()
    return out


def _safe_literal(raw: Optional[str]) -> Any:
    """Parse a Python-repr value (dict/list/bool/None/float) recovered by
    _parse_details, without executing arbitrary code (ast.literal_eval only
    -- these strings come from this project's own logger, but there is no
    reason to use eval() for it). Falls back to the raw string, or None."""
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        f = float(v)
        return None if f != f else f   # NaN -> None
    except (TypeError, ValueError):
        return None


def _to_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return None
    s = str(v).strip().lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    return None


# --------------------------------------------------------------------------
# DOF -> strain-component targeting, derived from ControllerConfig.GAINS.
# --------------------------------------------------------------------------

def _dof_to_angle_names(cfg=ControllerConfig) -> Dict[str, List[str]]:
    """Invert cfg.GAINS (angle_name -> {dof: gain}) into dof -> [angle_name, ...]."""
    out: Dict[str, List[str]] = {}
    for angle_name, couplings in cfg.GAINS.items():
        for dof in couplings:
            out.setdefault(dof, []).append(angle_name)
    return out


_SEGMENT_OF_ANGLE = {
    "trunk_flexion_deg": "trunk", "trunk_sidebend_deg": "trunk", "trunk_twist_deg": "trunk",
    "neck_flexion_deg": "neck", "neck_sidebend_deg": "neck", "neck_twist_deg": "neck",
}
_ALL_SEGMENTS_STRAIN_CAN_ACT_ON = {"trunk", "neck"}   # matches GAINS: no arm coupling exists
_UNREACHABLE_SEGMENTS = {"upper_arm", "lower_arm"}    # never in GAINS; structurally untouchable


def _targeted_segments(dof_moved: Dict[str, float], cfg=ControllerConfig):
    """Returns (targeted_segments: sorted list, untouched_segments: sorted list)
    for a set of nonzero DOF deltas from one intervention."""
    dof_map = _dof_to_angle_names(cfg)
    targeted = set()
    for dof in dof_moved:
        for angle_name in dof_map.get(dof, []):
            seg = _SEGMENT_OF_ANGLE.get(angle_name)
            if seg:
                targeted.add(seg)
    untouched = (_ALL_SEGMENTS_STRAIN_CAN_ACT_ON - targeted) | _UNREACHABLE_SEGMENTS
    return sorted(targeted), sorted(untouched)


# --------------------------------------------------------------------------
# Per-event explanation
# --------------------------------------------------------------------------

_DOF_COLS = ("dz", "dy", "dtilt", "drot", "dx")


def explain_intervention(event: Dict[str, Any], cfg=ControllerConfig) -> Dict[str, Any]:
    """
    `event`: one events.csv row as a plain dict (e.g. from csv.DictReader,
    or pandas_row.to_dict()). Works on an intervention event from ANY
    session already on disk -- no re-run.

    Returns a structured dict (every field also feeds the natural-language
    `explanation` string) plus `_details_parsed` (bool), an honesty flag: if
    False, clamped/singularity/policy/targeting-detail fields could not be
    recovered from this event (typically a synthetic session, which logs an
    empty `details` field) and are reported as unknown, not guessed.
    """
    parsed_raw = _parse_details(str(event.get("details", "") or ""))
    parsed = {k: _safe_literal(v) for k, v in parsed_raw.items()}

    pss_at_event = _to_float(event.get("pss_at_event"))
    pss_before = _to_float(event.get("predicted_pss_before"))
    pss_after = _to_float(event.get("predicted_pss_after"))

    dof_moved = {}
    for dof in _DOF_COLS:
        v = _to_float(event.get(dof))
        if v is not None and abs(v) > 1e-4:
            dof_moved[dof] = v

    targeted, untouched = _targeted_segments(dof_moved, cfg)

    clamped = bool(parsed.get("clamped")) or bool(parsed.get("rot_clamped"))
    ok = parsed.get("ok")
    rot_ok = parsed.get("rot_ok")
    singularity = parsed.get("singularity")
    rot_singularity = parsed.get("rot_singularity")

    refused = False
    refusal_reason = None
    if singularity:
        refused = True
        refusal_reason = f"refused near a {singularity} singularity"
    elif rot_singularity:
        refused = True
        refusal_reason = f"rotation refused near a {rot_singularity} singularity"
    elif ok is False or rot_ok is False:
        refused = True
        refusal_reason = "move refused (reason not a detected singularity; see raw details)"

    policy = parsed.get("policy")
    angles = parsed.get("angles") if isinstance(parsed.get("angles"), dict) else None
    command = parsed.get("command") if isinstance(parsed.get("command"), dict) else None

    result = {
        "pss_at_event": pss_at_event,
        "policy": policy,
        "measured_angles": angles,
        "dof_moved": dof_moved,
        "command_delta": command,
        "predicted_pss_before": pss_before,
        "predicted_pss_after": pss_after,
        "moved": _to_bool(event.get("moved")),
        "clamped": clamped,
        "refused": refused,
        "refusal_reason": refusal_reason,
        "targeted_strain": targeted,
        "untouched_strain": untouched,
        "_details_parsed": bool(parsed_raw),
    }
    result["explanation"] = _render_text(result)
    return result


def _render_text(r: Dict[str, Any]) -> str:
    parts = []

    trig = f"pss={r['pss_at_event']:.3f}" if r["pss_at_event"] is not None else "pss=unknown"
    if r["policy"]:
        trig += f" (policy: {r['policy']})"
    parts.append(f"Triggered at {trig}.")

    if r["dof_moved"]:
        dof_txt = ", ".join(f"{k}={v:+.4f}" for k, v in r["dof_moved"].items())
        parts.append(f"Commanded: {dof_txt}.")
    else:
        parts.append("No DOF commanded a nonzero move this event.")

    if r["predicted_pss_before"] is not None and r["predicted_pss_after"] is not None:
        parts.append(f"Predicted PSS {r['predicted_pss_before']:.3f} -> "
                     f"{r['predicted_pss_after']:.3f}.")

    if r["refused"]:
        parts.append(f"The move was {r['refusal_reason']} and did not execute.")
    elif r["clamped"]:
        parts.append("The move executed but was clamped by the workspace/orientation envelope.")
    elif r["moved"] is False:
        parts.append("The move did not execute (moved=False); reason not recorded in details.")
    elif r["_details_parsed"]:
        parts.append("The move executed without clamping and without a singularity refusal.")

    if r["targeted_strain"]:
        parts.append(f"This targeted {', '.join(r['targeted_strain'])} strain.")
    parts.append(
        "It did not, and structurally cannot, address arm-load strain "
        "(upper_arm/lower_arm posture) -- no controller DOF is coupled to "
        "those angles (see ControllerConfig.GAINS)."
    )

    if not r["_details_parsed"]:
        parts.append(
            "Note: this event's `details` field was empty or unrecognised "
            "(typical of a synthetic session), so policy/clamped/singularity "
            "status above could not be recovered and are not reported.")

    return " ".join(parts)


# --------------------------------------------------------------------------
# Whole-session rendering from an existing events.csv, no re-run.
# --------------------------------------------------------------------------

def render_session_explanations(events_csv_path: str, out_path: Optional[str] = None) -> str:
    """Reads an existing *_events.csv, explains every `intervention` row in
    order, and returns (optionally also writes) a Markdown trace. Pure
    post-hoc read -- no camera, robot, or re-simulation required, so this
    works on sessions collected long before this module existed."""
    import csv

    lines = [f"# Intervention explanation trace\n", f"source: `{events_csv_path}`\n"]
    n = 0
    with open(events_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("event_type") != "intervention":
                continue
            n += 1
            ex = explain_intervention(row)
            iid = row.get("intervention_id") or n
            ts = row.get("timestamp_s", "?")
            lines.append(f"## Intervention {iid} (t={ts}s)\n")
            lines.append(ex["explanation"] + "\n")
    lines.insert(2, f"interventions explained: {n}\n")
    text = "\n".join(lines)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
    return text


# --------------------------------------------------------------------------
# Self-test: crafted event rows, no camera/robot/CSV file required to test
# the core parsing+explanation logic (render_session_explanations itself is
# exercised against a real file in tests/test_all.py using a tempfile).
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("intervention_explainer.py self-test (crafted event rows)\n")

    clean = {
        "pss_at_event": "0.534", "predicted_pss_before": "0.534",
        "predicted_pss_after": "0.412", "moved": "True",
        "dz": "0.0", "dy": "0.0", "dtilt": "0.0", "drot": "0.15", "dx": "0.0",
        "details": ("clamped=False applied=[0.0, 0.0, 0.0, 0.0] ok=True "
                    "singularity=None rot_applied=[0.0, 0.0, 0.0, 0.15] "
                    "rot_ok=True rot_clamped=False rot_singularity=None "
                    "policy=head_gaze_rotation command={'dx': -0.0, 'dy': 0.0, "
                    "'dz': 0.0, 'dtilt': 0.0, 'drot': 0.15} "
                    "angles={'trunk_flexion_deg': 5.0, 'neck_flexion_deg': 6.0, "
                    "'neck_twist_deg': 22.0}"),
    }
    ex = explain_intervention(clean)
    print("clean rotation event:")
    print(f"  targeted={ex['targeted_strain']} untouched={ex['untouched_strain']} "
          f"clamped={ex['clamped']} refused={ex['refused']}")
    print(f"  {ex['explanation']}\n")
    # drot couples to neck_sidebend/neck_twist AND trunk_twist (see
    # ControllerConfig.GAINS), so a pure rotation targets both segments.
    assert ex["targeted_strain"] == ["neck", "trunk"]
    assert "upper_arm" in ex["untouched_strain"] and "lower_arm" in ex["untouched_strain"]
    assert ex["clamped"] is False and ex["refused"] is False

    singular = dict(clean)
    singular["dz"] = "0.0"; singular["drot"] = "0.0"
    singular["dy"] = "0.06"
    singular["details"] = ("clamped=True applied=[0.0, 0.0, 0.0, 0.0] ok=False "
                           "singularity=shoulder rot_applied=[0.0, 0.0, 0.0, 0.0] "
                           "rot_ok=True rot_clamped=False rot_singularity=None "
                           "policy=head_gaze_rotation command={'dy': 0.06} "
                           "angles={'trunk_flexion_deg': 50.0}")
    ex2 = explain_intervention(singular)
    print("singularity-refusal event:")
    print(f"  refused={ex2['refused']} reason={ex2['refusal_reason']}")
    print(f"  {ex2['explanation']}\n")
    assert ex2["refused"] is True and "shoulder" in ex2["refusal_reason"]

    synthetic = {
        "pss_at_event": "0.55", "predicted_pss_before": "0.55",
        "predicted_pss_after": "0.35", "moved": "True",
        "dz": "0.08", "dy": "0.03", "dtilt": "0.15", "drot": "0.0", "dx": "-0.03",
        "details": "",
    }
    ex3 = explain_intervention(synthetic)
    print("synthetic (empty details) event:")
    print(f"  details_parsed={ex3['_details_parsed']} targeted={ex3['targeted_strain']}")
    print(f"  {ex3['explanation']}\n")
    assert ex3["_details_parsed"] is False
    assert set(ex3["targeted_strain"]) == {"trunk", "neck"}   # from dz/dy/dtilt alone, no details needed

    print("all intervention_explainer checks passed: True")
