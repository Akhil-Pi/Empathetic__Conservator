"""
feasibility_report.py
======================
Part A: pre-procurement decision report.

Answers, for a cohort of sessions collected on one task: "what fraction of
observed physical strain can this class of automation (vision-based posture
sensing + goal-based artifact repositioning) plausibly remove, and does that
justify deploying it here versus a static fixture?"

Reuses evaluation.analyze_paradox() -- the EXISTING posture-mode clustering
and the REAL controller optimizer -- rather than reimplementing either. This
module only post-processes that function's output into a decision-report
shape: it does not change analyze_paradox, and running the existing
evaluation.main() pipeline is unaffected by anything here.

INTEGRITY GUARD (the whole point of this module existing as a separate,
careful layer rather than a one-off script):
  - Every number this module emits is labelled with its data origin (real
    vs synthetic) and its sample size (participants, sessions, high-strain
    frames).
  - A cohort that mixes real and synthetic sessions is REFUSED outright
    (see cohort_origin / MixedOriginError), never silently pooled.
  - The reducible/irreducible decomposition was historically demonstrated
    on synthetic sessions BY CONSTRUCTION (tools/make_synthetic_sessions.py
    deliberately builds trunk-dominated vs arm-dominated participants). When
    this module runs on REAL sessions, the same computation is a pilot-grade
    EMPIRICAL estimate, not a demonstration -- the N is always shown, and
    the caveats list always states which case applies.
  - No RULA integer appears anywhere in this module's output: it is built
    entirely on PSS_v2 bands (analyze_paradox's _BAND_COLS), never on
    rula.py, so the "no RULA integer before TABLES_VERIFIED" rule is
    satisfied by construction rather than by an extra gate that could be
    forgotten.

Origin labelling has to work without changing either existing session
writer (session_logger_v2.SessionLoggerV2 for real sessions,
tools/make_synthetic_sessions.py for synthetic ones): see session_origin().

Participant-ID note: evaluation.load_sessions() requires "P[0-9]+"-style
IDs, which excludes rig-test/pilot session files (e.g. DEBUG18, TEST01,
HEAD01) that exist on this project before a formal P##-numbered study
cohort does. load_sessions_lenient() below is a SEPARATE loader for this
module only -- evaluation.load_sessions and everything that depends on it
(all five existing analyses) is untouched.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from evaluation import Session, analyze_paradox, _parse_meta
from intervention_explainer import explain_intervention


# --------------------------------------------------------------------------
# Config: the decision-rule cutoffs. Named constants, not magic numbers.
# --------------------------------------------------------------------------

class FeasibilityConfig:
    # `addressable_fraction` = 1 - analyze_paradox()'s residual_unfixable_fraction:
    # the share of pooled high-strain TIME (by frame count, experimental
    # condition only) whose dominant RULA-style band is trunk or neck, i.e.
    # reachable by ControllerConfig.GAINS' DOF couplings (see
    # intervention_explainer._targeted_segments for the same fact derived
    # at single-intervention granularity).
    #
    # GO_THRESHOLD = 0.50: a plain majority-rule bar. If over half of
    # observed high-strain time belongs to a mechanism this class of
    # automation can address, adaptive repositioning is justified for this
    # task. This is a REASONED DEFAULT, the same epistemic status as
    # ControllerConfig.GAINS being "geometric estimates" before being fitted
    # from real intervention data (see evaluation.fit_response_gains) --
    # there is no prior deployment outcome data available to calibrate
    # this cutoff against. Revisit once procurement outcomes exist to check
    # it against.
    GO_THRESHOLD = 0.50

    # NOGO_THRESHOLD = 0.25: below one quarter of high-strain time being
    # addressable, even a PERFECTLY-executing adaptive system only ever
    # touches a small minority of the observed strain -- the deployment
    # complexity (cameras, a cobot, calibration, an ongoing ergonomics
    # program) is hard to justify against a static fixture for that
    # magnitude of return. Also a reasoned default, not empirically fitted.
    NOGO_THRESHOLD = 0.25

    assert 0.0 <= NOGO_THRESHOLD < GO_THRESHOLD <= 1.0

    OUTPUT_JSON_NAME = "feasibility_summary.json"
    OUTPUT_SUMMARY_NAME = "feasibility_summary.md"


def decide_verdict(addressable_fraction: float, cfg=FeasibilityConfig) -> str:
    """Pure decision rule, factored out so it is directly unit-testable at
    and around the two named cutoffs without needing a full session cohort.
    GO_THRESHOLD is inclusive (>=), NOGO_THRESHOLD is exclusive on its
    upper bound (<) -- so the two cutoffs partition [0, 1] with no gap and
    no overlap."""
    if addressable_fraction >= cfg.GO_THRESHOLD:
        return "GO"
    if addressable_fraction < cfg.NOGO_THRESHOLD:
        return "NO-GO"
    return "INCONCLUSIVE_AT_THIS_N"


# --------------------------------------------------------------------------
# Origin labelling and the mixed-cohort refusal.
# --------------------------------------------------------------------------

class MixedOriginError(RuntimeError):
    """Raised when a cohort directory contains both real and synthetic
    sessions. This is refused, not resolved automatically -- see the module
    docstring's INTEGRITY GUARD."""


def session_origin(meta: dict) -> str:
    """'synthetic' if the session's _meta.txt carries the EXISTING synthetic
    marker that tools/make_synthetic_sessions.py already writes (parsed by
    evaluation._parse_meta into meta['note']); 'real' otherwise. Neither
    existing session writer is changed to make this determination -- real
    sessions (session_logger_v2.SessionLoggerV2) simply never produce a
    matching note, so they fall through to 'real' by default."""
    note = str(meta.get("note", "") or "")
    return "synthetic" if "SYNTHETIC" in note.upper() else "real"


def cohort_origin(sessions: list) -> str:
    """Returns 'real' or 'synthetic' for a whole cohort, or raises
    MixedOriginError if sessions disagree. This is the INTEGRITY GUARD:
    every number downstream is labelled with whichever origin this
    returns, and nothing downstream runs at all if the cohort is mixed."""
    if not sessions:
        return "unknown"
    origins = {session_origin(s.meta) for s in sessions}
    if len(origins) > 1:
        raise MixedOriginError(
            f"session cohort mixes origins {sorted(origins)}; refusing to "
            f"compute a feasibility report that would implicitly pool real "
            f"and synthetic data. Split the input directory by origin (they "
            f"are already kept in separate directories by project "
            f"convention -- see README's working agreements) and run this "
            f"report separately for each.")
    return origins.pop()


# --------------------------------------------------------------------------
# Lenient session loader (this module only; does not touch evaluation.py).
# --------------------------------------------------------------------------

_FNAME_RE = re.compile(r"^([A-Za-z0-9]+)_(control|experimental)_")


def load_sessions_lenient(sessions_dir: str) -> list:
    """Same file-pairing logic as evaluation.load_sessions, with a
    permissive participant-ID pattern (any alnum token) instead of
    "P[0-9]+", so rig-test/pilot sessions (DEBUG18, TEST01, HEAD01, ...)
    load. Participant IDs are used verbatim (not evaluation._canon_participant,
    which assumes a numeric-only ID and would collide e.g. TEST01 and
    HEAD01 both to "P1") -- they only need to be distinct pooling keys
    within one cohort, not globally canonical."""
    sessions = []
    for fpath in sorted(glob.glob(os.path.join(sessions_dir, "*_frames.csv"))):
        base = fpath[: -len("_frames.csv")]
        fname = os.path.basename(base)
        m = _FNAME_RE.match(fname)
        if not m:
            continue
        participant, condition = m.group(1), m.group(2)
        frames = pd.read_csv(fpath)
        epath = base + "_events.csv"
        events = pd.read_csv(epath) if os.path.exists(epath) else pd.DataFrame()
        mpath = base + "_meta.txt"
        meta = _parse_meta(mpath) if os.path.exists(mpath) else {}
        sessions.append(Session(participant, condition, frames, events, meta))
    return sessions


# --------------------------------------------------------------------------
# Uncertainty: a simple, honestly-caveated Wilson interval.
# --------------------------------------------------------------------------

def _wilson_interval(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n. Deliberately simple
    (no library dependency beyond numpy) -- appropriate to "a simple
    uncertainty indication", not a rigorous account of within-session frame
    non-independence. See the caveats list for that limitation stated
    explicitly."""
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


# --------------------------------------------------------------------------
# Per-session explainability summary (Part C integration into Part A).
# --------------------------------------------------------------------------

def _explainability_summary(sessions: list) -> dict:
    """Of every logged intervention across the cohort, how many targeted
    which strain component, and how many were clamped or refused. Built
    entirely from events already on disk via intervention_explainer --
    nothing here re-runs a session."""
    n_total = 0
    targeted_counts: dict = {}
    refused = 0
    clamped = 0
    for s in sessions:
        if s.events is None or s.events.empty or "event_type" not in s.events:
            continue
        ev = s.events[s.events["event_type"] == "intervention"]
        for _, row in ev.iterrows():
            n_total += 1
            ex = explain_intervention(row.to_dict())
            for seg in ex["targeted_strain"]:
                targeted_counts[seg] = targeted_counts.get(seg, 0) + 1
            if ex["refused"]:
                refused += 1
            if ex["clamped"]:
                clamped += 1
    return {"n_interventions": n_total, "targeted_counts": targeted_counts,
            "refused": refused, "clamped": clamped}


# --------------------------------------------------------------------------
# Caveats always attached to a report.
# --------------------------------------------------------------------------

def _standard_caveats(origin: str, n_participants: int) -> list:
    out = []
    if origin == "synthetic":
        out.append(
            "Computed on SYNTHETIC session data (pipeline demonstration/"
            "validation only, generated by tools/make_synthetic_sessions.py "
            "or a sibling generator). This is NOT evidence about any real "
            "task, population, or deployment -- it demonstrates that the "
            "method runs and produces the expected shape of result on data "
            "built to have a known decomposition.")
    else:
        out.append(
            f"Computed on REAL session data. N={n_participants} "
            f"participant(s) -- a PILOT-GRADE empirical estimate, not a "
            f"validated deployment forecast. Treat directionally, not as a "
            f"precise number, until a larger cohort exists.")
    out.append(
        "Wrist, wrist-twist, and legs are not measured by this pipeline "
        "(see LEARNINGS.md, 'What v2 still cannot do'). The automated "
        "strain score therefore UNDER-estimates true strain, so this "
        "report's addressable-fraction estimate is conservative (more "
        "likely to understate than overstate how much strain a deployment "
        "could remove).")
    out.append(
        "No sham/placebo control exists in this study design. The belief "
        "that an adaptive system is helping can itself shift subjective "
        "(and some objective) measures independent of the physical "
        "mechanism -- this report only speaks to whether the POSTURAL "
        "mechanism is present in the data, not the full causal benefit of "
        "deploying the system.")
    out.append(
        "This decomposition is a TIME share (fraction of high-strain "
        "frames), not a magnitude-weighted share of total strain -- it "
        "answers 'what fraction of high-strain time falls in an "
        "addressable posture mode', not 'what fraction of cumulative "
        "strain magnitude'.")
    out.append(
        "The 95% interval is an approximate Wilson interval over "
        "high-strain FRAMES, not participants or sessions; frames within a "
        "session are not independent samples, so treat this interval as "
        "indicative only, especially at small participant N.")
    return out


# --------------------------------------------------------------------------
# Core computation.
# --------------------------------------------------------------------------

def compute_feasibility(sessions_dir: str, cfg=FeasibilityConfig,
                        task_name: str = "this task") -> dict:
    sessions = load_sessions_lenient(sessions_dir)
    if not sessions:
        return {"error": f"no usable sessions found in {sessions_dir}",
                "sessions_dir": sessions_dir}

    origin = cohort_origin(sessions)   # raises MixedOriginError on a mix

    exp_sessions = [s for s in sessions if s.condition == "experimental"]
    n_participants = len({s.participant for s in exp_sessions})
    n_sessions = len(exp_sessions)

    par = analyze_paradox(sessions)    # THE EXISTING function -- not reimplemented
    if "error" in par:
        return {
            "error": par["error"], "sessions_dir": sessions_dir,
            "task_name": task_name, "origin": origin,
            "n_participants": n_participants, "n_sessions": n_sessions,
        }

    residual = par["residual_unfixable_fraction"]
    addressable_fraction = 1.0 - residual
    n_highstrain = par["n_highstrain_frames"]
    k_addressable = int(round(addressable_fraction * n_highstrain))
    ci_lo, ci_hi = _wilson_interval(k_addressable, n_highstrain)

    verdict = decide_verdict(addressable_fraction, cfg)

    posture_modes = [{
        "cluster": c["cluster"],
        "dominant_segment": c["dominant_segment"],
        "reducible_by_repositioning": c["reducible_by_repositioning"],
        "frac_of_highstrain_time": c["frac_of_highstrain"],
        "mean_pss": c["mean_pss"],
        "achievable_pss_reduction": c["achievable_pss_reduction"],
    } for c in par["clusters"]]

    return {
        "sessions_dir": sessions_dir,
        "task_name": task_name,
        "origin": origin,
        "n_participants": n_participants,
        "n_sessions": n_sessions,
        "n_highstrain_frames": n_highstrain,
        "k_clusters_selected": par["k_selected"],
        "silhouette": par["silhouette"],
        "addressable_fraction": addressable_fraction,
        "addressable_fraction_ci95_wilson": [ci_lo, ci_hi],
        "irreducible_fraction": residual,
        "posture_modes": posture_modes,
        "verdict": verdict,
        "decision_thresholds": {"go_at_or_above": cfg.GO_THRESHOLD,
                                "nogo_below": cfg.NOGO_THRESHOLD},
        "explainability": _explainability_summary(sessions),
        "caveats": _standard_caveats(origin, n_participants),
    }


# --------------------------------------------------------------------------
# Output artefacts.
# --------------------------------------------------------------------------

def _json_clean(o):
    if isinstance(o, dict):
        return {k: _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def render_json(result: dict, out_path: str) -> str:
    with open(out_path, "w") as f:
        json.dump(_json_clean(result), f, indent=2)
    return out_path


_VERDICT_TEXT = {
    "GO": "Adaptive repositioning is justified for this task.",
    "NO-GO": "A static fixture likely suffices -- adaptive repositioning is "
             "not worth deploying for this task.",
    "INCONCLUSIVE_AT_THIS_N": "The data collected so far is inconclusive. "
                              "Collect more sessions before making a "
                              "deployment call.",
}


def render_plain_language(result: dict, out_path: str) -> str:
    """One-page, non-jargon summary for a procurement/operations reader.
    Short sentences. No RULA integer appears here under any circumstance --
    this module never imports rula.py."""
    if "error" in result:
        text = (f"# Feasibility summary -- INCOMPLETE\n\n"
                f"Could not produce a report: {result['error']}\n")
        with open(out_path, "w") as f:
            f.write(text)
        return text

    lo, hi = result["addressable_fraction_ci95_wilson"]
    lines = [
        f"# Pre-procurement feasibility summary -- {result['task_name']}",
        "",
        f"**Data**: {result['origin'].upper()} sessions. "
        f"{result['n_participants']} participant(s), "
        f"{result['n_sessions']} session(s), "
        f"{result['n_highstrain_frames']} high-strain frames analysed.",
        "",
        "## Bottom line",
        "",
        f"Of the time this task spent under elevated physical strain, about "
        f"**{result['addressable_fraction']*100:.0f}%** "
        f"(approximate range {lo*100:.0f}-{hi*100:.0f}%) was strain that "
        f"this class of robotic repositioning system can plausibly reduce.",
        "",
        f"The remaining **{result['irreducible_fraction']*100:.0f}%** comes "
        f"from arm posture the task itself requires. Moving the workpiece "
        f"does not change how the arm is held, so no repositioning system "
        f"-- this one or any other -- can remove that part.",
        "",
        f"## Recommendation: {result['verdict']}",
        "",
        _VERDICT_TEXT[result["verdict"]],
        "",
        f"(Decision rule: GO at or above "
        f"{result['decision_thresholds']['go_at_or_above']*100:.0f}% "
        f"addressable strain-time; NO-GO below "
        f"{result['decision_thresholds']['nogo_below']*100:.0f}%; "
        f"inconclusive in between, or simply "
        f"\"needs more data\" at very small N.)",
        "",
        "## What was observed",
        "",
    ]
    for m in result["posture_modes"]:
        tag = "ADDRESSABLE" if m["reducible_by_repositioning"] else "NOT addressable"
        seg_label = m["dominant_segment"].replace("_", " ")
        line = (f"- {m['frac_of_highstrain_time']*100:.0f}% of high-strain time: "
               f"dominated by {seg_label} strain -- **{tag}**")
        if m["reducible_by_repositioning"]:
            line += (f" (a repositioning move could reduce its strain score "
                     f"by about {m['achievable_pss_reduction']:.2f}, on a "
                     f"0-1 scale)")
        lines.append(line)

    ex = result["explainability"]
    lines += ["", "## How the system explains itself", ""]
    if ex["n_interventions"] == 0:
        lines.append("No interventions were logged in this cohort.")
    else:
        tgt_txt = ", ".join(f"{v} targeted {k}" for k, v in ex["targeted_counts"].items()) or "none"
        lines.append(
            f"Of {ex['n_interventions']} logged interventions in this "
            f"cohort: {tgt_txt}. {ex['refused']} were refused (workspace "
            f"envelope or a named singularity), {ex['clamped']} were "
            f"clamped. Every intervention has a per-event, plain-language "
            f"explanation available (see docs/DECISION_REPORT.md and "
            f"intervention_explainer.render_session_explanations).")

    lines += ["", "## Caveats -- read before acting on this", ""]
    for c in result["caveats"]:
        lines.append(f"- {c}")

    lines += [
        "",
        "_No RULA integer appears in this report unless RULA Tables A/B "
        "have been formally verified against a printed worksheet (see "
        "docs/RULA_VERIFICATION.md); this report is built entirely from "
        "the PSS_v2 strain score, not RULA scoring._",
    ]

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    return out_path


# --------------------------------------------------------------------------
# CLI entry point.
# --------------------------------------------------------------------------

def main(sessions_dir: str, out_dir: str = "eval_out", task_name: str = "this task") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    result = compute_feasibility(sessions_dir, task_name=task_name)
    json_path = render_json(result, os.path.join(out_dir, FeasibilityConfig.OUTPUT_JSON_NAME))
    md_path = render_plain_language(result, os.path.join(out_dir, FeasibilityConfig.OUTPUT_SUMMARY_NAME))
    result["_json_path"] = json_path
    result["_summary_path"] = md_path
    return result


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "data/sessions"
    out = sys.argv[2] if len(sys.argv) > 2 else "eval_out"
    task = sys.argv[3] if len(sys.argv) > 3 else "this task"
    try:
        res = main(d, out, task)
    except MixedOriginError as e:
        print(f"REFUSED: {e}")
        sys.exit(1)
    if "error" in res:
        print(f"error: {res['error']}")
        sys.exit(1)
    print(open(res["_summary_path"]).read())
