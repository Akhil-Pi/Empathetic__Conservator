"""
make_synthetic_sessions_manufacturing.py
=========================================
Sibling to make_synthetic_sessions.py (NOT a modification of it -- that
file is untouched) for task_profile.MANUFACTURING_ASSEMBLY, Part B of the
generalisation work.

Reuses make_synthetic_sessions.simulate_session() unchanged: the
trunk-dominated-vs-arm-dominated posture simulation is task-agnostic
PostureAngles synthesis, not conservation-specific, so there is nothing
manufacturing-specific to change about the strain math itself.

What IS specific to this profile:
  - Output directory: data/synthetic_manufacturing/, kept separate from
    real data and from conservation-synthetic data (data/synthetic/), per
    the project's existing "do not pool sessions from different sources"
    convention (README working agreements).
  - Generated events are restricted to
    task_profile.MANUFACTURING_ASSEMBLY.relevant_dof (dz, dy, dx only --
    no dtilt/drot), so the generated data concretely reflects the
    profile's DOF restriction rather than only asserting it in a docstring.
  - Participant IDs are P1..P6, reused from make_synthetic_sessions.py's
    own TRUNK_DOMINATED/ARM_DOMINATED split (which is keyed on those
    literal names). These IDs are LOCAL to this directory's cohort, same as
    the conservation-synthetic generator's P1..P6 are local to
    data/synthetic/ -- they are not a claim that "P1" here is the same
    person as "P1" in data/synthetic/. Keeping the two corpora in separate
    directories (as required above) is what makes this safe; the two
    should never be loaded together.
  - _meta.txt keeps the SAME "note: SYNTHETIC ..." marker convention the
    conservation generator already uses, so
    feasibility_report.session_origin() recognises this data as synthetic
    without needing a second marker format.

SYNTHETIC data for pipeline demonstration only. No real manufacturing site
or participant.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_SRC = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_SRC, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import csv
import os

from make_synthetic_sessions import simulate_session
from session_logger_v2 import FRAME_COLUMNS, EVENT_COLUMNS, PIPELINE_VERSION
from task_profile import MANUFACTURING_ASSEMBLY

PARTICIPANTS = [f"P{i}" for i in range(1, 7)]   # keyed by simulate_session's own dominance split


def _restrict_events_to_profile(events, relevant_dof):
    """Zeroes out (blanks) any DOF column not in relevant_dof, so the
    generated events.csv concretely reflects the profile's DOF restriction
    (no dtilt/drot for manufacturing_assembly) rather than only documenting
    it. Non-destructive: returns new dicts, does not mutate the input."""
    allowed = set(relevant_dof)
    out = []
    for e in events:
        e = dict(e)
        for dof in ("dz", "dy", "dtilt", "drot", "dx"):
            if dof not in allowed and e.get(dof):
                e[dof] = ""
        out.append(e)
    return out


def write_manufacturing_session(out_dir, participant, condition, frames, events, seed):
    """Same file shape as make_synthetic_sessions.write_session, with an
    added task_profile meta line. Kept as a separate function (rather than
    reusing write_session) only so that extra line can be added without
    touching the existing writer."""
    ts = f"20260922_{seed:06d}"
    base = os.path.join(out_dir, f"{participant}_{condition}_{ts}")
    with open(base + "_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_COLUMNS); w.writeheader(); w.writerows(frames)
    with open(base + "_events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_COLUMNS); w.writeheader(); w.writerows(events)
    with open(base + "_meta.txt", "w") as f:
        f.write(f"pipeline_version: {PIPELINE_VERSION}\n")
        f.write(f"participant_id: {participant}\ncondition: {condition}\n")
        f.write(f"frames_logged: {len(frames)}\nevents_logged: {len(events)}\n")
        f.write(f"task_profile: {MANUFACTURING_ASSEMBLY.name}\n")
        f.write("note: SYNTHETIC data for pipeline validation only "
               f"(task_profile={MANUFACTURING_ASSEMBLY.name})\n")


def generate(out_dir="data/synthetic_manufacturing"):
    os.makedirs(out_dir, exist_ok=True)
    seed = 500
    for p in PARTICIPANTS:
        for cond in ("control", "experimental"):
            seed += 1
            frames, events = simulate_session(p, cond, seed)
            events = _restrict_events_to_profile(events, MANUFACTURING_ASSEMBLY.relevant_dof)
            write_manufacturing_session(out_dir, p, cond, frames, events, seed)
    return out_dir


if __name__ == "__main__":
    out = generate()
    print(f"generated manufacturing-profile synthetic sessions in {out}\n")
    from feasibility_report import main as feasibility_main
    res = feasibility_main(out, out_dir="eval_out_manufacturing",
                           task_name="manufacturing_assembly (synthetic demo)")
    print(open(res["_summary_path"]).read())
