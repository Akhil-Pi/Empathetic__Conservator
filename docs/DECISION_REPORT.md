# Pre-procurement decision report

`src/feasibility_report.py` turns a cohort of sessions into a go/no-go
recommendation: **for this task, what fraction of observed physical strain
can vision-based posture sensing + goal-based artifact repositioning
plausibly remove, and does that justify deploying it over a static
fixture?**

It does not run a new analysis. It calls the EXISTING
`evaluation.analyze_paradox()` — the same posture-mode clustering and the
same real controller optimizer (`goal_controller.optimize_target`) the five
existing analyses already use — and reshapes that function's output into a
decision-report shape. Nothing about `evaluation.py`'s five analyses
changes by this module existing.

---

## What it claims, precisely

`analyze_paradox()` clusters high-strain frames (pooled across the
experimental-condition cohort) into posture modes by their RULA-style band
profile, and for each mode asks the real optimizer: "if the controller
commanded its single best move for the average posture in this mode, how
much would PSS_v2 drop?" A mode is **addressable** if its dominant band is
trunk or neck (the only angles `ControllerConfig.GAINS` couples any DOF
to); it is **not addressable** if arm posture dominates, since no DOF the
controller commands has any effect on arm angles at all — that's not a
tuning gap, it's a fact about the DOF/GAINS structure.

`addressable_fraction` is `1 - residual_unfixable_fraction`: the **share of
high-strain TIME** (by frame count, not by strain magnitude) that falls in
an addressable posture mode. It is a time share, not a magnitude-weighted
share — the report says this explicitly in its caveats every time.

## The cutoff, and why it's a config constant, not a magic number

```python
class FeasibilityConfig:
    GO_THRESHOLD = 0.50     # majority rule: over half addressable -> justified
    NOGO_THRESHOLD = 0.25   # under a quarter -> even perfect execution barely helps
```

Both are **reasoned defaults**, the same epistemic status
`ControllerConfig.GAINS` had before anyone fit it from real intervention
data (see `evaluation.fit_response_gains`). There is no prior deployment
outcome data to calibrate a cutoff against yet. `decide_verdict()` in
`feasibility_report.py` is the one place this logic lives — see
`tests/test_all.py::test_feasibility_decision_rule_at_cutoffs` for the
exact boundary behavior (GO_THRESHOLD is inclusive, the band between the
two cutoffs is `INCONCLUSIVE_AT_THIS_N`).

Revisit these numbers once real procurement outcomes exist to check them
against. Until then, treat a report sitting close to either cutoff as
"probably needs more data," not as a precise line.

## The two artefacts

1. **`feasibility_summary.json`** — machine-readable: every number above,
   plus `n_participants`, `n_sessions`, `n_highstrain_frames`, the Wilson
   interval, per-posture-mode breakdown, and the explainability summary.
2. **`feasibility_summary.md`** — the one-page plain-language summary,
   written for a procurement/operations reader, not an academic. Short
   sentences. No RULA integer ever appears in it — the module never imports
   `rula.py` at all, so this holds regardless of `rula.TABLES_VERIFIED`.

## The caveats, always printed

Every report — real or synthetic, GO or NO-GO — always states:

- **Data origin and N**, explicitly. Synthetic data is labelled as
  "pipeline demonstration only, NOT evidence about any real task."
- **Wrist/wrist-twist/legs are not measured** (see LEARNINGS.md), so the
  addressable-fraction estimate is conservative — more likely to
  understate than overstate what a deployment could remove.
- **No sham/placebo control** exists in this study design; the report only
  speaks to whether the postural mechanism is present, not the full causal
  benefit of deploying an adaptive system.
- **Time share, not magnitude share** — restated inline, see above.
- **The 95% interval is an approximate Wilson interval over high-strain
  FRAMES**, not participants or sessions. Frames within a session are not
  independent, so this is explicitly labelled "indicative only, especially
  at small participant N" — a simple uncertainty indication appropriate to
  a pilot sample size, not a rigorous mixed-effects account of
  within-session correlation.

## The integrity guard: real vs synthetic, and refusing to mix

Neither existing session writer (`session_logger_v2.SessionLoggerV2` for
real sessions, `tools/make_synthetic_sessions.py` for synthetic ones) was
changed to support this. `feasibility_report.session_origin()` detects
synthetic sessions via the marker `tools/make_synthetic_sessions.py`
**already** writes (`note: SYNTHETIC data for pipeline validation only` in
`_meta.txt`); a real session simply never has that marker, so it falls
through to `"real"` by default.

`cohort_origin()` requires every session in a directory to agree, and
raises `MixedOriginError` — refuses outright, does not attempt to split or
warn-and-continue — if a directory mixes real and synthetic sessions. See
`tests/test_all.py::test_feasibility_refuses_mixed_origin`.

## Rig-test / pilot data is not a formal study cohort

`evaluation.load_sessions()` requires `P[0-9]+`-style participant IDs.
Rig-test files that predate a formal numbered cohort (`DEBUG18`, `TEST01`,
`HEAD01`, ...) don't match that and are silently skipped by it.
`feasibility_report.load_sessions_lenient()` is a **separate** loader, used
only by this module, that accepts any alphanumeric participant token — it
does not change `evaluation.load_sessions` or anything that depends on it.

When you run this report against `test_data/`, the summary correctly says
`REAL sessions` (it genuinely is real human movement) — but treat that as
**pilot/rig-test data, not a validated study cohort**, and say so out loud
whenever you show the number to someone who wasn't in the room for how it
was collected.

## Per-session explainability integration

Every report includes a short section built from
`intervention_explainer.py` (Part C): of N logged interventions in the
cohort, how many targeted which strain component, how many were refused
(envelope or a named singularity), how many were clamped — all derived
from events already on disk, no re-run. See `docs/GENERALISATION.md`'s
sibling doc note, or just `intervention_explainer.py`'s own docstring, for
how the per-event explanation itself works.

## Running it

```bash
python src/feasibility_report.py <sessions_dir> <out_dir> "<task name>"
```

Or programmatically:

```python
from feasibility_report import main
result = main("test_data", "eval_out", task_name="art conservation bench work")
print(open(result["_summary_path"]).read())
```
