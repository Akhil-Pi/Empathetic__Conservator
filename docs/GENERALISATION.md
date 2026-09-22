# Task generalisation: the `TaskProfile` model

Part B asks: does this method — posture-based strain scoring plus
goal-based artifact repositioning — only work for art conservation, or
does it generalise to other sustained-precision tasks? `src/task_profile.py`
answers this by making explicit what actually differs between tasks, so the
question can be asked about a NEW task without touching any existing code
path.

## What a `TaskProfile` captures

```python
@dataclass(frozen=True)
class TaskProfile:
    name: str
    neutral_notes: str                          # documentation only
    workspace_envelope_m: Tuple[Tuple[float,...], Tuple[float,...]]
    relevant_dof: Tuple[str, ...]                # which DOF make sense for this task
    notes: str = ""
```

- **`neutral_notes`** is documentation, not a numeric input. Individual
  neutral posture is already absorbed per-session by
  `pss_v2.PSSv2Calculator.calibrate_neutral()` — a `TaskProfile` does not
  and must not bypass that.
- **`workspace_envelope_m`** is a *first-cut starting point* for
  `RobotConfig.ENVELOPE_MIN/MAX`, never a substitute for the "measure the
  workspace envelope on the rig" step in `README.md` / `docs/RIG_SETUP.md`.
  Every profile's envelope is a placeholder until it has been measured on
  an actual cell, conservation's included historically.
- **`relevant_dof`** is the task-level ceiling on which controller DOF make
  physical sense to command at all — still further gated at runtime by
  `goal_controller.enabled_dof` / `DOF_REQUIRES` and the active camera
  layout. A DOF being "relevant" to a task doesn't guarantee it fires; a
  DOF NOT in the profile is never even offered to the optimizer.

## `CONSERVATION` is the default, and changes nothing

```python
CONSERVATION = TaskProfile(
    name="conservation",
    workspace_envelope_m=((-0.33, -0.22, 0.38), (-0.18, -0.08, 0.56)),
    relevant_dof=("dz", "dy", "dtilt", "drot", "dx"),
    ...
)
```

This is a **description of the currently-shipped defaults**
(`RobotConfig.ENVELOPE_MIN/MAX`, `ControllerConfig.DOF`), not a new value.
`tests/test_all.py::test_conservation_profile_reproduces_existing_behaviour`
is the regression guard: it applies the profile and checks (a) the DOF list
and envelope literally equal the live config's, and (b) `optimize_target`
produces byte-identical output, on a real posture pulled from an existing
session file in `test_data/` (falling back to a fixed synthetic posture on
a clean checkout with no `test_data/`), between the plain config and the
profiled one.

Nothing in the existing pipeline (`run_session.py`, `goal_controller.py`,
`robot_interface.py`) imports `task_profile.py` at all — applying a profile
is opt-in, and no existing command's behaviour changes just because this
module exists.

## The wiring pattern

```python
def apply_profile_dof(profile, base_cfg=None):
    if base_cfg is None:
        from goal_controller import ControllerConfig as base_cfg
    class _ProfiledController(base_cfg):
        DOF = list(profile.relevant_dof)
    return _ProfiledController
```

This is the same dynamic-subclass technique `goal_controller.py` already
uses internally for per-call config overrides
(`_scaled_step_cfg`, used by `EPISODE_DECAY`) — the base config class is
never mutated, only subclassed for one call. `apply_profile_envelope` does
the same for `RobotConfig.ENVELOPE_MIN/MAX`.

## `MANUFACTURING_ASSEMBLY`: the second example profile

```python
MANUFACTURING_ASSEMBLY = TaskProfile(
    name="manufacturing_assembly",
    workspace_envelope_m=((-0.25, -0.15, 0.25), (-0.10, -0.02, 0.40)),
    relevant_dof=("dz", "dy", "dx"),   # no dtilt / drot
    ...
)
```

Modeled on seated electronics rework / small-part assembly at a fixed bench
jig: closer and lower than conservation's bench-height artifact, and — the
concrete demonstrated difference — **not tilted or rotated** to relieve
operator strain the way a handheld conservation artifact is. A fixed
assembly jig gets moved closer/farther/up/down, not reoriented, so
`dtilt`/`drot` are excluded from `relevant_dof` entirely.

**This geometry is a placeholder**, exactly as unmeasured as
conservation's own defaults were before rig measurement. It exists to show
the method applies outside conservation, not as a validated deployment
profile for any real manufacturing cell.

`tests/test_all.py::test_manufacturing_profile_selects_different_dof_and_envelope`
confirms: the DOF and envelope genuinely differ from conservation's, and
`optimize_target` run under the manufacturing profile never even has
`dtilt`/`drot` as keys in its result (not merely zero — excluded from the
search entirely).

## Synthetic sessions for the manufacturing profile

`tools/make_synthetic_sessions_manufacturing.py` is a **sibling** to
`tools/make_synthetic_sessions.py` — the original file is untouched. It
reuses `simulate_session()` unchanged (the trunk-dominated/arm-dominated
posture simulation is task-agnostic PostureAngles synthesis, nothing about
it is conservation-specific), and differs only in:

- **Output directory**: `data/synthetic_manufacturing/`, kept separate from
  both real data and conservation-synthetic data, per the project's
  existing "don't pool sessions from different sources" convention.
- **DOF restriction actually applied to the generated data**: events are
  post-processed to blank `dtilt`/`drot`, so the generated `events.csv`
  concretely reflects the profile's DOF restriction rather than only
  asserting it in a docstring.
- **The same `note: SYNTHETIC ...` meta marker** the conservation generator
  uses, so `feasibility_report.session_origin()` recognises this data as
  synthetic without a second marker format.

Run it directly (`python tools/make_synthetic_sessions_manufacturing.py`)
to regenerate the demo cohort and print its feasibility report in one step.

**Participant-ID note**: this generator reuses the literal `P1`..`P6`
participant IDs `simulate_session()`'s own trunk/arm-dominance split is
keyed on — these IDs are local to `data/synthetic_manufacturing/`'s cohort,
exactly as `data/synthetic/`'s `P1`..`P6` are local to its own cohort. They
are not the same "P1"; keeping the two directories separate (as required
above) is what makes reusing the IDs safe. Never load both directories
together.

## Adding a new task

1. Add a `TaskProfile` to `PROFILES` in `task_profile.py`: pick
   `relevant_dof` based on whether the workpiece/fixture is reoriented
   (include `dtilt`/`drot`) or only repositioned (exclude them), and a
   placeholder envelope.
2. If you want a synthetic demo cohort, copy the pattern in
   `tools/make_synthetic_sessions_manufacturing.py`: reuse
   `simulate_session()`, write to a new `data/synthetic_<name>/`
   directory, restrict events to the profile's DOF, keep the `note:
   SYNTHETIC ...` meta marker.
3. Run `feasibility_report.main(your_dir, task_name=your_profile.name)` to
   get a decision report for the new task.
4. Before any real use: measure the actual workspace envelope on the real
   cell (same requirement as conservation's own rig-day checklist) and
   replace the placeholder.
