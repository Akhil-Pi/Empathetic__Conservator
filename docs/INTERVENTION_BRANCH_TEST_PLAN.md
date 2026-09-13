# Intervention Branch Test Plan: Head/Gaze Rotation Policy

This branch replaces the goal optimizer during intervention with a simple,
interpretable rig-test policy:

- Head or gaze turned to one side rotates only the fixture axis.
- Forward lean raises the fixture slowly.
- The branch intentionally avoids lateral translation, depth translation, and
  tilt during head-turn interventions.

## What Changed

- `neck_twist_deg` above `8 deg` (`HEAD_TURN_TRIGGER_DEG`) produces a `drot`
  command. Below that, `neck_sidebend_deg` (head tilt) can trigger it instead.
- **Signal selection (2026-09, DEBUG18 fix):** which signal supplies BOTH the
  sign and the magnitude is decided once, by `_head_turn_signal()`, gated on
  `HEAD_TWIST_NOISE_FLOOR_DEG = 2 deg` — deliberately smaller than the 8 deg
  action trigger. `neck_twist_deg` wins whenever it clears the 2 deg noise
  floor, even if it's still below the 8 deg trigger (in which case rotation
  simply does not fire this cycle — it does NOT fall back to
  `neck_sidebend_deg`'s magnitude while keeping twist's sign, which is what
  DEBUG17 event 7 did: twist=-5.57°, sidebend=+14.12°, sign and magnitude
  came from two different signals and disagreed).
- If there is no head-turn signal, trunk or neck flexion above `15 deg`
  produces a small positive `dz` command.
- `SEQUENTIAL_ACTIONS = True`: when both a rotate and a raise would fire in
  the same cycle, only the higher-priority one (`rotate`, by default) executes
  this cycle; the other waits for its own next trigger. So in practice almost
  every triggered event is a **pure** rotation or a **pure** raise, not both
  at once — see the logging note below, it matters for reading the CSV.
- Intervention event details include the active policy and signed
  `command_delta`.
- **Logging (DEBUG18 fix):** rotation and translation used to share one
  field (`robot.last_move`), so the second robot call of a cycle silently
  overwrote the first's result — a clamp/singularity refusal on the raise
  half made it impossible to tell whether the rotate half had already
  succeeded (DEBUG17 events 10-11). They are now two independent fields:
  `robot.last_move` (translation/tilt) and `robot.last_rotation` (rotation).
  `run_session.py` logs both under `applied=/ok=/clamped=/singularity=` (from
  `last_move`) and `rot_applied=/rot_ok=/rot_clamped=/rot_singularity=` (from
  `last_rotation`).
- **Stale-field fix:** whichever half did NOT run in a cycle is explicitly
  zeroed (`applied=[0,0,0,0]`, `ok=True`), never left holding a leftover
  value from a previous, unrelated event (DEBUG17 event 9, and — since
  `SEQUENTIAL_ACTIONS=True` makes single-action cycles the *common* case, not
  a corner case — this also covers every ordinary pure-rotation or pure-raise
  event, not just the "nothing happened at all" case).
- **`--simulate` dry runs now exercise this too:** `SimulatedRobot` did not
  override `adjust_rotation()` before, so it silently fell through to the
  base class's Cartesian version, which writes `last_move`, not
  `last_rotation` — meaning `rot_ok`/`rot_applied` were permanently
  blank/False in every dry run, the exact mode meant to sanity-check this
  logging before trusting it live. `SimulatedRobot` now has its own
  `adjust_rotation()` that populates `last_rotation` and caps cumulative
  rotation at `MAX_ROT_RAD` independently of tilt, mirroring `UR3Robot`'s
  contract. **Practical effect: step 1 below (`--simulate`) is now a real
  check of the new logging, not just of pose signs — do it before going
  live.**

Regression tests for all of the above live in `tests/test_all.py`:
`test_head_turn_signal_does_not_mix_signals`,
`test_head_turn_signal_falls_back_to_sidebend_below_noise_floor`,
`test_execute_does_not_leave_stale_data_in_untouched_half`,
`test_simulated_robot_rotation_updates_last_rotation`. Run
`python3 tests/test_all.py` and confirm these four pass (they do not need
cv2, pandas, or a robot) before touching the rig.

## Test Steps

1. Run a short simulated session first — this now genuinely validates the
   `rot_*` fields, not just poses:
   `PYTHONPATH=src python src/run_session.py --participant HEAD01 --condition experimental --simulate --minutes 1 --preview`
2. Run live only after the cell is clear:
   `PYTHONPATH=src python src/run_session.py --participant HEAD01 --condition experimental --live --minutes 1 --preview`
3. Hold these postures for at least three seconds each:
   neutral, head turned right, head turned left, head tilted right/left
   (small twist, standing straight — this exercises the noise-floor
   fallback), forward lean, and forward lean + turn/tilt together (the
   combined case).

## How To Analyze `events.csv`

For every `intervention` row, first decide which half fired: check
`command={...}` — `drot` non-zero and everything else zero is a pure
rotation; `dz` non-zero and `drot` zero is a pure raise.

- **Pure rotation events:** `rot_ok=True`, `rot_applied` non-zero and signed
  correctly; `applied=[0,0,0,0]` and `clamped=False` (the untouched linear
  half — if this is anything else, the stale-field fix regressed).
- **Pure raise events:** the mirror image — `applied` non-zero, `rot_applied`
  now correctly reads `[0,0,0,0]` (previously this was one of the fields left
  stale; now it should never be).
- **Combined events (both fired the same cycle — only possible if
  `SEQUENTIAL_ACTIONS` gets set to `False`, or in a future non-sequential
  policy):** check `rot_ok`/`rot_applied` and `ok`/`applied` independently. A
  refusal on one side (e.g. `singularity=radius`) must not by itself say
  anything about whether the other side succeeded — that's the whole point of
  the split. Confirm the two can disagree (one `ok=True`, the other
  `ok=False`) in at least one logged event before trusting the split.
- **Head-tilt-only trials (twist < 8°, sidebend ≥ 8°):** confirm `drot` fires
  with sidebend's sign, and that a *small* twist (2-8°, opposite-signed to
  sidebend) correctly suppresses the trigger rather than firing with a mixed
  sign — this was event 7 live.
- If the signed `drot` is correct but the fixture rotates the wrong physical
  way, flip `ControllerConfig.LATERAL_SIGN` or fix the robot-frame mapping.
- **Watch for raise starvation during combined trials.** `ACTION_PRIORITY`
  is a hard gate, re-checked every retrigger, not a one-time tie-break: as
  long as a posture keeps BOTH the twist/tilt trigger and the lean trigger
  true at once, rotate wins every single cycle and raise never fires — no
  decay, no turn-taking. Confirmed in simulation: 135s / 30 retriggers of
  sustained combined lean+twist produced 30 rotations and **zero** raises
  (see `test_sustained_combined_strain_can_starve_raise` in
  `tests/test_all.py`). This is the mirror image of the original bug this
  branch was created to fix (then: raise dominated, rotate never fired). If
  the combined trial (step 3 above) shows `dz` never firing while `drot`
  keeps firing every retrigger, this is why — it's a policy gap
  (`ACTION_PRIORITY`/`SEQUENTIAL_ACTIONS` in `ControllerConfig`), not a new
  logging bug, and needs a design decision (e.g. alternate priority per
  episode) rather than a quick patch.
- **Singularity refusals late in a session** (`singularity=radius` after
  several raises): expected if `dz` has accumulated toward the envelope edge
  — this is the safety clamp doing its job, not a fault. Don't spend time
  debugging it as if it were one; if it happens early or on the first
  intervention, that IS worth flagging.
