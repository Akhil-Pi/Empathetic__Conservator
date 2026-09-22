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
- Trunk or neck flexion above `FORWARD_LEAN_TRIGGER_DEG` produces a small
  positive `dz` command, independent of the rotate signal.
  **Lowered from 15 deg to 10 deg (2026-09-14, live tests DEBUG20-22):**
  `forward_signal` almost never reached 15 deg at the exact instant a cycle
  fired (closest miss was 14.28 deg), and no more than one continuous 2s
  dwell above 15 deg occurred across three full live sessions, so raise fired
  on only 2 of 40 triggered events. 10 deg is the current value; if you
  change it again, re-run the same instant-vs-threshold check against a
  fresh `events.csv` rather than assuming it worked.
- **`SEQUENTIAL_ACTIONS = False` (2026-09-14, live tests DEBUG23/24) — changed
  from `True`.** With `True`, when both a rotate and a raise condition were
  met in the same cycle, only the higher-priority one (`rotate`) executed;
  the other's delta was zeroed OUTRIGHT, not delayed. Live data showed this
  discarding a genuine, independently-satisfied raise condition often, not
  rarely: of 23 rotate-only events across DEBUG23/24, 11 (48%) had
  `forward_signal` already above `FORWARD_LEAN_TRIGGER_DEG` at that same
  instant (one as high as 35.2 deg of trunk flexion) but got thrown away
  because a small head-turn/tilt happened at the same time — a natural
  combined posture, not an edge case. This also violated the original spec:
  both actions should trigger when both conditions are present. With `False`,
  whenever both conditions are met in the same cycle, BOTH `delta['drot']`
  and `delta['dz']` stay non-zero and `_execute()` calls both
  `robot.adjust_rotation()` and `robot.move_relative()` — rotate still goes
  first, ordered by `ACTION_PRIORITY`, which keeps its original role
  (execution order) instead of acting as a veto. Practical effect on the rig:
  a combined-strain intervention is now usually **two** distinct robot
  motions back to back (rotate then raise) in one cycle, not one. This is
  more motion to watch/read per event but is what "both should trigger"
  actually requires. `SEQUENTIAL_ACTIONS=True` is still supported (not
  deleted) if a future rig session wants the single-clean-motion behavior
  back — see `test_sequential_actions_opt_in_still_hands_off_correctly`.
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
  value from a previous, unrelated event (DEBUG17 event 9). This still
  matters with `SEQUENTIAL_ACTIONS=False`: a rotate-only or raise-only event
  (only one condition met this cycle) is still common, it's just no longer
  the *only* outcome when both conditions are met — see below.
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
`test_simulated_robot_rotation_updates_last_rotation`,
`test_both_actions_fire_together_when_both_conditions_met`,
`test_sequential_actions_opt_in_still_hands_off_correctly`. Run
`python3 tests/test_all.py` and confirm these six pass (they do not need
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
- **Combined events (both `drot` and `dz` non-zero in `command={...}`) are
  now the EXPECTED outcome whenever a person's posture genuinely has both a
  head-turn/tilt AND a forward-lean above their triggers at once** — this is
  the normal case now, not a rare one, since `SEQUENTIAL_ACTIONS=False`.
  Check `rot_ok`/`rot_applied` and `ok`/`applied` independently. A refusal on
  one side (e.g. `singularity=radius`) must not by itself say anything about
  whether the other side succeeded — that's the whole point of the split.
  Confirm the two can disagree (one `ok=True`, the other `ok=False`) in at
  least one logged event before trusting the split. Also confirm the physical
  order on the rig matches the log: rotate should visibly start first, raise
  right after (same cycle, not the next trigger) — per `ACTION_PRIORITY`.
- **Head-tilt-only trials (twist < 8°, sidebend ≥ 8°):** confirm `drot` fires
  with sidebend's sign, and that a *small* twist (2-8°, opposite-signed to
  sidebend) correctly suppresses the trigger rather than firing with a mixed
  sign — this was event 7 live.
- If the signed `drot` is correct but the fixture rotates the wrong physical
  way, flip `ControllerConfig.LATERAL_SIGN` or fix the robot-frame mapping.
- **If you see a rotate-only event where the logged `angles.trunk_flexion_deg`
  or `neck_flexion_deg` is already above `FORWARD_LEAN_TRIGGER_DEG` (10°),
  that's a regression** — with `SEQUENTIAL_ACTIONS=False` this should no
  longer happen; `dz` should have fired alongside `drot` in that same event.
  This exact pattern (rotate wins, `dz`'s own condition was independently
  met and got thrown away) is what DEBUG23/24 showed under the old
  `SEQUENTIAL_ACTIONS=True` default 48% of the time — it's the thing this
  change was made to fix, so specifically check for it in the new
  `events.csv`.
- **Singularity refusals late in a session** (`singularity=radius` after
  several raises): expected if `dz` has accumulated toward the envelope edge
  — this is the safety clamp doing its job, not a fault. Don't spend time
  debugging it as if it were one; if it happens early or on the first
  intervention, that IS worth flagging.
