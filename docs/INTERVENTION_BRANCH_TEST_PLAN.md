# Intervention Branch Test Plan: Head/Gaze Rotation Policy

This branch replaces the goal optimizer during intervention with a simple,
interpretable rig-test policy:

- Head or gaze turned to one side rotates only the fixture axis.
- Forward lean raises the fixture slowly.
- The branch intentionally avoids lateral translation, depth translation, and
  tilt during head-turn interventions.

## What Changed

- `lateral_gaze_deg` or `neck_twist_deg` above `8 deg` produces a `drot` command.
- The command sign is chosen from the signed head/gaze/twist signal.
- If there is no head-turn signal, trunk or neck flexion above `15 deg` produces
  a small positive `dz` command.
- Intervention event details include the active policy and signed
  `command_delta`.

## Test Steps

1. Run a short simulated session first:
   `PYTHONPATH=src python src/run_session.py --participant HEAD01 --condition experimental --simulate --minutes 1 --preview`
2. Run live only after the cell is clear:
   `PYTHONPATH=src python src/run_session.py --participant HEAD01 --condition experimental --live --minutes 1 --preview`
3. Hold these postures for at least three seconds each:
   neutral, head turned right, head turned left, forward lean.
4. For head-turn trials, watch that the fixture rotates without lateral
   translation or tilt.
5. For forward-lean trials, watch that the fixture raises slowly.

## How To Analyze Results

Open the generated `*_events.csv`:

- For head-right and head-left trials, `details` should show
  `policy=head_gaze_rotation` and a non-zero signed `command_delta['drot']`.
- During head-turn trials, `command_delta['dx']`, `command_delta['dy']`,
  `command_delta['dz']`, and `command_delta['dtilt']` should stay at zero.
- During forward-lean trials, `command_delta['dz']` should be positive and
  `command_delta['drot']` should be zero.
- If the signed `drot` is correct but the fixture rotates the wrong physical
  way, flip `ControllerConfig.LATERAL_SIGN` or fix the robot-frame mapping.
- Ignore trials with `clamped=True` when judging direction.
