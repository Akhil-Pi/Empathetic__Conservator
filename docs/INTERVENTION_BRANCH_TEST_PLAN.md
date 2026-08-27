# Intervention Branch Test Plan: Calibrated Frame Mapping

This branch keeps the goal-based optimizer but adds an explicit mapping from
the controller's human/artifact frame to the UR base frame.

The controller command is interpreted as:

- `dx`: toward the person's left
- `dy`: toward the person's front
- `dz`: up
- `drot`: fixture rotation in the human-frame sign convention

`RobotConfig` then maps those semantic commands into UR base-frame motion.

## What Changed

- Adds `RobotConfig.HUMAN_LEFT_IN_BASE`.
- Adds `RobotConfig.HUMAN_FORWARD_IN_BASE`.
- Adds `RobotConfig.HUMAN_UP_IN_BASE`.
- Adds `RobotConfig.HUMAN_ROT_SIGN`.
- Adds `move_human_relative()` and `adjust_human_rotation()` to the robot layer.
- The controller now sends semantic human-frame commands when those robot
  methods are available.
- Event details include `command`, `human_requested`, and `base_mapped`.

## Frame Calibration Steps

Do these with very small live motions before testing posture interventions:

1. From the pendant or a controlled script, move the TCP slightly in UR base
   `+X`, `+Y`, and `+Z`.
2. Record which direction the fixture moves relative to the person:
   person's left/right, front/back, up/down.
3. Set `RobotConfig.HUMAN_LEFT_IN_BASE` to the UR base unit vector that moves
   toward the person's left.
4. Set `RobotConfig.HUMAN_FORWARD_IN_BASE` to the UR base unit vector that moves
   toward the person's front.
5. Keep `RobotConfig.HUMAN_UP_IN_BASE` as `(0, 0, 1)` unless the robot base is
   mounted unusually.
6. Command a tiny positive fixture rotation and set `HUMAN_ROT_SIGN` so positive
   human-frame rotation matches the intended physical convention.

Example: if UR base `-Y` moves toward the person's left, set
`HUMAN_LEFT_IN_BASE = (0, -1, 0)`.

## Test Steps

1. Run a simulated session first:
   `PYTHONPATH=src python src/run_session.py --participant MAP01 --condition experimental --simulate --minutes 1 --preview`
2. After calibrating the frame constants, run live:
   `PYTHONPATH=src python src/run_session.py --participant MAP01 --condition experimental --live --minutes 1 --preview`
3. Hold these postures for at least three seconds each:
   head turned right, head turned left, forward lean, left side bend,
   right side bend.
4. Watch whether the fixture moves in the expected physical direction, not only
   whether the optimizer selected the expected DOF.

## How To Analyze Results

Open the generated `*_events.csv`:

- `command` is the signed semantic human-frame command requested by the
  controller.
- `base_mapped` is the UR base-frame translation after calibration mapping.
- `applied` is what the robot interface reports after workspace clamping.
- If `command` is correct but `base_mapped` points to the wrong UR direction,
  fix the `RobotConfig.HUMAN_*_IN_BASE` constants.
- If `base_mapped` is correct but `applied` is different, check workspace
  clamping and singularity refusal.
- If `command` is wrong before mapping, the remaining issue is in posture signs
  or controller policy, not the UR coordinate frame.

Ignore direction judgments for rows with `clamped=True`.
