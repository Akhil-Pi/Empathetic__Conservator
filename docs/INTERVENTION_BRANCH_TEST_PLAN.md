# Intervention Branch Test Plan: Diagnostic Logging Only

This branch does not change the intervention policy. It only records enough
data to locate where the wrong motion direction enters the pipeline.

## What Changed

- Adds `command_delta` to each intervention result. This is the signed command
  actually sent to the robot after the controller converts the optimizer output
  using the observed posture sign.
- Adds `diagnostic_angles` and `diagnostic_confidence` to the intervention
  result.
- Adds the signed command and triggering angles to the event CSV `details`
  field.

## Test Steps

1. Start with the robot in simulate mode and run a short experimental session:
   `PYTHONPATH=src python src/run_session.py --participant DEBUG01 --condition experimental --simulate --minutes 1 --preview`
2. Repeat in live mode only after the workspace is clear and the UR speed is
   acceptable:
   `PYTHONPATH=src python src/run_session.py --participant DEBUG01 --condition experimental --live --minutes 1 --preview`
3. During the task phase, hold these postures for at least three seconds each:
   neutral, head turned right, head turned left, forward lean, left side bend,
   right side bend.
4. Save the generated `*_frames.csv`, `*_events.csv`, and `*_meta.txt`.

## How To Analyze Results

For each intervention row in `*_events.csv`, inspect the `details` field:

- If the triggering angle signs are wrong or unstable, the problem is in camera
  fusion or camera placement.
- If angle signs are correct but `command_delta` has the wrong sign, the problem
  is in controller sign logic.
- If `command_delta` is correct but `applied` motion is wrong, the problem is in
  the UR base-frame mapping or robot interface.
- If `clamped=True`, the workspace envelope changed the requested move and the
  trial should not be used to judge policy direction.

Expected behavior for this branch is the same as `main`; it may still move in
the wrong direction. The value of this branch is diagnostic visibility.
