# Rig setup and hardware-day procedure

Work through this in order. Steps 1 to 5 are blocking: the system will run
without them but the data will not be trustworthy.

---

## 0. Before you travel

```bash
python3 tests/test_all.py     # must be 30/30
```

Download `pose_landmarker.task` (MediaPipe Pose) and place it beside the source.
It is gitignored because of its size.

---

## 1. Cameras

Two USB webcams, ideally on separate USB controllers and USB 3.0. Two 720p
streams can saturate a shared USB 2.0 bus; MJPEG is forced in `camera_stream.py`
for exactly this reason.

- **Camera A (front/overhead-left):** gaze and lateral cervical. Keeps the v1 role.
- **Camera B (side, sagittal):** trunk and neck flexion. This is the camera that
  fixes the dead trunk signal, so its placement matters most: roughly
  perpendicular to the sagittal plane, at about trunk height, with a clear view
  of hip and shoulder.

```bash
python3 src/camera_stream.py     # lists indices that open, runs the threading test
```

Note which index is which physically. Indices are not stable across replug on
Windows; on Linux prefer `/dev/v4l/by-id/...` paths.

Check `sync_skew_ms()` is under about 30 ms. Above 50 ms, `run_session.py` warns.

---

## 2. Confirm world axes (blocking)

`FusionConfig.WORLD_UP_AXIS`, `WORLD_UP_SIGN`, `WORLD_FWD_AXIS`, `WORLD_LAT_AXIS`
are assumptions about MediaPipe's world-landmark convention. Confirm once:

1. Participant stands upright and still.
2. Participant leans forward to a clearly measured angle (use a goniometer or a
   phone inclinometer against the back, roughly 45 degrees).
3. Read the fused trunk angle.

If it reads about 45, the axes are right. If it reads near zero, the forward
axis is wrong. If it reads negative, flip `WORLD_UP_SIGN`. Adjust and repeat
until the fused angle tracks the measured one.

---

## 3. Confirm lateral sign (blocking)

Set `ControllerConfig.LATERAL_SIGN` so the artifact moves **toward** the
strained side. Have someone lean left with the controller running in simulate
mode and watch the commanded `dx`. If the artifact would move away from them,
flip the sign.

---

## 4. Measure the workspace envelope (blocking)

Jog the UR3 to the extremes of where the artifact may safely go, in the base
frame, and record x, y, z minimum and maximum. Set:

```python
RobotConfig.ENVELOPE_MIN = (xmin, ymin, zmin)
RobotConfig.ENVELOPE_MAX = (xmax, ymax, zmax)
RobotConfig.ENVELOPE_VERIFIED = True
```

`UR3Robot` raises on construction until you do this. That guard is deliberate:
the controller caps a single move but nothing else caps the accumulation, so
this envelope is the only bound on cumulative drift over a long session.

Leave margin. The envelope should be comfortably inside both the UR3's reach and
any region a person may occupy.

---

## 5. Set the matched baseline pose (blocking)

`RobotConfig.BASELINE_POSE` is where the artifact starts in **both** conditions.
Choose a pose that is a reasonable working presentation of the artifact, jog to
it, read `getActualTCPPose()`, and paste it in.

This is the fix for the v1 confound. Do not use a different starting pose for
control.

---

## 6. Payload and fixture

UR3 nominal: 3 kg payload, 500 mm reach. The artifact plus its fixture must fit
both. The fixture must hold the artifact securely through the full tilt range
(`RobotConfig.MAX_TILT_RAD`), because tilting a held object is the highest-risk
motion in this system. If in doubt, reduce `MAX_TILT_RAD` before reducing care.

Set the payload and tool centre point on the teach pendant to match the actual
fixture. A wrong payload setting degrades the robot's own force monitoring.

---

## 7. Safety

The collaborative operating mode, permitted speed, and any safety-rated
monitoring for this cell come from the Pilot Factory's risk assessment under
ISO 10218 and ISO/TS 15066. The conservative `SPEED_MS` and `ACCEL_MS2` defaults
in this repository are a starting point, not a compliance statement.

Practical points:

- Keep the emergency stop within reach of the operator, not the participant.
- Brief participants that the robot may move, and that they can stop at any time.
- `run_session.py` prints nothing that reveals the condition. Do not narrate the
  robot's behaviour to the participant.
- Once `SPEED_MS` is chosen, **do not change it between participants**. Motion
  time sits inside the measured intervention latency, which feeds the
  latency-versus-recovery analysis.

---

## 8. Full dry run

Before the first real participant, run both conditions end to end with a team
member, then analyse those files the same day:

```bash
python3 src/run_session.py --participant P00 --condition control      --simulate --minutes 3
python3 src/run_session.py --participant P00 --condition experimental --simulate --minutes 3
python3 src/evaluation.py data/sessions
```

Any schema, rig, or wiring surprise appears here rather than with a participant
in the room.

---

## 9. Per-participant checklist

- [ ] Participant ID in a consistent format (`P01`, not `P1` and `P001` mixed)
- [ ] Robot at baseline pose, verified visually
- [ ] Calibration completed with a genuinely neutral posture, watched
- [ ] At least 30 calibration samples accepted (the runner enforces this)
- [ ] Both conditions collected, order counterbalanced across participants
- [ ] `_meta.txt` opened and spot-checked after the first session of the day
- [ ] Session files backed up before moving on

---

## 10. After day one

Run the gain fit:

```python
from evaluation import load_sessions, fit_response_gains
fit_response_gains(load_sessions("data/sessions"))
```

If it returns fitted gains with reasonable R², decide **deliberately** whether to
adopt them for the remaining sessions, and write down that decision. Changing
`ControllerConfig.GAINS` mid-study changes controller behaviour and splits your
cohort. If it reports insufficient delta variation, that is the identifiability
guard doing its job, not a bug.
