# Camera placement: how many, where, and what it costs you

Short answer, if you only read one line:

- **One camera: put it at the side, perpendicular to the sagittal plane.**
- **Two cameras: side plus front, roughly orthogonal.**
- **Never a front camera alone.** That is the v1 configuration and it is why the
  trunk term collapsed to about 2 percent of the strain score.

Everything below is why, what it costs, and what the code does about it.

---

## 1. The geometry, and why placement is not a detail

Put the person in their own frame:

| Plane | What moves in it | RULA terms |
|---|---|---|
| Sagittal (splits left/right) | forward lean, head-down | trunk flexion, neck flexion |
| Frontal (splits front/back) | leaning sideways | trunk side-bend, neck side-bend |
| Transverse (horizontal) | twisting | trunk twist, neck twist |

A rotation is resolved well when it happens **across** the image and badly when
it happens **along** the optical axis. Let `phi` be the camera azimuth measured
from the person's forward direction, so `phi = 0` is face-on and `phi = 90` is a
pure side view. To first order:

```
sagittal sensitivity  ∝  sin(phi)
frontal  sensitivity  ∝  cos(phi)
```

| Azimuth | Sagittal | Frontal | |
|---|---|---|---|
| 0° (front) | **0.00** | 1.00 | the v1 failure: forward lean is along the optical axis |
| 15° | 0.26 | 0.97 | |
| 45° (oblique) | 0.71 | 0.71 | both planes, both contaminated |
| 75° | 0.97 | 0.26 | |
| 90° (side) | **1.00** | 0.00 | ideal for trunk and neck flexion |

**PSS_v2 puts 70 percent of its weight on neck and trunk, and both are
sagittal.** That single fact decides the whole placement question.

Run `python3 src/camera_config.py` to print this table for your own setup.

---

## 2. The tolerance result, which is not what most people expect

How precisely does the side camera need to be aimed? Decompose a misalignment
of `err` degrees:

```
signal kept  = cos(err)
cross-talk   = sin(err)
```

| Error | Signal kept | Cross-talk from the other plane |
|---|---|---|
| 5° | 100% | 9% |
| 10° | 98% | 17% |
| **15°** | **97%** | **26%** |
| 20° | 94% | 34% |
| 30° | 87% | 50% |

A 15 degree error costs **3 percent of the signal** but admits **26 percent
cross-talk**. The tolerance is set by cross-talk, not by signal loss.

This matters because of how calibration works. PSS_v2 learns a per-person
neutral offset at the start of every session, which **absorbs a constant bias**.
A slightly wrong mounting height mostly produces a constant bias, so calibration
handles it. Cross-talk and scale error are **posture-dependent**, so calibration
cannot touch them.

> **Azimuth alignment matters far more than exact height or distance.**
> Spend your setup time aiming, not measuring the tripod height.

---

## 3. Where to put it relative to the cobot

Constraints in priority order. The first one is the one people actually get
wrong.

**1. The hips must be visible.** Trunk angle is the vector from hip midpoint to
shoulder midpoint. If the bench edge occludes the hips there is no trunk angle
at all, whatever the azimuth. A side camera set at desk height very often sees
the bench, not the hips, and it **fails silently** because MediaPipe still emits
a landmark, just with low visibility. Check this first, before anything else.

**2. Stay out of the cobot's working volume.** Collision risk and a
safety-assessment problem.

**3. Keep the arm out of the line of sight.** The UR3 holds the artifact between
the person and the bench. A camera on the same side as the arm's dominant sweep
gets occluded exactly *during interventions*, which is the worst possible
moment, because that is when the recovery measurement happens. Put the side
camera on the side **opposite the robot base**, or raise it to look over the arm.
Set `CameraConfig.ROBOT_BASE_SIDE` so the config can warn you.

**4. Distance 1.5 to 2.5 m** at 720p. Closer clips the torso and adds radial
distortion at the frame edge; further shrinks the subject until landmark noise
is comparable to the angles being measured.

**5. Stay within about 20° of horizontal.** An elevated view foreshortens the
vertical component of the trunk vector. Calibration absorbs the constant part,
not the scale. MediaPipe is also trained mostly on roughly level viewpoints.

---

## 4. One camera

### Recommended: `SIDE_ONLY`

```python
CameraConfig.LAYOUT = "SIDE_ONLY"
```

Azimuth 90° ±15°, trunk mid-height ±20 cm, 1.5–2.5 m, opposite the robot base.

| | |
|---|---|
| Measures | trunk flexion, neck flexion, upper-arm elevation, elbow angle |
| Cannot measure | side-bend, twist, lateral gaze |

**Effect on the calculation.** The side-bend and twist adders are held at zero
rather than guessed, so PSS is a **reduced score**. It is internally consistent
and perfectly usable, but it is *not numerically comparable to a two-camera PSS*.
Do not pool SIDE_ONLY and SIDE_FRONT sessions in one analysis.

**Effect on the robot.** The controller loses lateral (`dx`) and rotation
(`drot`) authority, because the signals that would justify those moves are not
observed. It keeps height, depth and tilt, which is where most of the achievable
relief lives anyway. This is automatic; see `goal_controller.enabled_dof`.

### Not recommended: `FRONT_ONLY`

This is the v1 configuration. Trunk and neck flexion are unmeasurable, which
removes about 70 percent of the PSS_v2 weight. The layout is kept in the
registry only so the config can express it and refuse to pretend. It emits a
warning and the DOF interlock collapses the controller to `dx` and `drot` alone,
which is the code independently rediscovering the v1 defect.

### Last resort: `OBLIQUE_ONLY`

Azimuth 45°. Resolves both planes at about 71 percent, with every angle
contaminated by the other plane. Cross-talk is posture-dependent so calibration
cannot remove it. Use only if the rig physically cannot take a side view, and do
not use it for the RULA validation.

---

## 5. Two cameras

### Recommended: `SIDE_FRONT` (the design default)

```python
CameraConfig.LAYOUT = "SIDE_FRONT"
```

| Camera | Azimuth | Height | Provides |
|---|---|---|---|
| Side | 90° ±15° | trunk mid ±20 cm | trunk flexion, neck flexion, arm elevation, elbow |
| Front | 0° ±20° | trunk mid +10 cm ±25 cm | side-bend, twist, lateral gaze |

**Why orthogonal.** Each plane of motion is in-plane for exactly one camera, so
cross-talk between the two is near zero and best-view fusion has an unambiguous
winner per joint. Any other pairing means two mediocre views of the same thing.

Twist stays the weakest measurement even here: it is inferred from the projected
shoulder line against the hip line rather than measured directly, so it carries
reduced confidence (0.6).

### Fallback: `SIDE_FRONT_OBLIQUE`

If the bench or robot column blocks the true frontal position, move the second
camera to about 35°. The side camera is unchanged, so the sagittal angles that
dominate PSS_v2 are unaffected; only the frontal terms degrade, by about
cos(35°) = 0.82.

### What two cameras does *not* buy you

Two views at similar azimuths add little. The pipeline uses **monocular
world-landmark best-view fusion**, not stereo triangulation, so a second nearby
view is redundant rather than additive. Value comes from viewing a *different
plane*, which is why orthogonality is the whole point.

---

## 6. What the code does with this

The layout is not documentation, it is enforced:

| Mechanism | Where | Effect |
|---|---|---|
| Layout registry | `camera_config.py` | declares what each layout can and cannot measure |
| Angle masking | `pose_fusion.apply_layout_mask` | unmeasurable angles → 0.0 at 0.0 confidence, never guessed |
| Arm neutral fallback | same | unmeasurable arm angles fall back to anatomical neutral, not 0, which would read as a straight elbow |
| Front-sagittal fallback | `FusionConfig.ALLOW_FRONT_SAGITTAL_FALLBACK` | **off by default**; this was the v1 defect |
| DOF interlock | `goal_controller.enabled_dof` | disables robot DOF whose driving signal is unmeasured |
| Validation | `camera_config.validate()` | warns on unrecommended layouts, missing sources, misalignment |
| Provenance | session `_meta.txt` | records the layout actually used |

Resulting DOF by layout:

| Layout | Enabled DOF |
|---|---|
| SIDE_FRONT | dz, dy, dtilt, drot, dx |
| SIDE_ONLY | dz, dy, dtilt |
| OBLIQUE_ONLY | dz, dy, dtilt, drot, dx (all degraded) |
| FRONT_ONLY | drot, dx only |

---

## 7. Setup checklist

1. Choose the layout and set `CameraConfig.LAYOUT`.
2. Set `SIDE_SOURCE` / `FRONT_SOURCE`. On Linux use `/dev/v4l/by-id/...` paths;
   integer indices do not survive a replug.
3. Set `ROBOT_BASE_SIDE`.
4. Place the side camera. **Verify the hips are visible** before anything else.
5. Measure the actual azimuths, write them into `AS_BUILT_*_AZIMUTH_DEG`, and set
   `AS_BUILT_VERIFIED = True`. The config warns until you do.
6. Run `python3 src/camera_config.py` and read the validation messages.
7. Confirm the world axes with an upright-then-lean-45° capture
   (`docs/RIG_SETUP.md` step 2).
8. Check `sync_skew_ms()` under about 30 ms for two cameras.

Override the layout per run without editing the file:

```bash
python3 src/run_session.py --participant P01 --condition control \
        --camera-layout SIDE_ONLY --simulate
```

---

## 8. Honest limits of this model

- The sin/cos sensitivities are **first-order projective geometry**. MediaPipe
  world landmarks are a monocular 3D lift, so real behaviour is milder than pure
  projection but follows the same trend. Treat these as design guidance, not
  calibration coefficients.
- The confidence multipliers per layout are **reasoned defaults, not measured**.
  If you want defensible numbers, run the same posture under two layouts and
  compare; that is a small, self-contained validation study.
- Twist is weak in every layout. If twist matters to your claims, it needs a
  different sensor, not a different placement.
