# The Empathetic Conservator

A vision-based UR3 cobot that monitors an art conservator's posture and
repositions the artifact to reduce musculoskeletal strain.

Two cameras estimate the conservator's joint angles. A RULA-grounded Postural
Strain Score (PSS) turns those angles into a single strain value. When strain is
sustained above threshold, a goal-based controller solves for the artifact pose
that minimises predicted strain and executes one smooth move. Everything is
logged so the result can be reconstructed and validated afterwards.

This is version 2. Version 1 shipped a strain metric that turned out to measure
head orientation rather than postural strain, among other problems.
**[LEARNINGS.md](LEARNINGS.md) is the honest record of what was wrong and what
changed** — read it before changing anything here, and it is the source for the
"what we learned" part of any talk about this project.

---

## Quickstart

```bash
git clone <repo-url> && cd empathetic-conservator
pip install -r requirements.txt
python3 tests/test_all.py            # 30 tests, no hardware needed
```

Everything below runs without a camera or a robot.

```bash
# generate synthetic sessions and run the whole analysis pipeline
python3 tools/make_synthetic_sessions.py

# run any module's own demonstration
python3 src/pss_v2.py
python3 src/rula.py
python3 src/goal_controller.py
python3 src/robot_interface.py
python3 src/kinematics.py        # singularity margins, all three conditions
python3 src/camera_config.py     # placement sensitivity and layout validation
```

Collecting a session (simulated robot, still needs two cameras):

```bash
python3 src/run_session.py --participant P01 --condition control      --simulate
python3 src/run_session.py --participant P01 --condition experimental --simulate

# override the camera layout for a run (default is SIDE_FRONT)
python3 src/run_session.py --participant P01 --condition control \
        --camera-layout SIDE_ONLY --simulate
```

Driving the real UR3 (see the hardware checklist first):

```bash
python3 src/run_session.py --participant P01 --condition experimental --live
```

Analysing collected sessions:

```bash
python3 src/evaluation.py data/sessions expert_rula.csv
# report and figures land in eval_out/
```

---

## Architecture

```
  side camera ─┐
               ├─► pose_fusion ──► PostureAngles ──► pss_v2 ──► PSS
 front camera ─┘   (best view          (angles)      (RULA-      │
                    per joint)                       grounded)   │
                                                                 ▼
      session_logger_v2  ◄────────────────────────  goal_controller
      (frames + events                              (optimise target pose)
       + config snapshot)                                        │
               │                                                 ▼
               │                                       robot_interface
               │                                   (RTDE, envelope, baseline)
               ▼                                                 │
          evaluation                                             ▼
   (RULA validation, H1, paradox,                             UR3 cobot
    latency-recovery, gain fit)
```

**Separation that makes this testable.** `pss_v2` consumes anatomical angles,
not landmarks, so scoring is unit-testable without a camera. `goal_controller`
talks to an abstract robot contract, so control logic is testable without a
robot. `kinematics` is pure maths, so singularity margins are testable without
an arm. Every module has a `__main__` demonstration and a corresponding assert
in `tests/test_all.py`.

**Two configuration choices drive real behaviour, not just documentation:**

- `CameraConfig.LAYOUT` decides which angles can honestly be measured. Anything
  a layout cannot measure is reported as absent rather than guessed, and the
  controller loses the degrees of freedom that signal would justify. One camera
  should be a **side** view; see [docs/CAMERA_PLACEMENT.md](docs/CAMERA_PLACEMENT.md).
- `RobotConfig.AVOID_SINGULARITIES` makes the workspace envelope
  singularity-aware and adds a term to the controller's cost function, so the
  optimiser routes around wrist, elbow and shoulder singularities instead of
  driving into them; see [docs/SINGULARITY.md](docs/SINGULARITY.md).

---

## Repository layout

```
src/
  pss_v2.py             RULA-grounded strain score; PostureAngles is the shared type
  camera_config.py      1 or 2 cameras, placement registry, what each layout can measure
  pose_fusion.py        best-view fusion into PostureAngles, layout-masked
  camera_stream.py      threaded capture, 1 or 2 cameras (latest-frame, MJPEG)
  kinematics.py         UR FK, Jacobian, the three singularity margins
  goal_controller.py    optimises target pose; DOF interlock; singularity cost
  robot_interface.py    RTDE + simulated robot; rotation maths, envelope, singularity
  session_logger_v2.py  frames + events + reproducible config snapshot
  rula.py               automated RULA scoring with documented assumptions
  evaluation.py         the five analyses
  run_session.py        session entry point, both conditions
tests/
  test_all.py           30 assert-based regression tests
tools/
  make_synthetic_sessions.py   logger-shaped synthetic data for pipeline testing
docs/
  RIG_SETUP.md          hardware-day checklist and calibration procedure
  CAMERA_PLACEMENT.md   1 vs 2 cameras, where to put them, what each costs
  SINGULARITY.md        why the arm kept hitting singularities and how it is avoided
  RULA_VERIFICATION.md  the open Table A/B verification task
```

---

## Data format

Three files per session, named `<participant>_<condition>_<timestamp>_*`:

- **`_frames.csv`** — every input angle, per-axis confidence, all four RULA
  bands, both group scores, raw and smoothed PSS. The angle columns recompute
  the PSS beside them exactly; this is asserted in the tests.
- **`_events.csv`** — interventions with per-DOF delta, predicted PSS before and
  after, whether the move executed, whether the envelope clamped it, latency.
- **`_meta.txt`** — participant, timing, calibration offsets, and a snapshot of
  every config value that governed the run. A session is reproducible from this.

Recovery time is deliberately **not** a logged field. It is derived in
`evaluation.py` from event timestamps and the PSS series, so there is one source
of truth rather than two that can disagree.

---

## The five analyses

| Analysis | What it answers |
|---|---|
| RULA validation | Does the automated score agree with an expert? Bland-Altman, quadratic-weighted kappa, dual Spearman |
| H1 | Is PSS lower in the experimental condition? Shapiro-Wilk first, then Wilcoxon and paired t with effect sizes, plus a time-above-threshold secondary endpoint |
| Paradox | If H1 is null, why? Clusters high-strain frames into posture modes and computes the achievable PSS reduction for each using the real optimiser |
| Latency vs recovery | Do slower interventions produce slower recovery? Recovery derived, censored cases reported |
| Gain fit | What are the true human-response gains? Regresses observed angle reduction on applied DOF delta, refusing to fit when deltas do not vary |

---

## Before hardware day

Full procedure in [docs/RIG_SETUP.md](docs/RIG_SETUP.md). The blocking items:

1. **Choose the camera layout** and place the cameras. With one camera use
   `SIDE_ONLY`, never `FRONT_ONLY`. Verify the hips are visible before anything
   else. Record the as-built azimuths and set `AS_BUILT_VERIFIED = True`.
2. **Measure `RobotConfig.WORKING_RADIUS_M`** and run
   `robot.check_workspace()`; it flags an envelope that reaches into the central
   cylinder or past the working radius, and a baseline pose that is itself
   near-singular.
3. **Confirm `FusionConfig.WORLD_*` axis flags** with one upright-then-lean-45°
   capture. If fused trunk reads ~45 when the person leans 45, axes are right.
4. **Confirm `ControllerConfig.LATERAL_SIGN`**: person leans left, artifact
   should move toward them, not away.
5. **Measure the workspace envelope** and set `RobotConfig.ENVELOPE_MIN/MAX`,
   then set `ENVELOPE_VERIFIED = True`. `UR3Robot` refuses to construct until
   you do, on purpose: that envelope is the only thing bounding cumulative
   drift.
6. **Set `RobotConfig.BASELINE_POSE`** to the matched starting pose. It is used
   in *both* conditions.
7. **Check payload.** UR3 nominal is 3 kg and 500 mm reach. Artifact plus
   fixture must fit both, and the fixture must hold securely through tilt.
8. **Do a full dry run** with a team member as participant, both conditions, then
   run `evaluation.py` on those files the same day.

Speed and acceleration defaults are conservative but are **not** a substitute for
a risk assessment. The collaborative operating mode and speed limits for this
cell must come from the Pilot Factory's own assessment.

---

## Working agreements

- **Run `tests/test_all.py` before every commit.** Each test corresponds to a bug
  that was actually found; they are the regression record.
- **Freeze config after the first participant.** The meta snapshot makes any
  mid-study change visible, but a change still splits your cohort. If you must,
  document it and analyse groups separately.
- **Never report an analysis without an execution trail.** If it cannot be
  re-run from this repository, it does not go in the paper.
- **No RULA integers until Tables A/B are verified.** See
  [docs/RULA_VERIFICATION.md](docs/RULA_VERIFICATION.md).
- **Keep synthetic and real data in separate directories.** They share a schema
  by design.
- **Do not pool sessions collected under different camera layouts.** A
  single-camera PSS is a reduced score and is not numerically comparable to a
  two-camera PSS. The layout used is recorded in every session's `_meta.txt`.

---

## Team

M1 / Tech Lead: Akhil Pillai · M2: Yiyi Lei · M3: Muhammed Shahnewaz Bhuiyan ·
M4: Franziska Beyer · M5: Danish Ali

TU Wien, Institut für Managementwissenschaften.
