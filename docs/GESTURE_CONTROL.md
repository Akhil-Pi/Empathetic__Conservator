# Gesture control: hands-free pause/resume

Short answer, if you only read one line: **`GestureConfig.ENABLED` defaults to
`False`.** With it off, this feature does not exist at runtime — every code
path it touches (`run_session.py`'s task loop, `session_logger_v2.py`'s
`paused` column, `goal_controller.py`'s `reset_dwell_timer()`) is inert.
Everything below only matters once someone deliberately flips it on.

---

## 1. What it does

Open palm facing the camera → **PAUSE** the task workflow (no new
interventions). Closed fist → **RESUME**. Detection runs on whichever camera
frame the session loop has *already read* that cycle (front camera by
default, falling back to the side camera if the active layout has no front
camera, disabled for the run if neither exists) — it never opens a second
capture device.

Recognition uses MediaPipe Tasks' `GestureRecognizer` and its canned model,
which already classifies `Open_Palm` and `Closed_Fist` among its built-in
labels, loaded from `gesture_recognizer.task` in the repo root (same
loading convention as `pose_landmarker.task` for pose).

A gesture must be held continuously for `GestureConfig.HOLD_SECONDS` (default
0.6s, wall-clock, not frame count — same reasoning as `pss_v2`'s time-based
smoothing: the two cameras on this rig do not run at a fixed frame rate)
before it toggles anything, and state changes are further separated by
`GestureConfig.COOLDOWN_S`. Pausing while already paused, or resuming while
already running, are no-ops. All of this logic (`PauseStateMachine` in
`src/gesture_control.py`) is pure Python with no MediaPipe dependency, so it
is unit-tested in `tests/test_all.py` without a camera.

## 2. Why the debounce and cooldown exist: false positives and occlusion

**False-positive risk.** A conservator's hands are open and closed
constantly during ordinary work — picking up a brush, releasing a tool,
adjusting grip. A one-frame gesture match would misfire on normal work
motion, not just on a deliberate pause signal. `HOLD_SECONDS` exists
specifically to filter this out: it costs a small amount of latency on a
genuine pause/resume, in exchange for making incidental hand shapes far less
likely to trigger one. This has **not** been validated against real
conservation task footage; the default of 0.6s is a starting point, not a
measured value, and should be tuned (or the hold requirement re-examined
entirely) once real task video is available.

**Occlusion risk.** The two situations where a conservator would most want to
pause — hands full of a tool, or reaching into an awkward position — are
exactly the situations where a hand is most likely to be occluded, holding
something, or out of frame. Gesture control is a *convenience* pause path,
not a safety-critical one: it should not be relied on as the primary way to
stop the robot in an emergency. Nothing in this feature changes or replaces
any existing stop/safety mechanism.

## 3. Blinding

A pause only has a **robot-visible effect in the experimental condition** —
in `control`, `GoalBasedController.evaluate()` is already an unconditional
no-op regardless of pause state, so pausing changes nothing about what the
participant sees; only the logged `paused` column and the `pause`/`resume`
events differ. Console output during the task loop stays condition-silent
either way (see `run_session.py`'s existing blinding note) — nothing printed
by this feature reveals which condition is running.

## 4. Logging and analysis-exclusion requirement

- `_frames.csv` gets a `paused` column (`0`/`1`), defaulting to `0`, so
  existing readers and the frame self-consistency test are unaffected by its
  presence.
- `_events.csv` gets two new `event_type` values, `pause` and `resume`, each
  carrying the triggering gesture and its confidence score in `details`
  (e.g. `gesture=Open_Palm score=0.900`).
- `_meta.txt` snapshots `GestureConfig` alongside the other configs, so a
  session using this feature is still fully reproducible from its meta file.

**Paused time must not silently count as monitored task time or as
time-above-threshold.** No intervention can fire during a pause, so pooling
paused frames into `evaluation.py`'s existing PSS statistics (`mean_pss`,
`frac_above_threshold` in `_session_pss_summary`) would understate how much
of the session the system was actually able to act on. This has been left as
a **TODO marked directly in `evaluation.py`** rather than silently changed:
excluding paused spans changes an existing, already-relied-on statistic, and
that is a decision for the study lead, not a default this feature should
make unilaterally. Until that TODO is resolved, treat `frac_above_threshold`
and `mean_pss` from any session that used gesture control as including
paused time.

## 5. Open question for the study lead: which condition(s)?

This is implemented **condition-agnostic**: it runs identically in `control`
and `experimental`. In `control` there is nothing for a pause to visibly
change (the controller is already a no-op there), so the only difference is
that `pause`/`resume` events and the `paused` column appear in the logs.

**Whether gesture control should be offered in both conditions, or only in
experimental, was not decided by this implementation and needs a deliberate
call from the study lead before it is used in a real session:**

- Running it in both keeps the study symmetric — same available interaction
  in both arms — and avoids gesture control itself becoming a confound.
- Running it only in experimental ties it more tightly to "the adaptive
  system," which may or may not be the intended framing, and would need a
  parallel manual mechanism in control (or a documented asymmetry) so
  participants aren't just missing a feature with no equivalent.

Do not collect real study data with this feature until that decision is made
and recorded (config freeze rules in `README.md` apply here the same as
anywhere else: freeze after participant 1, document if you must change it
mid-study).

## 6. Files

- `src/gesture_control.py` — `GestureConfig`, `PauseStateMachine` (pure,
  unit-tested), `GestureRecognizerWrapper` (MediaPipe, guarded import).
- `src/run_session.py` — `build_gesture_control()` and the task-loop
  integration (task phase only — never during calibration).
- `src/goal_controller.py` — `GoalBasedController.reset_dwell_timer()`.
- `src/session_logger_v2.py` — `paused` column, `GestureConfig` snapshot.
- `src/evaluation.py` — TODO at `_session_pss_summary()`.
- `tests/test_all.py` — debounce, cooldown, idempotency, single-frame
  spurious-gesture, disabled-by-default, and dwell-timer-reset tests.
