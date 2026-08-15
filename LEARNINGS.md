# What we got wrong in v1, and what v2 does about it

This is the engineering record behind the rebuild. It is deliberately blunt,
because the interesting part of this project is not that a cobot moved an
artifact. It is that a first implementation shipped a metric that did not
measure what it claimed to, and that this was found and fixed by going back to
the logged data rather than by arguing about it.

Everything below is either measured from the v1 session logs or verifiable by
running `tests/test_all.py`.

---

## 1. The strain metric did not measure postural strain

**The claim.** The proposal defined the Postural Strain Score as trunk
inclination anchored to RULA/REBA thresholds, combined with cervical strain.
Trunk inclination was the centrepiece.

**What shipped.** A four-component weighted score:
`PSS = 0.55 gaze + 0.30 cervical + 0.10 lean + 0.05 trunk`.

**What the logs actually contained.** Recomputing each component's share of the
mean PSS across all v1 sessions:

| Component | Intended | Actual share of PSS |
|---|---|---|
| Gaze (head turn) | 0.55 | ~56% |
| Cervical (head lateral offset) | 0.30 | ~42% |
| Lean | 0.10 | ~0.5% |
| Trunk inclination | 0.05 | ~1.7% |

Trunk and lean together contributed roughly 2%. The score was, in practice, a
two-component head-orientation signal. Both surviving components derive from the
same anatomical region, so it was not even two independent postural factors.

**Root cause.** A single overhead-left 2D camera cannot resolve trunk flexion
toward the bench: leaning forward barely changes the projected shoulder-to-hip
vector. The signal was not noisy, it was absent. No amount of reweighting or
machine learning on that input would have recovered it.

**The lesson.** Sensor geometry decides which ergonomic claims are available to
you. Choose the claim after you know what the camera can see, not before.

---

## 2. The dominant signal was never logged

Gaze drove 55% of the score and does not appear in the v1 `_frames.csv` at all.
It could only be recovered by algebraically inverting the PSS formula from the
other three components. That means the largest driver of every intervention
decision could not be independently reconstructed, validated, or audited.

**The lesson.** Log inputs, not just outputs. A metric you cannot reconstruct
from your own files is a metric you cannot defend in review.

v2 writes every input angle, every per-axis confidence, every sub-band, both
group scores, and a full snapshot of every config value that governed the run.
`tests/test_all.py::test_logger_frames_are_self_consistent_under_calibration`
asserts that the logged angle columns recompute the logged score exactly.

---

## 3. The experimental design confounded adaptation with geometry

In control the artifact rested flat on a table. In experimental it was held on
the cobot end effector at a different height and angle. Any difference between
conditions therefore mixes the adaptive behaviour with a different baseline
presentation of the work. No PSS or NASA-TLX difference can be attributed to
adaptation from that design.

**v2 fix.** `robot_interface.move_to_baseline()` is called in *both* conditions.
The artifact starts at an identical pose either way; only whether the robot
subsequently adapts differs. Asserted by
`test_baseline_identical_across_conditions`.

**Still not controlled.** A motion-matched sham (robot moves the same amount but
not contingent on posture) would additionally control for the belief that an
adaptive system is helping, which the HRI literature shows can shift subjective
ratings on its own. At N=6 we cannot run three arms. This is a stated limitation,
not a solved problem.

---

## 4. "Faint, left-right movement" was an architecture problem, not an ML problem

v1 applied fixed increments (2 cm, 0.08 rad) whenever PSS crossed a threshold,
then waited out a cooldown and nudged again. It never computed a target pose.
Timid oscillation is exactly what that policy encodes, and a better strain
signal would not have changed it.

**v2 fix.** A goal-based controller: on sustained strain, solve for the artifact
pose that minimises predicted PSS over five DOF, then execute one smooth move.
The effort penalty keeps the move smooth rather than keeping it small.

An early version of this had `LAMBDA_EFFORT = 0.15`, which reproduced the
original problem: under deep strain the optimiser preferred a small move. It was
lowered to 0.01 after the self-test showed under-movement. That is the whole
lesson in miniature: the timidity lived in a cost weight, not in the intelligence
of the system.

---

## 5. Analyses were reported that had not been run

The v1 analysis plan listed a Bland-Altman comparison of automated strain
against expert RULA. It was never performed. Shapiro-Wilk did not precede the
parametric claims. One citation did not correspond to a real source.

**v2 fix.** `evaluation.py` implements the RULA validation that was promised,
runs Shapiro-Wilk before choosing between paired t and Wilcoxon, and reports
effect sizes and censored counts. `rula.py` ships with
`TABLES_VERIFIED = False` and a test that fails if anyone flips it without doing
the worksheet check.

**The lesson, and it is the important one.** Every claimed analysis needs an
execution trail. If it cannot be re-run from the repository, it did not happen.

---

## 6. The robot layer did not exist

Discovered during the final audit: the rebuilt controller calls
`robot.move_relative(...)` and `robot.adjust_rotation(...)`, but no
implementation of that contract existed anywhere in the rebuild except a test
stub. The pipeline could not have driven a UR3.

Writing it surfaced two further problems that would have shown up on hardware
day:

**Rotation composition.** A UR pose is `[x, y, z, rx, ry, rz]` where the last
three are an axis-angle vector, not roll/pitch/yaw. Adding a delta to `rx` is
only valid for very small angles about that same axis. On a typical tool-down
orientation, `rx += 0.30` produces **0.191 rad of actual rotation, not 0.30** —
a 36% under-tilt on every intervention, silently. v2 composes rotations properly
(Rodrigues) and asserts it in `test_rotation_composition_is_correct`.

**Cumulative drift.** The controller caps a *single* move (8 cm, 0.30 rad).
Nothing capped the accumulation. Eight consecutive maximum raises would command
+64 cm, well outside a UR3's 500 mm reach and toward the participant. v2 clamps
the *absolute* target into a measured workspace envelope on every move and
reports when it clamped, so the analysis can censor interventions the robot
could not fully execute.

**The lesson.** Per-step limits are not safety bounds. Integrate them and see
where the system ends up.

---

## 7. The arm kept reaching singularities, and a box does not prevent that

Reported from real sessions. It follows directly from section 6: repeated
relative nudges with no bound on where the arm ends up walk the tool either
outward until the elbow straightens, or inward over the base axis. The Cartesian
envelope added in section 6 bounds a **box**, and a box still contains the
central cylinder around the base axis and still touches the reach shell.

Three conditions apply to a UR arm: wrist at `q5 → 0` or `±π`, elbow at
`q3 → 0`, and shoulder when the wrist centre approaches the joint-1 axis. Each
was verified to drive the smallest singular value of the Jacobian to 0.0000, so
these are genuine rank losses rather than heuristics.

Two things were worth learning here beyond the fix itself:

- **A cost beats a veto.** The singularity term went into the controller's
  objective, not a gate after it. The optimiser then routes *around* singular
  regions while still relieving strain. Measured, with the robot near the base
  axis: without the term it commanded `dy = +0.060` and drove the radial
  distance to 0.140 m against a 0.112 m cylinder; with the term it commanded
  `dy = +0.012` and held at 0.188 m. A veto would have lost the intervention
  entirely.
- **Nominal kinematics are not your arm.** UR publishes DH parameters that are
  revision-controlled per serial-number range. An early version compared the
  frame-5 origin against the two-link planar reach, which mixes frames, and
  wrongly blocked a perfectly good pose. Sampling the nominal chain also gives
  TCP distances well past the published 500 mm figure, because the wrist offsets
  stack. So the working radius is now a **measured** value like the envelope, and
  the live path uses the robot's own IK rather than our table. The UR wrist is
  also not spherical, so spherical-wrist shortcuts do not apply.

**The lesson.** Per-step limits are not workspace limits, and workspace limits
are not singularity limits. Each has to be stated separately, and the geometry
that matters is the robot's own, not a textbook's.

---

## 8. Camera count and placement is a claim, not a deployment detail

Section 1 established that the v1 trunk signal was absent because a single front
camera cannot resolve forward lean. What was still missing was any mechanism
preventing that from happening again, or from happening silently.

The geometry is first-order and blunt. For a camera at azimuth `phi` from the
person's forward direction, sagittal sensitivity goes as `sin(phi)` and frontal
as `cos(phi)`. A front camera has **zero** sagittal sensitivity. PSS_v2 puts 70
percent of its weight on neck and trunk, both sagittal. So with one camera the
answer is a side view, and it is not close.

The result that was genuinely counter-intuitive concerns tolerance. A side camera
misaligned by 15 degrees keeps `cos(15) = 97%` of the signal but admits
`sin(15) = 26%` cross-talk from side-bend. **The tolerance is set by cross-talk,
not by signal loss.** And that interacts with calibration in a way worth stating
plainly: the per-session neutral calibration absorbs a **constant** bias, which is
mostly what a wrong mounting height produces, but it cannot absorb cross-talk or
scale error, which are posture-dependent. So azimuth alignment deserves the setup
time; tripod height does not.

The enforcement matters as much as the analysis. `camera_config.py` declares per
layout what can and cannot be measured; unmeasurable angles are returned as
absent at zero confidence rather than guessed; and the controller drops the
degrees of freedom whose driving signal is missing. Under `SIDE_ONLY` it keeps
height, depth and tilt but loses lateral and rotation, because moving sideways in
response to a structurally-zero side-bend signal is not a small error, it is
motion caused by nothing.

The check that this is working: point the config at `FRONT_ONLY` and the DOF
interlock collapses the controller to `dx` and `drot` alone. The code
rediscovers the v1 defect on its own.

**The lesson.** Encode what your sensors cannot see, not just what they can. A
pipeline that cannot represent "unmeasured" will eventually report a guess with
the same authority as a measurement, which is exactly what v1 did.

---

## 9. Not all strain is the robot's to fix

The clearest analytical finding of the rebuild. RULA-grounded scoring separates
trunk/neck load from arm load, and repositioning an artifact can only relieve
the former. A conservator holding a tool in a raised-arm posture carries strain
that no artifact pose removes.

This reframes the v1 null result. Session-mean PSS being flat is not necessarily
a failed intervention: it is the expected outcome when a large share of strain is
mechanically out of scope. `evaluation.analyze_paradox()` clusters high-strain
frames into posture modes and uses the actual controller optimiser to compute the
achievable PSS reduction for each, which turns "the effect was null" into "here
is the fraction of strain this system can address, and here is the fraction it
cannot."

**The lesson.** State the mechanism your intervention acts through, then measure
how much of the observed problem lies inside it. A bounded claim that survives
scrutiny beats a broad one that does not.

---

## Summary: v1 to v2

| Dimension | v1 | v2 |
|---|---|---|
| Strain metric | 56% gaze / 42% cervical, trunk ~2% | RULA-grounded trunk, neck, arm from measured angles |
| Trunk sensing | single 2D camera, signal absent | side (sagittal) camera, best-view fusion |
| Reconstructability | dominant component unlogged | every input, confidence, sub-score, and config snapshot |
| Condition baseline | table vs end effector (confounded) | identical baseline pose in both conditions |
| Control policy | fixed 2 cm nudges, no target | optimised target pose, one smooth move |
| Robot layer | absent from rebuild | RTDE, correct rotation composition, envelope clamping |
| Cumulative bounds | none | absolute workspace and orientation envelope |
| RULA validation | promised, not run | Bland-Altman, weighted kappa, dual Spearman |
| Normality check | absent | Shapiro-Wilk before test selection |
| Null H1 | reported as a failure | mechanism quantified via posture-mode analysis |
| Response gains | geometric guesses, no path to refine | fitted from logged interventions, with identifiability guard |
| Robot singularities | hit during sessions | avoided as a cost term, plus envelope and IK checks |
| Camera count | fixed at 1, undocumented | configurable 1 or 2, with placement registry |
| Unmeasurable angles | reported as if measured | reported as absent; DOF interlock follows |
| Regression safety | none | 46 assert-based tests |

---

## What v2 still cannot do

Stated plainly, because the limits are part of the contribution.

- **Wrist, wrist twist, and legs are not measurable** from two cameras with a
  tool in hand. RULA scoring assumes them. Assuming a near-neutral wrist
  *under*-estimates strain for fine conservation work, so the automated grand
  score is a conservative estimate and must be reported as one.
- **RULA Tables A and B are not yet verified** against a printed worksheet. No
  RULA integer should appear in the paper until they are.
- **The human-response model is linear and local.** Gains are geometric estimates
  until fitted from real interventions. The architecture does not change when
  they improve, but the magnitude of every commanded move depends on them.
- **MediaPipe world landmarks are a monocular 3D lift, not triangulated stereo.**
  Good enough for sagittal trunk and neck angles from a side view; not metrology.
- **The camera sensitivity model is first-order projective geometry**, and the
  per-layout confidence multipliers are reasoned defaults, not measured. Twist is
  weak in every layout; if twist matters to a claim it needs a different sensor,
  not a different placement.
- **Joint limits and self-collision are not handled.** Singularity avoidance is
  not collision avoidance. UR's own joint limits can permit configurations where
  links interfere; that stays with the robot controller and the cell's risk
  assessment.
- **N=6 is a pilot.** A null H1 at that size is inconclusive, not evidence of no
  effect. The power caveat is generated automatically in the report.
- **Placebo is not controlled.** See section 3.
- **The controller no longer rotates for gaze**, unlike the submitted paper,
  because PSS_v2 is RULA-grounded and gaze is not a RULA strain factor. This is a
  deliberate behavioural change and must be disclosed.

---

## Roadmap

1. **August, Pilot Factory.** Verify rig axes and lateral sign, measure the
   workspace envelope, run matched-baseline sessions in both conditions.
2. **Immediately after day one.** Run `fit_response_gains`. If it returns gains
   with good R², decide deliberately whether to adopt them, and document it.
3. **Before writing numbers.** Verify RULA Tables A/B; collect expert RULA
   coding using the same documented assumptions as the automated scorer.
4. **Beyond this study.** Motion-matched sham control; reinforcement learning in
   simulation only, positioned as future work, given the published sim-to-real
   gap in comparable ergonomic-cobot work.
