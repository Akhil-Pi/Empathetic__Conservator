# Singularity avoidance

Sessions kept ending in a singularity. This is not bad luck, and it is not fixed
by moving more slowly.

---

## Why it kept happening

Both v1 and the early v2 controller issue **repeated relative nudges with no
bound on where the arm ends up**. Over a session of dozens of interventions the
tool walks:

- outward, toward the workspace boundary, until the elbow straightens, or
- inward, over the base axis, into the shoulder singularity.

The Cartesian envelope added earlier bounds a **box**. A box still contains the
central cylinder around the base axis and still touches the reach shell, so
clamping to a box does not prevent either failure. The envelope had to become
singularity-aware.

There is a second, quieter cause: a flat artifact tray held with the tool
pointing straight down sits close to a **wrist** singularity, and the tilt
interventions push `q5` through zero.

---

## The three conditions on a UR arm

Universal Robots describe joints 2, 3 and 4 (shoulder, elbow, wrist 1) as
rotating in a common plane, with trouble when wrist 2 aligns to that same plane.

| Type | Condition | What it looks like |
|---|---|---|
| **Wrist** | `q5 → 0` or `±π` | axes of joints 4 and 6 become coincident; one DOF lost; joints 4 and 6 spin to compensate |
| **Elbow** | `q3 → 0` | upper arm and forearm collinear, arm "stretched too far", elbow branch flips |
| **Shoulder** | wrist centre approaches the joint-1 axis | joints 1 and 4 commanded toward huge velocities |

UR's own practical advice for the third one is to lay the task out so it is never
necessary to work in or near that **central cylinder**.

Verified in `src/kinematics.py`: each condition drives the smallest singular
value of the Jacobian to 0.0000, confirming genuine rank loss rather than a
heuristic.

---

## Two layers

### Layer 1: geometric, always on, no robot needed

Keep the **target position** out of the central cylinder and inside the working
radius. Needs only nominal geometry, works offline and in simulation, and is what
makes the workspace envelope singularity-aware instead of a plain box.

```python
from kinematics import suggest_working_annulus
r_min, r_max = suggest_working_annulus(RobotConfig.WORKING_RADIUS_M)
# lay the task out between these radial distances from the base axis
```

### Layer 2: joint space, live only

Ask the robot's **own IK** for the joint vector that reaches the target, then
measure exact margins on `q3` and `q5`.

This uses the arm's **calibrated** kinematics. It matters: UR publishes DH
parameters that are **revision-controlled per serial-number range**, so any
nominal table is an approximation of your individual arm. The nominal values in
`kinematics.py` are labelled as such and are used only for offline checks.

Worth knowing: the UR wrist is **not spherical**. The `d5` offset means the three
wrist axes do not meet at a point, so textbook spherical-wrist shortcuts do not
apply here.

---

## Cost, not veto

The singularity term enters the controller's objective function rather than
gating the move afterwards:

```
cost = predicted_PSS + effort + overcorrection + LAMBDA_SINGULARITY * penalty
```

The penalty is zero while every margin is clear, then rises steeply and is capped
at 10. With `LAMBDA_SINGULARITY = 0.20` a near-singular candidate costs about
2.0, which no achievable PSS gain (PSS lives in [0, 1]) can buy.

**Why a cost.** The optimiser routes *around* singular regions while still
relieving strain. A veto would let it pick a target and then have the move
refused, which loses the intervention entirely. Measured effect, with the robot
near the base axis:

| | commanded `dy` | resulting radial distance |
|---|---|---|
| term off | +0.060 | 0.140 m (cylinder is 0.112 m) |
| term on | +0.012 | 0.188 m |

It still moves to relieve strain; it just stops short of the cylinder.

A hard refusal remains as an **execution backstop** for moves requested
directly rather than through the optimiser.

---

## moveL versus moveJ

| Motion | Command | Why |
|---|---|---|
| Baseline approach | **moveJ** | plans in joint space, needs no Jacobian inverse, so it cannot fail on a singularity along the path. Large motion, no person interaction yet. |
| Adaptive interventions | **moveL** | a straight, predictable Cartesian path matters when someone is close. Small moves, and the envelope keeps them clear of singular regions. |

`BASELINE_USE_MOVEJ = True`. If IK resolves the baseline to a singular
configuration, the move is refused with an explicit message rather than attempted.

---

## Setup

```python
class RobotConfig:
    AVOID_SINGULARITIES = True
    WORKING_RADIUS_M = 0.42       # MEASURE THIS on the rig
    USE_ROBOT_IK = True
    BASELINE_USE_MOVEJ = True
```

`WORKING_RADIUS_M` is a **measurement**, not a derived number. Sampling the
nominal chain gives TCP distances well past the published 500 mm figure because
the wrist offsets stack in extended configurations, so do not compute it from DH.
Jog the arm to the furthest point the artifact is actually worked and record the
distance from the base origin.

Pre-flight, before the first participant:

```python
robot = make_robot(simulate=True)
for problem in robot.check_workspace():
    print(problem)
```

This flags an envelope that reaches into the central cylinder, one that extends
past the working radius, and a `BASELINE_POSE` that is itself near-singular. The
shipped placeholder envelope deliberately fails this check.

### Choosing a good baseline pose

- Keep `q5` well away from 0 and ±π. If the artifact must be held flat, offset
  the tool direction rather than pointing it straight down; this is UR's own
  recommendation.
- Keep the elbow visibly bent; do not start near full extension.
- Put the artifact in the singularity-free annulus from
  `suggest_working_annulus`.
- After moving to baseline, check `UR3Robot.actual_margins()` and record it.

If the layout forces work near the central cylinder, UR's other suggestion is to
mount the base on a horizontal surface, which rotates the cylinder from vertical
to horizontal and may move it out of the task area.

---

## What is logged

Every intervention records whether the envelope clamped it and whether it was
refused for a singularity:

```
clamped=True applied=[...] singularity=shoulder
```

So the analysis can censor interventions the robot could not fully execute,
rather than treating a refused move as a delivered one.

---

## Limits

- Layer 1 uses the **TCP** rather than the wrist centre, so the shoulder term is
  approximate. Thresholds carry margin and layer 2 is the exact check.
- Nominal DH is not your arm. Keep `USE_ROBOT_IK = True` for live runs.
- Joint limits and self-collision are **not** handled here. UR's own joint limits
  can permit configurations where links interfere; that is the robot controller's
  responsibility and the Pilot Factory's risk assessment, not this module's.
- None of this substitutes for the collaborative-mode and speed limits set by the
  cell's assessment under ISO 10218 / ISO TS 15066.
