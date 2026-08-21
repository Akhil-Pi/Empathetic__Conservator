"""
goal_controller.py
==================
Part 3: goal-based controller. Replaces the reactive nudge policy in
intervention_policy.py.

Old behaviour: when PSS crossed a threshold, nudge the artifact by a fixed small
step (2 cm, 0.08 rad), wait out a cooldown, nudge again. It never computed a
target, so movement was timid and oscillated ("faint left-right").

New behaviour: when strain is sustained above threshold, compute the artifact
pose that minimises the person's PREDICTED PSS_v2, then execute one smooth move
to that pose. The move is as large as needed to reach the ergonomic optimum,
regularised so it stays smooth, and clamped to safe bounds and per-move step
limits.

The human-response model is analytical and local (a linear Jacobian mapping
artifact degrees of freedom to changes in the person's joint angles). Its gains
live in ControllerConfig.GAINS and are the quantities to refine from the August
data; the control architecture does not change when they do.

Consistency note: this controller responds to the angles PSS_v2 actually scores
(trunk/neck flexion, side-bend, twist). It does NOT rotate for gaze/head-turn,
unlike the reactive policy in the submitted paper, because PSS_v2 is RULA
grounded and gaze is not a RULA strain factor. This is a deliberate behavioural
change and should be disclosed in the follow-up study.

The optimiser and model are pure and unit-tested here with synthetic postures
(run `python goal_controller.py`); no robot or camera required.
"""

from __future__ import annotations

import time
import logging
from dataclasses import replace
from typing import Optional, Dict, Tuple
import numpy as np

from pss_v2 import PSSv2Calculator, PostureAngles

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Config. The GAINS block is the analytical human-response model; refine it
# from August data. Everything else is standard safety / timing.
# --------------------------------------------------------------------------

class ControllerConfig:
    # Degrees of freedom the controller can move, and the per-move step cap
    # (metres for translations, radians for rotations). These bound a single
    # intervention so no move is ever a large jump.
    DOF = ["dz", "dy", "dtilt", "drot", "dx"]
    STEP_LIMIT = {
        "dz":    0.03,   # raise / lower, m
        "dy":    0.02,   # toward / away from worker, m
        "dtilt": 0.10,   # end-effector tilt, rad
        "drot":  0.08,   # rotation about vertical, rad
        "dx":    0.02,   # lateral, m
    }

    # Linear human-response model: how much each DOF reduces each scored angle.
    # Units: degrees of angle reduction per unit of DOF (per metre or per radian).
    # Positive gain means increasing that DOF reduces that angle.
    # THESE ARE GEOMETRIC ESTIMATES. Refine from August data (Part 5).
    GAINS = {
        "trunk_flexion_deg":  {"dz": 150.0, "dy": 100.0},
        "neck_flexion_deg":   {"dz": 60.0,  "dtilt": 40.0},
        "trunk_sidebend_deg": {"dx": 80.0},
        "neck_sidebend_deg":  {"dx": 40.0, "drot": 30.0},
        "trunk_twist_deg":    {"drot": 20.0},
        "neck_twist_deg":     {"drot": 50.0},
    }

    # Cost weights. PSS lives in [0, 1]. The effort penalty must stay well below
    # the PSS reductions a move buys, or the controller under-moves under high
    # strain (the faint-movement failure). At a full single-DOF step the effort
    # cost is only LAMBDA_EFFORT, so it breaks ties near neutral without ever
    # suppressing a decisive corrective move.
    LAMBDA_EFFORT = 0.01
    OVERCORRECT_PENALTY = 0.03  # per deg any flexion is driven past neutral
                               # (stops the arm from over-raising into extension)

    # Rotation is preferred over translation whenever either could relieve the
    # same strain: dz/dtilt both reduce neck_flexion_deg, dx/drot both reduce
    # neck_sidebend_deg (see GAINS). Weighting translation's effort higher
    # makes the optimiser reach for rotation first and only spend translation
    # on angles rotation genuinely cannot touch (trunk_flexion, trunk_sidebend).
    # Tuning: 1.0 = no preference (translation and rotation compete equally).
    # Raise the translation weights to push more of the shared correction onto
    # rotation; lower them toward 1.0 if rotation is being asked to do more
    # than its STEP_LIMIT/GAINS can actually deliver and translation should
    # pick up the slack sooner. At 2.5, drot fires at its full step on the
    # first trigger while dx only takes the residual rotation can't share.
    EFFORT_WEIGHT = {
        "dz": 2.5, "dy": 2.5, "dx": 2.5,     # translation
        "dtilt": 1.0, "drot": 1.0,           # rotation
    }

    # A retrigger that fires before the person's posture has actually improved
    # (they haven't had time to react, or GAINS overestimates the real effect)
    # would otherwise recompute and reapply the SAME full-size correction every
    # cycle, walking the artifact straight into the workspace/orientation
    # limits. Each unresolved retrigger within one episode (from the first
    # threshold-crossing to hysteresis-confirmed recovery) shrinks the step by
    # this factor, so repeated small moves taper off instead of piling up
    # without bound.
    # Tuning: 1.0 disables decay (every retrigger fires at full STEP_LIMIT
    # again -- the old walk-into-the-wall behaviour). Smaller values converge
    # faster (fewer retriggers before the step is negligible) but correct less
    # total strain per episode before giving up; 0.6 reaches ~0 by the 7th-8th
    # retrigger, which at SUSTAINED_S=COOLDOWN_S=2.0 is about 30 s into a
    # sustained episode. If real sessions show it giving up before the person
    # actually recovers, raise it toward 0.8; if it's still reaching the
    # envelope/orientation limits, lower it.
    #
    # The counter resets whenever pss drops back below THRESHOLD (a fresh
    # problem, not a continuation of the old one -- see evaluate()), so this
    # only compounds within ONE unresolved episode. Do not remove that reset:
    # without it the counter never comes back down across a whole session and
    # every move eventually shrinks to nothing, which is a real bug we hit
    # (steps became imperceptible and rotation looked like it had vanished
    # again, even though drot itself was working -- everything was decayed to
    # the same near-zero scale). EPISODE_DECAY_FLOOR below is a second,
    # independent safety net against the same failure mode.
    EPISODE_DECAY = 0.6

    # Belt-and-suspenders for the compounding-decay failure above: even if the
    # per-episode reset above is ever wrong (a new camera/PSS edge case, a
    # config change elsewhere), the step scale can never fall below this
    # fraction of STEP_LIMIT, so a move is always at least visibly happening.
    EPISODE_DECAY_FLOOR = 0.2

    # Direction the artifact moves to counter an observed lean/twist. The
    # optimiser picks the lateral/rotation MAGNITUDE; this sign sets the
    # physical direction. +1 means "shift/rotate opposite the observed sign"
    # (the intuitive counter-move). Confirm once on the rig: have someone lean
    # left and check the artifact tracks toward them, not away.
    LATERAL_SIGN = +1.0

    LAMBDA_SINGULARITY = 0.20   # weight on the singularity term. PSS lives in
                                # [0, 1] and the singularity penalty is capped
                                # at 10, so a near-singular candidate costs
                                # about 2.0: more than any achievable PSS gain.

    # Trigger logic (kept from the reactive policy; these parts were sound).
    THRESHOLD = 0.25
    HYSTERESIS = 0.08
    SUSTAINED_S = 2.0
    COOLDOWN_S = 2.0

    # Optimiser resolution.
    LINE_SEARCH_POINTS = 21
    COORD_DESCENT_PASSES = 3


# --------------------------------------------------------------------------
# Degree-of-freedom interlock.
#
# A DOF is only usable if at least one angle it acts on is actually measured.
# With a single side camera, side-bend and twist are unobservable, so lateral
# and rotational moves would be commanded from a signal that is structurally
# zero: the artifact would drift sideways in response to nothing. Disabling
# them is the honest behaviour and it is also the safer one.
# --------------------------------------------------------------------------

DOF_REQUIRES = {
    "dz":    ("trunk_flexion_deg", "neck_flexion_deg"),
    "dy":    ("trunk_flexion_deg",),
    "dtilt": ("neck_flexion_deg",),
    "dx":    ("trunk_sidebend_deg", "neck_sidebend_deg"),
    "drot":  ("trunk_twist_deg", "neck_twist_deg", "neck_sidebend_deg"),
}


def enabled_dof(measurable=None, cfg=ControllerConfig) -> list:
    """DOF that the current camera layout can justify commanding."""
    if measurable is None:
        from camera_config import CameraConfig
        measurable = CameraConfig.measurable()
    measurable = set(measurable)
    return [d for d in cfg.DOF
            if any(a in measurable for a in DOF_REQUIRES.get(d, ()))]


# --------------------------------------------------------------------------
# Human-response model and cost.
# --------------------------------------------------------------------------

def predict_angles(current: PostureAngles, delta: Dict[str, float],
                   cfg=ControllerConfig) -> PostureAngles:
    """
    Apply the linear response model: predicted_angle = current - sum(gain * dof).
    Flexions can be reduced toward neutral; side-bend and twist are magnitudes,
    floored at 0 (you cannot make them negative by moving the artifact).
    """
    out = {}
    for angle_name, couplings in cfg.GAINS.items():
        cur = getattr(current, angle_name)
        reduction = sum(gain * delta.get(dof, 0.0) for dof, gain in couplings.items())
        if angle_name.endswith("sidebend_deg") or angle_name.endswith("twist_deg"):
            out[angle_name] = max(0.0, abs(cur) - abs(reduction))
        else:
            out[angle_name] = cur - reduction
    return replace(
        current,
        trunk_flexion_deg=out.get("trunk_flexion_deg", current.trunk_flexion_deg),
        neck_flexion_deg=out.get("neck_flexion_deg", current.neck_flexion_deg),
        trunk_sidebend_deg=out.get("trunk_sidebend_deg", current.trunk_sidebend_deg),
        neck_sidebend_deg=out.get("neck_sidebend_deg", current.neck_sidebend_deg),
        trunk_twist_deg=out.get("trunk_twist_deg", current.trunk_twist_deg),
        neck_twist_deg=out.get("neck_twist_deg", current.neck_twist_deg),
    )


def _overcorrection(current: PostureAngles, predicted: PostureAngles) -> float:
    """Total degrees any flexion was driven below neutral (into the opposite strain)."""
    over = 0.0
    for a in ("trunk_flexion_deg", "neck_flexion_deg"):
        p = getattr(predicted, a)
        if p < 0.0:
            over += abs(p)
    return over


def ergonomic_cost(current: PostureAngles, delta: Dict[str, float],
                   pss_calc: PSSv2Calculator, cfg=ControllerConfig,
                   extra_cost=None) -> float:
    """
    PSS of the predicted posture, plus effort and overcorrection penalties, plus
    an optional `extra_cost(delta) -> float`.

    `extra_cost` is how singularity avoidance enters. robot_interface supplies a
    callable that scores how close the resulting robot pose would sit to a wrist,
    elbow or shoulder singularity. Making it a COST rather than a veto lets the
    optimiser route around singular regions while still relieving strain, instead
    of picking a target and then having the move refused at execution time.
    """
    predicted = predict_angles(current, delta, cfg)
    # Score the predicted posture directly (raw, uncalibrated single read).
    pss_pred = _score_once(predicted, pss_calc)
    effort = cfg.LAMBDA_EFFORT * sum(
        getattr(cfg, "EFFORT_WEIGHT", {}).get(d, 1.0) *
        (delta.get(d, 0.0) / cfg.STEP_LIMIT[d]) ** 2 for d in cfg.DOF)
    over = cfg.OVERCORRECT_PENALTY * _overcorrection(current, predicted)
    sing = cfg.LAMBDA_SINGULARITY * extra_cost(delta) if extra_cost else 0.0
    return pss_pred + effort + over + sing


def _score_once(angles: PostureAngles, pss_calc: PSSv2Calculator) -> float:
    """PSS_raw for a single posture, independent of the smoothing history.
    The scratch calculator is cached on the parent: the optimizer calls this
    ~300 times per intervention, and that compute time sits inside the measured
    intervention latency, so per-call construction is real latency."""
    scratch = getattr(pss_calc, "_scratch", None)
    if scratch is None:
        scratch = PSSv2Calculator(cfg=pss_calc.cfg, smoothing_window=1)
        pss_calc._scratch = scratch
    scratch._neutral = pss_calc._neutral
    scratch._calibrated = pss_calc._calibrated
    return scratch.compute(angles)["pss_raw"]


# --------------------------------------------------------------------------
# Optimiser: bounded coordinate descent. Deterministic, robust to the kinks in
# PSS_v2, numpy-only. Low DOF and a cheap cost make this fast enough per event.
# --------------------------------------------------------------------------

def _scaled_step_cfg(cfg, scale: float):
    """A cfg-like object whose STEP_LIMIT is scaled, everything else inherited.
    Used to taper repeated retriggers within one uncorrected episode (see
    ControllerConfig.EPISODE_DECAY) without touching the caller's cfg."""
    if scale >= 1.0:
        return cfg
    class _Scaled(cfg):
        STEP_LIMIT = {d: v * scale for d, v in cfg.STEP_LIMIT.items()}
    return _Scaled


def optimize_target(current: PostureAngles, pss_calc: PSSv2Calculator,
                    cfg=ControllerConfig, dof=None,
                    extra_cost=None) -> Tuple[Dict[str, float], float, float]:
    """
    Find the DOF delta minimising ergonomic_cost, within per-move step limits.

    `dof` restricts the search to the degrees of freedom the camera layout can
    justify; DOF outside it stay at exactly zero. `extra_cost` adds the
    singularity term. Returns (delta, cost_before, cost_after).
    """
    search = list(dof) if dof is not None else list(cfg.DOF)
    delta = {d: 0.0 for d in cfg.DOF}
    cost_before = ergonomic_cost(current, delta, pss_calc, cfg, extra_cost)

    for _ in range(cfg.COORD_DESCENT_PASSES):
        for d in search:
            lim = cfg.STEP_LIMIT[d]
            candidates = np.linspace(-lim, lim, cfg.LINE_SEARCH_POINTS)
            best_val, best_cost = delta[d], None
            for v in candidates:
                trial = dict(delta); trial[d] = float(v)
                c = ergonomic_cost(current, trial, pss_calc, cfg, extra_cost)
                if best_cost is None or c < best_cost:
                    best_cost, best_val = c, float(v)
            delta[d] = best_val

    cost_after = ergonomic_cost(current, delta, pss_calc, cfg, extra_cost)
    return delta, cost_before, cost_after


# --------------------------------------------------------------------------
# Controller: same evaluate(...) role as InterventionPolicy, so it drops into
# the session loop. Signature adds `angles` because the model needs them.
# --------------------------------------------------------------------------

class GoalBasedController:
    def __init__(self, condition: str = "experimental", pss_calc: Optional[PSSv2Calculator] = None,
                 cfg=ControllerConfig, measurable=None):
        assert condition in ("control", "experimental")
        self.condition = condition
        self.cfg = cfg
        self.pss_calc = pss_calc or PSSv2Calculator()
        # DOF the camera layout can justify. Anything else is held at zero.
        self.dof = enabled_dof(measurable, cfg)
        self.disabled_dof = [d for d in cfg.DOF if d not in self.dof]
        if self.disabled_dof:
            logger.info(f"[GOAL] DOF disabled by camera layout: "
                        f"{self.disabled_dof} (driving angles not measured)")
        self._above_since: Optional[float] = None
        self._above_since_wall: float = 0.0
        self._in_corrected = False
        self._last_intervention_at = float("-inf")
        self._count = 0
        self._episode_fires = 0

    def reset(self):
        self._above_since = None
        self._in_corrected = False
        self._last_intervention_at = float("-inf")
        self._count = 0
        self._episode_fires = 0

    def evaluate(self, pss_components: dict, angles: PostureAngles, robot,
                 now: Optional[float] = None) -> dict:
        now = now if now is not None else time.time()
        pss = pss_components.get("pss_smooth", 0.0)
        result = {"triggered": False, "reason": "", "interventions": [], "pss": pss}

        if self.condition == "control":
            result["reason"] = "control_group"
            return result

        # Hysteresis recovery.
        if self._in_corrected and pss < (self.cfg.THRESHOLD - self.cfg.HYSTERESIS):
            self._in_corrected = False
            self._above_since = None
            self._episode_fires = 0

        # Cooldown.
        if (now - self._last_intervention_at) < self.cfg.COOLDOWN_S:
            result["reason"] = "cooldown"
            return result

        # Below threshold: nothing to do. This also ends the current escalation
        # episode (see EPISODE_DECAY) -- the strain that started it is gone, so
        # a future trigger is a fresh problem, not a continuation, and should
        # get a full-size step again rather than an ever-shrinking leftover.
        if pss < self.cfg.THRESHOLD:
            self._above_since = None
            self._episode_fires = 0
            result["reason"] = "below_threshold"
            return result

        # Sustained-dwell timer (ignore momentary spikes).
        if self._above_since is None:
            self._above_since = now
            self._above_since_wall = time.time()
            result["reason"] = "monitoring"
            return result
        if (now - self._above_since) < self.cfg.SUSTAINED_S:
            result["reason"] = f"sustained({now - self._above_since:.2f}s)"
            return result

        # ---- trigger: solve for the target and move once ----
        t_arm = self._above_since
        t_arm_wall = self._above_since_wall
        # Ask the robot to score candidate moves for singularity proximity, so
        # the optimiser avoids singular regions rather than being refused later.
        extra_cost = getattr(robot, "singularity_cost", None)
        decay_scale = max(self.cfg.EPISODE_DECAY ** self._episode_fires,
                          self.cfg.EPISODE_DECAY_FLOOR)
        step_cfg = _scaled_step_cfg(self.cfg, decay_scale)
        self._episode_fires += 1
        delta, cost_before, cost_after = optimize_target(
            angles, self.pss_calc, step_cfg, dof=self.dof, extra_cost=extra_cost)

        # Pure predicted PSS for logging. The optimizer costs above include the
        # effort and overcorrection penalties, so they must not be reported as
        # PSS values.
        pss_before = _score_once(angles, self.pss_calc)
        pss_after = _score_once(predict_angles(angles, delta, self.cfg), self.pss_calc)

        moved = self._execute(delta, angles, robot)
        # latency is a physical measurement: always the real wall clock, even if
        # the trigger logic is driven by an injected `now` (replay / testing).
        latency = time.time() - t_arm_wall

        self._count += 1
        self._last_intervention_at = now
        self._above_since = None
        self._in_corrected = True

        result.update({
            "triggered": True,
            "reason": "intervention",
            "interventions": [(k, round(v, 4)) for k, v in delta.items() if abs(v) > 1e-4],
            "intervention_id": self._count,
            "delta": delta,
            "predicted_pss_before": round(pss_before, 4),
            "predicted_pss_after": round(pss_after, 4),
            "cost_before": round(cost_before, 4),
            "cost_after": round(cost_after, 4),
            "moved": moved,
            "t_arm": t_arm,
            "latency_s": latency,
            "disabled_dof": list(self.disabled_dof),
        })
        logger.info(f"[GOAL] #{self._count} pss={pss:.3f} "
                    f"delta={result['interventions']} "
                    f"predicted PSS {pss_before:.3f}->{pss_after:.3f} "
                    f"(cost {cost_before:.3f}->{cost_after:.3f})")
        return result

    def _execute(self, delta: Dict[str, float], angles: PostureAngles, robot) -> bool:
        """One smooth move to the target. Translations + tilt via move_relative;
        rotation about vertical via adjust_rotation. Lateral and rotation take
        their MAGNITUDE from the optimiser and their DIRECTION from the observed
        lean/twist, so the artifact tracks toward the strained side."""
        dz = delta.get("dz", 0.0)
        dy = delta.get("dy", 0.0)
        dtilt = delta.get("dtilt", 0.0)

        # Direction from observed signed lean/twist; magnitude from the optimiser.
        side_sign = np.sign(angles.trunk_sidebend_deg) or 1.0
        twist_sign = (np.sign(angles.trunk_twist_deg)
                      or np.sign(angles.neck_twist_deg)
                      or np.sign(angles.neck_sidebend_deg)
                      or 1.0)
        dx = abs(delta.get("dx", 0.0)) * (-side_sign) * self.cfg.LATERAL_SIGN
        drot = abs(delta.get("drot", 0.0)) * (-twist_sign) * self.cfg.LATERAL_SIGN

        ok = True
        if any(abs(v) > 1e-4 for v in (dx, dy, dz, dtilt)):
            ok = robot.move_relative(dx=dx, dy=dy, dz=dz, drx=dtilt, asynchronous=True)
        if abs(drot) > 1e-4:
            robot.adjust_rotation(drot)
        return bool(ok)


# --------------------------------------------------------------------------
# Self-test: synthetic postures, no robot. A minimal fake robot records moves.
# --------------------------------------------------------------------------

class _FakeRobot:
    def __init__(self):
        self.moves = []
    def move_relative(self, dx=0, dy=0, dz=0, drx=0, dry=0, drz=0, asynchronous=False):
        self.moves.append(("move_relative", dict(dx=dx, dy=dy, dz=dz, drx=drx)))
        return True
    def adjust_rotation(self, d):
        self.moves.append(("adjust_rotation", d)); return True


if __name__ == "__main__":
    pss = PSSv2Calculator()

    def show(name, ang):
        before = _score_once(ang, pss)
        delta, cb, ca = optimize_target(ang, pss)
        nz = {k: round(v, 3) for k, v in delta.items() if abs(v) > 1e-4}
        print(f"{name:<34} PSS {before:.3f} -> {ca:.3f}   move={nz}")

    print("optimiser direction and improvement checks:\n")
    show("neutral (should not move)",
         PostureAngles(trunk_flexion_deg=2, neck_flexion_deg=3,
                       upper_arm_flexion_deg=15, lower_arm_flexion_deg=90))
    show("high trunk flexion (raise/forward)",
         PostureAngles(trunk_flexion_deg=50, neck_flexion_deg=8,
                       upper_arm_flexion_deg=30, lower_arm_flexion_deg=85))
    show("high neck flexion (raise/tilt)",
         PostureAngles(trunk_flexion_deg=10, neck_flexion_deg=30,
                       upper_arm_flexion_deg=20, lower_arm_flexion_deg=90))
    show("neck twist (rotate)",
         PostureAngles(trunk_flexion_deg=8, neck_flexion_deg=8, neck_twist_deg=25,
                       upper_arm_flexion_deg=20, lower_arm_flexion_deg=90))
    show("trunk side-bend (lateral)",
         PostureAngles(trunk_flexion_deg=8, neck_flexion_deg=8, trunk_sidebend_deg=20,
                       upper_arm_flexion_deg=20, lower_arm_flexion_deg=90))
    show("combined deep strain",
         PostureAngles(trunk_flexion_deg=55, neck_flexion_deg=28, neck_twist_deg=20,
                       upper_arm_flexion_deg=90, lower_arm_flexion_deg=70))

    print("\nstep-limit check (extreme posture stays within caps):")
    delta, _, _ = optimize_target(
        PostureAngles(trunk_flexion_deg=85, neck_flexion_deg=35), pss)
    within = all(abs(delta[d]) <= ControllerConfig.STEP_LIMIT[d] + 1e-9 for d in ControllerConfig.DOF)
    print(f"  delta={ {k: round(v,3) for k,v in delta.items()} }  within limits: {within}")

    print("\nend-to-end evaluate() with dwell + fake robot:")
    ctrl = GoalBasedController(condition="experimental", pss_calc=pss)
    strained = PostureAngles(trunk_flexion_deg=50, neck_flexion_deg=25,
                             upper_arm_flexion_deg=40, lower_arm_flexion_deg=85)
    comp = {"pss_smooth": 0.45}
    robot = _FakeRobot()
    t = 100.0
    for step in range(4):
        r = ctrl.evaluate(comp, strained, robot, now=t)
        print(f"  t={t:.2f} reason={r['reason']:<16} triggered={r['triggered']}")
        t += 0.5
    print("  robot moves recorded:", len(robot.moves))
    print("  control condition is a no-op:",
          GoalBasedController('control', pss).evaluate(comp, strained, robot, now=t)["triggered"] is False)
