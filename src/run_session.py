"""
run_session.py
==============
The one entry point for collecting a session. Replaces run_dual.py, which only
showed the camera wiring and had no robot, no controller, and no logging.

    python3 run_session.py --participant P01 --condition control      --simulate
    python3 run_session.py --participant P01 --condition experimental --simulate
    python3 run_session.py --participant P01 --condition experimental --live

Design points that matter for the study, not just the code:

MATCHED BASELINE. The robot moves the artifact to RobotConfig.BASELINE_POSE in
BOTH conditions and holds it there. In control it never moves again; in
experimental it adapts. This is the fix for the original confound, where the
artifact sat on a table in control and on the end effector in experimental, so
condition differences mixed adaptation with a different starting geometry.

BLINDING. Nothing in the console output tells the participant which condition
is running. Do not narrate the robot's behaviour to them.

FIXED CONFIG. Every governing parameter is snapshotted into the session's
_meta.txt. Do not edit config between participants; if you must, note it, and
analyse the groups separately.

The calibration window and the task loop are separate phases. Calibration is
NOT optional: the strain score, the trigger threshold, and every logged angle
assume it. A session that skipped calibration is a corrupted session.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("session")


def _det(detector, frame, ms):
    """Detect if this view exists, else None so fusion treats it as absent."""
    return detector.detect(frame, ms) if (detector is not None and frame is not None) else None


def _have_required(sf, ff, det_side, det_front) -> bool:
    """A frame is usable once every view the layout actually uses has arrived."""
    if det_side is not None and sf is None:
        return False
    if det_front is not None and ff is None:
        return False
    return True


def build_stack(condition: str, simulate: bool, model_path: str,
                side_index=None, front_index=None):
    """Construct every component from the ACTIVE CAMERA LAYOUT. Imported here so
    --help works without cv2 installed."""
    from camera_stream import DualCamera, list_cameras
    from pose_fusion import PoseDetectorV2, fuse, apply_layout_mask
    from pss_v2 import PSSv2Calculator
    from goal_controller import GoalBasedController
    from robot_interface import make_robot
    import camera_config as cc

    layout = cc.CameraConfig.layout()
    roles = {p.role for p in layout.placements}
    logger.info(cc.describe())
    for msg in cc.validate():
        logger.warning(f"camera config: {msg}")

    # Resolve sources: CLI overrides, then config, then probing.
    src = cc.CameraConfig.sources()
    side_src = side_index if side_index is not None else src.get("side")
    front_src = front_index if front_index is not None else src.get("front")
    if "oblique" in roles:            # oblique occupies the side channel
        side_src = side_src if side_src is not None else src.get("oblique")
    if "side" not in roles and "oblique" not in roles:
        side_src = None
    if "front" not in roles:
        front_src = None

    needed = layout.n_cameras
    have = sum(v is not None for v in (side_src, front_src))
    if have < needed:
        cams = list_cameras()
        logger.info(f"cameras that opened: {cams}")
        if len(cams) < needed:
            raise RuntimeError(
                f"layout {layout.name} needs {needed} camera(s); {len(cams)} found")
        if side_src is None and ("side" in roles or "oblique" in roles):
            side_src = cams[0]
        if front_src is None and "front" in roles:
            front_src = cams[-1]

    dual = DualCamera(side_source=side_src, front_source=front_src).start()
    det_side = PoseDetectorV2(model_path=model_path) if side_src is not None else None
    det_front = PoseDetectorV2(model_path=model_path) if front_src is not None else None
    pss = PSSv2Calculator()
    ctrl = GoalBasedController(condition=condition, pss_calc=pss,
                               measurable=layout.measurable)
    robot = make_robot(simulate=simulate)

    for problem in robot.check_workspace():
        logger.warning(f"workspace: {problem}")

    def fuse_masked(sw, fw):
        return apply_layout_mask(fuse(sw, fw), layout)

    return dual, det_side, det_front, pss, ctrl, robot, fuse_masked


def run(participant: str, condition: str, minutes: float, simulate: bool,
        model_path: str, log_dir: str, calib_s: float,
        side_index=None, front_index=None, preview=False):
    from session_logger_v2 import SessionLoggerV2

    dual, det_side, det_front, pss, ctrl, robot, fuse = build_stack(
        condition, simulate, model_path, side_index, front_index)
    log = SessionLoggerV2(participant, condition, log_dir=log_dir)

    try:
        skew = dual.sync_skew_ms()
        logger.info(f"camera skew: {skew:.1f} ms")
        if skew > 50:
            logger.warning("skew above 50 ms; check USB bus and MJPEG settings")

        # ---- matched baseline, BOTH conditions (the confound fix) ----
        logger.info("moving artifact to baseline pose")
        if not robot.move_to_baseline(asynchronous=False):
            raise RuntimeError("could not reach baseline pose; aborting")
        log.log_event("baseline_set", 0.0,
                      details=f"pose={[round(v,4) for v in robot.get_pose()]}")
        time.sleep(1.0)

        # ---- calibration ----
        logger.info(f"calibration: hold a neutral upright posture for {calib_s:.0f}s")
        samples = []
        t0 = time.time()
        base_ms = int(t0 * 1000)
        while time.time() - t0 < calib_s:
            (sf, _), (ff, _) = dual.read_pair()
            if not _have_required(sf, ff, det_side, det_front):
                time.sleep(0.005)
                continue
            ms = int(time.time() * 1000) - base_ms
            sw = _det(det_side, sf, ms)
            fw = _det(det_front, ff, ms)
            samples.append(fuse(sw, fw))
            if preview:
                import cv2
                if sf is not None:
                    f1 = cv2.resize(sf, (480, 360))
                    cv2.putText(f1, f"CALIBRATING... {len(samples)} samples",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow("side camera", f1)
                if ff is not None:
                    f2 = cv2.resize(ff, (480, 360))
                    cv2.putText(f2, f"CALIBRATING... {len(samples)} samples",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow("front camera", f2)
                cv2.waitKey(1)
            time.sleep(0.02)
        if len(samples) < 30:
            raise RuntimeError(f"only {len(samples)} calibration samples; "
                               f"check the cameras can see the participant")
        pss.calibrate_neutral(samples)
        log.log_calibration(pss._neutral)
        logger.info(f"calibrated on {len(samples)} samples")

        # ---- task ----
        logger.info("task phase started")
        t0 = time.time()
        base_ms = int(t0 * 1000)
        frames = 0
        while time.time() - t0 < minutes * 60:
            (sf, s_ts), (ff, f_ts) = dual.read_pair()
            if not _have_required(sf, ff, det_side, det_front):
                time.sleep(0.005)
                continue
            ms = int(time.time() * 1000) - base_ms
            side_world = _det(det_side, sf, ms)
            front_world = _det(det_front, ff, ms)
            angles = fuse(side_world, front_world)
            comp = pss.compute(angles)

            log.log_frame(angles, comp,
                          arm_side=getattr(angles, "arm_side", ""),
                          skew_ms=abs(s_ts - f_ts) * 1000.0)
            frames += 1

            # The controller derives motion DIRECTION from angle sign, so it
            # must see the same neutral-corrected angles that drive the PSS
            # score, not the raw fuse() output.
            result = ctrl.evaluate(comp, pss.calibrated_angles(angles), robot)
            if result.get("triggered"):
                lm = getattr(robot, "last_move", {}) or {}
                lr = getattr(robot, "last_rotation", {}) or {}
                command = result.get("command_delta", {}) or {}
                diag_angles = result.get("diagnostic_angles", {}) or {}
                log.log_event("intervention", comp["pss_smooth"], result=result,
                              details=f"clamped={lm.get('clamped')} "
                                      f"applied={lm.get('applied')} "
                                      f"ok={lm.get('ok')} "
                                      f"singularity={lm.get('singularity')} "
                                      f"rot_applied={lr.get('applied')} "
                                      f"rot_ok={lr.get('ok')} "
                                      f"rot_clamped={lr.get('clamped')} "
                                      f"rot_singularity={lr.get('singularity')} "
                                      f"policy={result.get('policy')} "
                                      f"command={command} "
                                      f"angles={diag_angles}")
                if lm.get("singularity"):
                    logger.warning(f"intervention refused near a "
                                   f"{lm['singularity']} singularity")

            # neutral console output: gives no cue about the condition
            print(f"  t={time.time()-t0:5.0f}s  frames={frames}", end="\r")

            if preview:
                import cv2
                def _draw(label, frame, world_dict, ang, comp):
                    if frame is None:
                        return
                    frame = cv2.resize(frame, (480, 360))
                    h, w = frame.shape[:2]
                    # draw skeleton from normalized landmarks
                    if world_dict and "_normalized" in world_dict:
                        nl = world_dict["_normalized"]
                        connections = [
                            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                            (0, 7), (0, 8), (7, 11), (8, 12),
                        ]
                        pts = {}
                        for lm_idx in set(sum(connections, ())):
                            if lm_idx < len(nl):
                                lm = nl[lm_idx]
                                pts[lm_idx] = (int(lm.x * w), int(lm.y * h))
                        for a, b in connections:
                            if a in pts and b in pts:
                                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1)
                        for idx, pt in pts.items():
                            cv2.circle(frame, pt, 3, (0, 255, 0), -1)
                    # overlay
                    pss = comp['pss_smooth']
                    color = (0, 255, 0) if pss < 0.25 else \
                            (0, 165, 255) if pss < 0.5 else (0, 0, 255)
                    info = [
                        f"trunk: {ang.trunk_flexion_deg:.1f}",
                        f"neck:  {ang.neck_flexion_deg:.1f}",
                        f"side:  {ang.trunk_sidebend_deg:.1f}",
                        f"PSS:   {pss:.3f}",
                    ]
                    cv2.rectangle(frame, (0, 0), (200, 25 + len(info) * 22), (0, 0, 0), -1)
                    for i, txt in enumerate(info):
                        cv2.putText(frame, txt, (8, 20 + i * 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.imshow(label, frame)
                _draw("side camera", sf, side_world, angles, comp)
                _draw("front camera", ff, front_world, angles, comp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("\npreview closed by user")
                    break

        logger.info(f"\ntask phase complete: {frames} frames")

    except KeyboardInterrupt:
        logger.info("\ninterrupted by operator; closing session cleanly")
    finally:
        if preview:
            import cv2
            cv2.destroyAllWindows()
        try:
            import camera_config as _cc
            log.close(notes=f"condition={condition}; simulate={simulate}\n"
                            f"{_cc.describe()}")
        except Exception as e:
            logger.error(f"log close failed: {e}")
        for closer in (lambda: dual.stop(),
                       lambda: det_side and det_side.close(),
                       lambda: det_front and det_front.close(),
                       lambda: robot.close()):
            try:
                closer()
            except Exception:
                pass


def main(argv=None):
    p = argparse.ArgumentParser(description="Empathetic Conservator session")
    p.add_argument("--participant", required=True, help="e.g. P01")
    p.add_argument("--condition", required=True, choices=["control", "experimental"])
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--calibration-seconds", type=float, default=10.0)
    p.add_argument("--model", default="pose_landmarker.task")
    p.add_argument("--log-dir", default="data/sessions")
    p.add_argument("--camera-layout", default=None,
                   help="override CameraConfig.LAYOUT, e.g. SIDE_ONLY")
    p.add_argument("--side-index", type=int, default=None)
    p.add_argument("--front-index", type=int, default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--simulate", action="store_true", default=True,
                   help="no robot hardware (default)")
    g.add_argument("--live", dest="simulate", action="store_false",
                   help="drive the real UR3 over RTDE")
    p.add_argument("--preview", action="store_true", default=False,
                   help="show live camera feeds with pose skeleton overlay")
    a = p.parse_args(argv)

    if a.camera_layout:
        import camera_config as cc
        if a.camera_layout not in cc.LAYOUTS:
            raise SystemExit(f"unknown layout {a.camera_layout!r}; "
                             f"choose from {sorted(cc.LAYOUTS)}")
        cc.CameraConfig.LAYOUT = a.camera_layout

    if not a.simulate:
        logger.warning("LIVE MODE: the UR3 will move. Confirm the workspace is "
                       "clear and RobotConfig has been set for this rig.")
    run(a.participant, a.condition, a.minutes, a.simulate, a.model,
        a.log_dir, a.calibration_seconds, a.side_index, a.front_index, preview=a.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
