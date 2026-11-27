"""Robot-side OpenPI client: observe -> infer -> execute -> measured feedback."""

import logging
import numpy as np
import threading
import time

from ..policy_protocol import parse_joint_actions, parse_policy_chunk, state_vector
from ..robot_ctrl.controller import ActionReplanRequired, JointStepLimitError
from ..utils.websocket import PolicyClient

log = logging.getLogger(__name__)


def observation(controller, cameras, config, sequence, prompt, *, last_action=None):
    state, poses = controller.snapshot()
    frames = cameras.snapshot()
    settings = config["policy"]
    result = {settings["state_key"]: state_vector(state, poses, settings)}
    if prompt and prompt.strip():
        result[settings["prompt_key"]] = prompt
    images = {settings["image_keys"][name]: frame["image"] for name, frame in frames.items()}
    if settings.get("image_container_key"):
        result[settings["image_container_key"]] = images
    else:
        result.update(images)
    now = time.monotonic()
    state_received = state.get("received_monotonic_s", now)
    earliest = min(state_received, *(f["received_monotonic_s"] for f in frames.values()))
    result["_unitree"] = {
        "sequence": sequence, "received_at": time.time(), "frame_id": controller.kin.frame_id,
        "state_representation": "absolute_joint_motor" if settings.get("action_space") == "joint" else "absolute_xyz_rotation_vector_radians",
        "action_representation": "absolute_joint_motor" if settings.get("action_space") == "joint" else "delta_xyz_rotation_vector_radians",
        "joint_action_representation": "absolute_joint_radians_left7_right7",
        "action_frame": settings["action_frame"],
        "joint_positions": state["q14"].tolist(), "joint_velocities": state["dq14"].tolist(),
        "waist_q3": state["waist_q3"].tolist(), "grip_q2": state["grip_q2"].tolist(),
        "state_age_s": now - state_received,
        "cameras": {name: {key: value for key, value in frame.items() if key not in ("image", "head_views")}
                    for name, frame in frames.items()},
        "last_action": last_action if last_action is not None else {"status": "idle"},
    }
    return result, earliest


def execute_policy(controller, response, sequence, expires, stop, *, recorder=None):
    config = controller.config
    settings = config["policy"]
    last_action = None
    if settings.get("action_space") == "joint":
        actions = parse_joint_actions(response, max_steps=settings["max_chunk_steps"]).copy()
        actions[:, [7, 15]] -= 0.5
        if not controller.execute:
            if recorder is not None:
                recorder.target = actions[0].copy()
            print(f"Cloud actions [{sequence}] shape={actions.shape} first={actions[0]}", flush=True)
            return {"status": "observing", "executed": False}
        period = 1.0 / settings.get("target_hz", 50.0)
        chunk = actions[:settings["execute_steps"]]
        for i, row in enumerate(chunk):
            if stop.is_set():
                break
            started = time.monotonic()
            deadline = min(expires, started + config["motion"]["action_timeout_s"])
            try:
                status = controller.apply(None, deadline, joint_positions=np.r_[row[:7], row[8:15]],
                                          gripper_positions=row[[7, 15]], wait_for_arrival=False)
            except (ActionReplanRequired, JointStepLimitError) as exc:
                if settings.get("rtc"):
                    raise
                controller.freeze()
                last_action = {"status": "error", "observation_id": sequence, "step": i, "message": str(exc)}
                log.warning("Skipped observation %d row %d: %s", sequence, i, exc)
            else:
                if recorder is not None:
                    recorder.target = row.copy()
                last_action = dict(status, observation_id=sequence, step=i)
            if i + 1 < len(chunk):
                stop.wait(max(0.0, min(started + period, expires) - time.monotonic()))
        return last_action
    eef, joints = parse_policy_chunk(response, settings)
    if eef is not None:
        controller.validate_deltas(eef)
    length = len(eef if eef is not None else joints)
    count = min(length, settings["execute_steps"])
    for index in range(count):
        if stop.is_set():
            break
        status = controller.apply(None if eef is None else eef[index], min(expires,
            time.monotonic() + config["motion"]["action_timeout_s"]),
            joint_positions=None if joints is None else joints[index])
        last_action = dict(status, observation_id=sequence, step=index, discarded_steps=length-count)
        if index + 1 < count:
            stop.wait(1.0 / config["motion"]["command_hz"])
    return last_action


def rtc_loop(config, controller, cameras, client, prompt, max_steps, stop, handle_response, scanner, feedback):
    from concurrent.futures import ThreadPoolExecutor
    import math

    settings = config["policy"]
    period = 1 / settings["target_hz"]
    sequence = completed = 0
    plan = pending = None
    discard_reply = False
    prefix_valid = False
    delay = 1
    worker = ThreadPoolExecutor(max_workers=1)
    try:
        spec = client.connect()["g1d"]
        expected = dict(rtc=True, action_space="joint", action_representation="absolute",
                        action_dim=16, state_dim=17 if scanner else 16, gripper_units="motor_position")
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError("RTC requires a matching G1-D joint server started with --rtc")
        horizon = spec["action_horizon"]
        if not 2 <= horizon <= settings["max_chunk_steps"]:
            raise ValueError("Invalid action horizon for RTC")
        chunk_len = min(horizon, settings["execute_steps"])
        prefix_len = settings["rtc_prefix_len"]
        if prefix_len < 1:
            raise ValueError("--rtc-prefix-len must be positive")
        replan_at = max(0, chunk_len - prefix_len)
        while not stop.is_set():
            now = time.monotonic()
            flag = bool(scanner and scanner.frame(time.monotonic_ns())["scan_success"])
            if plan is not None and (flag != plan_flag or now >= expires):
                controller.freeze()
                plan = None

            if pending is not None and pending.done():
                reply = pending.result()
                pending = None
                if discard_reply:
                    discard_reply = False
                    continue
                actions = parse_joint_actions(reply, max_steps=horizon, dtype=np.float32)
                if len(actions) != horizon or reply.get("observation_id", sequence) != sequence:
                    raise ValueError("Invalid RTC policy response")
                offset = max(row - overlap_start, int((now - requested) / period)) if overlapping else 0
                delay = max(1, math.ceil((now - requested) / period))
                if now >= deadline:
                    controller.freeze()
                    plan = None
                    log.warning("Discarding expired RTC reply; requesting a fresh observation")
                    continue
                if flag == requested_flag and offset < chunk_len:
                    plan, row, plan_flag, expires = actions[:chunk_len], offset, flag, deadline
                    origin = requested if overlapping else now
                    prefix_valid = True
                    plan_sequence = sequence
                    completed += 1

            if plan is not None:
                row = max(row, int((now - origin) / period))
                if row >= chunk_len:
                    controller.freeze()
                    plan = None
            finished = max_steps and completed >= max_steps
            if finished and plan is None:
                break

            if not finished and pending is None and (plan is None or row >= replan_at):
                sequence += 1
                obs, observed = observation(
                    controller, cameras, config, sequence, prompt, last_action=feedback.get("last_action"),
                )
                if scanner:
                    key = settings["state_key"]
                    obs[key] = np.concatenate(([np.float32(flag)], obs[key]))
                    obs["_unitree"]["scan_success"] = flag
                requested = time.monotonic()
                deadline = observed + settings["max_observation_age_s"]
                if plan is not None:
                    row = max(row, int((requested - origin) / period))
                overlapping = plan is not None and row < chunk_len
                if overlapping:
                    overlap_start = row
                guided = overlapping and prefix_valid
                if guided:
                    prefix = plan[row:].copy()
                    obs["rtc_prefix"] = prefix
                    obs["rtc_delay"] = min(delay, len(prefix))
                requested_flag = flag
                pending = worker.submit(client.infer, obs)

            if plan is None:
                stop.wait(min(period, 0.005))
                continue
            if row < chunk_len and time.monotonic() < expires:
                if not scanner or bool(scanner.frame(time.monotonic_ns())["scan_success"]) == plan_flag:
                    try:
                        status = handle_response(controller, {"actions": plan[row:row+1]}, plan_sequence,
                                                 expires, stop)
                    except (ActionReplanRequired, JointStepLimitError) as exc:
                        controller.freeze()
                        # Keep trying later rows, but do not guide with this plan again.
                        prefix_valid = False
                        if pending is not None and guided:
                            discard_reply = True
                        feedback["last_action"] = {"status": "error", "observation_id": plan_sequence,
                                                   "step": row, "message": str(exc)}
                        log.warning("RTC skipped observation %d row %d: %s",
                                    plan_sequence, row, exc)
                    else:
                        if status is not None:
                            feedback["last_action"] = status
            row += 1
            stop.wait(max(0, origin + row * period - time.monotonic()))
        return completed
    finally:
        controller.freeze()
        try:
            client.close()
        finally:
            worker.shutdown(wait=True, cancel_futures=True)


def _latched_stop(controller):
    health = controller.backend.health()
    return health.get("fault") or health.get("stopped")


def _hold_after_failure(config, controller, client, stop, exc):
    controller.freeze()
    try:
        client.close()
    except Exception as close_error:
        log.warning("Policy connection cleanup failed: %s", close_error)
    fault = _latched_stop(controller)
    if fault:
        log.error("DDS stop is latched (%s); holding last published targets until stopped: %s", fault, exc)
        stop.wait()
    else:
        log.warning("Holding; retrying from fresh feedback after %s: %s", type(exc).__name__, exc)
        stop.wait(config["policy"].get("reconnect_s", 1.0))


def _retry_holding(operation, config, controller, client, stop, max_steps, feedback):
    while not stop.is_set():
        try:
            return operation()
        except Exception as exc:
            if max_steps:
                raise
            feedback["last_action"] = {"status": "error", "message": str(exc)}
            _hold_after_failure(config, controller, client, stop, exc)
    return 0


def _infer_loop(config, controller, cameras, client, prompt, max_steps, stop, handle_response, scanner, feedback):
    if config["policy"].get("rtc"):
        return _retry_holding(
            lambda: rtc_loop(config, controller, cameras, client, prompt, max_steps, stop,
                             handle_response, scanner, feedback),
            config, controller, client, stop, max_steps, feedback,
        )
    sequence, completed = 0, 0
    while not stop.is_set() and (not max_steps or completed < max_steps):
        started = time.monotonic()
        try:
            if client.connection is None:
                metadata = client.connect()
                if config["policy"].get("action_space") == "joint":
                    spec = metadata.get("g1d", {})
                    if any(spec.get(key) != value for key, value in {
                        "action_space": "joint", "action_representation": "absolute", "state_dim": 17 if scanner else 16,
                        "action_dim": 16, "gripper_units": "motor_position",
                    }.items()):
                        raise ValueError("server does not advertise the absolute G1-D joint protocol")
                log.info("Connected to policy (%s)", metadata)
            sequence += 1
            obs, observed = observation(
                controller, cameras, config, sequence, prompt, last_action=feedback.get("last_action"),
            )
            flag = False
            if scanner:
                flag = scanner.frame(time.monotonic_ns())["scan_success"]
                obs["_unitree"]["scan_success"] = flag
                key = config["policy"]["state_key"]
                obs[key] = np.concatenate(([np.float32(flag)], obs[key]))
            sent = time.monotonic()
            response = client.infer(obs)
            log.info("Observation %d inference round trip %.1f ms", sequence, 1000 * (time.monotonic() - sent))
            settings = config["policy"]
            if response.get("observation_id", sequence) != sequence:
                raise ValueError("action observation_id does not match request")
            expires = observed + settings["max_observation_age_s"]
            if time.monotonic() >= expires and (controller.execute or settings.get("action_space") != "joint"):
                raise TimeoutError("policy response is older than max_observation_age_s")
            action_stop = stop
            if scanner and not flag:
                # A scan during inference or execution invalidates the old phase.
                class UntilScan:
                    def is_set(self):
                        return stop.is_set() or scanner.frame(time.monotonic_ns())["scan_success"]

                    def wait(self, timeout):
                        return stop.wait(timeout)
                action_stop = UntilScan()
                if action_stop.is_set():
                    continue
            status = handle_response(controller, response, sequence, expires, action_stop)
            if status is not None:
                feedback["last_action"] = status
            completed += 1
            log.info("Observation %d: %s", sequence, feedback.get("last_action", {}).get("status", "idle"))
        except Exception as exc:
            controller.freeze()
            feedback["last_action"] = {"status": "error", "observation_id": sequence,
                                      "message": str(exc)}
            if max_steps:
                raise
            _hold_after_failure(config, controller, client, stop, exc)
        period = 1.0 / config["motion"]["command_hz"]
        stop.wait(max(0, period - (time.monotonic() - started)))
    return completed


def run(config, controller, cameras, *, prompt="", max_steps=0, stop=None, handle_response=execute_policy, reset_first=False, record_dir=None):
    stop = threading.Event() if stop is None else stop
    client = PolicyClient(config["policy"])
    feedback = {}
    completed = 0

    def infer(scanner):
        if record_dir is None:
            return _infer_loop(config, controller, cameras, client, prompt, max_steps, stop, handle_response, scanner, feedback)
        from functools import partial
        from ._review_video import ReviewVideo
        with ReviewVideo(record_dir, cameras, controller, prompt) as recorder:
            handler = partial(handle_response, recorder=recorder)
            try:
                return _infer_loop(config, controller, cameras, client, prompt, max_steps, stop, handler, scanner, feedback)
            except KeyboardInterrupt:
                controller.freeze()
                raise

    def session():
        if not config["policy"].get("barcode_scanner"):
            return infer(None)
        from teleop.utils.barcode import BarcodeReceiver
        scanner = BarcodeReceiver()
        try:
            scanner.begin(time.monotonic_ns())
            return infer(scanner)
        finally:
            scanner.close()

    try:
        controller.connect()
        if not reset_first:
            _retry_holding(cameras.start, config, controller, client, stop, max_steps, feedback)
            return _retry_holding(session, config, controller, client, stop, max_steps, feedback)

        from ..robot_ctrl.joint_motion import joint_limits
        from .reset_dual_arms import move_to_reset

        limits = joint_limits(config)
        cameras_started = False
        while not stop.is_set():
            try:
                move_to_reset(controller.backend, controller.kin, config, 5.0, limits)
            except (OSError, RuntimeError, ValueError) as exc:
                if max_steps:
                    raise
                _hold_after_failure(config, controller, client, stop, exc)
                if stop.is_set():
                    break
                try:
                    input("Holding after a failed reset. Enter retries reset; Ctrl+C exits. ")
                except EOFError:
                    break
                continue
            try:
                input("Holding reset. Enter starts inference; Ctrl+C exits. ")
            except EOFError:
                break
            if not cameras_started:
                _retry_holding(cameras.start, config, controller, client, stop, max_steps, feedback)
                if stop.is_set():
                    break
                cameras_started = True
            try:
                log.info("Inference active. Ctrl+C cancels the chunk and returns arms to reset.")
                completed += _retry_holding(session, config, controller, client, stop, max_steps, feedback)
            except KeyboardInterrupt:
                if _latched_stop(controller):
                    raise
                log.info("Inference cancelled; returning to reset.")
            controller.freeze()
            client.close()
            if max_steps and completed >= max_steps:
                break
    finally:
        # Each resource is released even when another close operation fails.
        try:
            controller.freeze()
        finally:
            try:
                client.close()
            finally:
                try:
                    cameras.close()
                finally:
                    controller.close()
    return completed
