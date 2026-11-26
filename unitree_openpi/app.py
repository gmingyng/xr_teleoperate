"""Robot-side OpenPI client: observe -> infer -> execute -> measured feedback."""

import logging
import numpy as np
import threading
import time

from .robot_ctrl.controller import parse_policy_chunk
from .utils.websocket import PolicyClient

log = logging.getLogger(__name__)


def observation(controller, cameras, config, sequence, prompt):
    state, poses = controller.snapshot()
    frames = cameras.snapshot()
    settings = config["policy"]
    result = {settings["state_key"]: controller.state_vector(state, poses),
              settings["prompt_key"]: prompt}
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
        "cameras": {name: {key: value for key, value in frame.items() if key not in ("image", "review_right_eye")}
                    for name, frame in frames.items()},
        "last_action": controller.last_action,
    }
    return result, earliest


def execute_policy(controller, response, sequence, expires, stop, *, recorder=None):
    config = controller.config
    settings = config["policy"]
    if settings.get("action_space") == "joint":
        actions = np.asarray(response["actions"], dtype=float)
        if (actions.ndim != 2 or actions.shape[1] != 16 or
                not 1 <= len(actions) <= settings["max_chunk_steps"] or not np.isfinite(actions).all()):
            raise ValueError("expected a finite (T,16) absolute joint action chunk")
        if not controller.execute:
            if recorder is not None:
                recorder.target = actions[0].copy()
            print(f"Cloud actions [{sequence}] shape={actions.shape} first={actions[0]}", flush=True)
            controller.last_action = {"status": "observing", "executed": False}
            return
        period = 1.0 / settings.get("target_hz", 50.0)
        chunk = actions[:settings["execute_steps"]]
        for i, row in enumerate(chunk):
            if stop.is_set():
                break
            started = time.monotonic()
            deadline = min(expires, started + config["motion"]["action_timeout_s"])
            status = controller.apply(None, deadline, joint_positions=np.r_[row[:7], row[8:15]],
                                      gripper_positions=row[[7, 15]], wait_for_arrival=False)
            if recorder is not None:
                recorder.target = row.copy()
            controller.last_action = dict(status, observation_id=sequence, step=i)
            if i + 1 < len(chunk):
                stop.wait(max(0.0, min(started + period, expires) - time.monotonic()))
        return
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
        controller.last_action = dict(status, observation_id=sequence, step=index,
                                      discarded_steps=length-count)
        if index + 1 < count:
            stop.wait(1.0 / config["motion"]["command_hz"])


def _infer_loop(config, controller, cameras, client, prompt, max_steps, stop, handle_response, scanner=None):
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
            obs, observed = observation(controller, cameras, config, sequence, prompt)
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
            handle_response(controller, response, sequence, expires, action_stop)
            completed += 1
            log.info("Observation %d: %s", sequence, controller.last_action["status"])
        except Exception as exc:
            controller.freeze()
            controller.last_action = {"status": "error", "observation_id": sequence,
                                      "message": str(exc)}
            log.warning("Holding; inference/action failed: %s", exc)
            client.close()  # Discard delayed replies; reconnect with a new observation.
            if max_steps:
                raise
            stop.wait(config["policy"].get("reconnect_s", 1.0))
        period = 1.0 / config["motion"]["command_hz"]
        stop.wait(max(0, period - (time.monotonic() - started)))
    return completed


def run(config, controller, cameras, *, prompt="", max_steps=0, stop=None, handle_response=execute_policy, reset_first=False, record_dir=None):
    stop = threading.Event() if stop is None else stop
    client = PolicyClient(config["policy"])
    completed = 0

    def infer(scanner):
        if record_dir is None:
            return _infer_loop(config, controller, cameras, client, prompt, max_steps, stop, handle_response, scanner)
        from functools import partial
        from .utils.review_video import ReviewVideo
        with ReviewVideo(record_dir, cameras, controller, prompt) as recorder:
            handler = partial(handle_response, recorder=recorder)
            try:
                return _infer_loop(config, controller, cameras, client, prompt, max_steps, stop, handler, scanner)
            except KeyboardInterrupt:
                controller.freeze()
                raise

    def session():
        if not config["policy"].get("barcode_scanner"):
            return infer(None)
        from teleop.utils.recording_schema import BarcodeReceiver
        scanner = BarcodeReceiver()
        try:
            scanner.begin(time.monotonic_ns())
            return infer(scanner)
        finally:
            scanner.close()

    try:
        controller.connect()
        if not reset_first:
            cameras.start()
            return session()

        from .robot_ctrl.joint_motion import joint_limits
        from .scripts.reset_dual_arms import move_to_reset

        limits = joint_limits(config)
        cameras_started = False
        while not stop.is_set():
            move_to_reset(controller.backend, controller.kin, config, 5.0, limits)
            try:
                input("Holding reset. Enter starts inference; Ctrl+C exits. ")
            except EOFError:
                break
            if not cameras_started:
                cameras.start()
                cameras_started = True
            try:
                log.info("Inference active. Ctrl+C cancels the chunk and returns arms to reset.")
                completed += session()
            except KeyboardInterrupt:
                log.info("Inference cancelled; returning to reset.")
            controller.freeze()
            client.close()
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
