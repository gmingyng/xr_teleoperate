"""Robot-side OpenPI client: observe -> infer -> execute -> measured feedback."""

import logging
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
        "state_representation": "absolute_xyz_rotation_vector_radians",
        "action_representation": "delta_xyz_rotation_vector_radians",
        "joint_action_representation": "absolute_joint_radians_left7_right7",
        "action_frame": settings["action_frame"],
        "joint_positions": state["q14"].tolist(), "joint_velocities": state["dq14"].tolist(),
        "waist_q3": state["waist_q3"].tolist(), "grip_q2": state["grip_q2"].tolist(),
        "state_age_s": now - state_received,
        "cameras": {name: {key: value for key, value in frame.items() if key != "image"}
                    for name, frame in frames.items()},
        "last_action": controller.last_action,
    }
    return result, earliest


def execute_policy(controller, response, sequence, expires, stop):
    config = controller.config
    settings = config["policy"]
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


def run(config, controller, cameras, *, prompt="", max_steps=0, stop=None, handle_response=execute_policy):
    stop = threading.Event() if stop is None else stop
    client = PolicyClient(config["policy"])
    sequence, completed = 0, 0
    try:
        controller.connect()
        cameras.start()
        while not stop.is_set() and (not max_steps or completed < max_steps):
            started = time.monotonic()
            try:
                if client.connection is None:
                    metadata = client.connect()
                    log.info("Connected to policy (%s)", metadata)
                sequence += 1
                obs, observed = observation(controller, cameras, config, sequence, prompt)
                response = client.infer(obs)
                settings = config["policy"]
                if response.get("observation_id", sequence) != sequence:
                    raise ValueError("action observation_id does not match request")
                expires = observed + settings["max_observation_age_s"]
                if time.monotonic() >= expires:
                    raise TimeoutError("policy response is older than max_observation_age_s")
                handle_response(controller, response, sequence, expires, stop)
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
