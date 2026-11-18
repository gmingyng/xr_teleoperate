"""Robot-side OpenPI client: observe -> infer -> execute -> measured feedback."""

import argparse
from contextlib import nullcontext
import logging
from pathlib import Path
import threading
import time

from .robot_ctrl.controller import EEFController, parse_policy_chunk
from .utils.cameras import CameraCapture, MockCameras
from .utils.config import load_config
from .utils.ownership import ControllerLease
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
        "state_representation": "absolute_xyz_euler_xyz_radians",
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


def run(config, controller, cameras, *, prompt="", max_steps=0, stop=None):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config/g1d.json"))
    parser.add_argument("--url", help="OpenPI policy ws://HOST:PORT URL")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--execute", action="store_true", help="acquire and execute on the real robot")
    parser.add_argument("--mock", action="store_true", help="synthetic robot/cameras; never connects to DDS")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N inferences; 0 means continuous")
    args = parser.parse_args(argv)
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if args.url:
        if not args.url.startswith(("ws://", "wss://")):
            parser.error("--url must start with ws:// or wss://")
        config["policy"]["url"] = args.url
    if args.mock:
        from .robot_ctrl.mock import MockBackend, MockKinematics
        backend, kin = MockBackend(config), MockKinematics()
        cameras = MockCameras(config["cameras"])
    else:
        from .robot_ctrl.dds import G1DDSBackend
        from .robot_ctrl.kinematics import G1DKinematics
        backend, kin = G1DDSBackend(config), G1DKinematics(config)
        cameras = CameraCapture(config["cameras"])
    controller = EEFController(config, backend, kin, execute=args.execute or args.mock)
    lease = ControllerLease(config["device"]["controller_lock"]) if args.execute and not args.mock else nullcontext()
    try:
        with lease:
            run(config, controller, cameras, prompt=args.prompt, max_steps=args.max_steps)
    except KeyboardInterrupt:
        log.info("Stopped")
    return 0
