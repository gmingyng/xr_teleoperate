"""Standalone joint exercises using the same DDS backend as the policy client."""
import argparse
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import ARM_JOINT_NAMES
from ..utils.config import load_config
from ..utils.ownership import ControllerLease


def joint_limits(config):
    joints = {j.attrib["name"]: j for j in ET.parse(config["model"]["urdf"]).getroot().findall("joint")}
    limits = [joints[name].find("limit").attrib for name in ARM_JOINT_NAMES]
    return np.array([float(j["lower"]) for j in limits]), np.array([float(j["upper"]) for j in limits])


def check_target(q, limits):
    q = np.asarray(q, dtype=float)
    lower, upper = limits
    if q.shape != (14,) or not np.isfinite(q).all() or np.any(q < lower) or np.any(q > upper):
        raise ValueError("joint target must contain 14 finite angles within URDF limits")
    return q.copy()


def swing_target(center, amplitude, phase):
    q = center.copy()
    # Both shoulder-pitch joints, left7/right7 ordering. Start/end at center.
    q[[0, 7]] += amplitude * math.sin(phase) ** 3
    return q


def run_motion(backend, kin, config, mode, duration, amplitude, period, cycles, hold, limits):
    dt = 1.0 / config["motion"]["command_hz"]
    def send(q):
        state = backend.read()
        kin.update_state(state)
        backend.command(check_target(q, limits), kin.gravity(state["q14"]))

    state = backend.read()
    center = check_target(state["q14"], limits)
    kin.update_state(state)
    if mode == "reset":
        goal = check_target(np.zeros(14), limits)
        # Cubic interpolation has peak speed 1.5 * displacement / duration.
        duration = max(duration, 1.5 * np.max(np.abs(goal-center)) / config["motion"]["max_speed_rad_s"])
    else:
        for phase in (math.pi / 2, 3 * math.pi / 2):
            check_target(swing_target(center, amplitude, phase), limits)
        if 3 * amplitude * 2 * math.pi / period > config["motion"]["max_speed_rad_s"]:
            raise ValueError("swing period too short for configured speed limit")
        duration = cycles * period
        goal = center
    backend.acquire(state["grip_q2"], state["waist_q3"], kin.gravity(center))
    start = time.monotonic()
    while True:
        elapsed = min(time.monotonic() - start, duration)
        if mode == "reset":
            fraction = elapsed / duration
            q = center + (goal-center) * (3*fraction**2 - 2*fraction**3)
        else:
            q = swing_target(center, amplitude, 2*math.pi*elapsed/period)
        send(q)
        if elapsed >= duration:
            break
        time.sleep(dt)
    deadline = time.monotonic() + config["motion"]["settle_timeout_s"]
    while True:
        send(goal)
        if np.max(np.abs(backend.read()["q14"] - goal)) <= config["motion"]["arrival_tolerance_rad"]:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("arms did not reach the final joint target")
        time.sleep(dt)
    print("Motion complete; holding target. Ctrl+C releases DDS." if hold else "Motion complete; releasing DDS.", flush=True)
    while hold:
        send(goal)
        time.sleep(dt)


def main(mode):
    parser = argparse.ArgumentParser(description=f"G1D dual-arm {mode}; joint-space control without OpenPI/cameras/IK")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/g1d.json"))
    parser.add_argument("--interface", help="DDS network interface override")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true", help="publish commands to the real robot")
    group.add_argument("--mock", action="store_true", help="offline fake plant; no DDS")
    parser.add_argument("--release-on-finish", action="store_true", help="close DDS after completion; no powered hold afterwards")
    if mode == "reset":
        parser.add_argument("--duration", type=float, default=5.0, help="minimum reset duration in seconds")
    else:
        parser.add_argument("--amplitude-deg", type=float, default=3.0)
        parser.add_argument("--period", type=float, default=6.0, help="seconds per cycle")
        parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    duration = getattr(args, "duration", 5.0)
    amplitude = math.radians(getattr(args, "amplitude_deg", 3.0))
    period = getattr(args, "period", 6.0)
    cycles = getattr(args, "cycles", 3)
    if not all(math.isfinite(x) and x > 0 for x in (duration, amplitude, period)) or cycles <= 0:
        parser.error("durations, amplitude and cycles must be finite and positive")
    if amplitude > math.radians(10):
        parser.error("small swing amplitude must not exceed 10 degrees")
    config = load_config(args.config)
    if args.interface:
        config["device"]["interface"] = args.interface
    limits = joint_limits(config)
    if args.mock:
        from .mock import MockBackend, MockKinematics
        backend, kin = MockBackend(config), MockKinematics()
    else:
        from .dds import G1DDSBackend
        from .kinematics import G1DKinematics
        backend, kin = G1DDSBackend(config), G1DKinematics(config)
    from contextlib import nullcontext
    lease = nullcontext() if args.mock else ControllerLease(config["device"]["controller_lock"])
    with lease:
        try:
            backend.connect()
            run_motion(backend, kin, config, mode, duration, amplitude, period, cycles,
                       not (args.mock or args.release_on_finish), limits)
        except KeyboardInterrupt:
            print("Interrupted; releasing DDS. No powered hold after exit.", flush=True)
        finally:
            backend.hold()
            backend.close()
    return 0
