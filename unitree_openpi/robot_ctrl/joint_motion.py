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


HANGING = np.array([.0948, -.0242, .1446, 1.3979, -.2679, -.0964, -.1156,
                    .0739, .0234, .0317, 1.3795, -.2829, .0129, -.0980])


MOVE_SECONDS = 4.0
HOLD_SECONDS = 1.0
RAMP_SECONDS = 0.75


def swing_blend(elapsed):
    half_cycle = MOVE_SECONDS + HOLD_SECONDS
    phase = elapsed % (2 * half_cycle)
    moving_time = min(phase % half_cycle, MOVE_SECONDS)
    effective_time = MOVE_SECONDS - RAMP_SECONDS
    edge_time = min(moving_time, MOVE_SECONDS - moving_time)
    if edge_time < RAMP_SECONDS:
        distance = (edge_time - RAMP_SECONDS / math.pi *
                    math.sin(math.pi * edge_time / RAMP_SECONDS)) / 2
        progress = distance / effective_time
        if moving_time > MOVE_SECONDS / 2:
            progress = 1 - progress
    else:
        progress = (moving_time - RAMP_SECONDS / 2) / effective_time
    return progress if phase < half_cycle else 1 - progress


def run_swing(backend, kin, config, limits):
    hanging = check_target(HANGING, limits)
    reset = check_target(np.zeros(14), limits)
    motion = config["motion"]
    policy = config["policy"]
    default_internal = config["device"]["gripper_interface"] == "internal"

    def endpoint(key, default):
        value = policy.get(key)
        if value is None and default_internal:
            value = [default, default]
        result = np.asarray(value, dtype=float)
        if result.shape != (2,) or not np.isfinite(result).all():
            raise ValueError(f"configure two calibrated motor positions in policy.{key}")
        return result

    closed = endpoint("closed_q2", 0.0)
    opened = endpoint("open_q2", 2.3)
    if np.any(closed == opened):
        raise ValueError("each gripper needs distinct open and closed targets")
    arm_speed = motion["max_speed_rad_s"]
    grip_speed = motion["gripper_speed"]
    effective_time = MOVE_SECONDS - RAMP_SECONDS
    if (np.max(abs(reset - hanging)) / effective_time > arm_speed or
            np.max(abs(opened - closed)) / effective_time > grip_speed):
        raise ValueError("10-second cycle exceeds configured joint or gripper speed")
    state = backend.read()
    start_arm = check_target(state["q14"], limits)
    start_grip = state["grip_q2"].copy()
    kin.update_state(state)
    backend.acquire(start_grip, state["waist_q3"], kin.gravity(start_arm), control_waist=False)
    approach = max(.1, 1.5 * np.max(abs(start_arm-hanging)) / arm_speed,
                   1.5 * np.max(abs(start_grip-closed)) / grip_speed)
    start = time.monotonic()
    cycle_start = None
    while True:
        now = time.monotonic()
        state = backend.read()
        kin.update_state(state)
        if cycle_start is None:
            progress = min(1.0, (now-start) / approach)
            blend = progress**2 * (3 - 2*progress)
            arm = start_arm + blend * (hanging-start_arm)
            grip = start_grip + blend * (closed-start_grip)
            if progress == 1 and np.max(abs(state["q14"]-hanging)) <= motion["arrival_tolerance_rad"] and np.max(abs(state["grip_q2"]-closed)) <= motion["grip_tolerance"]:
                cycle_start = now
            elif now-start > approach + motion["settle_timeout_s"]:
                raise TimeoutError("arms or grippers did not reach the swing starting pose")
        else:
            blend = swing_blend(now - cycle_start)
            arm = hanging + blend * (reset-hanging)
            grip = closed + blend * (opened-closed)
        backend.command(arm, kin.gravity(state["q14"]), grip)
        time.sleep(max(0.0, 1 / motion["command_hz"] - (time.monotonic()-now)))


def main(mode):
    parser = argparse.ArgumentParser(description=f"G1D dual-arm {mode}; joint-space control without OpenPI/cameras/IK")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/g1d.json"))
    parser.add_argument("--interface", help="DDS network interface override")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true", help="publish commands to the real robot")
    group.add_argument("--mock", action="store_true", help="offline fake plant; no DDS")
    if mode == "reset":
        parser.add_argument("--release-on-finish", action="store_true", help="release after reset")
        parser.add_argument("--duration", type=float, default=5.0, help="minimum reset duration in seconds")
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
            if mode == "swing":
                run_swing(backend, kin, config, limits)
            else:
                run_motion(backend, kin, config, mode, duration, amplitude, period, cycles,
                           not (args.mock or args.release_on_finish), limits)
        except KeyboardInterrupt:
            print("Interrupted; releasing DDS. No powered hold after exit.", flush=True)
        finally:
            backend.hold()
            backend.close()
    return 0
