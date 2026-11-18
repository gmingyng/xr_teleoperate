"""G1-D dual-arm swing entry point."""
from pathlib import Path
import sys
import math
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unitree_openpi.robot_ctrl.joint_motion import check_target, motion_parser, motion_session
from unitree_openpi.robot_ctrl.kinematics import ARM_JOINT_NAMES


from unitree_openpi.robot_ctrl.postures import HANGING


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
    raised = np.zeros(14)
    for side in ("left", "right"):
        raised[ARM_JOINT_NAMES.index(f"{side}_shoulder_pitch_joint")] = math.radians(-30)
    raised = check_target(raised, limits)
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
    if (np.max(abs(raised - hanging)) / effective_time > arm_speed or
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
            arm = hanging + blend * (raised-hanging)
            grip = closed + blend * (opened-closed)
        backend.command(arm, kin.gravity(state["q14"]), grip)
        time.sleep(max(0.0, 1 / motion["command_hz"] - (time.monotonic()-now)))


def main():
    args = motion_parser("swing").parse_args()
    with motion_session(args) as session:
        run_swing(*session)


if __name__ == "__main__":
    main()
