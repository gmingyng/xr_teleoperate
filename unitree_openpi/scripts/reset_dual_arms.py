"""G1-D dual-arm reset entry point."""
from pathlib import Path
import sys
import math
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unitree_openpi.robot_ctrl.joint_motion import check_target, motion_parser, motion_session


def run_reset(backend, kin, config, duration, hold, limits):
    dt = 1.0 / config["motion"]["command_hz"]
    def send(q):
        state = backend.read()
        kin.update_state(state)
        backend.command(check_target(q, limits), kin.gravity(state["q14"]))

    state = backend.read()
    center = check_target(state["q14"], limits)
    kin.update_state(state)
    goal = check_target(np.zeros(14), limits)
    # Cubic interpolation has peak speed 1.5 * displacement / duration.
    duration = max(duration, 1.5 * np.max(np.abs(goal-center)) / config["motion"]["max_speed_rad_s"])
    backend.acquire(state["grip_q2"], state["waist_q3"], kin.gravity(center))
    start = time.monotonic()
    while True:
        elapsed = min(time.monotonic() - start, duration)
        fraction = elapsed / duration
        q = center + (goal-center) * (3*fraction**2 - 2*fraction**3)
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


def main():
    parser = motion_parser("reset")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--release-on-finish", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")
    with motion_session(args) as (backend, kin, config, limits):
        run_reset(backend, kin, config, args.duration, not (args.mock or args.release_on_finish), limits)


if __name__ == "__main__":
    main()
