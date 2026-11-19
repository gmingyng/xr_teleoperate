"""Print cloud actions and drive only the local hanging posture with --execute."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import numpy as np
from unitree_openpi.scripts._client import launch


def hanging_step(controller, expires):
    from unitree_openpi.robot_ctrl.postures import HANGING
    state = controller.snapshot()
    motion = controller.config["motion"]
    delta = HANGING - state["q14"]
    distance = float(np.max(np.abs(delta)))
    step = min(motion["max_joint_step_rad"], motion["max_speed_rad_s"] / motion["command_hz"])
    target = HANGING if distance <= step else state["q14"] + delta * (step / distance)
    return controller.apply_joints(target, expires)


def handle_response(controller, response, sequence, expires, stop):
    if "actions" not in response:
        raise ValueError("policy response is missing actions")
    print(f"Cloud actions [{sequence}]: {response['actions']}", flush=True)
    deadline = min(expires, time.monotonic() + controller.config["motion"]["action_timeout_s"])
    status = hanging_step(controller, deadline)
    controller.last_action = dict(status, observation_id=sequence, target="local_hanging")


def main(argv=None):
    return launch(argv, prompt="test", handle_response=handle_response, control_waist=False)


if __name__ == "__main__":
    raise SystemExit(main())
