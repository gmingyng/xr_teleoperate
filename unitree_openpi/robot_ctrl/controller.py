"""Absolute joint-state observations and bounded joint-target execution."""

import time

import numpy as np

from ..utils.transforms import vector


def parse_actions(response, settings):
    """Reject a complete malformed chunk before sending its first command."""
    if "actions" not in response:
        raise ValueError("policy response is missing actions")
    if isinstance(response["actions"], dict):
        raise ValueError("actions must be an absolute joint array")
    actions = np.asarray(response["actions"], dtype=float)
    width = 16 if settings["include_grippers"] else 14
    if actions.ndim == 1:
        actions = actions[None, :]
    if (actions.ndim != 2 or actions.shape[1] != width or not len(actions) or
            len(actions) > settings["max_chunk_steps"] or not np.isfinite(actions).all()):
        raise ValueError(f"actions must be a finite (T,{width}) array")
    if settings["include_grippers"]:
        validate_grip_targets(actions[:, [7, 15]], settings)
    return actions


def validate_grip_targets(grip, settings):
    lower, upper = np.asarray(settings["gripper_command_limits"], dtype=float)
    # Accept the wire's float32 representation of an endpoint, then send the exact bound.
    if np.any((grip < np.minimum(lower, lower.astype(np.float32))) |
              (grip > np.maximum(upper, upper.astype(np.float32)))):
        raise ValueError("gripper motor target outside calibrated limits")
    return np.clip(grip, lower, upper)


class JointController:
    def __init__(self, config, backend, kinematics, execute=False, *, control_waist=True):
        self.config, self.backend, self.kin = config, backend, kinematics
        self.execute = execute
        self.control_waist = control_waist
        self.last_action = {"status": "idle"}

    def connect(self):
        self.backend.connect()
        state = self.snapshot()
        if self.execute:
            self.backend.acquire(state["grip_q2"], state["waist_q3"],
                                 self.kin.gravity(state["q14"]), control_waist=self.control_waist)

    def snapshot(self):
        state = self.backend.read()
        self.kin.update_state(state)
        return state

    def state_vector(self, state):
        values = []
        settings = self.config["policy"]
        for index in range(2):
            values.extend(state["q14"][7*index:7*index+7])
            if settings["include_grippers"]:
                values.append(state["grip_q2"][index])
        return np.asarray(values, dtype=np.float32)

    def apply(self, action, expires_at):
        settings = self.config["policy"]
        row = parse_actions({"actions": action}, settings)[0]
        if settings["include_grippers"]:
            return self.apply_joints(np.r_[row[:7], row[8:15]], expires_at, row[[7, 15]])
        return self.apply_joints(row, expires_at)

    def apply_joints(self, joints, expires_at, grip=None):
        settings, motion = self.config["policy"], self.config["motion"]
        if time.monotonic() >= expires_at:
            raise TimeoutError("joint target expired")
        q = vector(joints, 14, "absolute arm joint positions")
        state = self.snapshot()
        for side, target in self.kin.fk(q).items():
            lower, upper = np.asarray(self.config["model"]["workspace"][side])
            if np.any(target[:3, 3] < lower) or np.any(target[:3, 3] > upper):
                raise ValueError(f"{side} target exceeds configured EEF workspace")
        if np.max(np.abs(q - state["q14"])) > motion["max_joint_step_rad"]:
            raise ValueError("joint step exceeds max_joint_step_rad")
        tau = self.kin.gravity(q)
        if time.monotonic() >= expires_at:
            raise TimeoutError("action expired during target validation")
        # Recheck feedback after a potentially slow numerical solve.
        latest = self.backend.read()
        if np.max(np.abs(latest["q14"] - state["q14"])) > motion["arrival_tolerance_rad"]:
            raise RuntimeError("robot moved during target validation; target must be replanned")
        if np.max(np.abs(latest["waist_q3"] - state["waist_q3"])) > motion["waist_tolerance_rad"]:
            raise RuntimeError("waist moved during target validation; target must be replanned")
        if grip is not None:
            grip = vector(grip, 2, "absolute gripper motor positions")
            grip = validate_grip_targets(grip, settings)
        if not self.execute:
            return {"status": "observing", "executed": False}
        self.backend.command(q, tau, grip)
        deadline = min(expires_at, time.monotonic() + motion["settle_timeout_s"])
        while True:
            measured = self.backend.read()
            arm_ok = np.max(np.abs(measured["q14"] - q)) <= motion["arrival_tolerance_rad"]
            grip_ok = grip is None or np.max(np.abs(measured["grip_q2"] - grip)) <= motion["grip_tolerance"]
            if arm_ok and grip_ok:
                return {"status": "reached", "executed": True,
                        "q14": measured["q14"].tolist()}
            if time.monotonic() >= deadline:
                self.backend.freeze()
                raise TimeoutError("action timed out before measured arrival")
            time.sleep(1.0 / motion["publish_hz"])

    def freeze(self):
        if self.execute:
            self.backend.freeze()

    def close(self):
        try:
            if self.execute:
                self.backend.hold()
        finally:
            self.backend.close()
