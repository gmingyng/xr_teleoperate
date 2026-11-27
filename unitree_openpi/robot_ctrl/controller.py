"""Measured absolute EEF observations and bounded relative-action execution."""

import time

import numpy as np

from ..utils.transforms import relative_target, vector


class JointStepLimitError(ValueError):
    """A target exceeds the permitted change from measured arm positions."""


class ActionReplanRequired(RuntimeError):
    """The feedback used to calculate a target changed before dispatch."""


class EEFController:
    def __init__(self, config, backend, kinematics, execute=False, *, control_waist=True):
        self.config, self.backend, self.kin = config, backend, kinematics
        self.execute = execute
        self.control_waist = control_waist

    def connect(self):
        self.backend.connect()
        state, _ = self.snapshot()
        if self.execute:
            self.backend.acquire(state["grip_q2"], state["waist_q3"],
                                 self.kin.gravity(state["q14"]), control_waist=self.control_waist)

    def snapshot(self):
        state = self.backend.read()
        self.kin.update_state(state)
        poses = self.kin.fk(state["q14"])
        return state, poses

    def validate_deltas(self, actions):
        stride = 7 if self.config["policy"]["include_grippers"] else 6
        motion = self.config["motion"]
        for start in (0, stride):
            if np.any(np.linalg.norm(actions[:, start:start+3], axis=1) > motion["max_translation_m"]):
                raise ValueError("relative translation exceeds max_translation_m")
            if np.any(np.linalg.norm(actions[:, start+3:start+6], axis=1) > motion["max_rotation_rad"]):
                raise ValueError("relative rotation exceeds max_rotation_rad")

    def apply(self, action, expires_at, joint_positions=None, *, gripper_positions=None, wait_for_arrival=True):
        settings, motion = self.config["policy"], self.config["motion"]
        width, stride = (14, 7) if settings["include_grippers"] else (12, 6)
        if action is not None:
            action = vector(action, width, "relative EEF action")
            self.validate_deltas(action[None])
        if time.monotonic() >= expires_at:
            raise TimeoutError("action expired before IK")
        state, poses = self.snapshot()
        targets = {}
        if action is not None:
            targets = {side: relative_target(poses[side], action[start:start+6], settings["action_frame"])
                       for side, start in (("left", 0), ("right", stride))}
        if joint_positions is None:
            if not targets:
                raise ValueError("missing joint and EEF targets")
            q = None
        else:
            q = vector(joint_positions, 14, "absolute joint positions")
            actual = self.kin.fk(q)
            for side, target in targets.items():
                delta = target[:3, :3].T @ actual[side][:3, :3]
                angle = np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1))
                if (np.linalg.norm(actual[side][:3, 3]-target[:3, 3]) >
                        self.config["model"].get("position_tolerance_m", .003) or
                        angle > self.config["model"].get("rotation_tolerance_rad", .03)):
                    raise ValueError("joint and EEF targets disagree")
            targets = actual
        for side, target in targets.items():
            lower, upper = np.asarray(self.config["model"]["workspace"][side])
            if np.any(target[:3, 3] < lower) or np.any(target[:3, 3] > upper):
                raise ValueError(f"{side} target exceeds configured EEF workspace")
        if q is None:
            q = self.kin.solve(targets, state["q14"])
        tau = self.kin.gravity(q)
        if time.monotonic() >= expires_at:
            raise TimeoutError("action expired while validating the target")
        latest = self.backend.read()
        # Relative targets depend on the initial arm pose. Absolute targets do not.
        if (action is not None and
                np.max(np.abs(latest["q14"] - state["q14"])) > motion["arrival_tolerance_rad"]):
            raise ActionReplanRequired("robot moved during IK; target must be replanned")
        if np.max(np.abs(latest["waist_q3"] - state["waist_q3"])) > motion["waist_tolerance_rad"]:
            raise ActionReplanRequired("waist moved during target validation; target must be replanned")
        joint_step = np.abs(q - latest["q14"])
        largest_step = float(np.max(joint_step))
        if largest_step > motion["max_joint_step_rad"]:
            joint = int(np.argmax(joint_step))
            raise JointStepLimitError(
                f"joint step {largest_step:.3f} rad exceeds max_joint_step_rad "
                f"{motion['max_joint_step_rad']:g} at arm joint index {joint} "
                f"(measured {latest['q14'][joint]:.3f}, target {q[joint]:.3f})"
            )
        grip = None
        if gripper_positions is not None:
            grip = vector(gripper_positions, 2, "absolute gripper positions")
            lo, hi = np.asarray(settings["gripper_command_limits"], dtype=float)
            grip = np.clip(grip, lo, hi)
        if action is not None and settings["include_grippers"]:
            normalized = action[[6, 13]]
            if np.any((normalized < 0) | (normalized > 1)):
                raise ValueError("normalized gripper values must be in [0,1]")
            grip = np.array(settings["closed_q2"]) + normalized * (
                np.array(settings["open_q2"]) - settings["closed_q2"])
        if not self.execute:
            return {"status": "observing", "executed": False}
        if time.monotonic() >= expires_at:
            raise TimeoutError("action expired before dispatch")
        self.backend.command(q, tau, grip)
        if not wait_for_arrival:
            return {"status": "target_sent", "executed": True}
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
