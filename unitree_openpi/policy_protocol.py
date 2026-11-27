"""Encode measured G1-D state and validate OpenPI action payloads."""

import numpy as np
from scipy.spatial.transform import Rotation


def joint_state_vector(state):
    q, grip = state["q14"], state["grip_q2"]
    return np.asarray([*q[:7], grip[0], *q[7:], grip[1]], dtype=np.float32)


def state_vector(state, poses, settings):
    values = []
    if settings.get("action_space") == "joint":
        return joint_state_vector(state)
    for index, side in enumerate(("left", "right")):
        values.extend(poses[side][:3, 3])
        values.extend(Rotation.from_matrix(poses[side][:3, :3]).as_rotvec())
        if settings["include_grippers"]:
            closed, opened = settings["closed_q2"][index], settings["open_q2"][index]
            values.append(float(np.clip((state["grip_q2"][index]-closed)/(opened-closed), 0, 1)))
    return np.asarray(values, dtype=np.float32)


def parse_actions(response, settings):
    """Reject a complete malformed chunk before sending its first command."""
    if "actions" not in response:
        raise ValueError("policy response is missing actions")
    actions = np.asarray(response["actions"], dtype=float)
    width = 14 if settings["include_grippers"] else 12
    if actions.ndim == 1:
        actions = actions[None, :]
    if (actions.ndim != 2 or actions.shape[1] != width or not len(actions) or
            len(actions) > settings["max_chunk_steps"] or not np.isfinite(actions).all()):
        raise ValueError(f"actions must be a finite (T,{width}) array")
    if settings["include_grippers"] and np.any((actions[:, [6, 13]] < 0) |
                                                               (actions[:, [6, 13]] > 1)):
        raise ValueError("gripper actions must be absolute normalized values in [0,1]")
    return actions


def parse_policy_chunk(response, settings):
    """Accept standard OpenPI EEF arrays or explicitly named joint/EEF trajectories."""
    payload = response.get("actions")
    if not isinstance(payload, dict):
        return parse_actions(response, settings), None
    if not payload or set(payload) - {"eef", "joint_positions"}:
        raise ValueError("named actions require eef and/or joint_positions")
    eef = parse_actions({"actions": payload["eef"]}, settings) if "eef" in payload else None
    joints = None
    if "joint_positions" in payload:
        joints = np.asarray(payload["joint_positions"], dtype=float)
        if joints.ndim == 1:
            joints = joints[None, :]
        if (joints.ndim != 2 or joints.shape[1] != 14 or not np.isfinite(joints).all()
                or not 1 <= len(joints) <= settings["max_chunk_steps"]):
            raise ValueError("joint_positions must be a finite (T,14) array")
        if eef is not None and len(eef) != len(joints):
            raise ValueError("joint and EEF trajectories must have equal length")
    return eef, joints



def parse_joint_actions(response, *, max_steps=None, dtype=float):
    """Validate interleaved arm and gripper motor positions from a joint policy."""
    payload = response.get("actions")
    error = "expected a finite (T,16) absolute joint action chunk"
    if payload is None or isinstance(payload, dict):
        raise ValueError(error)
    try:
        actions = np.asarray(payload, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if (actions.ndim != 2 or actions.shape[1] != 16 or not len(actions)
            or (max_steps is not None and len(actions) > max_steps)
            or not np.isfinite(actions).all()):
        raise ValueError(error)
    return actions
