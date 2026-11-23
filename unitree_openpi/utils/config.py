"""Deployment configuration without imports from robotskills or the SDK."""

import copy
import json
import math
from pathlib import Path

import numpy as np

from .transforms import vector

MOTION_DEFAULTS = {
    "publish_hz": 250.0, "command_hz": 20.0, "state_timeout_s": 0.5,
    "connect_timeout_s": 10.0, "max_speed_rad_s": 0.5,
    "tracking_tolerance_rad": 0.15, "tracking_window_s": 0.3,
    "grip_tolerance": 0.3, "waist_tolerance_rad": 0.03,
    "max_translation_m": 0.03, "max_rotation_rad": 0.15,
    "max_joint_step_rad": 0.3, "action_timeout_s": 1.0,
    "arrival_tolerance_rad": 0.04, "settle_timeout_s": 4.0,
    "gripper_speed": 1.0,
}


def load_config(path):
    path = Path(path).resolve()
    return normalize_config(json.loads(path.read_text(encoding="utf-8")), path.parent)


def normalize_config(value, base=None):
    config = copy.deepcopy(value)
    base = Path.cwd() if base is None else Path(base)
    device = config["device"]
    if not isinstance(device.get("interface"), str) or not device["interface"].strip():
        raise ValueError("device.interface must name the actual DDS network interface")
    if type(device.get("domain_id")) is not int or device["domain_id"] < 0:
        raise ValueError("device.domain_id must be a nonnegative integer")
    if device.get("gripper_interface") not in ("internal", "external"):
        raise ValueError("device.gripper_interface must be internal or external")
    if device["gripper_interface"] == "external":
        for side in ("left", "right"):
            for name in ("state", "command"):
                if not device.get("external_grippers", {}).get(side, {}).get(name):
                    raise ValueError("external gripper state/command topics are required")
    config["motion"] = {**MOTION_DEFAULTS, **config.get("motion", {})}
    for name, number in config["motion"].items():
        if type(number) not in (float, int) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"motion.{name} must be finite and positive")
    if config["motion"]["command_hz"] > config["motion"]["publish_hz"]:
        raise ValueError("command_hz cannot exceed publish_hz")
    for name, count in (("arm_kp", 14), ("arm_kd", 14), ("waist_kp", 3),
                        ("waist_kd", 3), ("gripper_kp", 2), ("gripper_kd", 2)):
        values = vector(config["gains"][name], count, name)
        if np.any(values < 0) or (name.endswith("kp") and np.any(values <= 0)):
            raise ValueError(f"invalid gains: {name}")
        config["gains"][name] = values.tolist()
    for section in ("model", "policy", "cameras"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing configuration section: {section}")
    model, policy, cameras = (config[k] for k in ("model", "policy", "cameras"))
    urdf = Path(model["urdf"]).expanduser()
    model["urdf"] = str((urdf if urdf.is_absolute() else base / urdf).resolve())
    if not Path(model["urdf"]).is_file():
        raise ValueError(f"URDF does not exist: {model['urdf']}")
    if policy["protocol"] not in ("openpi", "json"):
        raise ValueError("policy.protocol must be openpi or json")
    if not policy["url"].startswith(("ws://", "wss://")):
        raise ValueError("policy.url must start with ws:// or wss://")
    for key in ("timeout_s", "max_observation_age_s", "reconnect_s"):
        number = policy[key]
        if type(number) not in (float, int) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"policy.{key} must be positive")
    for key in ("execute_steps", "max_chunk_steps"):
        if type(policy[key]) is not int or policy[key] <= 0:
            raise ValueError(f"policy.{key} must be a positive integer")
    if policy["execute_steps"] > policy["max_chunk_steps"]:
        raise ValueError("execute_steps cannot exceed max_chunk_steps")
    if type(policy.get("include_grippers")) is not bool:
        raise ValueError("include_grippers must be a boolean")
    if policy["include_grippers"]:
        for key in ("open_q2", "closed_q2"):
            policy[key] = vector(policy[key], 2, key).tolist()
        if np.any(np.array(policy["open_q2"]) == policy["closed_q2"]):
            raise ValueError("gripper open and closed endpoints must differ")
        if device["gripper_interface"] == "internal":
            defaults = [[0, 0], [5.4, 5.4]]
        else:
            endpoints = [policy["open_q2"], policy["closed_q2"]]
            defaults = np.sort(endpoints, axis=0)
        limits = np.asarray(policy.get("gripper_command_limits", defaults), dtype=float)
        if (limits.shape != (2, 2) or not np.isfinite(limits).all()
                or np.any(limits[0] >= limits[1])):
            raise ValueError("gripper_command_limits requires [min_q2, max_q2]")
        policy["gripper_command_limits"] = limits.tolist()
    if set(cameras["streams"]) != {"head", "left_wrist", "right_wrist"}:
        raise ValueError("exactly head, left_wrist, right_wrist camera streams are required")
    for endpoint in cameras["streams"].values():
        if not isinstance(endpoint, str) or not endpoint.startswith("tcp://"):
            raise ValueError("camera streams must be TCP ZMQ endpoints")
    if cameras.get("head_view") not in ("left", "right", "full"):
        raise ValueError("cameras.head_view must be left, right or full")
    for key in ("stale_after_s", "startup_timeout_s", "max_skew_s"):
        number = cameras[key]
        if type(number) not in (float, int) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"cameras.{key} must be positive")
    if (len(cameras["image_size"]) != 2 or
            any(type(v) is not int or v <= 0 for v in cameras["image_size"])):
        raise ValueError("image_size must be [positive width, positive height]")
    if set(policy["image_keys"]) != set(cameras["streams"]):
        raise ValueError("policy.image_keys must map all three cameras")
    image_keys = list(policy["image_keys"].values())
    container = policy.get("image_container_key")
    if container is not None and (not isinstance(container, str) or not container):
        raise ValueError("image_container_key must be a nonempty string or null")
    keys = [policy["state_key"], policy["prompt_key"]] + ([container] if container else image_keys)
    if (any(not isinstance(k, str) or not k for k in keys + image_keys)
            or len(set(keys)) != len(keys) or "_unitree" in keys
            or len(set(image_keys)) != len(image_keys)):
        raise ValueError("observation keys must be unique nonempty strings")
    for side in ("left", "right"):
        box = np.asarray(model["workspace"][side], dtype=float)
        if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] >= box[1]):
            raise ValueError(f"invalid {side} workspace [min_xyz, max_xyz]")
    return config
