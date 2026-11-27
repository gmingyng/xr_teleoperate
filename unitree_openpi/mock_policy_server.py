"""Local OpenPI-wire test server; emits measured joint positions or a small test step."""

import numpy as np

from .utils.wire import packb, unpackb, json_encode, json_decode


def make_handler(protocol="openpi", delta_joint=0.0):
    encode, decode = (packb, unpackb) if protocol == "openpi" else (json_encode, json_decode)

    metadata = {
        "name": "unitree_openpi mock",
        "action_space": "absolute_joint_positions",
        "simulation": True,
        "g1d": {
            "rtc": False,
            "action_space": "joint",
            "action_representation": "absolute",
            "state_dim": 16,
            "action_dim": 16,
            "gripper_units": "motor_position",
            "action_horizon": 1,
        },
    }

    def handle(connection):
        connection.send(encode(metadata))
        for payload in connection:
            obs = decode(payload)
            state = np.asarray(obs["observation/state"] if "observation/state" in obs else obs["state"])
            if state.shape not in ((14,), (16,)):
                raise ValueError("mock server expects state shape (14,) or (16,)")
            action = state.astype(np.float32, copy=True)[None, :]
            action[0, 0] += delta_joint
            connection.send(encode({"actions": action,
                                    "observation_id": obs["_unitree"]["sequence"]}))
    return handle

