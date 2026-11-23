"""Local OpenPI-wire test server; emits measured joint positions or a small test step."""

import numpy as np

from .utils.wire import packb, unpackb, json_encode, json_decode


def make_handler(protocol="openpi", delta_joint=0.0):
    encode, decode = (packb, unpackb) if protocol == "openpi" else (json_encode, json_decode)

    def handle(connection):
        connection.send(encode({"name": "unitree_openpi mock", "action_space": "absolute_joint_positions",
                                "simulation": True}))
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

