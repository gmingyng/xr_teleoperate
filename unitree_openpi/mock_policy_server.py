"""Local OpenPI-wire test server; emits zero EEF deltas or a small test step."""

import numpy as np

from .utils.wire import packb, unpackb, json_encode, json_decode


def make_handler(protocol="openpi", delta_x=0.0):
    encode, decode = (packb, unpackb) if protocol == "openpi" else (json_encode, json_decode)

    def handle(connection):
        connection.send(encode({"name": "unitree_openpi mock", "action_space": "relative_eef",
                                "simulation": True}))
        for payload in connection:
            obs = decode(payload)
            state = np.asarray(obs["observation/state"] if "observation/state" in obs else obs["state"])
            if state.shape not in ((12,), (14,)):
                raise ValueError("mock server expects state shape (12,) or (14,)")
            action = np.zeros((1, len(state)), dtype=np.float32)
            action[0, 0] = delta_x
            if len(state) == 14:
                action[0, [6, 13]] = state[[6, 13]]
            connection.send(encode({"actions": action,
                                    "observation_id": obs["_unitree"]["sequence"]}))
    return handle

