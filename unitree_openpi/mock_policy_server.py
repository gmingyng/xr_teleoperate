"""Local OpenPI-wire test server; emits zero EEF deltas or a small test step."""

import argparse
import logging
import signal

import numpy as np

from .utils.wire import packb, unpackb, json_encode, json_decode


def make_handler(protocol="openpi", delta_x=0.0):
    encode, decode = (packb, unpackb) if protocol == "openpi" else (json_encode, json_decode)

    def handle(connection):
        connection.send(encode({"name": "unitree_openpi mock", "action_space": "relative_eef",
                                "simulation": True}))
        for payload in connection:
            obs = decode(payload)
            state = np.asarray(obs["state"])
            if state.shape not in ((12,), (14,)):
                raise ValueError("mock server expects state shape (12,) or (14,)")
            action = np.zeros((1, len(state)), dtype=np.float32)
            action[0, 0] = delta_x
            if len(state) == 14:
                action[0, [6, 13]] = state[[6, 13]]
            connection.send(encode({"actions": action,
                                    "observation_id": obs["_unitree"]["sequence"]}))
    return handle


def main(argv=None):
    from websockets.sync.server import serve

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--protocol", choices=("openpi", "json"), default="openpi")
    parser.add_argument("--delta-x", type=float, default=0.0,
                        help="left-arm delta meters per observation, for mock testing")
    args = parser.parse_args(argv)
    if not np.isfinite(args.delta_x) or abs(args.delta_x) > 0.01:
        parser.error("--delta-x must be finite and within +/-0.01m")
    logging.basicConfig(level=logging.INFO)
    with serve(make_handler(args.protocol, args.delta_x), args.host, args.port,
               compression=None, max_size=64*1024*1024) as server:
        signal.signal(signal.SIGINT, lambda *_: server.shutdown())
        logging.info("Mock policy listening on ws://%s:%d", args.host, args.port)
        server.serve_forever()

