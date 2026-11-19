"""Start a local mock policy server for the G1-D WebSocket client."""

from pathlib import Path
import sys

# Direct execution also works after copying this package outside the repository.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
import signal
import numpy as np

from unitree_openpi.mock_policy_server import make_handler


def main(argv=None):
    from websockets.sync.server import serve

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--protocol", choices=("openpi", "json"), default="openpi")
    parser.add_argument("--delta-joint", type=float, default=0.0,
                        help="left shoulder-pitch offset in radians per observation, for mock testing")
    args = parser.parse_args(argv)
    if not np.isfinite(args.delta_joint) or abs(args.delta_joint) > 0.01:
        parser.error("--delta-joint must be finite and within +/-0.01m")
    logging.basicConfig(level=logging.INFO)
    with serve(make_handler(args.protocol, args.delta_joint), args.host, args.port,
               compression=None, max_size=64*1024*1024) as server:
        signal.signal(signal.SIGINT, lambda *_: server.shutdown())
        logging.info("Mock policy listening on ws://%s:%d", args.host, args.port)
        server.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
