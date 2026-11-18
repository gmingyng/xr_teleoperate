"""Probe the G1-D AGV Move RPC with zero velocity or one small forward pulse."""
import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unitree_openpi.utils.config import load_config


def move(client, speed):
    result = client.Move(speed, 0.0, 0.0)
    code = result[0] if isinstance(result, tuple) else result
    if code != 0:
        raise RuntimeError(f"AGV Move({speed}) returned {result!r}")


def probe(client, forward):
    move(client, 0.0)
    if forward:
        try:
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                move(client, 0.04)
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        finally:
            move(client, 0.0)
    print("AGV Move acknowledged zero velocity.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "config/g1d.json")
    parser.add_argument("--interface")
    parser.add_argument("--forward", action="store_true", help="send 0.04 m/s for 0.5 s then stop")
    args = parser.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_openpi.robot_ctrl.agv import AgvClient

    device = load_config(args.config)["device"]
    ChannelFactoryInitialize(device["domain_id"], args.interface or device["interface"])
    client = AgvClient()
    client.SetTimeout(0.3)
    client.Init()
    probe(client, args.forward)


if __name__ == "__main__":
    main()
