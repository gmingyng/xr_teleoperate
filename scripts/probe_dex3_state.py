#!/usr/bin/env python3
"""Read-only Dex3 state probe for the G1-D DDS topics."""

import argparse
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_


LEFT_STATE_TOPIC = "rt/lf/dex3/left/state"
RIGHT_STATE_TOPIC = "rt/lf/dex3/right/state"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)

    lock = threading.Lock()
    ready = threading.Event()
    states = {}

    def make_handler(side):
        def handle(message):
            with lock:
                states[side] = [float(motor.q) for motor in message.motor_state[:7]]
                if "left" in states and "right" in states:
                    ready.set()

        return handle

    left_subscriber = ChannelSubscriber(LEFT_STATE_TOPIC, HandState_)
    right_subscriber = ChannelSubscriber(RIGHT_STATE_TOPIC, HandState_)
    left_subscriber.Init(make_handler("left"), 1)
    right_subscriber.Init(make_handler("right"), 1)

    try:
        if not ready.wait(args.timeout_s):
            missing = [side for side in ("left", "right") if side not in states]
            print(f"[blocked] no Dex3 state received for: {', '.join(missing)}")
            return 1
        with lock:
            print(f"[ok] left Dex3 state:  {states['left']}")
            print(f"[ok] right Dex3 state: {states['right']}")
        print("[ok] read-only Dex3 state probe passed; no command publisher was created")
        return 0
    finally:
        left_subscriber.Close()
        right_subscriber.Close()


if __name__ == "__main__":
    raise SystemExit(main())
