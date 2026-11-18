"""Keyboard control of G1-D chassis velocity and column height."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import select
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unitree_openpi.utils.config import load_config


@contextmanager
def keyboard(device):
    if device:
        import fcntl
        codes = {"w": 17, "s": 31, "a": 30, "d": 32, "k": 37, "l": 38, "q": 16}
        with open(device, "rb", buffering=0) as stream:
            def read_keys():
                bits = bytearray(96)
                # Linux EVIOCGKEY(96) reads currently depressed keys.
                fcntl.ioctl(stream, 0x80604518, bits, True)
                return {key for key, code in codes.items() if bits[code // 8] & (1 << (code % 8))}
            yield read_keys
        return

    import termios
    import tty
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        raise RuntimeError("Interactive terminal or --keyboard /dev/input/eventN required")
    previous = termios.tcgetattr(fd)
    last_seen = {}
    try:
        tty.setcbreak(fd)
        def read_keys():
            now = time.monotonic()
            if select.select([fd], [], [], 0)[0]:
                data = os.read(fd, 4096)
                if not data:
                    raise EOFError("Terminal disconnected")
                for key in data.decode("ascii", errors="ignore").lower():
                    if key in "wasdklq":
                        last_seen[key] = now
            return {key for key, stamp in last_seen.items() if now - stamp < 0.15}
        yield read_keys
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def checked(result):
    code = result[0] if isinstance(result, tuple) else result
    if code != 0:
        raise RuntimeError(f"AGV RPC failed: {result!r}")


def drive(client, read_keys, speed, yaw_speed):
    try:
        while True:
            keys = read_keys()
            if "q" in keys:
                break
            forward = ("w" in keys) - ("s" in keys)
            turn = ("a" in keys) - ("d" in keys)
            lift = ("k" in keys) - ("l" in keys)
            checked(client.Move(forward * speed, 0.0, turn * yaw_speed))
            checked(client.HeightAdjust(lift * speed / 0.0765))
            time.sleep(0.05)
    finally:
        try:
            checked(client.Move(0.0, 0.0, 0.0))
        finally:
            checked(client.HeightAdjust(0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "config/g1d.json")
    parser.add_argument("--interface")
    parser.add_argument("--keyboard", help="local Linux keyboard event device for exact held-key state")
    parser.add_argument("--speed", type=float, default=0.03, help="forward and lift speed in m/s")
    parser.add_argument("--yaw-speed", type=float, default=0.05, help="turn speed in rad/s")
    args = parser.parse_args()
    if not (0 < args.speed <= 0.0765 and 0 < args.yaw_speed <= 0.6):
        parser.error("speed must be in (0, 0.0765] and yaw-speed in (0, 0.6]")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_openpi.robot_ctrl.agv import AgvClient
    device = load_config(args.config)["device"]
    ChannelFactoryInitialize(device["domain_id"], args.interface or device["interface"])
    client = AgvClient()
    client.SetTimeout(0.3)
    client.Init()
    print("Hold W/S forward/back, A/D turn, K/L lift/lower. Q or Ctrl+C exits.", flush=True)
    try:
        with keyboard(args.keyboard) as read_keys:
            drive(client, read_keys, args.speed, args.yaw_speed)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
