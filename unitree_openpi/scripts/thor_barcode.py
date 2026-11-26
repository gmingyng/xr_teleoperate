"""Forward numeric product scans from the THOR USB HID scanner to the recorder."""
import argparse
from pathlib import Path
import socket
import struct
import sys

EVENT = struct.Struct("@llHHi")  # native Linux struct input_event on THOR
DIGITS = {**dict(zip(range(2, 12), "1234567890")),
          **dict(zip((82, 79, 80, 81, 75, 76, 77, 71, 72, 73), "0123456789"))}


def barcodes(stream):
    digits = ""
    invalid = False
    while packet := stream.read(EVENT.size):
        if len(packet) != EVENT.size:
            raise OSError("Truncated input_event")
        _, _, kind, key, value = EVENT.unpack(packet)
        if kind == 0 and key == 3:  # SYN_DROPPED: discard partial barcode
            invalid = True
        if kind != 1 or value != 1:
            continue
        if key in (28, 96):  # Enter or keypad Enter
            if not invalid and len(digits) in (8, 12, 13, 14):
                yield digits
            digits, invalid = "", False
        elif key in DIGITS and len(digits) < 14:
            digits += DIGITS[key]
        else:
            invalid = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/input/event6")
    parser.add_argument("--robot", default="192.168.123.164")
    parser.add_argument("--port", type=int, default=48166)
    args = parser.parse_args()
    device = Path(args.device).resolve()
    ids = Path("/sys/class/input") / device.name / "device/id"
    if tuple((ids / k).read_text().strip().lower() for k in ("vendor", "product")) != ("0218", "0210"):
        parser.error(f"{device} is not the 0218:0210 scanner")
    with device.open("rb", buffering=0) as stream:
        print(f"Reading {device}; sending scans to {args.robot}:{args.port}", flush=True)
        for code in barcodes(stream):
            try:
                with socket.create_connection((args.robot, args.port), timeout=1) as conn:
                    conn.sendall(code.encode("ascii") + b"\n")
                    with conn.makefile("rb") as reply:
                        status = reply.readline(64).decode("ascii").strip()
                    print(code, status or "no acknowledgement", flush=True)
            except OSError as exc:
                print(f"{code}: no acknowledgement ({exc}); check recorder and rescan", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
