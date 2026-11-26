"""Scan the full-resolution head and wrist streams and signal each newly seen code."""
import argparse
from contextlib import ExitStack
import json
import logging
from pathlib import Path
import time

import cv2
import numpy as np
import zxingcpp
import zmq


RETAIL_CODES = zxingcpp.BarcodeFormats([zxingcpp.BarcodeFormat.EAN13, zxingcpp.BarcodeFormat.EAN8,
                                    zxingcpp.BarcodeFormat.UPCA, zxingcpp.BarcodeFormat.UPCE])


def image_codes(gray):
    return {(symbol.format.name, symbol.bytes) for symbol in
            zxingcpp.read_barcodes(np.ascontiguousarray(gray), formats=RETAIL_CODES)}


def stereo_codes(gray):
    left, right = np.hsplit(gray, 2)
    return image_codes(left) | image_codes(right)


class ScanGate:
    def __init__(self):
        self.previous = {}
        self.active = {}
        self.updated = {}

    def update(self, source, codes, now):
        previous = self.previous.get(source, set())
        if now - self.updated.get(source, now) > 0.5:
            previous = set()
        self.updated[source] = now
        self.previous[source] = codes
        for code, last_seen in list(self.active.items()):
            if now - last_seen >= 1.0:
                del self.active[code]
        new = (previous & codes) - self.active.keys()
        for code in (codes & self.active.keys()) | new:
            self.active[code] = now
        return sorted(new)


def beep_pcm():
    t = np.arange(1920) / 16000
    envelope = np.clip(np.minimum(t, 0.12 - t) / 0.005, 0, 1)
    return (10000 * envelope * np.sin(2 * np.pi * 1600 * t)).astype("<i2").tobytes()


def set_led(client, rgb):
    rc = client.LedControl(*rgb)
    if rc:
        logging.warning("LED %s returned %s", rgb, rc)


def notify(client, pcm):
    try:
        set_led(client, (0, 255, 0))
        rc, _ = client.PlayStream("g1d_scan", str(time.monotonic_ns()), pcm)
        if rc:
            logging.warning("PlayStream returned %s", rc)
        time.sleep(0.15)
    finally:
        set_led(client, (0, 0, 255))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/g1d.json")
    parser.add_argument("--dry-run", action="store_true", help="decode and print; skip speaker and LED")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    streams = config["cameras"]["streams"]
    client = None
    if not args.dry_run:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        device = config["device"]
        ChannelFactoryInitialize(device["domain_id"], device["interface"])
        client = AudioClient()
        client.SetTimeout(2.0)
        client.Init()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    gate, pcm = ScanGate(), beep_pcm()
    try:
        if client is not None:
            set_led(client, (0, 0, 255))
        with zmq.Context() as context, ExitStack() as stack:
            poller = zmq.Poller()
            names = {}
            for name, endpoint in streams.items():
                socket = stack.enter_context(context.socket(zmq.SUB))
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.CONFLATE, 1)
                socket.setsockopt(zmq.SUBSCRIBE, b"")
                socket.connect(endpoint)
                names[socket] = name
                poller.register(socket, zmq.POLLIN)
                logging.info("Scanning %s on %s", name, endpoint)
            while True:
                for socket, event in poller.poll(500):
                    if not event & zmq.POLLIN:
                        continue
                    name = names[socket]
                    payload = socket.recv()
                    gray = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE) if payload else None
                    if gray is None or (name == "head" and gray.shape[1] % 2):
                        gate.previous.pop(name, None)
                        continue
                    codes = stereo_codes(gray) if name == "head" else image_codes(gray)
                    for kind, data in gate.update(name, codes, time.monotonic()):
                        logging.info("scan_success source=%s type=%s data=%r", name, kind, data)
                        if client is not None:
                            notify(client, pcm)
    finally:
        if client is not None:
            set_led(client, (0, 0, 0))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
