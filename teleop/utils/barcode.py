"""Receive and latch barcode events for recording or inference."""

import re
import socketserver
import time
from threading import Lock, Thread


class BarcodeReceiver:
    """Latch the first scan of each open episode, using the robot clock."""
    def __init__(self, host="192.168.123.164", port=48166, peer="192.168.123.166", on_scan=None):
        self.on_scan = on_scan
        self.lock = Lock()
        self.start_ns = None
        self.event = None
        self.reported = False
        receiver = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                if self.client_address[0] != peer:
                    return
                self.connection.settimeout(1)
                try:
                    raw = self.rfile.readline(16)
                    if re.fullmatch(rb"(?:[0-9]{8}|[0-9]{12,14})\n", raw):
                        status = receiver.accept(raw[:-1].decode("ascii"))
                    else:
                        status = "invalid"
                    self.wfile.write(status.encode("ascii") + b"\n")
                except OSError:
                    pass

        class Server(socketserver.TCPServer):
            allow_reuse_address = True

        self.server = Server((host, port), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def begin(self, start_ns):
        with self.lock:
            self.start_ns = start_ns
            self.event = None
            self.reported = False

    def accept(self, code):
        with self.lock:
            if self.start_ns is None:
                return "outside_episode"
            if self.event is not None:
                return "already_scanned"
            now = time.monotonic_ns()
            self.event = dict(barcode=code, received_monotonic_ns=now,
                              episode_s=(now - self.start_ns) / 1e9)
        if self.on_scan:
            self.on_scan()
        return "accepted"

    def frame(self, sample_ns):
        with self.lock:
            scanned = self.event is not None and self.event["received_monotonic_ns"] <= sample_ns
            pulse = scanned and not self.reported
            if pulse:
                self.reported = True
            return dict(scan_success=scanned, scan_trigger=pulse,
                        barcode=self.event["barcode"] if scanned else None)

    def finish(self):
        with self.lock:
            self.start_ns = None
            return dict(scan_success=self.event is not None, scan_event=self.event)

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
