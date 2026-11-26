"""JSON-safe snapshots for rich raw G1-D recordings.

The legacy LeRobot converter intentionally continues to read only the existing
16-D qpos state/action paths. Everything built here is additive raw telemetry.
"""

from __future__ import annotations

import re
import socketserver
import time
from threading import Lock, Thread
from copy import deepcopy
from typing import Any, Iterable


G1D_BODY_JOINT_NAMES = [
    "kLeftHipPitch",
    "kLeftHipRoll",
    "kLeftHipYaw",
    "kLeftKnee",
    "kLeftAnklePitch",
    "kLeftAnkleRoll",
    "kRightHipPitch",
    "kRightHipRoll",
    "kRightHipYaw",
    "kRightKnee",
    "kRightAnklePitch",
    "kRightAnkleRoll",
    "kWaistYaw",
    "kWaistRollReserved",
    "kWaistPitchReserved",
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
    "kReserved29",
    "kReserved30",
    "kLeftDex1",
    "kReserved32",
    "kRightDex1",
    "kReserved34",
]

LEFT_ARM_INDICES = tuple(range(15, 22))
RIGHT_ARM_INDICES = tuple(range(22, 29))
LEFT_EE_INDICES = (31,)
RIGHT_EE_INDICES = (33,)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _sequence(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return list(value)
    except TypeError:
        return [value]


def lowstate_to_record(lowstate: Any, motor_count: int = 35) -> dict:
    """Convert a Unitree LowState-like object to JSON-safe telemetry."""

    motor_states = list(getattr(lowstate, "motor_state", []))[:motor_count]
    if len(motor_states) != motor_count:
        raise ValueError(
            f"Expected {motor_count} motor states, got {len(motor_states)}"
        )

    def motor_values(name: str, converter=_number) -> list:
        return [converter(getattr(motor, name, 0)) for motor in motor_states]

    motor = {
        "qpos": motor_values("q"),
        "qvel": motor_values("dq"),
        "qacc": motor_values("ddq"),
        "torque": motor_values("tau_est"),
        "mode": motor_values("mode", _integer),
        "temperature": [
            [_integer(item) for item in _sequence(getattr(state, "temperature", []))]
            for state in motor_states
        ],
        "voltage": motor_values("vol"),
        "sensor": [
            [_integer(item) for item in _sequence(getattr(state, "sensor", []))]
            for state in motor_states
        ],
        "motor_state": motor_values("motorstate", _integer),
        "reserve": [
            [_integer(item) for item in _sequence(getattr(state, "reserve", []))]
            for state in motor_states
        ],
    }

    imu_state = getattr(lowstate, "imu_state", None)
    imu = {
        "quaternion": [_number(item) for item in _sequence(getattr(imu_state, "quaternion", []))],
        "gyroscope": [_number(item) for item in _sequence(getattr(imu_state, "gyroscope", []))],
        "accelerometer": [_number(item) for item in _sequence(getattr(imu_state, "accelerometer", []))],
        "rpy": [_number(item) for item in _sequence(getattr(imu_state, "rpy", []))],
        "temperature": _integer(getattr(imu_state, "temperature", 0)),
    }

    robot = {
        "version": [_integer(item) for item in _sequence(getattr(lowstate, "version", []))],
        "mode_pr": _integer(getattr(lowstate, "mode_pr", 0)),
        "mode_machine": _integer(getattr(lowstate, "mode_machine", 0)),
        "tick": _integer(getattr(lowstate, "tick", 0)),
        "wireless_remote": [
            _integer(item) for item in _sequence(getattr(lowstate, "wireless_remote", []))
        ],
        "reserve": [
            _integer(item) for item in _sequence(getattr(lowstate, "reserve", []))
        ],
        "crc": _integer(getattr(lowstate, "crc", 0)),
    }
    return {"motor": motor, "imu": imu, "robot": robot}


def lowcmd_to_record(lowcmd: Any, motor_count: int = 35) -> dict:
    """Convert the last LowCmd-like message to JSON-safe commanded values."""

    motor_cmd = list(getattr(lowcmd, "motor_cmd", []))[:motor_count]
    if len(motor_cmd) != motor_count:
        raise ValueError(f"Expected {motor_count} motor commands, got {len(motor_cmd)}")

    def values(name: str, converter=_number) -> list:
        return [converter(getattr(command, name, 0)) for command in motor_cmd]

    return {
        "qpos": values("q"),
        "qvel": values("dq"),
        "torque": values("tau"),
        "kp": values("kp"),
        "kd": values("kd"),
        "mode": values("mode", _integer),
        "motor_reserve": values("reserve", _integer),
        "mode_pr": _integer(getattr(lowcmd, "mode_pr", 0)),
        "mode_machine": _integer(getattr(lowcmd, "mode_machine", 0)),
        "reserve": [
            _integer(item) for item in _sequence(getattr(lowcmd, "reserve", []))
        ],
        "crc": _integer(getattr(lowcmd, "crc", 0)),
    }


def select_motor_fields(motor: dict, indices: Iterable[int]) -> dict:
    """Select dynamic motor fields for an arm or end-effector group."""

    selected = {}
    for key in ("qpos", "qvel", "qacc", "torque"):
        values = motor.get(key, [])
        selected[key] = [values[index] for index in indices]
    return selected


def matrix_to_list(matrix: Any) -> list:
    if matrix is None:
        return []
    if hasattr(matrix, "tolist"):
        return matrix.tolist()
    return deepcopy(matrix)


def build_operator_record(tele_data: Any, controller_actions: Any, xr_live: bool) -> dict:
    """Capture processed and raw XR controller inputs without affecting control."""

    return {
        "head_pose_robot": matrix_to_list(getattr(tele_data, "head_pose", None)),
        "left_controller_pose_robot": matrix_to_list(getattr(tele_data, "left_wrist_pose", None)),
        "right_controller_pose_robot": matrix_to_list(getattr(tele_data, "right_wrist_pose", None)),
        "head_pose_openxr": matrix_to_list(getattr(tele_data, "raw_head_pose", None)),
        "left_controller_pose_openxr": matrix_to_list(getattr(tele_data, "raw_left_wrist_pose", None)),
        "right_controller_pose_openxr": matrix_to_list(getattr(tele_data, "raw_right_wrist_pose", None)),
        "left_trigger_pressed": bool(getattr(tele_data, "left_ctrl_trigger", False)),
        "left_trigger_value": _number(getattr(tele_data, "left_ctrl_triggerValue", 0.0)),
        "right_trigger_pressed": bool(getattr(tele_data, "right_ctrl_trigger", False)),
        "right_trigger_value": _number(getattr(tele_data, "right_ctrl_triggerValue", 0.0)),
        "left_grip_pressed": bool(getattr(tele_data, "left_ctrl_squeeze", False)),
        "left_grip_value": _number(getattr(tele_data, "left_ctrl_squeezeValue", 0.0)),
        "right_grip_pressed": bool(getattr(tele_data, "right_ctrl_squeeze", False)),
        "right_grip_value": _number(getattr(tele_data, "right_ctrl_squeezeValue", 0.0)),
        "left_thumbstick_pressed": bool(getattr(tele_data, "left_ctrl_thumbstick", False)),
        "left_thumbstick": [_number(item) for item in _sequence(getattr(tele_data, "left_ctrl_thumbstickValue", []))],
        "right_thumbstick_pressed": bool(getattr(tele_data, "right_ctrl_thumbstick", False)),
        "right_thumbstick": [_number(item) for item in _sequence(getattr(tele_data, "right_ctrl_thumbstickValue", []))],
        "buttons": {
            "x": bool(getattr(tele_data, "left_ctrl_aButton", False)),
            "y": bool(getattr(tele_data, "left_ctrl_bButton", False)),
            "a": bool(getattr(tele_data, "right_ctrl_aButton", False)),
            "b": bool(getattr(tele_data, "right_ctrl_bButton", False)),
        },
        "left_grip_locked": bool(getattr(controller_actions, "left_grip_locked", False)),
        "right_grip_locked": bool(getattr(controller_actions, "right_grip_locked", False)),
        "controller_events": {
            "arm_command_keys": list(getattr(controller_actions, "arm_commands", ())),
            "start_requested": bool(getattr(controller_actions, "start_requested", False)),
            "record_toggle_requested": bool(getattr(controller_actions, "record_toggle_requested", False)),
            "reset_requested": bool(getattr(controller_actions, "reset_requested", False)),
            "quit_requested": bool(getattr(controller_actions, "quit_requested", False)),
            "left_grip_lock_toggled": bool(getattr(controller_actions, "left_grip_lock_toggled", False)),
            "right_grip_lock_toggled": bool(getattr(controller_actions, "right_grip_lock_toggled", False)),
        },
        "xr_live": bool(xr_live),
    }


def camera_quality_record(tele_image: Any, sample_monotonic_ns: int) -> dict:
    received_ns = getattr(tele_image, "received_monotonic_ns", None) if tele_image is not None else None
    age_s = None
    if received_ns is not None:
        age_s = max(0.0, (sample_monotonic_ns - int(received_ns)) / 1_000_000_000.0)
    jpg = getattr(tele_image, "jpg", None) if tele_image is not None else None
    bgr = getattr(tele_image, "bgr", None) if tele_image is not None else None
    return {
        "valid": bool(bgr is not None),
        "received_monotonic_ns": int(received_ns) if received_ns is not None else None,
        "age_s": age_s,
        "fps": _number(getattr(tele_image, "fps", 0.0)) if tele_image is not None else 0.0,
        "jpg_bytes": len(jpg) if jpg is not None else 0,
    }


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


