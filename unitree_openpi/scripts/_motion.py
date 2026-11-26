"""Shared CLI setup for the reset and swing scripts."""
import argparse
from contextlib import contextmanager, nullcontext
from pathlib import Path

from unitree_openpi.robot_ctrl.joint_motion import joint_limits
from unitree_openpi.utils.config import load_config
from unitree_openpi.utils.ownership import ControllerLease


def motion_parser(mode):
    parser = argparse.ArgumentParser(description=f"G1D dual-arm {mode}; joint-space control without OpenPI/cameras/IK")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/g1d.json"))
    parser.add_argument("--interface", help="DDS network interface override")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true", help="publish commands to the real robot")
    group.add_argument("--mock", action="store_true", help="offline fake plant; no DDS")
    return parser


@contextmanager
def motion_session(args):
    config = load_config(args.config)
    if args.interface:
        config["device"]["interface"] = args.interface
    limits = joint_limits(config)
    if args.mock:
        from unitree_openpi.robot_ctrl.mock import MockBackend, MockKinematics
        backend, kin = MockBackend(config), MockKinematics()
    else:
        from unitree_openpi.robot_ctrl.dds import G1DDSBackend
        from unitree_openpi.robot_ctrl.kinematics import G1DKinematics
        backend, kin = G1DDSBackend(config), G1DKinematics(config)
    lease = nullcontext() if args.mock else ControllerLease(config["device"]["controller_lock"])
    with lease:
        try:
            backend.connect()
            yield backend, kin, config, limits
        except KeyboardInterrupt:
            print("Interrupted; releasing DDS. No powered hold after exit.", flush=True)
            raise SystemExit(130)
        finally:
            backend.hold()
            backend.close()
