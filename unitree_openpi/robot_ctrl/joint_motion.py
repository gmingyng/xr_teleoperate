"""Standalone joint exercises using the same DDS backend as the policy client."""
import argparse
from contextlib import contextmanager, nullcontext
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import ARM_JOINT_NAMES
from ..utils.config import load_config
from ..utils.ownership import ControllerLease


def joint_limits(config):
    joints = {j.attrib["name"]: j for j in ET.parse(config["model"]["urdf"]).getroot().findall("joint")}
    limits = [joints[name].find("limit").attrib for name in ARM_JOINT_NAMES]
    return np.array([float(j["lower"]) for j in limits]), np.array([float(j["upper"]) for j in limits])


def check_target(q, limits):
    q = np.asarray(q, dtype=float)
    lower, upper = limits
    if q.shape != (14,) or not np.isfinite(q).all() or np.any(q < lower) or np.any(q > upper):
        raise ValueError("joint target must contain 14 finite angles within URDF limits")
    return q.copy()


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
        from .mock import MockBackend, MockKinematics
        backend, kin = MockBackend(config), MockKinematics()
    else:
        from .dds import G1DDSBackend
        from .kinematics import G1DKinematics
        backend, kin = G1DDSBackend(config), G1DKinematics(config)
    lease = nullcontext() if args.mock else ControllerLease(config["device"]["controller_lock"])
    with lease:
        try:
            backend.connect()
            yield backend, kin, config, limits
        except KeyboardInterrupt:
            print("Interrupted; releasing DDS. No powered hold after exit.", flush=True)
        finally:
            backend.hold()
            backend.close()
