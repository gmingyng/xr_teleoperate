"""Standalone joint exercises using the same DDS backend as the policy client."""
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import ARM_JOINT_NAMES


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
