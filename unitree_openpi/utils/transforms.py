"""Meters, XYZ Euler absolute poses, and rotation-vector relative actions."""

import numpy as np
from scipy.spatial.transform import Rotation


def vector(value, length, name="vector"):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} requires {length} finite numbers")
    return array.copy()


def pose_matrix(pose):
    pose = vector(pose, 6, "absolute xyz+rpy")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def matrix_pose(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")]


def relative_target(current, delta, frame="base"):
    """base: p'=p+dp, R'=Exp(dr)R; tool: p'=p+Rdp, R'=R Exp(dr)."""
    delta = vector(delta, 6, "relative xyz+rotation-vector")
    result = np.array(current, dtype=float, copy=True)
    rotation = Rotation.from_rotvec(delta[3:]).as_matrix()
    if frame == "base":
        result[:3, 3] += delta[:3]
        result[:3, :3] = rotation @ current[:3, :3]
    elif frame == "tool":
        result[:3, 3] += current[:3, :3] @ delta[:3]
        result[:3, :3] = current[:3, :3] @ rotation
    else:
        raise ValueError("action_frame must be base or tool")
    return result


def waist_vector(value, name="waist_q3"):
    result = vector(value, 3, name)
    if result[1] != 0:
        raise ValueError("G1-D waist uses [yaw, 0, pitch]; roll must be zero")
    return result
