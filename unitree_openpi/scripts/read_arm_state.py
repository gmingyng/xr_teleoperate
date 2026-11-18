"""Display one G1-D feedback snapshot with measured joints and FK."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unitree_openpi.robot_ctrl.dds import G1DDSBackend
from unitree_openpi.robot_ctrl.kinematics import ARM_JOINT_NAMES, G1DKinematics
from unitree_openpi.utils.config import load_config
from unitree_openpi.utils.transforms import matrix_pose


def read_state(backend, kinematics):
    state = backend.read()
    kinematics.update_state(state)
    poses = kinematics.fk(state["q14"])
    result = {"received_at": state["received_at"], "reference_frame": kinematics.frame_id}
    for index, side in enumerate(("left", "right")):
        joints = slice(7 * index, 7 * (index + 1))
        pose = matrix_pose(poses[side])
        result[side] = {
            "joint_names": list(ARM_JOINT_NAMES[joints]),
            "q_rad": state["q14"][joints].tolist(),
            "dq_rad_s": state["dq14"][joints].tolist(),
            "gripper_q_raw": float(state["grip_q2"][index]),
            "gripper_dq_raw": float(state["grip_dq2"][index]),
            "eef_xyz_m": pose[:3].tolist(),
            "eef_rpy_rad": pose[3:].tolist(),
        }
    return result


def print_state(state):
    print(f"Received (Unix s): {state['received_at']:.3f}    EEF frame: {state['reference_frame']}")
    for side in ("left", "right"):
        arm = state[side]
        print(f"\n{side.capitalize()} arm")
        print(f"{'Joint':30} {'q (rad)':>10} {'dq (rad/s)':>12}")
        for name, angle, speed in zip(arm["joint_names"], arm["q_rad"], arm["dq_rad_s"]):
            print(f"{name:30} {angle:10.4f} {speed:12.4f}")
        print(f"Gripper (raw): q={arm['gripper_q_raw']:.4f}  dq={arm['gripper_dq_raw']:.4f}")
        print("EEF XYZ (m):    " + "  ".join(f"{v:.4f}" for v in arm["eef_xyz_m"]))
        print("EEF RPY (rad):  " + "  ".join(f"{v:.4f}" for v in arm["eef_rpy_rad"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "config/g1d.json")
    parser.add_argument("--interface", help="override device.interface from the config")
    parser.add_argument("--mock", action="store_true", help="use offline sample feedback and mock FK")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.interface:
        config["device"]["interface"] = args.interface
    if args.mock:
        from unitree_openpi.robot_ctrl.mock import MockBackend, MockKinematics
        backend, kinematics = MockBackend(config), MockKinematics()
    else:
        backend, kinematics = G1DDSBackend(config), G1DKinematics(config)
    try:
        if args.mock:
            backend.connect()
        else:
            backend.connect(snapshot_only=True)
        print_state(read_state(backend, kinematics))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
