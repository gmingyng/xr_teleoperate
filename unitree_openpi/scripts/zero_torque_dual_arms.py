"""Continuously request zero torque on the fourteen G1-D arm motors."""
import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unitree_openpi.robot_ctrl.dds import ARM_SLOTS
from unitree_openpi.utils.config import load_config
from unitree_openpi.utils.ownership import ControllerLease


def arm_command(message, mode_machine):
    message.mode_pr = 0
    message.mode_machine = mode_machine
    for slot, motor in enumerate(message.motor_cmd):
        motor.mode = int(slot in ARM_SLOTS)
        motor.q = motor.dq = motor.kp = motor.kd = motor.tau = 0.0
    return message


def publish(config):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    device, motion = config["device"], config["motion"]
    ChannelFactoryInitialize(device["domain_id"], device["interface"])
    reader = ChannelSubscriber(device["lowstate_topic"], LowState_)
    writer = ChannelPublisher(device["lowcmd_topic"], LowCmd_)
    from threading import Condition
    condition = Condition()
    latest_mode = None
    received = 0.0
    initial_mode = None
    mode_changed = False

    def receive(state):
        nonlocal latest_mode, received, mode_changed
        with condition:
            latest_mode = int(state.mode_machine)
            if initial_mode is not None and latest_mode != initial_mode:
                mode_changed = True
            received = time.monotonic()
            condition.notify()

    try:
        reader.Init(receive)
        with condition:
            if not condition.wait_for(lambda: latest_mode is not None, motion["connect_timeout_s"]):
                raise TimeoutError("No G1-D LowState received")
            initial_mode = latest_mode
        writer.Init()
        message = arm_command(unitree_hg_msg_dds__LowCmd_(), initial_mode)
        message.crc = CRC().Crc(message)
        period = 1.0 / motion["publish_hz"]
        print("Requesting zero torque on both arms. Ctrl+C stops publication.", flush=True)
        while True:
            started = time.monotonic()
            with condition:
                if mode_changed or started - received > motion["state_timeout_s"]:
                    raise RuntimeError("LowState expired or robot mode changed")
            if writer.Write(message) is False:
                raise RuntimeError("DDS rejected arm command")
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        try:
            writer.Close()
        finally:
            reader.Close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True,
                        help="support both arms before enabling zero torque")
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "config/g1d.json")
    parser.add_argument("--interface")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.interface:
        config["device"]["interface"] = args.interface
    with ControllerLease(config["device"]["controller_lock"]):
        try:
            publish(config)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
