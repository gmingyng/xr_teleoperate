import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO, 
                       file=True, 
                       file_path="/home/unitree/unitree_eai_environment/logs",
                       backup_count=100,
                       max_file_size=50*1024*1024,
                       file_name_format="{prog_name}_%Y%m%d.log",
                       )
logger_mp = logging_mp.getLogger(__name__)
import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import numpy as np
import pinocchio as pin
import os 
import subprocess
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from teleop.televuer.tv_wrapper import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_29_Arm_Internal_Dex1_Controller
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Dex1_1_Gripper_Controller
from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX, Inspire_Controller_FTP, Inspire_Controller_DFX_ctrl, Inspire_Controller_FTP_ctrl
from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand, Brainco_Controller_ctrl
from teleop.robot_control.mobile_control import G1D_Mobile_Lift_Controller, G1D_Mobile_Waist_Pitch_Controller
from teleop.utils.instruction_map import ControlDataMapper, HandleInstruction
from teleop.utils.robot_config import (
    ARM_ARGUMENT_CHOICES,
    SUPPORTED_ARM,
    normalize_arm_name,
    unsupported_arm_message,
)
from teleop.utils.output_gate import motion_input_is_live, set_shared_flag, shared_flag_is_set
from teleop.utils.controller_controls import ControllerButtons, ControllerControlMapper
from teleop.utils.recording_schema import (
    G1D_BODY_JOINT_NAMES,
    LEFT_ARM_INDICES,
    LEFT_EE_INDICES,
    RIGHT_ARM_INDICES,
    RIGHT_EE_INDICES,
    build_operator_record,
    camera_quality_record,
    select_motor_fields,
)

from teleop.teleimager.src.teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
# from teleop.utils.motion_switcher import MotionSwitcher
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")


def get_source_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=parent_dir,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).strip()
    except Exception:
        return None

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
EPISODE_ID     = 0      # Episode ID (int) for IPC communication
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key, episode_id=None):
    global STOP, START, RECORD_TOGGLE, EPISODE_ID
    if key == 'r':
        if 'arm_modes' in globals():
            arm_modes[:] = ['track', 'track']
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and (START or (READY and args.control_source == "both")):
        if episode_id is not None:
            EPISODE_ID = episode_id
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=ARM_ARGUMENT_CHOICES, default='G1', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex1_internal', 'dex3', 'brainco', 'inspire_ftp', 'inspire_dfx'], help='Select end effector controller')
    parser.add_argument('--dual-trigger-dex1-mode', action = 'store_true', help = 'Control dex1 gripper with both controller triggers: index trigger closes, side squeeze opens')
    # mobile base, elevation and waist control
    parser.add_argument('--base-type', type=str, choices=['mobile_lift', 'lift','legs'], default='mobile_lift', help='Select lower body type')
    parser.add_argument('--use-waist-yaw', action = 'store_true', help = 'Enable waist yaw control with the right controller thumbstick')
    parser.add_argument('--use-waist-pitch', action = 'store_true', help = 'Enable waist pitch control with the right controller B/A buttons')
    # mode flags
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--control-source', choices=['keyboard', 'controller', 'both'], default='keyboard',
                        help='Use keyboard r/s/q, PICO controller chords, or both for lifecycle control.')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    parser.add_argument('--xr-deadman-s', type=float, default=0.3, help='Disable hardware output when XR motion input is stale for this many seconds.')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = '/home/unitree/unitree_eai_environment/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    parser.add_argument("--barcode-scanner", action="store_true", help="Record THOR product scans in _unitree")
    parser.add_argument("--arm-reset-speed", type=float, default=0.5, metavar="RAD_S",
                        help="Per-joint X/A reset target increment per second (default: 0.5 rad/s)")
    parser.add_argument("--record-format", choices=("raw","lerobot_v3"), default="raw")
    args = parser.parse_args()
    if args.record_format == "lerobot_v3" and (not args.record or args.ee != "dex1_internal" or args.ipc):
        parser.error("G1-D v3 recording requires --record --ee dex1_internal and local episode numbering")
    if not np.isfinite(args.arm_reset_speed) or args.arm_reset_speed <= 0:
        parser.error("--arm-reset-speed must be finite and positive")
    if args.barcode_scanner and not args.record:
        parser.error("--barcode-scanner requires --record")
    if args.xr_deadman_s <= 0:
        parser.error("--xr-deadman-s must be positive")
    args.arm = normalize_arm_name(args.arm)
    if args.arm == "G1_23":
        parser.error(unsupported_arm_message(args.arm))
    if args.ee == "dex3" and args.input_mode != "hand":
        parser.error("--ee dex3 requires --input-mode hand")
    if args.control_source in ("controller", "both") and args.input_mode != "controller":
        parser.error("--control-source controller/both requires --input-mode controller")
    if args.dual_trigger_dex1_mode and (args.ee not in ("dex1", "dex1_internal") or args.input_mode != "controller"):
        parser.error("--dual-trigger-dex1-mode requires --ee dex1/dex1_internal and --input-mode controller")
    if args.dual_trigger_dex1_mode and args.control_source in ("controller", "both"):
        parser.error("--dual-trigger-dex1-mode conflicts with controller Grip hold locking; omit it for controller lifecycle control")
    args.task_name = args.task_name.strip()
    args.task_goal = args.task_goal.strip()
    if args.record and not args.task_name:
        parser.error("--task-name must not be empty when --record is enabled")
    if args.record and not args.task_goal:
        parser.error("--task-goal must not be empty when --record is enabled")
    logger_mp.info(f"args: {args}")
    if args.record:
        logger_mp.info(
            f"Recording dataset task_name={args.task_name!r}, current batch task_goal={args.task_goal!r}"
        )

    # Bound before the try so the finally can stop it even if startup fails part way.
    waist_pitch_ctrl = None
    mobile_ctrl = None
    arm_ctrl = None
    output_enabled = Value('b', False, lock=True)
    ipc_server = None
    listen_keyboard_thread = None
    img_client = None
    tv_wrapper = None
    recorder = None

    try:
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        elif args.control_source in ("keyboard", "both"):
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     arm_reference_mode="head_yaw" # another choice is "head_position".
                                     )
        controller_mapper = (
            ControllerControlMapper()
            if args.control_source in ("controller", "both")
            else None
        )

        # Enter debug mode
        # motion_switcher = MotionSwitcher()
        # status, result = motion_switcher.Enter_Debug_Mode()
        # logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")
        
        xr_motion_data_ready = Value('b', False, lock=True)
        # arm
        if args.arm == SUPPORTED_ARM:
            arm_ik = G1_29_ArmIK()
            if args.ee == "dex1_internal":
                left_gripper_value = Value('d', 0.0, lock=True)         # [input]
                right_gripper_value = Value('d', 0.0, lock=True)        # [input]
                left_gripper_squeeze_value = Value('d', 0.0, lock=True)  # [input] only used in dual trigger mode
                right_gripper_squeeze_value = Value('d', 0.0, lock=True) # [input] only used in dual trigger mode
                dual_gripper_data_lock = Lock()
                dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
                dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
                arm_ctrl = G1_29_Arm_Internal_Dex1_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock,
                                                              dual_gripper_state_array, dual_gripper_action_array,
                                                              simulation_mode=args.sim,
                                                              xr_motion_data_ready_in=xr_motion_data_ready,
                                                              left_gripper_squeeze_in=left_gripper_squeeze_value,
                                                              right_gripper_squeeze_in=right_gripper_squeeze_value,
                                                              dual_trigger_mode=args.dual_trigger_dex1_mode,
                                                              output_enabled_in=output_enabled)
            else:
                arm_ctrl = G1_29_ArmController(simulation_mode=args.sim, output_enabled_in=output_enabled)
        elif args.arm == "H2":
            message = unsupported_arm_message(args.arm)
            logger_mp.warning(message)
            raise NotImplementedError(message)
        elif args.arm == "R1":
            message = unsupported_arm_message(args.arm)
            logger_mp.warning(message)
            raise NotImplementedError(message)

        # end-effector
        if args.ee == "dex3":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready,
                                          output_enabled_in=output_enabled)
        elif args.ee == "dex1": # external
            left_gripper_value = Value('d', 0.0, lock=True)         # [input]
            right_gripper_value = Value('d', 0.0, lock=True)        # [input]
            left_gripper_squeeze_value = Value('d', 0.0, lock=True)  # [input] only used in dual trigger mode
            right_gripper_squeeze_value = Value('d', 0.0, lock=True) # [input] only used in dual trigger mode
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready,
                                                     left_gripper_squeeze_in=left_gripper_squeeze_value, right_gripper_squeeze_in=right_gripper_squeeze_value,
                                                     dual_trigger_mode=args.dual_trigger_dex1_mode, output_enabled_in=output_enabled)
        elif args.ee == "inspire_dfx" and args.input_mode == "hand":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx" and args.input_mode == "controller":
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                    dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_ftp" and args.input_mode == "hand":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp" and args.input_mode == "controller":
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                    dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "hand":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_hand(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                                dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "controller":
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        else:
            pass

        # For mobile base and elevation control
        if args.base_type != "legs":
            try:
                mobile_ctrl = G1D_Mobile_Lift_Controller(args.base_type, args.input_mode == "hand", output_enabled_in=output_enabled)
            except Exception as e:
                STOP = True
                logger_mp.error(f"Failed to initialize mobile base/lift controller: {e}")
                raise
        else:
            mobile_ctrl=None
        # For waist pitch control. The pitch joint is a standalone motor on its own DDS
        # topic, independent of the lowcmd path that carries waist yaw.
        if args.use_waist_pitch:
            try:
                waist_pitch_ctrl = G1D_Mobile_Waist_Pitch_Controller(arm_ctrl, output_enabled_in=output_enabled)
            except Exception as e:
                STOP = True
                logger_mp.error(f"Failed to initialize waist pitch controller: {e}")
                raise
        else:
            waist_pitch_ctrl = None
        control_data_mapper = ControlDataMapper(arm_ctrl.get_current_waist_yaw_q()[0])
        handle_instruction = HandleInstruction(args.input_mode == "hand", tv_wrapper, mobile_ctrl)

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recording_info = {
                "schema_version": "1.1.0",
                "joint_names": {
                    "left_arm": [G1D_BODY_JOINT_NAMES[i] for i in LEFT_ARM_INDICES],
                    "left_ee": [G1D_BODY_JOINT_NAMES[i] for i in LEFT_EE_INDICES],
                    "right_arm": [G1D_BODY_JOINT_NAMES[i] for i in RIGHT_ARM_INDICES],
                    "right_ee": [G1D_BODY_JOINT_NAMES[i] for i in RIGHT_EE_INDICES],
                    "body": G1D_BODY_JOINT_NAMES,
                },
                "units": {
                    "qpos": "rad",
                    "qvel": "rad/s",
                    "qacc": "rad/s^2",
                    "torque": "N*m",
                    "time": "s",
                },
                "camera_mapping": {
                    "color_0": "cam_left_high",
                    "color_1": "cam_right_high",
                    "color_2": "cam_left_wrist",
                    "color_3": "cam_right_wrist",
                },
                "camera_config": camera_config,
                "runtime": {
                    "git_commit": get_source_git_commit(),
                    "command_line": sys.argv,
                    "arguments": vars(args),
                },
            }
            writer_type = EpisodeWriter
            name = args.task_name
            if args.record_format == "lerobot_v3":
                from teleop.utils.v3_writer import V3Writer
                writer_type = V3Writer
                name += "_lerobot_v3"
            recorder = writer_type(task_dir = os.path.join(args.task_dir, name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless,
                                     # Author: mingyang - Enable robot voice prompts for controller recording.
                                     voice_prompts = not args.sim and args.control_source in ("controller", "both"),
                                     extra_info = recording_info, barcode_scanner=args.barcode_scanner, scan_beep=not args.sim)

        arm_modes = ["hold", "hold"]
        hold_targets = [None, None]
        wait_neutral = False

        discard_after_reset_command = False

        def process_controller_controls(tele_data, xr_motion_live):
            global START, RECORD_RUNNING, RECORD_TOGGLE, wait_neutral, discard_after_reset_command
            if controller_mapper is None:
                return None

            buttons = ControllerButtons.from_tele_data(tele_data)
            actions = controller_mapper.update(buttons)
            if not xr_motion_live:
                arm_modes[:] = ["hold", "hold"]
                hold_targets[:] = [None, None]
                wait_neutral = True
            if wait_neutral:
                if xr_motion_live and not any((buttons.x, buttons.y, buttons.a, buttons.b)):
                    wait_neutral = False
                if not actions.quit_requested:
                    return actions
            if actions.left_grip_lock_toggled:
                logger_mp.info(f"[Controller] Left Dex1 hold {'LOCKED' if actions.left_grip_locked else 'UNLOCKED'}")
            if actions.right_grip_lock_toggled:
                logger_mp.info(f"[Controller] Right Dex1 hold {'LOCKED' if actions.right_grip_locked else 'UNLOCKED'}")

            if actions.quit_requested:
                START = False
                arm_modes[:] = ["hold", "hold"]
                hold_targets[:] = [None, None]
                if RECORD_RUNNING:
                    RECORD_RUNNING = False
                    RECORD_TOGGLE = False
                    if xr_motion_live and not wait_neutral:
                        arm_modes[:] = ["reset", "reset"]
                        # Submit a reset target before waiting for the recorder's disk queue.
                        discard_after_reset_command = True
                        logger_mp.info("[Controller] A+X canceled recording; returning both arms to reset")
                    else:
                        set_shared_flag(output_enabled, False)
                        recorder.discard_episode()
                    return actions
                logger_mp.info("[Controller] A+X pressed; graceful process shutdown requested")
                on_press('q')
                return actions

            for key in actions.arm_commands:
                side = 0 if key in ("x", "y") else 1
                arm_modes[side] = "track" if key in ("y", "b") else "reset"
                hold_targets[side] = None
                logger_mp.info(f"[Controller] {'Left' if side == 0 else 'Right'} arm: {arm_modes[side]}")

            if actions.record_toggle_requested:
                if not args.record:
                    logger_mp.warning("[Controller] B+Y ignored because --record is not enabled")
                else:
                    logger_mp.info("[Controller] B+Y record toggle requested")
                    RECORD_TOGGLE = True
            return actions

        def update_dex1_controller_inputs(tele_data, controller_actions):
            if args.input_mode != "controller" or args.ee not in ("dex1", "dex1_internal"):
                return

            left_locked = controller_actions.left_grip_locked if controller_actions is not None else False
            right_locked = controller_actions.right_grip_locked if controller_actions is not None else False
            if not left_locked:
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
            if not right_locked:
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue

            # Legacy dual-trigger mode remains available with keyboard lifecycle
            # control. Controller lifecycle mode reserves Grip for hold locking.
            if args.dual_trigger_dex1_mode:
                with left_gripper_squeeze_value.get_lock():
                    left_gripper_squeeze_value.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_squeeze_value.get_lock():
                    right_gripper_squeeze_value.value = tele_data.right_ctrl_squeezeValue

        if args.control_source == "controller":
            logger_mp.info("Controller controls: Y/B=track left/right, X/A=reset left/right, B+Y=record toggle/save, A+X=discard and pause while recording, otherwise quit, Grip=toggle Dex1 hold")
        elif args.control_source == "both":
            logger_mp.info("Controls: keyboard r/s/q or controller Y/B=track left/right, X/A=reset left/right, B+Y=record toggle/save, A+X=discard and pause while recording, otherwise quit; Grip toggles Dex1 hold")
        else:
            logger_mp.info("Keyboard controls: r=start/re-arm, s=record toggle, q=quit")

        logger_mp.info("Startup ready-pose motion disabled for G1-D; waiting for start")
        set_shared_flag(output_enabled, False)

        READY = True                  # now ready to (1) enter START state
        while controller_mapper is None and not START and not STOP:
            wait_start_time = time.time()
            time.sleep(max(0, (1 / args.frequency) - (time.time() - wait_start_time)))

        logger_mp.info("---------------------🚀start program🚀-------------------------")
        waiting_for_xr_logged = False
        waiting_for_rearm_logged = False
        head_img = None
        left_wrist_img = None
        right_wrist_img = None
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record:
                    head_img = img_client.get_head_frame()
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode(episode_id=EPISODE_ID if args.ipc else None):
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)
            
            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            xr_motion_live = motion_input_is_live(
                tele_data.motion_data_ready,
                tele_data.motion_data_timestamp,
                args.xr_deadman_s,
            )
            controller_actions = process_controller_controls(tele_data, xr_motion_live)
            if STOP:
                break

            # logger_mp.info(f"tele_data: {tele_data}")
            if args.ee in ("dex3", "inspire_ftp", "inspire_dfx", "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee in ("brainco", "inspire_dfx", "inspire_ftp") and args.input_mode == "controller":
                with left_gripper_trigger_in.get_lock():
                    left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                with left_gripper_squeeze_in.get_lock():
                    left_gripper_squeeze_in.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_trigger_in.get_lock():
                    right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                with right_gripper_squeeze_in.get_lock():
                    right_gripper_squeeze_in.value = tele_data.right_ctrl_squeezeValue
            elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                update_dex1_controller_inputs(tele_data, controller_actions)
            elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = xr_motion_live

            if not xr_motion_live:
                set_shared_flag(output_enabled, False)
                if mobile_ctrl is not None:
                    mobile_ctrl.g1_height_action_array_in[0] = 0.0
                    if mobile_ctrl.g1_move_action_array_in is not None:
                        mobile_ctrl.g1_move_action_array_in[:] = [0.0, 0.0]
                if START:
                    START = False
                    logger_mp.warning("XR motion input is stale; hardware output disabled. Press start again after XR recovers.")
                if not waiting_for_xr_logged:
                    logger_mp.warning("XR motion data is not live; hardware output remains disabled.")
                    waiting_for_xr_logged = True
                time_elapsed = time.time() - start_time
                time.sleep(max(0, (1 / args.frequency) - time_elapsed))
                continue
            waiting_for_xr_logged = False

            arm_control_active = START or any(mode != "hold" for mode in arm_modes)
            if not arm_control_active and not RECORD_RUNNING:
                if not waiting_for_rearm_logged:
                    logger_mp.info("XR motion input is live; press start to re-enable hardware output.")
                    waiting_for_rearm_logged = True
                time_elapsed = time.time() - start_time
                time.sleep(max(0, (1 / args.frequency) - time_elapsed))
                continue
            waiting_for_rearm_logged = False
            
            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            # current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            left_arm_pose_state, right_arm_pose_state = arm_ik.solve_fk(current_lr_arm_q)
            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            if arm_control_active:
                sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q)
                if controller_mapper is not None and arm_modes != ["track", "track"]:
                    sol_q = sol_q.copy()
                    for side, mode in enumerate(arm_modes):
                        joints = slice(side * 7, (side + 1) * 7)
                        measured = current_lr_arm_q[joints]
                        if mode == "reset":
                            # Limit the requested offset from feedback to one reset update.
                            step = args.arm_reset_speed / args.frequency
                            sol_q[joints] = measured - np.clip(measured, -step, step)
                        elif mode == "hold":
                            if hold_targets[side] is None:
                                hold_targets[side] = measured.copy()
                            sol_q[joints] = hold_targets[side]
                    robot = arm_ik.reduced_robot
                    sol_tauff = pin.computeGeneralizedGravity(robot.model, robot.data, sol_q).copy()
            else:
                # An episode may start before either arm is taken over.
                sol_q = current_lr_arm_q.copy()
            left_arm_pose_action, right_arm_pose_action = arm_ik.solve_fk(sol_q)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")

            # For mobile base and elevation control
            height_state = None
            height_action = [0.0]
            move_state = None
            move_action = [0.0, 0.0]
            waist_yaw_state = None
            waist_yaw_action = None
            waist_pitch_state = None
            waist_pitch_action = None
            # One read per frame, so every consumer below sees the same controller sample.
            handle_instruction_data = handle_instruction.get_instruction()
            if  mobile_ctrl is not None:
                height_state = mobile_ctrl.g1_height_state_array_out
                vel_data = control_data_mapper.update(ry=handle_instruction_data['ry'])
                height_action = np.array([vel_data['g1_height']]).tolist()
                mobile_ctrl.g1_height_action_array_in[0] = height_action[0]  
                if args.base_type == "mobile_lift":
                    move_state = mobile_ctrl.g1_move_state_array_out
                    vel_data = control_data_mapper.update(lx=handle_instruction_data['lx'], ly=handle_instruction_data['ly'])
                    move_action = np.array([vel_data['mobile_x_vel'], vel_data['mobile_yaw_vel']]).tolist()
                    mobile_ctrl.g1_move_action_array_in[0] = move_action[0]  
                    mobile_ctrl.g1_move_action_array_in[1] = move_action[1] 

            if args.use_waist_yaw:
                waist_yaw_state = arm_ctrl.get_current_waist_yaw_q()

                vel_data = control_data_mapper.update(rx=handle_instruction_data['rx'], current_waist_yaw=waist_yaw_state[0])
                waist_yaw_action = np.array([vel_data['waist_yaw_pos']], dtype=float)
                
                sol_q = np.concatenate([sol_q, waist_yaw_action])

            if waist_pitch_ctrl is not None:
                waist_pitch_ctrl.set_waist_pitch_buttons(button_bend=handle_instruction_data['rbutton_B'],
                                                         button_raise=handle_instruction_data['rbutton_A'])
                waist_pitch_state = [waist_pitch_ctrl.get_current_waist_pitch_state()]
                waist_pitch_action = [waist_pitch_ctrl.get_current_waist_pitch_action()]

            try:
                if arm_control_active:
                    arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                    if not shared_flag_is_set(output_enabled):
                        set_shared_flag(output_enabled, True)
                        logger_mp.info("Hardware output enabled after explicit start and live XR input.")
            except Exception as e:
                logger_mp.error(f"Failed to control arms with ik solution: {e}")
                raise e
            if discard_after_reset_command:
                recorder.discard_episode()
                discard_after_reset_command = False

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                elif (args.ee in ("brainco", "inspire_dfx", "inspire_ftp") and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[7:7+7]
                if RECORD_RUNNING:
                    lowstate_record = None
                    commanded_record = None
                    get_recording_snapshot = getattr(arm_ctrl, "get_recording_snapshot", None)
                    if callable(get_recording_snapshot):
                        lowstate_record, commanded_record = get_recording_snapshot()
                        if not arm_control_active:
                            commanded_record = None

                    left_motor = None
                    right_motor = None
                    left_ee_motor = None
                    right_ee_motor = None
                    if lowstate_record is not None:
                        motor_record = lowstate_record["motor"]
                        left_motor = select_motor_fields(motor_record, LEFT_ARM_INDICES)
                        right_motor = select_motor_fields(motor_record, RIGHT_ARM_INDICES)
                        left_ee_motor = select_motor_fields(motor_record, LEFT_EE_INDICES)
                        right_ee_motor = select_motor_fields(motor_record, RIGHT_EE_INDICES)

                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   left_motor["qvel"] if left_motor else [],
                            "qacc":   left_motor["qacc"] if left_motor else [],
                            "torque": left_motor["torque"] if left_motor else [],
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   right_motor["qvel"] if right_motor else [],
                            "qacc":   right_motor["qacc"] if right_motor else [],
                            "torque": right_motor["torque"] if right_motor else [],
                        },       
                        "left_arm_pose": {
                            "qpos": left_arm_pose_state.tolist(),
                            "qvel": [],
                            "torque": [],
                        },
                        "right_arm_pose": {
                            "qpos": right_arm_pose_state.tolist(),
                            "qvel": [],
                            "torque": [],
                        },                  
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   left_ee_motor["qvel"] if left_ee_motor else [],
                            "qacc":   left_ee_motor["qacc"] if left_ee_motor else [],
                            "torque": left_ee_motor["torque"] if left_ee_motor else [],
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   right_ee_motor["qvel"] if right_ee_motor else [],
                            "qacc":   right_ee_motor["qacc"] if right_ee_motor else [],
                            "torque": right_ee_motor["torque"] if right_ee_motor else [],
                        }, 
                        "body": {
                            "qpos": lowstate_record["motor"]["qpos"] if lowstate_record else current_body_state,
                            "qvel": lowstate_record["motor"]["qvel"] if lowstate_record else [],
                            "qacc": lowstate_record["motor"]["qacc"] if lowstate_record else [],
                            "torque": lowstate_record["motor"]["torque"] if lowstate_record else [],
                        }, 

                    }
                    if lowstate_record is not None:
                        states["body"].update({
                            "mode": lowstate_record["motor"]["mode"],
                            "temperature": lowstate_record["motor"]["temperature"],
                            "voltage": lowstate_record["motor"]["voltage"],
                            "sensor": lowstate_record["motor"]["sensor"],
                            "motor_state": lowstate_record["motor"]["motor_state"],
                            "reserve": lowstate_record["motor"]["reserve"],
                        })
                        states["imu"] = lowstate_record["imu"]
                        states["robot"] = lowstate_record["robot"]
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },     
                        "left_arm_pose": {
                            "qpos": left_arm_pose_action.tolist(),
                            "qvel": [],
                            "torque": [],
                        },
                        "right_arm_pose": {
                            "qpos": right_arm_pose_action.tolist(),
                            "qvel": [],
                            "torque": [],
                        },                     
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 

                        
                    }
                    if commanded_record is not None:
                        actions["commanded"] = commanded_record
                    if mobile_ctrl != None:
                        states["torso"] = {
                            "height": np.array(height_state[0]).tolist(),
                            "qvel": np.array(height_state[1]).tolist()
                        }
                        actions["torso"] = {
                            "qvel": np.array(height_action[0]).tolist()
                        }
                        if args.base_type == "mobile_lift":
                            states["chassis"] = {
                                "qvel": np.array(move_state).tolist()  # [x_vel, yaw_vel]
                            }
                            actions["chassis"] = {
                                "qvel": np.array(move_action).tolist()   # [x_vel, yaw_vel]
                            }
                    if args.use_waist_yaw and waist_yaw_state is not None and waist_yaw_action is not None:
                        states["waist_yaw"] = {
                            "qpos": waist_yaw_state.tolist(),  # [yaw]
                        }
                        actions["waist_yaw"] = {
                            "qpos": waist_yaw_action.tolist(),  # [yaw]
                        }
                    if args.use_waist_pitch and waist_pitch_state is not None and waist_pitch_action is not None:
                        # Already plain lists, unlike the yaw pair which comes off a numpy array.
                        states["waist_pitch"] = {
                            "qpos": waist_pitch_state,  # [pitch]
                        }
                        actions["waist_pitch"] = {
                            "qpos": waist_pitch_action,  # [pitch]
                        }

                    sample_monotonic_ns = time.monotonic_ns()
                    robot_tick = None
                    if lowstate_record is not None:
                        robot_tick = lowstate_record["robot"]["tick"]
                    timestamps = {
                        "episode_s": recorder.get_episode_elapsed_s(sample_monotonic_ns),
                        "monotonic_ns": sample_monotonic_ns,
                        "wall_time_ns": time.time_ns(),
                        "robot_tick": robot_tick,
                        "xr_motion_monotonic_s": (
                            float(tele_data.motion_data_timestamp)
                            if tele_data.motion_data_timestamp is not None
                            else None
                        ),
                    }
                    operator = build_operator_record(
                        tele_data,
                        controller_actions,
                        xr_motion_live,
                    )
                    operator["base_command"] = current_body_action
                    operator["base_command_applied"] = args.base_type != "legs"
                    operator["teleoperation_started"] = bool(arm_control_active)
                    quality = {
                        "xr_live": bool(xr_motion_live),
                        "cameras": {
                            "color_0": camera_quality_record(head_img, sample_monotonic_ns),
                            "color_1": camera_quality_record(head_img, sample_monotonic_ns),
                            "color_2": camera_quality_record(left_wrist_img, sample_monotonic_ns),
                            "color_3": camera_quality_record(right_wrist_img, sample_monotonic_ns),
                        },
                    }

                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(
                            colors=colors,
                            depths=depths,
                            states=states,
                            actions=actions,
                            sim_state=sim_state,
                            timestamps=timestamps,
                            operator=operator,
                            quality=quality,
                        )
                    else:
                        recorder.add_item(
                            colors=colors,
                            depths=depths,
                            states=states,
                            actions=actions,
                            timestamps=timestamps,
                            operator=operator,
                            quality=quality,
                        )

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    except Exception as e:
        logger_mp.error(f"Error: {e}")
    finally:
        if mobile_ctrl is not None:
            mobile_ctrl.g1_height_action_array_in[0] = 0.0
            if mobile_ctrl.g1_move_action_array_in is not None:
                mobile_ctrl.g1_move_action_array_in[:] = [0.0, 0.0]

        try:
            if waist_pitch_ctrl is not None:
                waist_pitch_ctrl.close()
        except Exception as e:
            logger_mp.error(f"Failed to home and stop waist pitch controller: {e}")

        set_shared_flag(output_enabled, False)
        
        try:
            if ipc_server is not None:
                ipc_server.stop()
            if listen_keyboard_thread is not None:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            if tv_wrapper is not None:
                tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        # try:
        #     if not args.motion:
        #         status, result = motion_switcher.Exit_Debug_Mode()
        #         logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        # except Exception as e:
        #     logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if recorder is not None:
                if globals().get("discard_after_reset_command", False):
                    recorder.discard_episode()
                if RECORD_RUNNING:
                    logger_mp.info("Finalizing and saving the active episode before exit...")
                    RECORD_RUNNING = False
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("Finally, exiting program.")
