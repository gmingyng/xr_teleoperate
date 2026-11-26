from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber # dds
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_ # idl
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Point32_,geometry_msgs_msg_dds__Twist_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__MotorCmd_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorCmd_, MotorState_
import threading
import time
import numpy as np
import pinocchio as pin
from multiprocessing import Process, Array, Value

import logging_mp
logger_mp = logging_mp.getLogger(__name__)
from teleop.robot_control.dds_utils import wait_for_dds
from teleop.utils.output_gate import shared_flag_is_set



kTopicHeightCmd = "rt/cmd_hispeed"
kTopicHeightState = "rt/hispeed_state"
kTopicG1MoveCmd = "rt/cmd_vel_no_limit"
kTopicG1MoveState = "rt/slamware_ros_sdk_server_node/odom"

kTopicR3Controller = "rt/wirelesscontroller"

kTopicWaistPitchCmd = "rt/waistpitchcmd"
kTopicWaistPitchState = "rt/waistpitchstate"

class G1D_Mobile_Lift_Controller:
    def __init__(self, base_type, r3_controller, fps = 30.0, output_enabled_in = None):
        """
        Initialize G1 mobile base and elevation controller
        
        Args:
            base_type: "only_height" for height only, "with_move" for height + movement
            r3_controller: "true" for R3 controller, "false" for XR controller
            fps: Control frequency
        """
        logger_mp.info("Initialize G1D_Mobile_Lift_Controller...")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = fps
        self.base_type = base_type 
        self.r3_controller = r3_controller
        self.output_enabled_in = output_enabled_in
        # Data reception flags
        self.height_data_received = False
        self.move_data_received = False

        # init buffer
        # For controlling height and movement commands
        self.g1_height_action_array_in = Array('d', 1, lock = True) 
        self.g1_move_action_array_in = Array('d', 2, lock = True)


        # For receiving height and movement state
        self.g1_height_state_array_out  = Array('d', 2, lock=True)  
        self.g1_height_action_array_out = Array('d', 1, lock=True)  # For receiving published height action values, ready to save to dataset
        self.g1_move_state_array_out = None
        self.g1_move_action_array_out = None  # For receiving published movement action values, ready to save to dataset

        self.r3_controller_state_array_out = None

        # Height control publisher
        self.HeightCmb_publisher = ChannelPublisher(kTopicHeightCmd, Point32_)
        self.HeightCmb_publisher.Init()
        # Height state subscriber
        self.HeightState_subscriber = ChannelSubscriber(kTopicHeightState, Point32_)
        self.HeightState_subscriber.Init()

        self.g1_height_msg = geometry_msgs_msg_dds__Point32_()

        # When base_type is with_move, use movement control; otherwise only use height control
        if self.base_type == "mobile_lift":
            self.g1_move_state_array_out = Array('d', 2, lock=True)
            self.g1_move_action_array_out = Array('d', 2, lock=True)  # For receiving published movement action values, ready to save to dataset
            # Movement control publisher
            self.G1MoveCmb_publisher = ChannelPublisher(kTopicG1MoveCmd, Twist_)
            self.G1MoveCmb_publisher.Init()
            self.g1_move_msg = geometry_msgs_msg_dds__Twist_()
            # Movement state subscriber
            self.G1MoveState_subscriber = ChannelSubscriber(kTopicG1MoveState, Odometry_)
            self.G1MoveState_subscriber.Init()

        self.subscribe_g1_mobilebase_state_thread = threading.Thread(target=self._subscribe_g1_mobilebase_state)
        self.subscribe_g1_mobilebase_state_thread.daemon = True
        self.subscribe_g1_mobilebase_state_thread.start()

        wait_for_dds(
            lambda: self.height_data_received and (self.base_type != "mobile_lift" or self.move_data_received),
            "G1D_Mobile_Lift_Controller"
        )
        # If not using Unitree controller, start control process
        if self.r3_controller:
            self.r3_controller_state_array_out = Array('d', 5, lock=True)
            self.r3_controller_state_subscriber = ChannelSubscriber(kTopicR3Controller, WirelessController_)
            self.r3_controller_state_subscriber.Init()
            self.subscribe_r3_controller_state_thread = threading.Thread(target=self._subscribe_r3_controller_state)
            self.subscribe_r3_controller_state_thread.daemon = True
            self.subscribe_r3_controller_state_thread.start()
        self.running = True
        mobile_control_process = Process(target=self.control_process, args=(self.base_type, self.output_enabled_in))
        mobile_control_process.daemon = True
        mobile_control_process.start()

        logger_mp.info("Initialize G1D_Mobile_Lift_Controller OK!\n")

    def _subscribe_r3_controller_state(self):
        while True:
            try:
                r3_controller_msg = self.r3_controller_state_subscriber.Read()
                if r3_controller_msg is not None:
                    self.r3_controller_state_array_out[1] = r3_controller_msg.lx
                    self.r3_controller_state_array_out[0] = r3_controller_msg.ly
                    self.r3_controller_state_array_out[2] = r3_controller_msg.rx
                    self.r3_controller_state_array_out[3] = r3_controller_msg.ry
                    self.r3_controller_state_array_out[4] = r3_controller_msg.keys
            except Exception as e:
                logger_mp.info(f"[_subscribe_r3_controller_state] Exception: {e}")
                time.sleep(0.1)
            time.sleep(0.01)

    def _subscribe_g1_mobilebase_state(self):
        while True:
            try:
                height_msg = self.HeightState_subscriber.Read()
                if height_msg is not None:
                    self.g1_height_state_array_out[0] = height_msg.y  # in meters
                    self.g1_height_state_array_out[1] = height_msg.z
                    
                    if not self.height_data_received:
                        self.height_data_received = True
                        
                if self.base_type == "mobile_lift":
                    move_msg = self.G1MoveState_subscriber.Read()
                    if move_msg is not None:
                        self.g1_move_state_array_out[0] = move_msg.twist.twist.linear.x
                        self.g1_move_state_array_out[1] = move_msg.twist.twist.angular.z
                        
                        if not self.move_data_received:
                            self.move_data_received = True

                time.sleep(0.01)
                
            except Exception as e:
                logger_mp.info(f"[_subscribe_g1_mobilebase_state] Exception: {e}")
                time.sleep(0.1) 

    def ctrl_g1_height(self, g1_height_target):
        self.g1_height_msg.z = g1_height_target
        self.HeightCmb_publisher.Write(self.g1_height_msg)

    def ctrl_g1_move(self, g1_move_target):
        self.g1_move_msg.linear.x = g1_move_target[0]
        self.g1_move_msg.angular.z = g1_move_target[1]
        self.G1MoveCmb_publisher.Write(self.g1_move_msg)

    def control_process(self, base_type, output_enabled_in = None):
        was_enabled = False
        try:
            while self.running:
                self.start_time = time.time()
                output_enabled = shared_flag_is_set(output_enabled_in)
                if not output_enabled:
                    if was_enabled:
                        self.ctrl_g1_height(0.0)
                        if base_type == "mobile_lift":
                            self.ctrl_g1_move([0.0, 0.0])
                        logger_mp.warning("Mobile/lift output disabled; zero-speed stop sent.")
                    was_enabled = False
                    time.sleep(1 / self.fps)
                    continue
                was_enabled = True
                target_height = self.g1_height_action_array_in[0]
                self.ctrl_g1_height(target_height)
                
                if base_type == "mobile_lift":
                    g1_move_target = self.g1_move_action_array_in
                    self.ctrl_g1_move(g1_move_target)
                
                current_time = time.time()
                time_elapsed = current_time - self.start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("G1D_Mobile_Lift_Controller has been closed.")

class G1D_WaistPitchGravityFeedforward:
    """Gravity torque on the waist pitch joint, computed from the G1-D URDF.

    Only the pitch joint and the two arms are kept. The wheels and the lift column sit
    below the pitch joint and carry no weight it has to lift, and locking the wheels also
    keeps their continuous joints out of the configuration vector. The arms are kept
    because holding them out in front raises the torque from 44 to 64 N*m.
    """
    joint_names = ["torso_Joint"] + [f"{side}_{name}_joint" for side in ("left", "right")
                                     for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                                                  "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]

    def __init__(self):
        model = pin.buildModelFromUrdf("../assets/g1_d/g1_d.urdf")
        locked = [model.getJointId(name) for name in model.names[1:] if name not in self.joint_names]
        self.model = pin.buildReducedModel(model, locked, pin.neutral(model))
        self.data = self.model.createData()
        self.idxs = np.array([self.model.joints[self.model.getJointId(name)].idx_q
                              for name in self.joint_names])
        self.zeros = np.zeros(self.model.nv)

    def compute(self, q):
        """q: [pitch, 14 arm joints] in URDF order. Returns the pitch holding torque."""
        full_q = pin.neutral(self.model)
        full_q[self.idxs] = q
        return pin.rnea(self.model, self.data, full_q, self.zeros, self.zeros)[self.idxs[0]]

class G1D_Mobile_Waist_Pitch_Controller:
    kp = 500.0
    kd = 12.0
    mode = 1                 # FOC with command-timeout protection
    min_angle = 0.0          # rad, upright
    max_angle = 2.3562       # rad, bent over
    travel_seconds = 5.0     # s to sweep the full range while a button is held
    DELTA_WAIST_PITCH_CMD = 0.10  # rad, cap on how far the command may lead the measured angle

    def __init__(self, arm_ctrl, fps = 200.0, output_enabled_in = None):
        logger_mp.info("Initialize G1D_Mobile_Waist_Pitch_Controller...")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = fps
        # The arm angles swing the gravity load on the pitch joint by about 20 N*m, so the
        # feedforward reads them live from the arm controller.
        self.arm_ctrl = arm_ctrl
        self.output_enabled_in = output_enabled_in
        self.waist_pitch_data_received = False
        # Holding a button sweeps the whole range in travel_seconds, so the speed follows
        # from the limits rather than being tuned separately.
        self.velocity = (self.max_angle - self.min_angle) / self.travel_seconds
        self.gravity_ff = G1D_WaistPitchGravityFeedforward()

        # Direction: -1 = pitch up, 0 = hold, +1 = bend over.
        self.g1_waist_pitch_action_value_in = Value('d', 0.0, lock = True)
        self.g1_waist_pitch_state_value_out  = Value('d', 0.0, lock=True)
        self.g1_waist_pitch_action_value_out = Value('d', 0.0, lock=True)

        # Waist pitch command publisher
        self.WaistPitchCmd_publisher = ChannelPublisher(kTopicWaistPitchCmd, MotorCmd_)
        self.WaistPitchCmd_publisher.Init()
        # Waist pitch state subscriber
        self.WaistPitchState_subscriber = ChannelSubscriber(kTopicWaistPitchState, MotorState_)
        self.WaistPitchState_subscriber.Init()

        self.subscribe_waist_pitch_state_thread = threading.Thread(target=self._subscribe_waist_pitch_state)
        self.subscribe_waist_pitch_state_thread.daemon = True
        self.subscribe_waist_pitch_state_thread.start()

        wait_for_dds(lambda: self.waist_pitch_data_received, "G1D_Mobile_Waist_Pitch_Controller")

        self.g1_waist_pitch_action_value_in.value = 0.0
        self.g1_waist_pitch_action_value_out.value = self.g1_waist_pitch_state_value_out.value
        logger_mp.info(f"Current waist pitch q: {self.g1_waist_pitch_state_value_out.value:.4f} rad")

        self.running = True
        self.waist_pitch_control_thread = threading.Thread(target=self.control_thread)
        self.waist_pitch_control_thread.daemon = True
        self.waist_pitch_control_thread.start()

        logger_mp.info("Initialize G1D_Mobile_Waist_Pitch_Controller OK!\n")

    def _subscribe_waist_pitch_state(self):
        while True:
            try:
                waist_pitch_msg = self.WaistPitchState_subscriber.Read()
                if waist_pitch_msg is not None:
                    self.g1_waist_pitch_state_value_out.value = waist_pitch_msg.q

                    if not self.waist_pitch_data_received:
                        self.waist_pitch_data_received = True
            except Exception as e:
                logger_mp.info(f"[_subscribe_waist_pitch_state] Exception: {e}")
                time.sleep(0.1)
            time.sleep(0.005)

    def get_current_waist_pitch_state(self):
        return self.g1_waist_pitch_state_value_out.value

    def get_current_waist_pitch_action(self):
        return self.g1_waist_pitch_action_value_out.value

    def set_waist_pitch_buttons(self, button_bend, button_raise):
        # Pressing both buttons, or neither, is no request.
        direction = 0.0 if bool(button_bend) == bool(button_raise) else (1.0 if button_bend else -1.0)
        self.g1_waist_pitch_action_value_in.value = direction
        return direction

    def ctrl_waist_pitch_go_home(self, tolerance = 0.05, timeout = 6.0):
        logger_mp.info("[G1D_Mobile_Waist_Pitch_Controller] ctrl_waist_pitch_go_home start...")
        self.set_waist_pitch_buttons(button_bend = False, button_raise = True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if abs(self.g1_waist_pitch_state_value_out.value - self.min_angle) < tolerance:
                logger_mp.info("[G1D_Mobile_Waist_Pitch_Controller] Waist pitch has reached the home position.")
                break
            time.sleep(0.02)
        else:
            logger_mp.warning(f"[G1D_Mobile_Waist_Pitch_Controller] Waist pitch did not reach home within "
                              f"{timeout}s, stopping at {self.g1_waist_pitch_state_value_out.value:.4f} rad.")
        self.set_waist_pitch_buttons(button_bend = False, button_raise = False)
        time.sleep(0.1)   # let the control thread publish the settled command before it stops

    def close(self):
        try:
            if shared_flag_is_set(self.output_enabled_in):
                self.ctrl_waist_pitch_go_home()
        finally:
            self.running = False
            self.waist_pitch_control_thread.join(timeout = 0.5)
            logger_mp.info("[G1D_Mobile_Waist_Pitch_Controller] closed.")

    def ctrl_g1_waist_pitch(self, q_target):
        self.g1_waist_pitch_msg.q = q_target
        self.WaistPitchCmd_publisher.Write(self.g1_waist_pitch_msg)

    def control_thread(self):
        self.g1_waist_pitch_msg = unitree_hg_msg_dds__MotorCmd_()
        self.g1_waist_pitch_msg.dq = 0.0
        self.g1_waist_pitch_msg.kp = self.kp
        self.g1_waist_pitch_msg.kd = self.kd
        self.g1_waist_pitch_msg.mode = self.mode
        q_cmd = self.g1_waist_pitch_state_value_out.value  # what was last sent, and where the next integration starts
        try:
            last_time = time.time()
            while self.running:
                self.start_time = time.time()
                if not shared_flag_is_set(self.output_enabled_in):
                    last_time = self.start_time
                    time.sleep(1 / self.fps)
                    continue
                # Clamp dt so a scheduling hiccup cannot integrate a large jump.
                dt = min(self.start_time - last_time, 5.0 / self.fps)
                last_time = self.start_time

                direction = self.g1_waist_pitch_action_value_in.value
                measured = self.g1_waist_pitch_state_value_out.value

                if direction > 0.5:
                    velocity = self.velocity
                elif direction < -0.5:
                    velocity = -self.velocity
                else:
                    velocity = 0.0

                q_target = np.clip(q_cmd + velocity * dt, self.min_angle, self.max_angle)
                q_cmd = np.clip(q_target, measured - self.DELTA_WAIST_PITCH_CMD, measured + self.DELTA_WAIST_PITCH_CMD)

                # Gravity is fed forward so kp only has to cover friction: it peaks at
                # 64 N*m, more than the DELTA_WAIST_PITCH_CMD clamp leaves kp to give.
                self.g1_waist_pitch_msg.tau = self.gravity_ff.compute(np.concatenate(([measured], self.arm_ctrl.get_current_dual_arm_q())))
                self.ctrl_g1_waist_pitch(q_cmd)
                self.g1_waist_pitch_action_value_out.value = q_cmd

                current_time = time.time()
                time_elapsed = current_time - self.start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("G1D_Mobile_Waist_Pitch_Controller has been closed.")
