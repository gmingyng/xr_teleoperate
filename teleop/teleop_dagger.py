"""Interactive G1-D DAgger: hold a PICO Grip to demonstrate a correction."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import time

import numpy as np

from teleop.utils.output_gate import motion_input_is_live
from teleop.utils.barcode import BarcodeReceiver
from unitree_openpi.policy_protocol import joint_state_vector, parse_joint_actions
from unitree_openpi.robot_ctrl.controller import EEFController
from unitree_openpi.robot_ctrl.dds import G1DDSBackend
from unitree_openpi.robot_ctrl.kinematics import G1DKinematics
from unitree_openpi.utils.cameras import CameraCapture
from unitree_openpi.utils.config import load_config
from unitree_openpi.utils.ownership import ControllerLease
from unitree_openpi.utils.websocket import PolicyClient

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
CAMERAS = dict(zip(("color_0","color_1","color_2","color_3"),
                   ("cam_left_high","cam_right_high","cam_left_wrist","cam_right_wrist")))


class ActionArbiter:
    def __init__(self):
        self.held = (False, False)
        self.epoch = 0
        self.chunk = deque()
        self.reference = {}

    def invalidate(self):
        self.epoch += 1
        self.chunk.clear()

    def transition(self, held, xr, state, poses):
        if held == self.held:
            return False
        self.invalidate()
        self.held = held
        self.reference.clear()
        self.hold_q = state["q14"].copy()
        self.hold_g = state["grip_q2"].copy()
        for i, side in enumerate(SIDES):
            if held[i]:
                self.reference[side] = (getattr(xr, side+"_wrist_pose").copy(), poses[side].copy(),
                                        getattr(xr, side+"_ctrl_triggerValue"))
        return True

    def correction(self, xr, seed, kin, limits, first):
        # A mode edge commands exactly the measured joint angles (no IK branch change).
        q = self.hold_q.copy()
        g = self.hold_g.copy()
        if not first:
            targets = {}
            for i, side in enumerate(SIDES):
                if not self.held[i]:
                    continue
                hand0, tool0, trigger0 = self.reference[side]
                hand = getattr(xr, side+"_wrist_pose")
                target = tool0.copy()
                target[:3, 3] += hand[:3, 3] - hand0[:3, 3]
                target[:3, :3] = hand[:3, :3] @ hand0[:3, :3].T @ tool0[:3, :3]
                targets[side] = target
                delta = getattr(xr, side+"_ctrl_triggerValue") - trigger0
                g[i] += delta * (limits[1,i] - limits[0,i]) / 10
            solved = kin.solve(targets, seed)
            for i, held in enumerate(self.held):
                if held:
                    q[7*i:7*i+7] = solved[7*i:7*i+7]
        g = np.clip(g, *limits)
        return np.r_[q[:7], g[0], q[7:], g[1]]


def parts(row):
    return {k: {"qpos": row[s].tolist()} for k,s in
            (("left_arm",slice(0,7)), ("left_ee",slice(7,8)),
             ("right_arm",slice(8,15)), ("right_ee",slice(15,16)))}


def predict(client, obs, state_dim):
    if client.connection is None:
        spec = client.connect()["g1d"]
        if (spec["action_space"], spec["action_representation"], spec["state_dim"],
            spec["action_dim"], spec.get("gripper_units")) != ("joint","absolute",state_dim,16,"motor_position"):
            raise ValueError("Incompatible G1-D joint policy")
    response = client.infer(obs)
    if response.get("observation_id", obs["_unitree"]["sequence"]) != obs["_unitree"]["sequence"]:
        raise ValueError("Mismatched observation_id")
    return parse_joint_actions(response)


def run(args):
    from teleop.televuer.tv_wrapper import TeleVuerWrapper
    from teleop.utils.episode_writer import EpisodeWriter
    cfg = load_config(ROOT/"unitree_openpi/config/g1d.json")
    cfg["policy"].update(url=args.url, action_space="joint")
    period = 1 / args.hz
    limits = np.asarray(cfg["policy"]["gripper_command_limits"])
    arb = ActionArbiter()
    with ExitStack() as stack:
        if args.execute:
            stack.enter_context(ControllerLease(cfg["device"]["controller_lock"]))
        ctrl = EEFController(cfg,G1DDSBackend(cfg),G1DKinematics(cfg),execute=args.execute,control_waist=False)
        stack.callback(ctrl.close)
        ctrl.connect()
        cam = CameraCapture(cfg["cameras"], head_views=("right",))
        stack.callback(cam.close)
        cam.start()
        xr_source = TeleVuerWrapper(False, display_mode="pass-through",arm_reference_mode="head_yaw")
        stack.callback(xr_source.close)
        scanner = BarcodeReceiver() if args.barcode_scanner else None
        if scanner:
            stack.callback(scanner.close)
            scanner.begin(time.monotonic_ns())
        def new_writer():
            return EpisodeWriter(str(args.task_dir), task_goal=args.prompt, frequency=args.hz,
                                 image_size=cfg["cameras"]["image_size"], rerun_log=False,
                                 extra_info={"camera_mapping": CAMERAS})

        # Each intervention owns its writer. A quick re-grip can open the next
        # episode while the previous writer drains its JPEG queue.
        finalizer = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        shared_writer = None
        if args.record_format == "lerobot_v3" and args.execute:
            from teleop.utils.v3_writer import V3Writer
            shared_writer = V3Writer(str(args.task_dir)+"_lerobot_v3", task_goal=args.prompt,
                                     frequency=args.hz, extra_info={"camera_mapping": CAMERAS})
            stack.callback(shared_writer.close)

        def finish_segment(writer):
            if writer is shared_writer:
                writer.save_episode()
            else:
                finalizer.submit(writer.close)

        recorder = None

        def close_active_writer():
            if recorder is not None and recorder is not shared_writer:
                recorder.close()

        stack.callback(close_active_writer)
        client = PolicyClient(cfg["policy"])
        stack.callback(client.close)
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        stack.callback(ctrl.freeze)
        pending = None
        active = armed = False
        prev_y = True
        last_flag = False
        seq = 0
        print("Release both Grips and press Y with live XR to arm. Hold Grip to take over. Ctrl+C exits.",flush=True)
        while True:
            tick = time.monotonic()
            xr = xr_source.get_tele_data()
            live = motion_input_is_live(xr.motion_data_ready,xr.motion_data_timestamp,.3)
            held = (bool(xr.left_ctrl_squeeze),bool(xr.right_ctrl_squeeze))
            y = bool(xr.left_ctrl_bButton)
            start = live and y and not prev_y and not any(held)
            prev_y = y if live else True
            if not live or (not armed and not start):
                if armed:
                    ctrl.freeze()
                    arb.invalidate()
                    arb.held = (False,False)
                    if active:
                        finish_segment(recorder)
                        recorder = None
                        active = False
                armed = False
                time.sleep(period)
                continue
            armed = True
            state, _ = ctrl.snapshot()
            frames = cam.snapshot()
            ns = time.monotonic_ns()
            meta = scanner.frame(ns) if scanner else {"scan_success":False,"scan_trigger":False,"barcode":None}
            flag = meta["scan_success"]
            if flag != last_flag:
                arb.invalidate()
                last_flag = flag
            edge = arb.transition(held,xr,state,poses)
            if edge:
                ctrl.freeze()
            human = any(held)
            if active and not human:
                finish_segment(recorder)
                recorder = None
                active = False
            feedback = joint_state_vector(state)
            images = {cfg["policy"]["image_keys"][k]: v["image"] for k,v in frames.items()}
            expires = min(state["received_monotonic_s"],*(v["received_monotonic_s"] for v in frames.values())) + cfg["policy"]["max_observation_age_s"]
            if pending and pending.done():
                if epoch == arb.epoch and not human and time.monotonic() < deadline:
                    arb.chunk.extend(pending.result()[:args.max_execute])
                pending = None
            if human:
                if not active and args.execute:
                    recorder = shared_writer or new_writer()
                    active = recorder.create_episode()
                    if not active:
                        raise RuntimeError("Unable to open correction episode")
                row = arb.correction(xr,state["q14"],ctrl.kin,limits,edge)
                expires = min(expires,xr.motion_data_timestamp+.3)
            elif arb.chunk and time.monotonic() < deadline:
                row = arb.chunk.popleft()
                expires = min(expires,deadline)
            else:
                arb.chunk.clear()
                if pending is None:
                    seq += 1
                    obs = {**images,"observation/state":np.r_[np.float32(flag),feedback] if scanner else feedback,
                           "prompt":args.prompt,"_unitree":{"sequence":seq,"scan_success":flag}}
                    epoch,deadline = arb.epoch,expires
                    pending = pool.submit(predict,client,obs,len(obs["observation/state"]))
                time.sleep(period)
                continue
            ctrl.apply(None,expires,joint_positions=np.r_[row[:7],row[8:15]],
                       gripper_positions=row[[7,15]],wait_for_arrival=False)
            if human and active:
                rgb = (images["observation/images/high"],frames["head"]["head_views"]["right"],
                       images["observation/images/left_wrist"],images["observation/images/right_wrist"])
                recorder.add_item({k:im[:,:,::-1].copy() for k,im in zip(CAMERAS,rgb)},
                                  states=parts(feedback),actions=parts(row),unitree=meta,
                                  timestamps={"monotonic_ns":ns,"episode_s":recorder.get_episode_elapsed_s(ns)},
                                  operator={"action_source":"human","left_clutch":held[0],"right_clutch":held[1]})
            time.sleep(max(0,period-(time.monotonic()-tick)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url",required=True)
    p.add_argument("--prompt",required=True)
    p.add_argument("--task-dir",type=Path,default=Path.home()/"g1d_dataset/dagger")
    p.add_argument("--hz",type=int,default=30)
    p.add_argument("--max-execute",type=int,default=20)
    p.add_argument("--barcode-scanner",action="store_true")
    p.add_argument("--execute",action="store_true",help="enable hardware actions and record executed corrections")
    p.add_argument("--record-format", choices=("raw","lerobot_v3"), default="raw")
    args = p.parse_args()
    if not 0 < args.hz <= 250 or not 0 < args.max_execute <= 64 or not args.prompt.strip():
        p.error("Invalid frequency, action count or empty prompt")
    try:
        run(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
