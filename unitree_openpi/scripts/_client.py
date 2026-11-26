"""Shared CLI setup for the G1-D WebSocket clients."""
import argparse
from contextlib import nullcontext
import logging
from pathlib import Path

from unitree_openpi.app import run, execute_policy
from unitree_openpi.robot_ctrl.controller import EEFController
from unitree_openpi.utils.cameras import CameraCapture, MockCameras
from unitree_openpi.utils.config import load_config
from unitree_openpi.utils.ownership import ControllerLease

log = logging.getLogger(__name__)


def launch(argv=None, *, prompt="", handle_response=execute_policy, control_waist=True, default_max_execute=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/g1d.json"))
    parser.add_argument("--url", help="OpenPI policy ws://HOST:PORT URL")
    parser.add_argument("--prompt", default=prompt)
    parser.add_argument("--execute", action="store_true", help="acquire and execute on the real robot")
    parser.add_argument("--mock", action="store_true", help="synthetic robot/cameras; never connects to DDS")
    parser.add_argument("--hz", type=float, default=50.0, help="absolute joint target updates per second (default: 50)")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N inferences; 0 means continuous")
    if default_max_execute is not None:
        parser.add_argument("--reset-first", action="store_true", help="reset and hold arms before inference")
        parser.add_argument("--max-execute", type=int, default=default_max_execute,
                            help="execute at most this many leading actions per chunk (default: %(default)s)")
    parser.add_argument("--record-mp4", nargs="?", const="recordings", metavar="DIR",
                        help="save one review MP4 per inference session (default directory: recordings)")
    parser.add_argument("--barcode-scanner", action="store_true", help="send a latched scan flag to a 17-D joint policy")
    args = parser.parse_args(argv)
    reset_first = getattr(args, "reset_first", False)
    if reset_first and not (args.execute or args.mock):
        parser.error("--reset-first requires --execute or --mock")
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    config["policy"]["barcode_scanner"] = args.barcode_scanner
    if default_max_execute is not None:
        if args.max_execute < 1:
            parser.error("--max-execute must be positive")
        config["policy"]["execute_steps"] = args.max_execute
    if not 0 < args.hz <= config["motion"]["publish_hz"]:
        parser.error("--hz must be positive and cannot exceed the DDS publish rate")
    config["policy"]["target_hz"] = args.hz
    config["policy"]["action_space"] = "joint"
    if not args.execute:
        config["policy"]["timeout_s"] = max(config["policy"]["timeout_s"], 120)
    if args.url:
        if not args.url.startswith(("ws://", "wss://")):
            parser.error("--url must start with ws:// or wss://")
        config["policy"]["url"] = args.url
    if args.record_mp4:
        config["cameras"]["record_right_eye"] = True
    if args.mock:
        from unitree_openpi.robot_ctrl.mock import MockBackend, MockKinematics
        backend, kin = MockBackend(config), MockKinematics()
        cameras = MockCameras(config["cameras"])
    else:
        from unitree_openpi.robot_ctrl.dds import G1DDSBackend
        from unitree_openpi.robot_ctrl.kinematics import G1DKinematics
        backend, kin = G1DDSBackend(config), G1DKinematics(config)
        cameras = CameraCapture(config["cameras"])
    controller = EEFController(config, backend, kin, execute=args.execute or args.mock,
                               control_waist=control_waist)
    lease = ControllerLease(config["device"]["controller_lock"]) if args.execute and not args.mock else nullcontext()
    try:
        with lease:
            run(config, controller, cameras, prompt=args.prompt, max_steps=args.max_steps,
                handle_response=handle_response, reset_first=reset_first, record_dir=args.record_mp4)
    except KeyboardInterrupt:
        log.info("Stopped")
    return 0
