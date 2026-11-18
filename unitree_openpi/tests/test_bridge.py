import copy
import importlib.util
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation
from websockets.sync.server import serve

from unitree_openpi.app import observation, run
from unitree_openpi.mock_policy_server import make_handler
from unitree_openpi.robot_ctrl.controller import EEFController, parse_actions
from unitree_openpi.robot_ctrl.dds import G1DDSBackend
from unitree_openpi.robot_ctrl.mock import MockBackend, MockKinematics
from unitree_openpi.utils.cameras import CameraCapture, MockCameras, prepare_rgb
from unitree_openpi.utils.config import load_config, normalize_config
from unitree_openpi.utils.transforms import relative_target
from unitree_openpi.utils.wire import packb, unpackb, json_encode, json_decode

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "config/g1d.json")


def controller(settings=None, execute=True):
    settings = config() if settings is None else settings
    backend = MockBackend(settings)
    ctrl = EEFController(settings, backend, MockKinematics(), execute=execute)
    ctrl.connect()
    return ctrl, backend


class RelativeActions(unittest.TestCase):
    def test_base_and_tool_translation_and_noncommuting_rotation(self):
        current = np.eye(4)
        current[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        delta = [.01, 0, 0, .1, 0, 0]
        base, tool = (relative_target(current, delta, frame) for frame in ("base", "tool"))
        np.testing.assert_allclose(base[:3, 3], [.01, 0, 0])
        np.testing.assert_allclose(tool[:3, 3], [0, .01, 0], atol=1e-12)
        dr = Rotation.from_rotvec([.1, 0, 0]).as_matrix()
        np.testing.assert_allclose(base[:3, :3], dr @ current[:3, :3])
        np.testing.assert_allclose(tool[:3, :3], current[:3, :3] @ dr)

    def test_relative_move_uses_measured_pose_and_preserves_other_arm(self):
        ctrl, backend = controller()
        try:
            backend.q[0] = .02
            before = ctrl.kin.fk(backend.q)
            action = np.zeros(14)
            action[0] = .01
            result = ctrl.apply(action, time.monotonic() + 1)
            after = ctrl.kin.fk(backend.q)
            self.assertEqual(result["status"], "reached")
            np.testing.assert_allclose(after["left"][:3, 3] - before["left"][:3, 3], [.01, 0, 0])
            np.testing.assert_allclose(before["right"], after["right"])
        finally:
            ctrl.close()

    def test_invalid_stale_and_outside_workspace_do_not_publish(self):
        ctrl, backend = controller()
        try:
            for action in (np.full(14, np.nan), np.r_[.5, np.zeros(13)]):
                with self.assertRaises(ValueError):
                    ctrl.apply(action, time.monotonic() + 1)
            with self.assertRaises(TimeoutError):
                ctrl.apply(np.zeros(14), time.monotonic() - 1)
            ctrl.config["model"]["workspace"]["left"][1][0] = .1
            with self.assertRaises(ValueError):
                ctrl.apply(np.zeros(14), time.monotonic() + 1)
            self.assertFalse(backend.commands)
        finally:
            ctrl.close()

    def test_action_expires_during_ik_and_does_not_publish(self):
        ctrl, backend = controller()
        original = ctrl.kin.solve
        def slow(target, seed):
            time.sleep(.03)
            return original(target, seed)
        ctrl.kin.solve = slow
        try:
            with self.assertRaises(TimeoutError):
                ctrl.apply(np.zeros(14), time.monotonic() + .01)
            self.assertFalse(backend.commands)
        finally:
            ctrl.close()

    def test_joint_jump_is_rejected(self):
        ctrl, backend = controller()
        ctrl.kin.solve = lambda targets, seed: seed + 1
        try:
            with self.assertRaisesRegex(ValueError, "joint step"):
                ctrl.apply(np.zeros(14), time.monotonic() + 1)
            self.assertFalse(backend.commands)
        finally:
            ctrl.close()

    def test_observation_mode_never_acquires_or_commands(self):
        ctrl, backend = controller(execute=False)
        try:
            result = ctrl.apply(np.zeros(14), time.monotonic() + 1)
            self.assertEqual(result["status"], "observing")
            self.assertFalse(backend.acquired)
            self.assertFalse(backend.commands)
        finally:
            ctrl.close()

    def test_grippers_are_absolute_and_calibrated(self):
        settings = config()
        settings["policy"].update(include_grippers=True, open_q2=[2., -2.], closed_q2=[0., 0.])
        ctrl, backend = controller(settings)
        try:
            action = np.zeros(14)
            action[[6, 13]] = [.25, .75]
            ctrl.apply(action, time.monotonic() + 1)
            np.testing.assert_allclose(backend.grip, [.5, -1.5])
            obs, _ = observation(ctrl, MockCameras(settings["cameras"]), settings, 1, "test")
            np.testing.assert_allclose(obs["observation/state"][[6, 13]], [.25, .75])
            bad = action.copy()
            bad[6] = 2
            with self.assertRaises(ValueError):
                parse_actions({"actions": [bad]}, settings["policy"])
        finally:
            ctrl.close()

    def test_complete_chunk_checked_before_execution(self):
        settings = config()["policy"]
        for actions in ([[]], np.zeros((65, 12)), np.zeros((1, 13)), [[np.nan]*12]):
            with self.assertRaises(ValueError):
                parse_actions({"actions": actions}, settings)


class WireProtocol(unittest.TestCase):
    def test_roundtrip_uses_official_openpi_numpy_tags(self):
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        packed = packb({"image": image, "state": np.array([.1, .2], np.float32)})
        raw = msgpack.unpackb(packed, raw=False)
        self.assertTrue(raw["image"][b"__ndarray__"])
        self.assertEqual(raw["image"][b"dtype"], "|u1")
        result = unpackb(packed)
        np.testing.assert_array_equal(result["image"], image)
        np.testing.assert_allclose(result["state"], [.1, .2])

    def test_rejects_object_dtypes_and_bad_array_lengths(self):
        with self.assertRaises(ValueError):
            packb(np.array([object()], dtype=object))
        for dtype, shape, payload in (("O", [1], b"12345678"), ("<f8", [2], b"0")):
            bad = msgpack.packb({b"__ndarray__": True, b"dtype": dtype,
                                b"shape": shape, b"data": payload}, use_bin_type=True)
            with self.assertRaises(ValueError):
                unpackb(bad)

    def test_observation_has_absolute_state_rgb_and_ack(self):
        ctrl, backend = controller()
        try:
            backend.q[0] = .05
            backend.q[3:6] = [.2, -.3, .4]
            settings = config()
            ctrl.last_action = {"status": "reached", "observation_id": 1}
            obs, created = observation(ctrl, MockCameras(settings["cameras"]), settings, 2, "pick")
            self.assertAlmostEqual(float(obs["observation/state"][0]), .35)
            self.assertEqual(obs["observation/state"].shape, (14,))
            np.testing.assert_allclose(obs["observation/state"][3:6], Rotation.from_euler("xyz", [.2, -.3, .4]).as_rotvec(), atol=1e-6)
            self.assertEqual({key for key in obs if key.startswith("observation/images/")},
                             {"observation/images/high", "observation/images/left_wrist", "observation/images/right_wrist"})
            self.assertEqual(obs["observation/images/high"].shape, (224, 224, 3))
            self.assertEqual(obs["observation/images/high"].dtype, np.uint8)
            self.assertEqual(obs["_unitree"]["last_action"]["status"], "reached")
            self.assertLessEqual(created, time.monotonic())
        finally:
            ctrl.close()

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV unavailable")
    def test_json_jpeg_is_rgb_and_state_stays_numeric(self):
        image = np.zeros((16, 16, 3), np.uint8)
        image[:, :, 0] = 255
        result = json_decode(json_encode({"image": image, "state": np.zeros(12)}))
        self.assertGreater(result["image"][0, 0, 0], 245)
        self.assertLess(result["image"][0, 0, 2], 10)
        self.assertEqual(result["state"], [0.]*12)


class CameraTests(unittest.TestCase):
    def test_stale_camera_and_skew_reject_observation(self):
        settings = config()["cameras"]
        capture = CameraCapture(settings)
        now = time.monotonic()
        for name in settings["streams"]:
            capture._frames[name] = dict(image=np.zeros((2, 2, 3), np.uint8),
                sequence=1, received_monotonic_s=now, received_at=time.time())
        capture.snapshot()
        capture._frames["head"]["received_monotonic_s"] = now - 10
        with self.assertRaisesRegex(RuntimeError, "stale"):
            capture.snapshot()
        capture._frames["head"]["received_monotonic_s"] = now - .3
        with self.assertRaisesRegex(RuntimeError, "skew"):
            capture.snapshot()

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV unavailable")
    def test_head_crop_rgb_conversion_and_letterbox(self):
        settings = config()["cameras"]
        settings["image_size"] = [4, 4]
        bgr = np.zeros((2, 8, 3), np.uint8)
        bgr[:, :4, 2] = 255
        bgr[:, 4:, 0] = 255
        rgb = prepare_rgb(bgr, "head", settings)
        np.testing.assert_array_equal(rgb[1, 0], [255, 0, 0])
        np.testing.assert_array_equal(rgb[0], np.zeros((4, 3), np.uint8))


class DDSChecks(unittest.TestCase):
    def test_two_axis_waist_command_mapping(self):
        from unitree_openpi.robot_ctrl.dds import WAIST_SLOTS, ARM_SLOTS, GRIP_SLOTS
        backend = G1DDSBackend(config())
        commands = [SimpleNamespace(mode=0, q=0., kp=0., kd=0.) for _ in range(35)]
        backend._message = SimpleNamespace(motor_cmd=commands)
        backend._enable_slots(ARM_SLOTS, "arm", np.full(14, .2))
        backend._enable_slots(WAIST_SLOTS, "waist", [.4, 0, .1])
        backend._enable_slots(GRIP_SLOTS, "gripper", [.7, -.7])
        self.assertEqual(commands[12].q, .4)
        self.assertEqual(commands[13].q, .1)
        self.assertEqual(commands[14].mode, 0)
        self.assertTrue(all(commands[i].mode == 0 for i in range(12)))
        self.assertEqual(commands[15].q, .2)
        self.assertEqual(commands[33].q, -.7)

    def test_g1d_slots_and_receive_timestamps_do_not_advance_on_read(self):
        backend = G1DDSBackend(config())
        motors = [SimpleNamespace(q=float(i)/100, dq=float(i)/1000) for i in range(35)]
        backend._receive("arm", SimpleNamespace(motor_state=motors, mode_machine=5))
        backend._connected = True
        first = backend.read()
        time.sleep(.002)
        second = backend.read()
        np.testing.assert_allclose(first["q14"], np.arange(15, 29)/100)
        np.testing.assert_allclose(first["waist_q3"], [.12, 0, .13])
        np.testing.assert_allclose(first["grip_q2"], [.31, .33])
        self.assertEqual(first["received_monotonic_s"], second["received_monotonic_s"])
        backend._received["arm"] = time.monotonic() - 10
        with self.assertRaisesRegex(RuntimeError, "stale"):
            backend.read()

    def test_watchdog_cancels_unsent_motion_and_leaves_unrelated_motors_disabled(self):
        settings = config()
        backend = G1DDSBackend(settings)
        motors = [SimpleNamespace(q=0., dq=0.) for _ in range(35)]
        backend._receive("arm", SimpleNamespace(motor_state=motors, mode_machine=5))
        backend._connected = True
        backend._initial_q = backend._initial_tau = np.zeros(14)
        backend._last_sent_q = backend._last_sent_tau = np.zeros(14)
        backend._target_q, backend._target_tau = np.ones(14), np.zeros(14)
        backend._grip_target = backend._last_sent_grip = np.zeros(2)
        backend._waist_target = np.zeros(3)
        backend._last_command = time.monotonic() - 10
        commands = [SimpleNamespace(mode=0, q=0., dq=0., tau=0.) for _ in range(35)]
        backend._message = SimpleNamespace(motor_cmd=commands, crc=0)
        backend._crc = SimpleNamespace(Crc=lambda msg: 0)
        sent = []
        backend._arm_publisher = SimpleNamespace(Write=lambda msg: sent.append(msg))
        backend._grip_publishers = {}
        backend._publish_once()
        np.testing.assert_array_equal(backend._last_sent_q, np.zeros(14))
        self.assertEqual(len(sent), 1)
        self.assertTrue(all(commands[i].mode == 0 for i in range(15)))


class EndToEnd(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV unavailable")
    def test_json_socket_transmits_three_images_and_executes(self):
        settings = config()
        settings["policy"]["protocol"] = "json"
        with serve(make_handler("json", .001), "127.0.0.1", 0, compression=None) as server:
            settings["policy"]["url"] = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            backend = MockBackend(settings)
            ctrl = EEFController(settings, backend, MockKinematics(), execute=True)
            try:
                self.assertEqual(run(settings, ctrl, MockCameras(settings["cameras"]), max_steps=2), 2)
            finally:
                server.shutdown()
                thread.join(timeout=2)
        self.assertEqual(len(backend.commands), 2)
        self.assertAlmostEqual(backend.q[0], .002, places=7)

    def test_named_joint_and_eef_over_socket(self):
        settings = config()
        def policy(connection):
            connection.send(packb({}))
            for index in range(3):
                obs = unpackb(connection.recv())
                joints = np.array(obs["_unitree"]["joint_positions"])
                joints[0] += .01
                actions = {"joint_positions": joints[None]}
                if index:
                    eef = np.zeros((1, 14))
                    eef[0, 0] = .01 if index == 1 else -.01
                    actions["eef"] = eef
                connection.send(packb({"actions": actions}))
        backend = MockBackend(settings)
        ctrl = EEFController(settings, backend, MockKinematics(), execute=True)
        with serve(policy, "127.0.0.1", 0) as server:
            settings["policy"]["url"] = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with self.assertRaisesRegex(ValueError, "targets disagree"):
                    run(settings, ctrl, MockCameras(settings["cameras"]), max_steps=3)
            finally:
                server.shutdown()
                thread.join(timeout=2)
        self.assertEqual(len(backend.commands), 2)
        self.assertAlmostEqual(backend.q[0], .02)

    def test_openpi_metadata_images_action_and_feedback_over_real_socket(self):
        settings = config()
        received = []
        handler = make_handler(delta_x=.001)
        def inspect(connection):
            # Exercise the real binary handshake and observation wire format.
            class Proxy:
                def send(self, payload):
                    connection.send(payload)
                def __iter__(self):
                    for payload in connection:
                        received.append(unpackb(payload))
                        yield payload
            handler(Proxy())
        with serve(inspect, "127.0.0.1", 0, compression=None) as server:
            settings["policy"]["url"] = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            backend = MockBackend(settings)
            ctrl = EEFController(settings, backend, MockKinematics(), execute=True)
            try:
                self.assertEqual(run(settings, ctrl, MockCameras(settings["cameras"]), max_steps=3), 3)
            finally:
                server.shutdown()
                thread.join(timeout=2)
        self.assertEqual(len(received), 3)
        self.assertEqual(len(backend.commands), 3)
        self.assertAlmostEqual(backend.q[0], .003, places=7)
        self.assertEqual(received[1]["_unitree"]["last_action"]["status"], "reached")
        self.assertEqual(received[0]["observation/images/right_wrist"].shape, (224, 224, 3))

    def test_late_policy_reply_never_reaches_robot(self):
        settings = config()
        settings["policy"]["max_observation_age_s"] = .02
        def delayed(connection):
            connection.send(packb({"mock": True}))
            connection.recv()
            time.sleep(.05)
            connection.send(packb({"actions": np.zeros((1, 12))}))
        with serve(delayed, "127.0.0.1", 0, compression=None) as server:
            settings["policy"]["url"] = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            backend = MockBackend(settings)
            ctrl = EEFController(settings, backend, MockKinematics(), execute=True)
            try:
                with self.assertRaises(TimeoutError):
                    run(settings, ctrl, MockCameras(settings["cameras"]), max_steps=1)
            finally:
                server.shutdown()
                thread.join(timeout=2)
        self.assertFalse(backend.commands)


class DeploymentConfig(unittest.TestCase):
    def test_bundled_model_and_frames_are_g1d(self):
        import xml.etree.ElementTree as ET
        settings = config()
        root = ET.parse(settings["model"]["urdf"]).getroot()
        self.assertEqual(root.attrib["name"], "G1_D")
        links = {link.attrib["name"] for link in root.findall("link")}
        for name in ("reference_frame", "left_frame", "right_frame"):
            self.assertIn(settings["model"][name], links)
        joints = {joint.attrib["name"]: joint.attrib["type"] for joint in root.findall("joint")}
        self.assertEqual(joints["Left_Wheel_Joint"], "continuous")
        self.assertEqual(joints["Right_Wheel_Joint"], "continuous")
        for name in settings["model"]["waist_joint_map"]:
            self.assertIn(name, joints)

    def test_invalid_wire_keys_and_uncalibrated_grippers_fail_configuration(self):
        raw = config()
        for change in (dict(image_container_key="observation/state"), dict(state_key="_unitree"),
                       dict(open_q2=None), dict(reconnect_s=-1)):
            settings = copy.deepcopy(raw)
            settings["policy"].update(change)
            with self.assertRaises((ValueError, TypeError)):
                normalize_config(settings)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV unavailable")
class ZMQCapture(unittest.TestCase):
    def test_three_jpeg_streams_are_received_and_expire(self):
        import cv2
        import zmq
        settings = config()["cameras"]
        settings.update(stale_after_s=.15, startup_timeout_s=3, max_skew_s=.5,
                        head_view="full", image_size=[16, 16])
        ready, stopped = threading.Event(), threading.Event()
        endpoints, errors = {}, []
        def publish():
            context = zmq.Context()
            sockets = []
            try:
                for name in settings["streams"]:
                    socket = context.socket(zmq.PUB)
                    sockets.append(socket)
                    port = socket.bind_to_random_port("tcp://127.0.0.1")
                    endpoints[name] = f"tcp://127.0.0.1:{port}"
                ready.set()
                bgr = np.zeros((16, 16, 3), np.uint8)
                bgr[:, :, 2] = 255
                ok, jpeg = cv2.imencode(".jpg", bgr)
                assert ok
                while not stopped.is_set():
                    for socket in sockets:
                        socket.send(jpeg.tobytes())
                    stopped.wait(.02)
            except Exception as exc:
                errors.append(exc)
                ready.set()
            finally:
                for socket in sockets:
                    socket.close(0)
                context.term()
        thread = threading.Thread(target=publish, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(3))
        self.assertFalse(errors)
        settings["streams"] = endpoints.copy()
        capture = CameraCapture(settings)
        try:
            capture.start()
            frames = capture.snapshot()
            self.assertEqual(set(frames), {"head", "left_wrist", "right_wrist"})
            for frame in frames.values():
                self.assertGreater(frame["image"][0, 0, 0], 245)
                self.assertLess(frame["image"][0, 0, 2], 10)
            stopped.set()
            thread.join(timeout=2)
            time.sleep(.25)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                capture.snapshot()
        finally:
            stopped.set()
            thread.join(timeout=2)
            capture.close()


@unittest.skipUnless(importlib.util.find_spec("pinocchio"), "Pinocchio unavailable")
class NumericalKinematics(unittest.TestCase):
    def test_official_g1d_model_fk_ik_roundtrip_and_continuous_wheels(self):
        from unitree_openpi.robot_ctrl.kinematics import G1DKinematics
        kin = G1DKinematics(config())
        state = {"waist_q3": [0, 0, .1]}
        kin.update_state(state)
        q = np.zeros(14)
        poses = kin.fk(q)
        target_q = q.copy()
        target_q[[0, 7]] = .02
        targets = kin.fk(target_q)
        result = kin.solve(targets, q)
        achieved = kin.fk(result)
        for side in targets:
            np.testing.assert_allclose(achieved[side][:3, 3], targets[side][:3, 3], atol=.003)
            self.assertEqual(poses[side].shape, (4, 4))
        self.assertTrue(np.isfinite(kin.gravity(q)).all())
        # A single-arm request must preserve the other arm exactly.
        result = kin.solve({"left": targets["left"]}, q)
        np.testing.assert_array_equal(result[7:], q[7:])
        actual = kin.fk(result)["left"]
        self.assertLessEqual(np.linalg.norm(actual[:3, 3] - targets["left"][:3, 3]),
                             kin.position_tolerance)
        self.assertLessEqual(np.linalg.norm(kin.pin.log3(targets["left"][:3, :3].T @ actual[:3, :3])),
                             kin.rotation_tolerance)
        impossible = targets["left"].copy()
        impossible[0, 3] = 100.0
        with self.assertRaises(ValueError):
            kin.solve({"left": impossible}, q)



if __name__ == "__main__":
    unittest.main()
