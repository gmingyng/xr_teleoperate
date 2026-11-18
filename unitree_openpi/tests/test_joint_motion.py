import unittest
import numpy as np
from .test_bridge import config
from unitree_openpi.robot_ctrl.joint_motion import run_motion, joint_limits, swing_target
from unitree_openpi.robot_ctrl.mock import MockBackend, MockKinematics
from unitree_openpi.robot_ctrl.dds import G1DDSBackend, ARM_SLOTS, WAIST_SLOTS, GRIP_SLOTS
from types import SimpleNamespace


class JointMotionTests(unittest.TestCase):
    def test_default_gains_match_teleop_g1_and_dds_mapping(self):
        cfg = config()
        backend = G1DDSBackend(cfg)
        backend._message = SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(35)])
        for slots, group, size in [(ARM_SLOTS, "arm", 14), (WAIST_SLOTS, "waist", 3), (GRIP_SLOTS, "gripper", 2)]:
            backend._enable_slots(slots, group, np.zeros(size))
        expected = [(80, 3)]*4 + [(40, 1.5)]*3
        self.assertEqual([(backend._message.motor_cmd[i].kp, backend._message.motor_cmd[i].kd) for i in ARM_SLOTS], expected*2)
        self.assertEqual([(backend._message.motor_cmd[i].kp, backend._message.motor_cmd[i].kd) for i in WAIST_SLOTS], [(300,3)]*2)
        self.assertEqual([(backend._message.motor_cmd[i].kp, backend._message.motor_cmd[i].kd) for i in GRIP_SLOTS], [(5,.05)]*2)

    def test_reset_from_nonzero_pose(self):
        cfg = config()
        backend = MockBackend(cfg)
        backend.connect()
        backend.q[[0, 7]] = .02
        run_motion(backend, MockKinematics(), cfg, "reset", .1, .01, 1, 1, False, joint_limits(cfg))
        np.testing.assert_array_equal(backend.q, np.zeros(14))
        self.assertGreater(len(backend.commands), 2)
        self.assertTrue(all(np.all(q[[0,7]] >= 0) and np.all(q[[0,7]] <= .02) for q in backend.commands))

    def test_swing_only_shoulders_and_returns_to_center(self):
        center = np.zeros(14)
        center[3] = .3
        peak = swing_target(center, .05, np.pi/2)
        np.testing.assert_allclose(peak[[0,7]], [.05,.05])
        np.testing.assert_array_equal(peak[[i for i in range(14) if i not in (0,7)]], center[[i for i in range(14) if i not in (0,7)]])
        cfg = config()
        backend = MockBackend(cfg)
        backend.connect()
        backend.q = center.copy()
        run_motion(backend, MockKinematics(), cfg, "swing", 1, .001, .15, 1, False, joint_limits(cfg))
        np.testing.assert_array_equal(backend.q, center)

    def test_swing_limit_rejected_before_acquisition(self):
        cfg = config()
        backend = MockBackend(cfg)
        backend.connect()
        limits = joint_limits(cfg)
        backend.q[0] = limits[1][0]
        with self.assertRaises(ValueError):
            run_motion(backend, MockKinematics(), cfg, "swing", 1, .05, 6, 1, False, limits)
        self.assertFalse(backend.acquired)
        self.assertFalse(backend.commands)


if __name__ == "__main__":
    unittest.main()
