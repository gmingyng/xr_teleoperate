import unittest
import numpy as np
from .test_bridge import config
from unitree_openpi.robot_ctrl.joint_motion import joint_limits
from unitree_openpi.scripts.reset_dual_arms import run_reset
from unitree_openpi.scripts.swing_dual_arms import swing_blend
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
        run_reset(backend, MockKinematics(), cfg, .1, False, joint_limits(cfg))
        np.testing.assert_array_equal(backend.q, np.zeros(14))
        self.assertGreater(len(backend.commands), 2)
        self.assertTrue(all(np.all(q[[0,7]] >= 0) and np.all(q[[0,7]] <= .02) for q in backend.commands))

    def test_swing_cycle_endpoints(self):
        for elapsed, expected in [(0, 0), (4, 1), (4.9, 1), (9, 0), (9.9, 0), (14, 1)]:
            self.assertAlmostEqual(swing_blend(elapsed), expected)


if __name__ == "__main__":
    unittest.main()
