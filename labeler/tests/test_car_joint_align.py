import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from smoke_labeler.car_joint_align import balanced_indices, corrected_floor_pose, trajectory_seed, gyro_timing, calibrate_yaw


class JointAlignmentTests(unittest.TestCase):
    def test_mount_calibration_supports_offsets_beyond_the_hall_search(self):
        class StraightOdometry:
            def pose(self, timestamp):
                result = np.eye(4)
                result[0, 3] = timestamp * .1
                return result

        rng = np.random.default_rng(42)
        world = rng.uniform([-4, -4, 1], [4, 4, 2], (400, 3))
        mount = Rotation.from_euler('z', 65, degrees=True).as_matrix()
        odom = StraightOdometry()
        frames = [dict(stamp=k, points=world - odom.pose(k)[:3, 3] @ mount)
                  for k in range(52)]
        result = calibrate_yaw(dict(frames=frames, odom=odom),
                               offsets_deg=np.arange(-180, 180, 5))
        self.assertEqual(result['best_all_odd_even'], [65, 65, 65])

    def test_detects_positive_gyro_clock_offset(self):
        wheel_time = np.arange(0, 8, .02)
        imu_time = np.arange(0, 8, .005)
        origin = 1_700_000_000_000_000_000
        yaw = .2*np.sin(2*wheel_time)
        gyro = np.zeros((len(imu_time), 3))
        gyro[:, 2] = .4*np.cos(2*(imu_time-.11))
        motion = dict(odom_time_ns=origin+np.rint(wheel_time*1e9).astype(np.int64),
                      imu_time_ns=origin+np.rint(imu_time*1e9).astype(np.int64),
                      odom_quaternion_xyzw=Rotation.from_euler('z', yaw[:, None]).as_quat(),
                      imu_angular_velocity_raw=gyro)
        result = gyro_timing(motion)
        self.assertAlmostEqual(result['offset_s'], .11, places=6)
        self.assertLess(result['rmse'], result['zero_rmse']/10)

    def test_refinement_retains_existing_drift_correction_and_new_anchor(self):
        previous = np.repeat(np.eye(4)[None], 3, axis=0)
        previous[:, 0, 3] = [1., 1.1, 1.3]
        np.testing.assert_allclose(trajectory_seed(previous, previous[0], 2), previous[2])
        anchor = previous[0].copy()
        anchor[1, 3] = 2.
        expected = previous[2].copy()
        expected[1, 3] = 2.
        np.testing.assert_allclose(trajectory_seed(previous, anchor, 2), expected)

    def test_sampling_keeps_all_laser_lines_and_identity(self):
        rng = np.random.default_rng(91)
        xyz = rng.uniform(-5, 5, (20000, 3))
        line = np.arange(len(xyz)) % 4
        offset = rng.uniform(0, .1, len(xyz))
        chosen = balanced_indices(xyz, line, offset, 1400, 42)
        self.assertEqual(len(chosen), 1400)
        self.assertTrue(np.all(np.diff(chosen) > 0))
        self.assertTrue(np.all(np.bincount(line[chosen], minlength=4) > 250))
        np.testing.assert_array_equal(chosen, balanced_indices(xyz, line, offset, 1400, 42))

    def test_floor_correction_recovers_tilt_and_height(self):
        rng = np.random.default_rng(40)
        floor = np.column_stack((rng.uniform(-4, 4, (1400, 2)), np.full(1400, -.35)))
        correct = np.eye(4)
        correct[:3, :3] = Rotation.from_euler('xyz', [2, -1, 25], degrees=True).as_matrix()
        correct[:3, 3] = [1, 2, .08]
        points = (floor-correct[:3, 3]) @ correct[:3, :3]
        pose = correct.copy()
        pose[:3, :3] = Rotation.from_euler('z', 25, degrees=True).as_matrix()
        pose[2, 3] = 0
        recovered, found = corrected_floor_pose(pose, points, (np.array([0., 0., 1.]), .35))
        self.assertTrue(found)
        world = points @ recovered[:3, :3].T + recovered[:3, 3]
        np.testing.assert_allclose(world[:, 2], -.35, atol=1e-8)
        np.testing.assert_allclose(recovered[:2, 3], pose[:2, 3])


if __name__ == '__main__':
    unittest.main()
