import math
import unittest

from detector.history import LandmarkHistory, PointSample


def sample(frame_idx, timestamp_sec, x):
    return PointSample(
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        x=float(x),
        y=0.0,
        z=0.0,
    )


class HistoryStatisticsTests(unittest.TestCase):
    def test_flat_baseline_then_change_is_anomalous(self):
        z = LandmarkHistory.robust_zscore_of_last([2.0, 2.0, 2.0, 2.0, 2.0, 3.0])
        self.assertTrue(math.isinf(z))
        self.assertGreater(z, 0)

    def test_flat_baseline_without_change_is_not_anomalous(self):
        self.assertEqual(
            LandmarkHistory.robust_zscore_of_last([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
            0.0,
        )

    def test_displacement_does_not_bridge_detection_gap(self):
        samples = [
            sample(0, 0.0, 0),
            sample(1, 0.1, 1),
            sample(5, 0.5, 50),
            sample(6, 0.6, 51),
        ]
        self.assertEqual(LandmarkHistory.displacement_series(samples), [1.0, 1.0])

    def test_rate_is_independent_of_frame_interval(self):
        slow_sampling = [sample(0, 0.0, 0), sample(1, 0.1, 1)]
        fast_sampling = [sample(0, 0.0, 0), sample(1, 0.05, 0.5)]
        self.assertAlmostEqual(
            LandmarkHistory.displacement_series(slow_sampling, as_rate=True)[0],
            LandmarkHistory.displacement_series(fast_sampling, as_rate=True)[0],
        )

    def test_consecutive_tail_starts_after_gap(self):
        samples = [
            sample(0, 0.0, 0),
            sample(1, 0.1, 1),
            sample(5, 0.5, 5),
            sample(6, 0.6, 6),
        ]
        tail = LandmarkHistory.consecutive_tail(samples)
        self.assertEqual([item.frame_idx for item in tail], [5, 6])


if __name__ == "__main__":
    unittest.main()
