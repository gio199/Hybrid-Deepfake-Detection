import unittest

from detector.history import Signal
from detector.scoring import AnomalyAggregator, combine_signals


class SignalFusionTests(unittest.TestCase):
    def test_correlated_body_signals_use_group_max(self):
        score = combine_signals(
            [Signal("bone_length", 0.5), Signal("joint_angle_motion", 0.5)],
            {"bone_length": 1.0, "joint_angle_motion": 1.0},
        )
        self.assertAlmostEqual(score, 0.5)

    def test_independent_signal_groups_compound(self):
        score = combine_signals(
            [Signal("bone_length", 0.5), Signal("pixel_flicker", 0.5)],
            {"bone_length": 1.0, "pixel_flicker": 1.0},
        )
        self.assertAlmostEqual(score, 0.75)


class EvidenceTests(unittest.TestCase):
    def test_short_clip_abstains(self):
        aggregator = AnomalyAggregator()
        for index in range(10):
            aggregator.add_frame(index, index / 30.0, [], evaluated=True)
        result = aggregator.finalize()
        self.assertEqual(result.verdict, "INSUFFICIENT EVIDENCE")
        self.assertIn("usable evidence", result.evidence_warning)

    def test_low_coverage_abstains(self):
        aggregator = AnomalyAggregator()
        for index in range(120):
            aggregator.add_frame(index, index / 30.0, [], evaluated=index < 10)
        result = aggregator.finalize()
        self.assertEqual(result.verdict, "INSUFFICIENT EVIDENCE")
        self.assertLess(result.evidence_coverage, 0.25)

    def test_sufficient_clean_evidence_can_be_likely_real(self):
        aggregator = AnomalyAggregator()
        for index in range(60):
            aggregator.add_frame(index, index / 30.0, [], evaluated=True)
        result = aggregator.finalize()
        self.assertEqual(result.verdict, "LIKELY REAL")
        self.assertAlmostEqual(result.evidence_coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
