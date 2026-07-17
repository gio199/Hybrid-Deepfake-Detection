import unittest

import numpy as np

from detector.learned_detector import LearnedFaceDetector, LearnedModelConfig


class LearnedDetectorTests(unittest.TestCase):
    def test_softmax_probability(self):
        score = LearnedFaceDetector._probability_from_output(
            np.array([0.0, 2.0]), "softmax", fake_index=1
        )
        self.assertGreater(score, 0.8)

    def test_sigmoid_probability(self):
        score = LearnedFaceDetector._probability_from_output(
            np.array([0.0]), "sigmoid", fake_index=0
        )
        self.assertAlmostEqual(score, 0.5)

    def test_probability_is_clipped(self):
        score = LearnedFaceDetector._probability_from_output(
            np.array([1.5]), "probability", fake_index=0
        )
        self.assertEqual(score, 1.0)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            LearnedModelConfig(output_type="unknown").validate()


if __name__ == "__main__":
    unittest.main()
