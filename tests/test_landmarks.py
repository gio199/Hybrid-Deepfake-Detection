import unittest

from detector.landmarks import _timestamp_to_ms, _visibility_or_default


class LandmarkInputTests(unittest.TestCase):
    def test_timestamp_uses_real_milliseconds(self):
        self.assertEqual(_timestamp_to_ms(1.5), 1500)
        self.assertEqual(_timestamp_to_ms(1 / 30), 33)

    def test_zero_visibility_is_preserved(self):
        self.assertEqual(_visibility_or_default(0.0), 0.0)
        self.assertEqual(_visibility_or_default(None), 1.0)


if __name__ == "__main__":
    unittest.main()
