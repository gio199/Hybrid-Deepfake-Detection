import unittest

from detector.quality_checks import (
    assess_blur_vs_bitrate,
    bits_per_pixel,
    bits_per_pixel_from_bitrate,
)


class QualityMetricTests(unittest.TestCase):
    def test_stream_bitrate_bpp_matches_size_formula(self):
        width, height, fps, seconds = 1920, 1080, 30, 10
        bitrate_kbps = 6000
        file_size_bytes = int(bitrate_kbps * 1000 * seconds / 8)
        frames = fps * seconds
        self.assertAlmostEqual(
            bits_per_pixel_from_bitrate(bitrate_kbps, width, height, fps),
            bits_per_pixel(file_size_bytes, width, height, frames),
        )

    def test_quality_reason_identifies_stream_bitrate(self):
        signal = assess_blur_vs_bitrate(
            [60.0] * 30,
            file_size_bytes=None,
            width=1280,
            height=720,
            frame_count=300,
            video_bitrate_kbps=5000,
            fps=30,
        )
        self.assertIsNotNone(signal)
        self.assertIn("video-stream bitrate", signal.reason)


if __name__ == "__main__":
    unittest.main()
