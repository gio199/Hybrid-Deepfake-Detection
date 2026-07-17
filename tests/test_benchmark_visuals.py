import io
import unittest
import zipfile

from benchmarks.prepare_aigvd_examples import (
    DATASET_REVISION,
    SAMPLE_ID,
    _archive_url,
    _find_member,
)
from benchmarks.render_dashboard import build_html


class AIGVDBenchPreparationTests(unittest.TestCase):
    def test_archive_url_pins_dataset_revision(self):
        rendered = _archive_url("AIGVDBench/OpenSource/T2V/HunyuanVideo.zip")

        self.assertIn(DATASET_REVISION, rendered)
        self.assertIn("HunyuanVideo.zip", rendered)

    def test_find_member_matches_basename_inside_archive_directory(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(f"nested/test/{SAMPLE_ID}", b"video")
        buffer.seek(0)

        with zipfile.ZipFile(buffer) as archive:
            member = _find_member(archive, SAMPLE_ID)

        self.assertEqual(member.filename, f"nested/test/{SAMPLE_ID}")


class DashboardTests(unittest.TestCase):
    def test_dashboard_contains_score_chart_and_video_inspector(self):
        sample = {
            "id": "sample-1",
            "name": "Known generated",
            "label": "generated",
            "label_value": 1,
            "score": 72.5,
            "verdict": "LIKELY FAKE",
            "correct_at_threshold": True,
            "threshold": 55.0,
            "group": "face_swap",
            "notes": "Test note",
            "source_url": "https://example.test/source",
            "report_href": "report.json",
            "timeline_image_href": "timeline.png",
            "annotated_href": "annotated.mp4",
            "annotated_expected": True,
            "annotated_available": True,
            "duration_sec": 4.0,
            "evidence_coverage": 1.0,
            "timeline": {"timestamps": [0.0, 1.0], "scores": [10.0, 80.0]},
            "categories": [{"name": "jitter", "mean": 30.0, "peak": 90.0, "frame_frac": 20.0}],
        }

        rendered = build_html([sample], 55.0)

        self.assertIn("Score comparison", rendered)
        self.assertIn("Known generated", rendered)
        self.assertIn("annotated.mp4", rendered)
        self.assertIn("const samples =", rendered)


if __name__ == "__main__":
    unittest.main()
