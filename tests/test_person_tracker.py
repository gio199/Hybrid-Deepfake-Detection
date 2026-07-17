import unittest

from detector.landmarks import FACE_NOSE_TIP, Landmark, RawFrameDetections
from detector.person_tracker import MultiPersonTracker


def face_at(x, y=50.0):
    face = [Landmark(float(x), float(y)) for _ in range(478)]
    face[FACE_NOSE_TIP] = Landmark(float(x), float(y))
    face[234] = Landmark(float(x - 10), float(y))
    face[454] = Landmark(float(x + 10), float(y))
    return face


def raw_frame(frame_idx, timestamp_sec, xs):
    return RawFrameDetections(
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        image_w=100,
        image_h=100,
        faces=[face_at(x) for x in xs],
    )


class PersonTrackerTests(unittest.TestCase):
    def test_velocity_prediction_preserves_crossing_tracks(self):
        tracker = MultiPersonTracker()
        first = tracker.update(raw_frame(0, 0.0, [20, 80]))
        left_id = min(first, key=lambda item: item[1].anchor_point()[0])[0]
        right_id = max(first, key=lambda item: item[1].anchor_point()[0])[0]

        tracker.update(raw_frame(1, 0.1, [40, 60]))
        crossed = dict(tracker.update(raw_frame(2, 0.2, [65, 35])))

        self.assertGreater(crossed[left_id].anchor_point()[0], 50)
        self.assertLess(crossed[right_id].anchor_point()[0], 50)


if __name__ == "__main__":
    unittest.main()
