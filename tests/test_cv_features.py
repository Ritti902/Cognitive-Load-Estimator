"""Pure feature-extraction checks that do not access a webcam or save frames."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from cv_pipeline.blink_detector import BlinkCounter, calculate_ear
from cv_pipeline.gaze_detector import get_gaze_direction


class CvFeatureTests(unittest.TestCase):
    def test_eye_aspect_ratio_and_degenerate_eye(self):
        points = [
            SimpleNamespace(x=0.0, y=0.0),
            SimpleNamespace(x=1.0, y=1.0),
            SimpleNamespace(x=2.0, y=1.0),
            SimpleNamespace(x=3.0, y=0.0),
            SimpleNamespace(x=2.0, y=-1.0),
            SimpleNamespace(x=1.0, y=-1.0),
        ]
        self.assertAlmostEqual(calculate_ear(points, [0, 1, 2, 3, 4, 5]), 2 / 3)
        self.assertEqual(calculate_ear(points, [0, 1, 2, 0, 4, 5]), 0.0)

    def test_blink_counter_and_per_minute_rate(self):
        counter = BlinkCounter()
        counter.update(0.1, 0.1)
        self.assertEqual(counter.update(0.3, 0.3), 1)
        self.assertEqual(counter.get_blinks_per_minute(60), 1)

    def test_gaze_categories_and_zero_width_fallback(self):
        landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(469)]
        landmarks[33] = SimpleNamespace(x=0.2, y=0.0)
        landmarks[133] = SimpleNamespace(x=0.4, y=0.0)
        landmarks[468] = SimpleNamespace(x=0.3, y=0.0)
        self.assertEqual(get_gaze_direction(landmarks), ("CENTER", 0.0))
        landmarks[468] = SimpleNamespace(x=0.45, y=0.0)
        direction, score = get_gaze_direction(landmarks)
        self.assertEqual(direction, "RIGHT")
        self.assertAlmostEqual(score, 0.75)
        landmarks[133] = SimpleNamespace(x=0.2, y=0.0)
        self.assertEqual(get_gaze_direction(landmarks), ("CENTER", 0.0))


if __name__ == "__main__":
    unittest.main()
