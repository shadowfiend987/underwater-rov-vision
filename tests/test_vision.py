"""Regression tests using generated inputs; no hardware or field metrics."""

import argparse
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from rov_vision import apriltag as tag, color_poles as color
from rov_vision.io_utils import read_image, write_image

ROOT = Path(__file__).resolve().parents[1]
RAW_CONFIG = json.loads((ROOT / "configs/mission5.json").read_text())
SETTINGS = color.validate_settings(RAW_CONFIG)
ORDER = color.parse_order("R-B-Y")


def rectangle(hue=0, saturation=255, x=80, width=40, height=180):
    hsv = np.zeros((480, 640, 3), np.uint8)
    hsv[160 : 160 + height, x : x + width] = (hue, saturation, 220)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def marker(family="36h11", number=7):
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, tag.FAMILIES[family])
    )
    gray = cv2.aruco.generateImageMarker(dictionary, number, 160)
    gray = cv2.copyMakeBorder(gray, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class TagTests(unittest.TestCase):
    def test_all_configured_families_decode_generated_marker(self):
        for family in tag.FAMILIES:
            with self.subTest(family=family):
                _, ids, _ = tag.detect_frame(marker(family), family)
                self.assertEqual(ids.flatten().tolist(), [7])

    def test_blank_image(self):
        self.assertIsNone(tag.detect_frame(np.full((240, 240, 3), 255, np.uint8))[1])

    def test_three_generated_markers(self):
        frame = np.hstack([marker(number=i) for i in (2, 7, 12)])
        _, ids, _ = tag.detect_frame(frame)
        self.assertEqual(set(ids.flatten()), {2, 7, 12})

    def test_min_max_lock_and_reset(self):
        for mode, expected in [("min", 2), ("max", 12)]:
            m = tag.Mission(mode)
            for _ in range(4):
                m.observe([12, 2, 7])
            self.assertFalse(m.confirm())
            m.observe([12, 2, 7])
            self.assertEqual(m.status, "READY_PRESS_ENTER")
            self.assertTrue(m.confirm())
            self.assertEqual(m.target, expected)
            m.observe([20])
            self.assertEqual(m.seen, {2, 7, 12})
            self.assertFalse(m.confirm())
            m.reset()
            self.assertEqual(m.status, "SCANNING")
            self.assertIsNone(m.target)

    def test_interruption_resets_unconfirmed_count(self):
        m = tag.Mission("min")
        for _ in range(4):
            m.observe([7])
        m.observe([])
        m.observe([7])
        self.assertEqual(m.seen, set())
        self.assertEqual(m.counts[7], 1)

    def test_sequential_scanning_and_fourth_id(self):
        m = tag.Mission("max")
        for i in [2, 7, 12, 20]:
            for _ in range(5):
                m.observe([i, i])
        self.assertEqual(m.status, "INVALID_RESET_REQUIRED")
        self.assertFalse(m.confirm())

    def test_unsorted_ids_keep_correct_corners(self):
        corners = [
            np.array([[[90, 10], [110, 10], [110, 30], [90, 30]]], np.float32),
            np.array([[[490, 10], [510, 10], [510, 30], [490, 30]]], np.float32),
        ]
        live = tag.locate_target(2, np.array([[12], [2]]), corners, 600)
        self.assertEqual(
            live, {"status": "TRACKING", "center": [500, 20], "direction": "RIGHT"}
        )

    def test_lost_waiting_and_duplicate_ids_clear_location(self):
        for target, ids, expected in [
            (None, None, "WAITING"),
            (7, None, "LOST"),
            (7, np.array([[7], [7]]), "AMBIGUOUS"),
        ]:
            live = tag.locate_target(target, ids, [], 640)
            self.assertEqual(live["status"], expected)
            self.assertIsNone(live["center"])
            self.assertIsNone(live["direction"])

    def test_image_third_boundaries(self):
        for x, expected in [
            (199, "LEFT"),
            (200, "CENTER"),
            (399, "CENTER"),
            (400, "RIGHT"),
        ]:
            corners = [np.array([[[x, 0], [x, 2], [x, 4], [x, 6]]], np.float32)]
            self.assertEqual(
                tag.locate_target(7, np.array([[7]]), corners, 600)["direction"],
                expected,
            )

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            tag.Mission("unknown")


class ColorTests(unittest.TestCase):
    def result(self, frame, order=ORDER, index=0, settings=SETTINGS):
        return color.process_frame(frame, settings, order, index)[1]

    def test_three_colors_and_both_red_bands(self):
        for hue, order in [
            (0, ORDER),
            (175, ORDER),
            (25, ["YELLOW", "RED", "BLUE"]),
            (110, ["BLUE", "RED", "YELLOW"]),
        ]:
            with self.subTest(hue=hue):
                r = self.result(rectangle(hue), order)
                self.assertTrue(r["found"])
                self.assertEqual(r["center_px"], [100, 250])

    def test_loss_clears_previous_coordinates(self):
        self.assertTrue(self.result(rectangle())["found"])
        r = self.result(np.zeros((480, 640, 3), np.uint8))
        self.assertFalse(r["found"])
        self.assertIsNone(r["center_px"])
        self.assertIsNone(r["error_x_px"])

    def test_area_and_aspect_rejections(self):
        for frame in [rectangle(width=4, height=20), rectangle(width=180, height=40)]:
            self.assertFalse(self.result(frame)["found"])

    def test_fill_rejection(self):
        hsv = np.zeros((480, 640, 3), np.uint8)
        points = np.array([[80, 160], [119, 160], [100, 340]], np.int32)
        cv2.fillPoly(hsv, [points], (0, 255, 220))
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.assertTrue(self.result(frame)["found"])
        stricter = deepcopy(SETTINGS)
        stricter["fill_percent"] = 80
        self.assertFalse(self.result(frame, settings=stricter)["found"])

    def test_largest_valid_candidate_wins(self):
        frame = rectangle(x=80, width=40, height=180)
        frame = cv2.add(frame, rectangle(x=400, width=50, height=200))
        self.assertEqual(self.result(frame)["center_px"], [425, 260])

    def test_center_tolerance_boundaries(self):
        for center, expected in [
            (289, "LEFT"),
            (290, "CENTERED"),
            (350, "CENTERED"),
            (351, "RIGHT"),
        ]:
            self.assertEqual(
                self.result(rectangle(x=center - 20))["alignment"], expected
            )

    def test_saved_blue_and_yellow_threshold_changes(self):
        old = color.validate_settings(
            json.loads((ROOT / "configs/before_lab.json").read_text())
        )
        blue = rectangle(90, 60)
        self.assertTrue(self.result(blue, ["BLUE", "RED", "YELLOW"])["found"])
        self.assertFalse(
            self.result(blue, ["BLUE", "RED", "YELLOW"], settings=old)["found"]
        )
        yellow = rectangle(25, 120)
        self.assertFalse(self.result(yellow, ["YELLOW", "RED", "BLUE"])["found"])
        self.assertTrue(
            self.result(yellow, ["YELLOW", "RED", "BLUE"], settings=old)["found"]
        )

    def test_completion_is_manual_and_input_is_unchanged(self):
        frame = rectangle()
        original = frame.copy()
        r = self.result(frame, index=3)
        self.assertTrue(r["all_manually_confirmed"])
        self.assertEqual(r["alignment"], "NO_ACTIVE_TARGET")
        self.assertFalse(r["found"])
        np.testing.assert_array_equal(frame, original)

    def test_bad_config_rejected(self):
        bads = [None, {}, dict(RAW_CONFIG, min_area=True), dict(RAW_CONFIG, min_area=0)]
        inverted = deepcopy(RAW_CONFIG)
        inverted["colors"]["BLUE"][0][0][0] = 150
        bads.append(inverted)
        for bad in bads:
            with (
                self.subTest(bad=bad),
                self.assertRaises((ValueError, KeyError, TypeError)),
            ):
                color.validate_settings(bad)

    def test_order_validation(self):
        self.assertEqual(color.parse_order(" r - b - y "), ORDER)
        for bad in ["R-R-Y", "R-B", "R-B-Y-Y", "red-blue-yellow"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                color.parse_order(bad)


class IntegrationTests(unittest.TestCase):
    def test_unicode_image_roundtrip_and_decode_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "\u6d4b\u8bd5.png"
            write_image(path, rectangle())
            np.testing.assert_array_equal(read_image(path), rectangle())
            path.write_bytes(b"broken")
            with self.assertRaises(ValueError):
                read_image(path)

    def test_cli_headless_and_clear_input_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_image(tmp / "color.png", rectangle())
            write_image(tmp / "tag.png", marker())
            commands = [
                (["color", "--order", "R-B-Y", "--image", str(tmp / "color.png")], 0),
                (
                    [
                        "apriltag",
                        "--mode",
                        "max",
                        "--family",
                        "36h11",
                        "--image",
                        str(tmp / "tag.png"),
                    ],
                    0,
                ),
                (["color", "--order", "R-R-Y"], 2),
                (["apriltag", "--camera", "-1"], 2),
                (["color", "--order", "R-B-Y", "--image", str(tmp / "missing.png")], 1),
            ]
            for args, expected in commands:
                with self.subTest(args=args):
                    proc = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "run.py"),
                            *args,
                            "--output",
                            str(tmp / "result"),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        cwd=tmp,
                    )
                    self.assertEqual(
                        proc.returncode, expected, proc.stdout + proc.stderr
                    )
                    self.assertNotIn("Traceback", proc.stderr)

    def test_import_has_no_camera_or_gui_side_effect(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                'import rov_vision.color_poles; import rov_vision.apriltag; print("IMPORT_OK")',
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "IMPORT_OK")

    def test_color_keys_and_settings_with_mock_camera_gui(self):
        # Real processing, fake capture/windows; this is not interactive GUI acceptance.
        bars = {}
        recorded = []

        class Capture:
            released = False

            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def read(self):
                return True, rectangle()

            def release(self):
                self.released = True

        cap = Capture()

        def create(name, window, value, *args):
            bars[(name, window)] = value

        def setbar(name, window, value):
            bars[(name, window)] = value

        def getbar(name, window):
            return bars[(name, window)]

        real_process = color.process_frame

        def process(*args):
            result = real_process(*args)
            recorded.append(result[1])
            return result

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "settings.json"
            config.write_text(json.dumps(RAW_CONFIG))
            with (
                patch.object(cv2, "VideoCapture", return_value=cap),
                patch.object(cv2, "namedWindow"),
                patch.object(cv2, "createTrackbar", side_effect=create),
                patch.object(cv2, "setTrackbarPos", side_effect=setbar),
                patch.object(cv2, "getTrackbarPos", side_effect=getbar),
                patch.object(cv2, "imshow"),
                patch.object(cv2, "destroyAllWindows"),
                patch.object(cv2, "waitKey", side_effect=list(map(ord, "npusdnq"))),
                patch.object(color, "process_frame", side_effect=process),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    color.main(["--order", "R-B-Y", "--config", str(config)]), 0
                )
            self.assertEqual(
                [r["current_index"] for r in recorded], [0, 1, 1, 0, 0, 0, 1]
            )
            self.assertEqual(json.loads(config.read_text()), RAW_CONFIG)
            self.assertTrue(cap.released)

    def test_camera_open_failure_releases_resources(self):
        from unittest.mock import MagicMock

        cap = MagicMock()
        cap.isOpened.return_value = False
        with (
            patch.object(cv2, "VideoCapture", return_value=cap),
            patch.object(cv2, "destroyAllWindows"),
        ):
            with self.assertRaises(RuntimeError):
                color.main(
                    [
                        "--order",
                        "R-B-Y",
                        "--config",
                        str(ROOT / "configs/mission5.json"),
                    ]
                )
        cap.release.assert_called_once()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cv2, "VideoCapture", return_value=cap),
            patch.object(cv2, "destroyAllWindows"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                tag.main(
                    [
                        "--camera",
                        "0",
                        "--mode",
                        "min",
                        "--family",
                        "36h11",
                        "--output",
                        tmp,
                    ]
                ),
                1,
            )

    def test_apriltag_capture_loop_saves_lock_and_logs_loss(self):
        from unittest.mock import MagicMock

        frame = np.hstack([marker(number=i) for i in (2, 7, 12)])
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, frame.copy()) for _ in range(5)] + [
            (True, np.full_like(frame, 255))
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cv2, "VideoCapture", return_value=cap),
            patch.object(cv2, "imshow"),
            patch.object(cv2, "destroyAllWindows"),
            patch.object(cv2, "waitKey", side_effect=[-1, -1, -1, -1, 13, ord("q")]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            tag.main(
                ["--camera", "0", "--mode", "max", "--family", "36h11", "--output", tmp]
            )
            result = json.loads(
                next(Path(tmp).rglob("round_1_result.json")).read_text()
            )
            self.assertEqual(result["target_id"], 12)
            self.assertEqual(result["confirmed_ids"], [2, 7, 12])
            events = [
                json.loads(line)
                for line in next(Path(tmp).rglob("events.jsonl"))
                .read_text()
                .splitlines()
            ]
            lost = [e for e in events if e.get("live_status") == "LOST"]
            self.assertEqual(len(lost), 1)
            self.assertIsNone(lost[0]["center"])
            self.assertEqual(lost[0]["status"], "RESULT_CONFIRMED")
            self.assertTrue(next(Path(tmp).rglob("round_1_result.png")).is_file())
        cap.release.assert_called_once()

    def test_camera_read_failure_returns_error_without_waiting(self):
        from unittest.mock import MagicMock

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cv2, "VideoCapture", return_value=cap),
            patch.object(cv2, "imshow"),
            patch.object(cv2, "destroyAllWindows"),
            patch.object(cv2, "waitKey") as key,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                tag.main(
                    [
                        "--camera",
                        "0",
                        "--mode",
                        "min",
                        "--family",
                        "36h11",
                        "--output",
                        tmp,
                    ]
                ),
                1,
            )
            key.assert_not_called()

    def test_hsv_band_switch_preserves_edits(self):
        bars = {("Band", color.HSV_WINDOW): 0}

        def setbar(name, window, value):
            bars[(name, window)] = value

        with (
            patch.object(cv2, "setTrackbarPos", side_effect=setbar),
            patch.object(cv2, "getTrackbarPos", side_effect=lambda n, w: bars[(n, w)]),
        ):
            color.active_band = 0
            color.apply_settings(SETTINGS)
            bars[("S low", color.HSV_WINDOW)] = 123
            bars[("Band", color.HSV_WINDOW)] = 3
            color.read_hsv_controls()
            self.assertEqual(color.COLOR_RANGES["RED"][0][0][1], 123)
            self.assertEqual(bars[("H low", color.HSV_WINDOW)], 85)
            bars[("H low", color.HSV_WINDOW)] = 135
            color.read_hsv_controls()
            self.assertEqual(color.COLOR_RANGES["BLUE"][0][0][0], 130)


if __name__ == "__main__":
    unittest.main(verbosity=2)
