"""HSV pole candidates and manual operator guidance, derived from Mission5/main.py."""

import argparse
import cv2
import json
from pathlib import Path
from copy import deepcopy
from .io_utils import read_image, write_image

MIN_AREA = 500
CENTER_TOLERANCE = 30
MIN_ASPECT_RATIO = 2.0
MIN_FILL_RATIO = 0.45
COLOR_RANGES = {
    "RED": [((0, 100, 60), (10, 255, 255)), ((170, 100, 60), (179, 255, 255))],
    "YELLOW": [((20, 100, 60), (35, 255, 255))],
    "BLUE": [((100, 100, 60), (130, 255, 255))],
}
DRAW_COLORS = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "BLUE": (255, 0, 0)}
LETTER_TO_COLOR = {"R": "RED", "Y": "YELLOW", "B": "BLUE"}
DEBUG_WINDOW = "Mission5 Tuning"
HSV_WINDOW = "Mission5 HSV"
BANDS = [("RED", 0), ("RED", 1), ("YELLOW", 0), ("BLUE", 0)]
HSV_CONTROLS = [
    ("H low", 179),
    ("S low", 255),
    ("V low", 255),
    ("H high", 179),
    ("S high", 255),
    ("V high", 255),
]
DEFAULT_SETTINGS = {
    "colors": deepcopy(COLOR_RANGES),
    "min_area": MIN_AREA,
    "aspect_x10": round(MIN_ASPECT_RATIO * 10),
    "fill_percent": round(MIN_FILL_RATIO * 100),
}
active_band = 0
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mission5.json"


def make_mask(hsv, ranges):
    """Combine the configured HSV bands into one binary mask."""
    mask = cv2.inRange(hsv, (0, 0, 0), (0, 0, 0))
    mask[:] = 0
    for lower, upper in ranges:
        part = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, part)
    return mask


def on_trackbar_change(value):
    pass


def show_band_values():
    """Display the selected HSV band in the trackbars."""
    color, index = BANDS[active_band]
    lower, upper = COLOR_RANGES[color][index]
    values = list(lower) + list(upper)
    for (name, _), value in zip(HSV_CONTROLS, values):
        cv2.setTrackbarPos(name, HSV_WINDOW, int(value))


def read_hsv_controls():
    """Read the previous band before switching the selected band."""
    global active_band
    values = [cv2.getTrackbarPos(name, HSV_WINDOW) for name, _ in HSV_CONTROLS]
    for channel in range(3):
        if values[channel] > values[channel + 3]:
            values[channel] = values[channel + 3]
            cv2.setTrackbarPos(HSV_CONTROLS[channel][0], HSV_WINDOW, values[channel])
    color, index = BANDS[active_band]
    COLOR_RANGES[color][index] = (tuple(values[:3]), tuple(values[3:]))
    selected_band = cv2.getTrackbarPos("Band", HSV_WINDOW)
    if selected_band != active_band:
        active_band = selected_band
        show_band_values()


def apply_settings(settings):
    """Apply settings and update the tuning controls."""
    global COLOR_RANGES
    COLOR_RANGES = deepcopy(settings["colors"])
    cv2.setTrackbarPos("Min area", DEBUG_WINDOW, settings["min_area"])
    cv2.setTrackbarPos("Aspect x10", DEBUG_WINDOW, settings["aspect_x10"])
    cv2.setTrackbarPos("Fill percent", DEBUG_WINDOW, settings["fill_percent"])
    show_band_values()


def validate_settings(data):
    """Validate the configuration before using it."""
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")

    def check_integer(value, low, high):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Parameter must be an integer from {low} to {high}")
        return value

    colors = data["colors"]
    if not isinstance(colors, dict):
        raise ValueError("Invalid colors object")
    checked_colors = {}
    for color, count in (("RED", 2), ("YELLOW", 1), ("BLUE", 1)):
        ranges = colors[color]
        if not isinstance(ranges, list) or len(ranges) != count:
            raise ValueError(f"{color} has an invalid band count")
        checked_colors[color] = []
        for pair in ranges:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Each HSV band must have a lower and upper bound")
            lower, upper = pair
            if (
                not isinstance(lower, list)
                or not isinstance(upper, list)
                or len(lower) != 3
                or (len(upper) != 3)
            ):
                raise ValueError("HSV bounds must each contain three values")
            for i, maximum in enumerate((179, 255, 255)):
                check_integer(lower[i], 0, maximum)
                check_integer(upper[i], 0, maximum)
                if lower[i] > upper[i]:
                    raise ValueError("HSV lower bounds must not exceed upper bounds")
            checked_colors[color].append((tuple(lower), tuple(upper)))
    return {
        "colors": checked_colors,
        "min_area": check_integer(data["min_area"], 1, 10000),
        "aspect_x10": check_integer(data["aspect_x10"], 0, 100),
        "fill_percent": check_integer(data["fill_percent"], 0, 100),
    }


def save_settings():
    """Write settings atomically using a temporary file."""
    settings = {
        "colors": COLOR_RANGES,
        "min_area": max(1, cv2.getTrackbarPos("Min area", DEBUG_WINDOW)),
        "aspect_x10": cv2.getTrackbarPos("Aspect x10", DEBUG_WINDOW),
        "fill_percent": cv2.getTrackbarPos("Fill percent", DEBUG_WINDOW),
    }
    temporary_path = CONFIG_PATH.with_suffix(".tmp")
    try:
        temporary_path.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(CONFIG_PATH)
        print(f"Settings saved: {CONFIG_PATH}")
    except OSError as error:
        print(f"Save failed: {error}")


def process_frame(frame, settings, order, current_index=0, band=0):
    """Process one BGR frame; no camera, GUI, or persistent target coordinates."""
    frame = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if current_index < len(order):
        target = order[current_index]
    else:
        target = None
    # Never carry a previous frame's position into the current result.
    target_center = None
    edit_color, edit_range_index = BANDS[band]
    MIN_AREA = settings["min_area"]
    MIN_ASPECT_RATIO = settings["aspect_x10"] / 10.0
    MIN_FILL_RATIO = settings["fill_percent"] / 100.0
    debug_mask = hsv[:, :, 0].copy()
    debug_mask[:] = 0
    debug_text = "No active target"
    for color_name, ranges in settings["colors"].items():
        mask = make_mask(hsv, ranges)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if color_name == edit_color:
            debug_mask = mask.copy()
            debug_text = f"{edit_color}: no color contours"
            if contours:
                raw_largest = max(contours, key=cv2.contourArea)
                raw_area = cv2.contourArea(raw_largest)
                rx, ry, rw, rh = cv2.boundingRect(raw_largest)
                raw_aspect = rh / rw
                raw_fill = raw_area / (rw * rh)
                failed = []
                if raw_area < MIN_AREA:
                    failed.append("AREA")
                if raw_aspect < MIN_ASPECT_RATIO:
                    failed.append("ASPECT")
                if raw_fill < MIN_FILL_RATIO:
                    failed.append("FILL")
                verdict = "PASS"
                if failed:
                    verdict = "FAIL: " + ",".join(failed)
                debug_text = (
                    f"A={raw_area:.0f} AR={raw_aspect:.2f} F={raw_fill:.2f} | {verdict}"
                )
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                continue
            aspect_ratio = h / w
            fill_ratio = area / (w * h)
            if aspect_ratio < MIN_ASPECT_RATIO:
                continue
            if fill_ratio < MIN_FILL_RATIO:
                continue
            candidates.append(contour)
        if not candidates:
            continue
        # Original policy: one largest valid candidate per color.
        largest = max(candidates, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cx = x + w // 2
        cy = y + h // 2
        if color_name == target:
            target_center = (cx, cy)
        draw_color = DRAW_COLORS[color_name]
        cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
        cv2.circle(frame, (cx, cy), 5, draw_color, -1)
        label = f"{color_name} ({cx}, {cy})"
        # Keep labels inside the frame and readable against light backgrounds.
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_x = max(0, min(x, frame.shape[1] - text_width - 2))
        label_y = max(text_height + 2, y - 10)
        cv2.rectangle(
            frame,
            (label_x, label_y - text_height - 2),
            (label_x + text_width + 1, label_y + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            draw_color,
            2,
        )
    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2
    cv2.line(
        frame,
        (frame_center_x, 0),
        (frame_center_x, frame_height - 1),
        (255, 255, 255),
        1,
    )
    for boundary_x in (
        frame_center_x - CENTER_TOLERANCE,
        frame_center_x + CENTER_TOLERANCE,
    ):
        cv2.line(
            frame, (boundary_x, 0), (boundary_x, frame_height - 1), (150, 150, 150), 1
        )
    if target is None:
        status = "All 3 manually confirmed"
        guidance = "No active target"
        guidance_color = (255, 255, 255)
    elif target_center is None:
        status = f"Target: {target} | Confirmed: {current_index}/3"
        guidance = "TARGET NOT FOUND"
        guidance_color = (0, 165, 255)
    else:
        status = f"Target: {target} | Confirmed: {current_index}/3"
        target_cx, target_cy = target_center
        error_x = target_cx - frame_center_x
        if error_x < -CENTER_TOLERANCE:
            direction = "TARGET LEFT"
            guidance_color = (0, 255, 255)
        elif error_x > CENTER_TOLERANCE:
            direction = "TARGET RIGHT"
            guidance_color = (0, 255, 255)
        else:
            direction = "HORIZONTALLY CENTERED"
            guidance_color = (0, 255, 0)
        guidance = f"{direction} | dx={error_x:+d}px"
        cv2.circle(frame, (target_cx, target_cy), 10, (255, 255, 255), 2)
    vision_result = {
        "mission": 5,
        "order": list(order),
        "current_index": current_index,
        "confirmed_count": current_index,
        "all_manually_confirmed": target is None,
        "target_color": target,
        "found": False,
        "center_px": None,
        "error_x_px": None,
        "alignment": "NOT_FOUND",
        "frame_size_px": [frame_width, frame_height],
    }
    if target is None:
        vision_result["alignment"] = "NO_ACTIVE_TARGET"
    elif target_center is not None:
        result_cx, result_cy = target_center
        result_dx = result_cx - frame_center_x
        if result_dx < -CENTER_TOLERANCE:
            alignment = "LEFT"
        elif result_dx > CENTER_TOLERANCE:
            alignment = "RIGHT"
        else:
            alignment = "CENTERED"
        vision_result.update(
            {
                "found": True,
                "center_px": [result_cx, result_cy],
                "error_x_px": result_dx,
                "alignment": alignment,
            }
        )
    return (
        frame,
        vision_result,
        debug_mask,
        debug_text,
        status,
        guidance,
        guidance_color,
    )


def parse_order(value):
    letters = value.upper().replace(" ", "").split("-")
    if len(letters) != 3 or set(letters) != {"R", "Y", "B"}:
        raise argparse.ArgumentTypeError(
            "Use R, Y and B exactly once, for example R-B-Y"
        )
    return [LETTER_TO_COLOR[letter] for letter in letters]


def current_settings():
    return {
        "colors": deepcopy(COLOR_RANGES),
        "min_area": max(1, cv2.getTrackbarPos("Min area", DEBUG_WINDOW)),
        "aspect_x10": cv2.getTrackbarPos("Aspect x10", DEBUG_WINDOW),
        "fill_percent": cv2.getTrackbarPos("Fill percent", DEBUG_WINDOW),
    }


def main(argv=None):
    global CONFIG_PATH, active_band
    parser = argparse.ArgumentParser(
        description="HSV color pole detection and manual guidance"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--order", type=parse_order)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--image", type=Path, help="Process one image without opening a camera or GUI"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/color"))
    args = parser.parse_args(argv)
    if args.camera < 0:
        parser.error("--camera must be non-negative")
    CONFIG_PATH = args.config
    try:
        settings = validate_settings(
            json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"Cannot load config {CONFIG_PATH}: {exc}")
    order = args.order
    if order is None and args.image:
        parser.error("--order is required with --image")
    while order is None:
        try:
            order = parse_order(input("Target order, e.g. R-B-Y: "))
        except argparse.ArgumentTypeError as exc:
            print(exc)
    letters = [name[0] for name in order]
    current_index = 0
    if args.image:
        annotated, result, *_ = process_frame(read_image(args.image), settings, order)
        args.output.mkdir(parents=True, exist_ok=True)
        write_image(args.output / "annotated.png", annotated)
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return 0
    cap = cv2.VideoCapture(args.camera)
    try:
        if not cap.isOpened():
            raise RuntimeError(
                "Cannot open camera. Check index, permissions and other camera apps."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        active_band = 0
        cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(
            "Min area", DEBUG_WINDOW, MIN_AREA, 10000, on_trackbar_change
        )
        cv2.createTrackbar(
            "Aspect x10",
            DEBUG_WINDOW,
            round(MIN_ASPECT_RATIO * 10),
            100,
            on_trackbar_change,
        )
        cv2.createTrackbar(
            "Fill percent",
            DEBUG_WINDOW,
            round(MIN_FILL_RATIO * 100),
            100,
            on_trackbar_change,
        )
        cv2.namedWindow(HSV_WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Band", HSV_WINDOW, 0, 3, on_trackbar_change)
        for name, maximum in HSV_CONTROLS:
            cv2.createTrackbar(name, HSV_WINDOW, 0, maximum, on_trackbar_change)
        apply_settings(settings)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError(
                    "Camera read failed; restart after checking the connection."
                )
            read_hsv_controls()
            (
                frame,
                vision_result,
                debug_mask,
                debug_text,
                status,
                guidance,
                guidance_color,
            ) = process_frame(
                frame, current_settings(), order, current_index, active_band
            )
            frame_height, frame_width = frame.shape[:2]
            edit_color, edit_range_index = BANDS[active_band]
            cv2.rectangle(frame, (0, 0), (frame_width - 1, 115), (0, 0, 0), -1)
            cv2.putText(
                frame,
                status,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Order: " + "-".join(letters),
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                guidance,
                (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                guidance_color,
                2,
            )
            cv2.putText(
                frame,
                "N next | U undo | S save | D reset | Q quit",
                (10, 103),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Mission5 | Color Candidates | Q to quit", frame)
            debug_view = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(
                debug_view, (0, 0), (debug_view.shape[1] - 1, 65), (0, 0, 0), -1
            )
            cv2.putText(
                debug_view,
                f"Editing: {edit_color} | Largest RAW contour",
                (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                debug_view,
                debug_text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            cv2.imshow(DEBUG_WINDOW, debug_view)
            hsv_panel = cv2.copyMakeBorder(
                debug_view, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            cv2.putText(
                hsv_panel,
                f"Band {active_band}: {edit_color} range {edit_range_index + 1} | S save | D defaults",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            cv2.imshow(HSV_WINDOW, hsv_panel)
            key = cv2.waitKey(1) & 255
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("p"), ord("P")):
                print("\nCurrent frame result:")
                print(json.dumps(vision_result, ensure_ascii=False, indent=2))
            elif key in (ord("s"), ord("S")):
                read_hsv_controls()
                save_settings()
            elif key in (ord("d"), ord("D")):
                apply_settings(DEFAULT_SETTINGS)
                print("Code defaults restored; press S to save them.")
            elif key in (ord("n"), ord("N")):
                if current_index < len(order):
                    completed_color = order[current_index]
                    current_index += 1
                    print(f"Manually confirmed: {completed_color}")
            elif key in (ord("u"), ord("U")):
                if current_index > 0:
                    current_index -= 1
                    print(f"Confirmation undone; target: {order[current_index]}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
