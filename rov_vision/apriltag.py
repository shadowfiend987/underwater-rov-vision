"""AprilTag scanning and image-space guidance for a human operator."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np

EXPECTED_COUNT = 3
CONFIRM_FRAMES = 5
TAG_FAMILY = "36h11"
FAMILIES = {
    "36h11": "DICT_APRILTAG_36h11",
    "36h10": "DICT_APRILTAG_36h10",
    "25h9": "DICT_APRILTAG_25h9",
    "16h5": "DICT_APRILTAG_16h5",
}


class Mission:
    """Keep scan history separate from the locked result."""

    def __init__(self, mode):
        if mode not in ("min", "max"):
            raise ValueError("Mode must be min or max")
        self.mode = mode
        self.round = 0
        self.reset()

    def reset(self):
        self.round += 1
        self.seen = set()
        self.counts = {}
        self.locked = False
        self.target = None

    def observe(self, current):
        if self.locked:
            return []
        current = set(current)
        self.counts = {i: n for i, n in self.counts.items() if i in current}
        added = []
        for tag_id in sorted(current):
            self.counts[tag_id] = min(self.counts.get(tag_id, 0) + 1, CONFIRM_FRAMES)
            if self.counts[tag_id] >= CONFIRM_FRAMES and tag_id not in self.seen:
                self.seen.add(tag_id)
                added.append(tag_id)
        return added

    @property
    def status(self):
        if self.locked:
            return "RESULT_CONFIRMED"
        if len(self.seen) > EXPECTED_COUNT:
            return "INVALID_RESET_REQUIRED"
        if len(self.seen) == EXPECTED_COUNT:
            return "READY_PRESS_ENTER"
        return "SCANNING"

    def confirm(self):
        if self.locked or len(self.seen) != EXPECTED_COUNT:
            return False
        self.target = min(self.seen) if self.mode == "min" else max(self.seen)
        self.locked = True
        return True

    @property
    def selected_target(self):
        if self.locked:
            return self.target
        if len(self.seen) == EXPECTED_COUNT:
            return min(self.seen) if self.mode == "min" else max(self.seen)
        return None


def locate_target(target_id, ids, corners, width):
    """Use original detection order; never pair sorted IDs with corners."""
    if target_id is None:
        return {"status": "WAITING", "center": None, "direction": None}
    matches = (
        []
        if ids is None
        else [i for i, tag in enumerate(ids.flatten()) if int(tag) == target_id]
    )
    if not matches:
        return {"status": "LOST", "center": None, "direction": None}
    if len(matches) != 1:
        return {"status": "AMBIGUOUS", "center": None, "direction": None}
    points = corners[matches[0]].reshape(4, 2)
    x, y = (float(points[:, 0].mean()), float(points[:, 1].mean()))
    direction = "LEFT" if x < width / 3 else "CENTER" if x < width * 2 / 3 else "RIGHT"
    return {"status": "TRACKING", "center": [int(x), int(y)], "direction": direction}


def draw_guidance(frame, live):
    h, w = frame.shape[:2]
    for x in (w // 3, 2 * w // 3):
        cv2.line(frame, (x, 0), (x, h - 1), (160, 160, 160), 1)
    if live["center"] is not None:
        cv2.circle(frame, tuple(live["center"]), 8, (0, 0, 255), -1)


def choose_options(argv=None):
    parser = argparse.ArgumentParser(
        description="AprilTag scan and live target guidance"
    )
    parser.add_argument("--camera", type=int)
    parser.add_argument("--mode", choices=("min", "max"))
    parser.add_argument("--family", choices=tuple(FAMILIES))
    parser.add_argument(
        "--image", type=Path, help="Detect one image without a camera or GUI"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/apriltag"))
    args = parser.parse_args(argv)
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be non-negative")
    if args.image:
        if args.mode is None or args.family is None:
            parser.error("--mode and --family are required with --image")
        return args
    if args.mode is None:
        print("\n=== AprilTag options ===")
        print("Select a rule: 1 = minimum ID, 2 = maximum ID")
        while True:
            value = input("Enter 1 or 2: ").strip()
            if value in ("1", "2"):
                args.mode = "min" if value == "1" else "max"
                break
            print("Please enter 1 or 2.")
    if args.camera is None:
        print("Camera indices vary. Verify the selected lens by covering it.")
        while True:
            value = input("Camera index [0]: ").strip() or "0"
            if value.isdecimal():
                args.camera = int(value)
                break
            print("Enter a non-negative integer.")
    if args.family is None:
        print("Tag families: 1=36h11 2=36h10 3=25h9 4=16h5")
        while True:
            value = input("Select the family printed on your tags [1]: ").strip() or "1"
            if value in ("1", "2", "3", "4"):
                args.family = list(FAMILIES)[int(value) - 1]
                break
            print("Please enter 1 to 4.")
    print(f"Rule={args.mode.upper()}, camera={args.camera}, family={args.family}")
    return args


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_image(path, frame):
    ok, buffer = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(f"Image encoding failed: {path}")
    buffer.tofile(str(path))


def panel_for(mission, frame, current, camera_ok=True, notice="", live=None):
    panel = np.zeros((720, 1000, 3), dtype=np.uint8)
    color = (60, 220, 80) if mission.locked else (0, 215, 255)
    rows = [
        ("APRILTAG SCANNING - OPERATOR ASSIST", 0.9, (255, 255, 255)),
        (f"RULE: {mission.mode.upper()}   |   ROUND: {mission.round}", 0.9, color),
        (f"CONFIRMED IDs: {sorted(mission.seen)}", 0.9, color),
        (f"STATE: {mission.status}", 0.75, color),
        (
            f"FINAL TARGET: {mission.target}"
            if mission.locked
            else f"SELECTED: {mission.selected_target} - ENTER to confirm",
            1.05,
            color,
        ),
    ]
    for index, (text, scale, ink) in enumerate(rows):
        cv2.putText(
            panel, text, (20, 40 + index * 42), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 2
        )
    if camera_ok and frame is not None:
        h, w = frame.shape[:2]
        scale = min(960 / w, 350 / h)
        preview = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        ph, pw = preview.shape[:2]
        panel[250 : 250 + ph, 20 : 20 + pw] = preview
        live = live or {"status": "WAITING", "direction": None, "center": None}
        live_text = f"TARGET: {mission.selected_target} | {live['status']} | {live['direction'] or 'NO DIRECTION'} | XY: {live['center']}"
    else:
        live_text = (
            "CAMERA_ERROR - no current image; any confirmed result is historical"
        )
    cv2.putText(
        panel, live_text, (20, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    cv2.putText(
        panel,
        "ENTER: confirm + save | S: screenshot | R: new round | Q/Esc: quit",
        (20, 660),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        panel, notice[:105], (20, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1
    )
    return panel


def make_detector(family):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, FAMILIES[family]))
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def detect_frame(frame, family="36h11"):
    return make_detector(family).detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))


def main(argv=None):
    args = choose_options(argv)
    if args.image:
        from .io_utils import read_image

        frame = read_image(args.image)
        corners, ids, _ = detect_frame(frame, args.family)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        result = {
            "input_kind": "single_image",
            "tag_family": args.family,
            "detected_ids": sorted((int(i) for i in ids.flatten()))
            if ids is not None
            else [],
            "mission_confirmed": False,
            "note": "One image does not verify consecutive-frame confirmation or a physical task.",
        }
        args.output.mkdir(parents=True, exist_ok=True)
        write_image(args.output / "annotated.png", frame)
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return 0
    mission = Mission(args.mode)
    session = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    directory = args.output / session
    directory.mkdir(parents=True, exist_ok=False)
    event_path = directory / "events.jsonl"
    evidence = {}
    seq = 0
    last_state = None
    notice = f"Camera: {args.camera} | Family: {args.family} | LEFT/CENTER/RIGHT = image position only"

    def emit(kind, **fields):
        nonlocal seq
        seq += 1
        record = {
            "event_seq": seq,
            "timestamp_utc": utc_now(),
            "event": kind,
            "round": mission.round,
            "mode": mission.mode,
            "camera_index": args.camera,
            "tag_family": args.family,
            **fields,
        }
        with event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

    def result_payload():
        return {
            "schema_version": 1,
            "session_id": session,
            "timestamp_utc": utc_now(),
            "mission": "2.1",
            "status": mission.status,
            "round": mission.round,
            "mode": mission.mode,
            "tag_family": args.family,
            "camera_index": args.camera,
            "confirmed_ids": sorted(mission.seen),
            "target_id": mission.target,
            "confirm_frames": CONFIRM_FRAMES,
            "tag_evidence": evidence.copy(),
            "note": "Software result only; not a referee score or proof of underwater validation.",
        }

    cap = None
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, FAMILIES[args.family])
        )
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            emit(
                "CAMERA_ERROR",
                reason="OPEN_FAILED",
                current_target_id=None,
                direction=None,
            )
            print(
                "Cannot open camera. Check the index, connection, permissions and other camera apps."
            )
            return 1
        print("Output directory: ", directory)
        emit("START")
        while True:
            try:
                ok, frame = cap.read()
            except cv2.error:
                ok, frame = (False, None)
            if not ok or frame is None or frame.size == 0:
                emit(
                    "CAMERA_ERROR",
                    reason="READ_FAILED",
                    current_target_id=None,
                    direction=None,
                )
                error_panel = panel_for(mission, None, [], camera_ok=False)
                write_image(
                    directory / f"round_{mission.round}_camera_error.png", error_panel
                )
                cv2.imshow("AprilTag Mission", error_panel)
                print("Camera read failed. An error panel was saved.")
                return 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            current = (
                sorted(set((int(i) for i in ids.flatten()))) if ids is not None else []
            )
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            added = mission.observe(current)
            for tag_id in added:
                name = f"round_{mission.round}_tag_{tag_id}.png"
                write_image(directory / name, frame)
                evidence[str(tag_id)] = name
                emit("TAG_CONFIRMED", tag_id=tag_id, evidence=name)
            live = locate_target(mission.selected_target, ids, corners, frame.shape[1])
            draw_guidance(frame, live)
            signature = (
                mission.round,
                mission.status,
                tuple(sorted(mission.seen)),
                tuple(current),
                live["status"],
                live["direction"],
            )
            if signature != last_state:
                emit(
                    "STATE",
                    status=mission.status,
                    confirmed_ids=sorted(mission.seen),
                    target_id=mission.target,
                    selected_target=mission.selected_target,
                    current_ids=current,
                    live_status=live["status"],
                    direction=live["direction"],
                    center=live["center"],
                )
                last_state = signature
            panel = panel_for(mission, frame, current, notice=notice, live=live)
            cv2.imshow("AprilTag Mission", panel)
            key = cv2.waitKey(1) & 255
            if key in (10, 13):
                if mission.confirm():
                    base = f"round_{mission.round}_result"
                    panel = panel_for(
                        mission,
                        frame,
                        current,
                        notice="Result confirmed; new tags will not change this result",
                        live=live,
                    )
                    write_image(directory / f"{base}.png", panel)
                    with (directory / f"{base}.json").open(
                        "w", encoding="utf-8"
                    ) as stream:
                        json.dump(
                            result_payload(), stream, ensure_ascii=False, indent=2
                        )
                    emit(
                        "RESULT_SAVED",
                        confirmed_ids=sorted(mission.seen),
                        target_id=mission.target,
                        image=f"{base}.png",
                        result=f"{base}.json",
                    )
                    notice = "RESULT SAVED - press R only when starting a new round"
                else:
                    notice = (
                        "Already confirmed"
                        if mission.locked
                        else "Need exactly 3 confirmed IDs; reset if more than 3"
                    )
            elif key == ord("s"):
                name = f"round_{mission.round}_screen_{datetime.now().strftime('%H%M%S_%f')}.png"
                write_image(directory / name, panel)
                emit("SCREENSHOT", image=name)
                notice = "SCREENSHOT SAVED"
            elif key == ord("r"):
                mission.reset()
                evidence = {}
                last_state = None
                emit("RESET")
                notice = (
                    "New round - previous saved results remain in the session folder"
                )
            elif key == ord("q") or key == 27:
                emit("STOPPED", reason="USER_QUIT")
                break
    except KeyboardInterrupt:
        emit("STOPPED", reason="KEYBOARD_INTERRUPT")
        return 130
    except Exception as exc:
        emit("PROGRAM_ERROR", error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
