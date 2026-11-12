"""Generate explicitly synthetic inputs for software checks, never field evidence."""

import argparse
from pathlib import Path
import cv2
import numpy as np
from rov_vision.io_utils import write_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/synthetic"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    color = np.full((480, 720, 3), 235, np.uint8)
    cv2.putText(
        color,
        "SYNTHETIC INPUT - NOT A CAMERA TEST",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    for x, bgr in [(100, (0, 0, 230)), (340, (0, 230, 230)), (580, (230, 0, 0))]:
        cv2.rectangle(color, (x, 150), (x + 40, 400), bgr, -1)
    write_image(args.output / "color_poles.png", color)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tags = np.full((360, 720, 3), 255, np.uint8)
    cv2.putText(
        tags,
        "SYNTHETIC 36h11 - IDs 2, 7, 12",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    for x, number in [(40, 2), (280, 7), (520, 12)]:
        marker = cv2.aruco.generateImageMarker(dictionary, number, 160)
        tags[100:260, x : x + 160] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        # Separate printable tags retain an unmodified white margin.
        bordered = cv2.copyMakeBorder(
            marker, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255
        )
        write_image(args.output / f"tag_36h11_{number}.png", bordered)
    write_image(args.output / "apriltags.png", tags)
    print(
        f"Synthetic inputs saved to {args.output}. These are not physical test results."
    )


if __name__ == "__main__":
    main()
