"""Unicode-safe image I/O for Windows paths."""

from pathlib import Path
import cv2
import numpy as np


def read_image(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Image file not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path, frame):
    path = Path(path)
    ok, data = cv2.imencode(path.suffix, frame)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    data.tofile(str(path))
