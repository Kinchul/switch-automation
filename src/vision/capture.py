from __future__ import annotations


def open_capture(device: str = "/dev/video0", width: int = 1280, height: int = 720, fps: int = 30):
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install `opencv-python` or `opencv-python-headless` "
            "before using video capture."
        ) from exc

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open capture device {device!r}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap
