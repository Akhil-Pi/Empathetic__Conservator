"""
camera_test.py
==============
Quick visual check: opens every camera it can find and shows them in
labelled windows. No mediapipe needed, just opencv.

    python3 tools/camera_test.py
    python3 tools/camera_test.py --indices 0 1
    python3 tools/camera_test.py --indices 0 "http://192.168.1.34:8080/video"

Press Q to quit. Note which index shows which physical view, then set
CameraConfig.SIDE_SOURCE and FRONT_SOURCE accordingly.
"""

import sys
import argparse

try:
    import cv2
except ImportError:
    print("opencv-python is not installed. Run:")
    print("  pip install opencv-python")
    sys.exit(1)


def probe_indices(max_idx=5):
    """Try indices 0..max_idx, return those that open."""
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                found.append(i)
        cap.release()
    return found


def main():
    p = argparse.ArgumentParser(description="Visual camera test")
    p.add_argument("--indices", nargs="+", default=None,
                   help="camera indices or URLs to open, e.g. 0 1 or "
                        "0 http://192.168.1.34:8080/video")
    a = p.parse_args()

    if a.indices is None:
        print("probing camera indices 0-4...")
        sources = probe_indices()
        if not sources:
            print("no cameras found. if using a phone, pass its URL:")
            print('  python3 tools/camera_test.py --indices 0 '
                  '"http://PHONE_IP:8080/video"')
            return
        print(f"found: {sources}")
    else:
        sources = []
        for s in a.indices:
            try:
                sources.append(int(s))
            except ValueError:
                sources.append(s)       # URL string

    caps = {}
    for src in sources:
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            caps[src] = cap
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  index {src}: opened, {w}x{h}")
        else:
            print(f"  index {src}: FAILED to open")

    if not caps:
        print("no cameras opened successfully")
        return

    print(f"\nshowing {len(caps)} stream(s). press Q to quit.\n"
          f"note which window shows which physical view, then set\n"
          f"CameraConfig.SIDE_SOURCE and FRONT_SOURCE accordingly.")

    while True:
        for src, cap in caps.items():
            ret, frame = cap.read()
            if ret and frame is not None:
                label = f"camera {src}"
                cv2.putText(frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow(label, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    print("done")


if __name__ == "__main__":
    main()
