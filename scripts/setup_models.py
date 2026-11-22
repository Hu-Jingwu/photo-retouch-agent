"""Download the face / pose / segmentation models into models/ (about 25 MB).

    python scripts/setup_models.py

They are not stored in the repository. Licences: YuNet (OpenCV Zoo, MIT);
MediaPipe Face Landmarker, Pose Landmarker and Selfie Multiclass Segmenter (Apache 2.0).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
URLS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "selfie_multiclass_256x256.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    for name, url in URLS.items():
        dst = MODELS / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"ok       {name}")
            continue
        print(f"download {name} ...", flush=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)
        print(f"         {dst.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
