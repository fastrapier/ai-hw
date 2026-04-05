"""Sample frames from video and run DETR object detection (Hugging Face)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/workspace/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))
MODEL_ID = os.environ.get("DETR_MODEL", "facebook/detr-resnet-50")
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "3"))
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "10"))
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.5"))


def make_synthetic_avi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter (codec MJPG)")
    for i in range(24):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (40 + (i * 7) % 200, 80, min(255, 50 + i * 8))
        cv2.rectangle(frame, (30 + i * 3, 30), (100 + i * 3, 180), (0, 220, 90), thickness=2)
        writer.write(frame)
    writer.release()


def iter_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    idx = 0
    taken = 0
    while taken < MAX_FRAMES:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % FRAME_STRIDE == 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield idx, Image.fromarray(rgb)
            taken += 1
        idx += 1
    cap.release()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exts = (".mp4", ".avi", ".mov", ".mkv")
    videos = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in exts)
    if videos:
        vid_path = videos[0]
        source = vid_path.name
    else:
        vid_path = Path("/tmp/synth_demo.avi")
        make_synthetic_avi(vid_path)
        source = "synthetic.avi"

    processor = DetrImageProcessor.from_pretrained(MODEL_ID)
    model = DetrForObjectDetection.from_pretrained(MODEL_ID)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    frame_results: list[dict] = []
    with torch.no_grad():
        for frame_idx, pil_image in iter_frames(vid_path):
            inputs = processor(images=pil_image, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=SCORE_THRESHOLD
            )[0]
            labels = results["labels"].tolist()
            scores = results["scores"].tolist()
            boxes = results["boxes"].tolist()
            objs = []
            for lab, sc, box in zip(labels, scores, boxes):
                lab_i = int(lab)
                objs.append(
                    {
                        "label_id": lab_i,
                        "label": id2label.get(lab_i, str(lab_i)),
                        "score": float(sc),
                        "box_xyxy": [float(x) for x in box],
                    }
                )
            frame_results.append({"frame_index": frame_idx, "detections": objs})

    payload = {
        "framework": "huggingface-transformers",
        "model": MODEL_ID,
        "source": source,
        "score_threshold": SCORE_THRESHOLD,
        "frames": frame_results,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
