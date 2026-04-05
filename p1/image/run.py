"""ImageNet classification: ViT-B/16 + 10-crop averaging (torchvision, PyTorch)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms as T

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/workspace/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))
TOP_K = 5
# ViT weights.meta в части версий torchvision без mean/std — те же значения, что у ResNet на ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def placeholder_image() -> Image.Image:
    """Simple gradient if no files (deterministic sanity check)."""
    w, h = 224, 224
    data = torch.zeros(h, w, 3, dtype=torch.uint8)
    for y in range(h):
        data[y, :, 0] = int(255 * y / max(h - 1, 1))
    for x in range(w):
        data[:, x, 1] = torch.maximum(data[:, x, 1], torch.tensor(int(255 * x / max(w - 1, 1))))
    data[:, :, 2] = 128
    return Image.fromarray(data.numpy(), mode="RGB")


def topk_for_image(
    model: torch.nn.Module,
    weights,
    categories: list[str],
    img: Image.Image,
) -> list[dict[str, float | int | str]]:
    # Ten-crop + mean logits: заметно стабильнее на крупных сценах / нестандартных кадрах,
    # чем один center crop у ResNet (живопись всё равно вне домена ImageNet).
    meta = weights.meta
    mean = meta.get("mean", IMAGENET_MEAN)
    std = meta.get("std", IMAGENET_STD)
    resize = T.Resize(256, interpolation=T.InterpolationMode.BILINEAR)
    ten_crop = T.TenCrop(224)
    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=mean, std=std)
    img_s = resize(img)
    crops = ten_crop(img_s)
    batch = torch.stack([normalize(to_tensor(c)) for c in crops])
    with torch.no_grad():
        logits = model(batch).mean(dim=0)
        probs = torch.softmax(logits, dim=0)
    top = torch.topk(probs, k=min(TOP_K, probs.numel()))
    rows: list[dict[str, float | int | str]] = []
    for rank, (score, idx) in enumerate(
        zip(top.values.tolist(), top.indices.tolist()),
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                "class_index": int(idx),
                "label": categories[int(idx)],
                "probability": float(score),
            }
        )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    paths = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in exts]
    paths.sort()

    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)
    model.eval()
    categories = weights.meta["categories"]

    results: list[dict[str, str | list[dict[str, float | int | str]]]] = []
    if paths:
        for path in paths:
            img = load_image(path)
            results.append(
                {
                    "source": path.name,
                    "top_k": topk_for_image(model, weights, categories, img),
                }
            )
    else:
        img = placeholder_image()
        results.append(
            {
                "source": "placeholder",
                "top_k": topk_for_image(model, weights, categories, img),
            }
        )

    payload = {
        "framework": "pytorch",
        "model": "vit_b_16_imagenet1k_v1",
        "inference": "ten_crop_mean_logits",
        "results": results,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
