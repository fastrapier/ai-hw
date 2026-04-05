"""Audio tagging with YAMNet (TensorFlow Hub)."""
from __future__ import annotations

import csv
import json
import os
import urllib.request
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/workspace/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))
SAMPLE_RATE = 16000
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "master/research/audioset/yamnet/yamnet_class_map.csv"
)


def load_class_map() -> dict[int, str]:
    with urllib.request.urlopen(CLASS_MAP_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(StringIO(text))
    return {int(row["index"]): row["display_name"] for row in reader}


def _resample_linear(wav_np: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear resample (no scipy). tf.signal.resample is missing on some TF arm64 wheels."""
    if sr_in == sr_out:
        return wav_np.astype(np.float32, copy=False)
    n_in = wav_np.shape[0]
    n_out = max(1, int(round(n_in * sr_out / sr_in)))
    x_old = np.linspace(0.0, 1.0, n_in, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, wav_np.astype(np.float64)).astype(np.float32)


def load_wav_16k_mono(path: Path) -> tf.Tensor:
    file_contents = tf.io.read_file(str(path))
    wav, sr = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sr_int = int(sr.numpy())
    wav_np = wav.numpy().astype(np.float32)
    wav_np = _resample_linear(wav_np, sr_int, SAMPLE_RATE)
    return tf.constant(wav_np)


def synthetic_waveform() -> tf.Tensor:
    seconds = 0.96
    n = int(seconds * SAMPLE_RATE)
    t = tf.linspace(0.0, seconds, n)
    return 0.1 * tf.sin(2.0 * np.pi * 440.0 * t)


def top5_for_waveform(
    model: Any,
    waveform: tf.Tensor,
    id_to_name: dict[int, str],
) -> list[dict[str, float | int | str]]:
    scores, _embeddings, _spectrogram = model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0)
    top_k = min(5, int(mean_scores.shape[0]))
    top_indices = tf.argsort(mean_scores, direction="DESCENDING")[:top_k]
    top: list[dict[str, float | int | str]] = []
    for idx in top_indices.numpy().tolist():
        top.append(
            {
                "class_index": int(idx),
                "name": id_to_name.get(int(idx), str(int(idx))),
                "score": float(mean_scores[idx].numpy()),
            }
        )
    return top


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(INPUT_DIR.glob("*.wav"))

    id_to_name = load_class_map()
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    results: list[dict[str, str | list[dict[str, float | int | str]]]] = []
    if wav_files:
        for path in wav_files:
            waveform = load_wav_16k_mono(path)
            results.append(
                {
                    "source": path.name,
                    "top5": top5_for_waveform(model, waveform, id_to_name),
                }
            )
    else:
        waveform = synthetic_waveform()
        results.append(
            {
                "source": "synthetic",
                "top5": top5_for_waveform(model, waveform, id_to_name),
            }
        )

    payload = {
        "framework": "tensorflow-hub",
        "model": "yamnet/1",
        "sample_rate_hz": SAMPLE_RATE,
        "results": results,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
