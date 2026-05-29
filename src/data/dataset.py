"""
dataset.py — FoR dataset loader, numpy-based (no PyTorch).

Teammates using TensorFlow/Keras can call build_tf_datasets().
Teammates using scikit-learn can call build_arrays() to get X, y directly.
"""

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from src.features.mel_spectrogram import extract, set_preset

REAL = 0
FAKE = 1

SPLIT_MAP = {
    "train":      "training",
    "training":   "training",
    "val":        "validation",
    "validation": "validation",
    "test":       "testing",
    "testing":    "testing",
}


class FoRDataset:
    """
    Loads all files for a given split into memory as numpy arrays.
    For 195k files this is memory-heavy — use build_tf_datasets()
    for lazy loading if RAM is limited.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        meta_csv: Optional[str] = None,
    ):
        split = SPLIT_MAP.get(split, split)
        self.samples: list[tuple[Path, int]] = []

        if meta_csv and Path(meta_csv).exists():
            self._load_from_csv(meta_csv, split)
        else:
            self._load_from_disk(Path(dataset_dir), split)

        if not self.samples:
            raise RuntimeError(f"No samples found for split='{split}' in {dataset_dir}")

        print(f"  {split:12s}: {len(self.samples):>6,} samples")

    def _load_from_disk(self, root: Path, split: str):
        for label_name, label in [("real", REAL), ("fake", FAKE)]:
            folder = root / split / label_name
            if not folder.exists():
                raise FileNotFoundError(f"Expected folder not found: {folder}")
            for wav in sorted(folder.glob("*.wav")):
                self.samples.append((wav, label))

    def _load_from_csv(self, csv_path: str, split: str):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == split and row["valid"] == "True":
                    self.samples.append((Path(row["path"]), int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def get_item(self, idx: int) -> tuple[np.ndarray, int]:
        path, label = self.samples[idx]
        audio, _ = sf.read(str(path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)   # stereo → mono
        mel = extract(audio)             # (80, target_frames)
        mel = mel[np.newaxis, ...]       # (1, 80, target_frames) channel dim
        return mel, label


# ── For TensorFlow / Keras teammates ─────────────────────────────────────────

def build_tf_datasets(
    dataset_dir: str,
    preset:      str = "for-2sec",
    batch_size:  int = 32,
    meta_csv:    Optional[str] = None,
):
    """
    Returns (train_ds, val_ds, test_ds) as tf.data.Dataset objects.

    Usage in trainer.py:
        from src.data.dataset import build_tf_datasets
        train_ds, val_ds, test_ds = build_tf_datasets("data/raw/for-2sec")
        model.fit(train_ds, validation_data=val_ds, epochs=20)
    """
    import tensorflow as tf
    set_preset(preset)

    def make_generator(split):
        ds = FoRDataset(dataset_dir, split=split, meta_csv=meta_csv)
        def gen():
            for i in range(len(ds)):
                x, y = ds.get_item(i)
                yield x, y
        return gen, len(ds)

    results = []
    for split, shuffle in [("training", True), ("validation", False), ("testing", False)]:
        gen, n = make_generator(split)

        from src.features.mel_spectrogram import _cfg
        shape = (1, _cfg["n_mels"], _cfg["target_frames"])

        tf_ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=shape, dtype=tf.float32),
                tf.TensorSpec(shape=(),    dtype=tf.int32),
            )
        )
        if shuffle:
            tf_ds = tf_ds.shuffle(buffer_size=2000, seed=42)

        tf_ds = tf_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        results.append(tf_ds)

    return tuple(results)


# ── For scikit-learn teammates (loads everything into RAM) ────────────────────

def build_arrays(
    dataset_dir: str,
    preset:      str = "for-2sec",
    split:       str = "training",
    meta_csv:    Optional[str] = None,
    flatten:     bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y) as numpy arrays — useful for sklearn models.

    flatten=True  → X shape: (N, 80 * target_frames)  for SVM / RandomForest
    flatten=False → X shape: (N, 1, 80, target_frames) for CNN

    Usage:
        from src.data.dataset import build_arrays
        X_train, y_train = build_arrays("data/raw/for-2sec", split="training")
        X_test,  y_test  = build_arrays("data/raw/for-2sec", split="testing")
    """
    set_preset(preset)
    ds = FoRDataset(dataset_dir, split=split, meta_csv=meta_csv)

    X, y = [], []
    for i in range(len(ds)):
        mel, label = ds.get_item(i)
        X.append(mel.flatten() if flatten else mel)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
