"""
preprocess.py  — FoR dataset explorer and validator.

Since for-norm / for-2sec are already normalized and split by the authors,
this script does NOT resample or segment. Instead it:
  1. Walks the dataset folder and builds a metadata CSV.
  2. Validates every file (readable, correct sr, non-silent).
  3. Optionally copies a clean subset to data/processed/ for fast iteration.

Usage:
    python src/data/preprocess.py \
        --dataset_dir  data/raw/for-norm \
        --out_csv       data/for_norm_meta.csv \
        --version       for-norm

    python src/data/preprocess.py \
        --dataset_dir  data/raw/for-2sec \
        --out_csv       data/for_2sec_meta.csv \
        --version       for-2sec
"""

import argparse
import logging
from pathlib import Path
import csv

import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# FoR folder layout (consistent across for-norm, for-2sec, for-rerec)
SPLITS  = ["training", "validation", "testing"]
CLASSES = {"real": 0, "fake": 1}

EXPECTED_SR = {
    "for-original": None,   # mixed, not normalized
    "for-norm":     16000,
    "for-2sec":     16000,  # for-2sec uses 44.1 kHz
    "for-rerec":    44100,
}


def scan_dataset(dataset_dir: Path, version: str) -> list[dict]:
    """
    Walk the FoR folder tree and return a list of metadata dicts.
    Each dict: {path, split, label_name, label, sr, duration, valid, issue}
    """
    expected_sr = EXPECTED_SR.get(version)
    records = []

    for split in SPLITS:
        for label_name, label in CLASSES.items():
            folder = dataset_dir / split / label_name
            if not folder.exists():
                logger.warning(f"Missing folder: {folder}")
                continue

            wavs = sorted(folder.glob("*.wav"))
            logger.info(f"  {split}/{label_name}: {len(wavs)} files")

            for wav in wavs:
                rec = {
                    "path":       str(wav),
                    "split":      split,
                    "label_name": label_name,
                    "label":      label,
                    "sr":         None,
                    "duration":   None,
                    "valid":      True,
                    "issue":      "",
                }
                try:
                    info = sf.info(str(wav))
                    rec["sr"]       = info.samplerate
                    rec["duration"] = round(info.duration, 3)

                    # Check sample rate if expected
                    if expected_sr and info.samplerate != expected_sr:
                        rec["valid"] = False
                        rec["issue"] = f"sr={info.samplerate} expected {expected_sr}"

                    # Check not silent
                    audio, _ = sf.read(str(wav), dtype="float32")
                    if np.max(np.abs(audio)) < 1e-6:
                        rec["valid"] = False
                        rec["issue"] = "silent file"

                except Exception as e:
                    rec["valid"] = False
                    rec["issue"] = str(e)

                records.append(rec)

    return records


def save_csv(records: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "split", "label_name", "label", "sr", "duration", "valid", "issue"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Saved metadata CSV → {out_path}")


def print_summary(records: list[dict]):
    from collections import Counter
    valid   = [r for r in records if r["valid"]]
    invalid = [r for r in records if not r["valid"]]

    logger.info("=" * 50)
    logger.info(f"Total files  : {len(records)}")
    logger.info(f"Valid        : {len(valid)}")
    logger.info(f"Invalid      : {len(invalid)}")
    if invalid:
        for r in invalid[:5]:
            logger.warning(f"  {r['path']} → {r['issue']}")

    logger.info("\nPer split / class breakdown:")
    counts = Counter((r["split"], r["label_name"]) for r in valid)
    for (split, cls), n in sorted(counts.items()):
        logger.info(f"  {split:12s} {cls:6s}: {n}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True,
                        help="Root of a FoR version, e.g. data/raw/for-norm")
    parser.add_argument("--out_csv",     required=True,
                        help="Where to save the metadata CSV")
    parser.add_argument("--version",     default="for-norm",
                        choices=list(EXPECTED_SR.keys()))
    args = parser.parse_args()

    logger.info(f"Scanning {args.dataset_dir} (version={args.version}) ...")
    records = scan_dataset(Path(args.dataset_dir), args.version)
    print_summary(records)
    save_csv(records, Path(args.out_csv))
