import numpy as np
import librosa

# ── Presets ───────────────────────────────────────────────────────────────────
PRESETS = {
    "for-norm": dict(
        sample_rate   = 16000,
        n_fft         = 512,
        hop_length    = 160,
        win_length    = 400,
        n_mels        = 80,
        f_min         = 20.0,
        f_max         = 8000.0,
        target_frames = 500,
    ),
    "for-2sec": dict(
        sample_rate   = 16000,
        n_fft         = 512,        # ← better for 16kHz
        hop_length    = 160,        # ← 10ms at 16kHz
        win_length    = 400,        # ← 25ms at 16kHz
        n_mels        = 80,
        f_min         = 20.0,
        f_max         = 8000.0,     # ← Nyquist for 16kHz
        target_frames = 201,        # ← exact for 2s @ 16kHz / hop 160
    ),
}
# ─────────────────────────────────────────────────────────────────────────────

_cfg = PRESETS["for-2sec"]   # default


def set_preset(name: str):
    global _cfg
    assert name in PRESETS, f"Unknown preset '{name}'. Choose from {list(PRESETS)}"
    _cfg = PRESETS[name]


def extract(audio: np.ndarray) -> np.ndarray:
    """
    audio   : 1D numpy float32 array (raw waveform)
    returns : (n_mels, target_frames) float32 array, normalized to [0, 1]
    """
    mel = librosa.feature.melspectrogram(
        y          = audio,
        sr         = _cfg["sample_rate"],
        n_fft      = _cfg["n_fft"],
        hop_length = _cfg["hop_length"],
        win_length = _cfg["win_length"],
        n_mels     = _cfg["n_mels"],
        fmin       = _cfg["f_min"],
        fmax       = _cfg["f_max"],
        power      = 2.0,
    )

    # Convert to dB scale
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

    # Pad or trim to fixed target_frames
    target = _cfg["target_frames"]
    if mel_db.shape[1] < target:
        mel_db = np.pad(mel_db, ((0, 0), (0, target - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :target]

    # Normalize to [0, 1]
    lo, hi = mel_db.min(), mel_db.max()
    if hi > lo:
        mel_db = (mel_db - lo) / (hi - lo)

    return mel_db.astype(np.float32)   # shape: (80, target_frames)