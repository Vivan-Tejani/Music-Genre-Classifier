"""
Data pipeline for the GTZAN Music Genre Classifier.

• Downloads and extracts the GTZAN dataset (≈1.2 GB).
• Provides a PyTorch Dataset that converts .wav → Mel-spectrograms on the fly.
"""
import os
import tarfile
import urllib.request

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

import config


# ─────────────────────── Download helpers ───────────────────────

GTZAN_URL = (
    "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
)


def prepare_dataset() -> str:
    """
    Download and extract the GTZAN dataset, ensuring the directory 
    structure matches config.GTZAN_DIR.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    archive_path = os.path.join(config.DATA_DIR, "genres.tar.gz")

    # 1. Check if already extracted and contains the genre folders
    if os.path.isdir(config.GTZAN_DIR):
        subdirs = [d for d in os.listdir(config.GTZAN_DIR) if os.path.isdir(os.path.join(config.GTZAN_DIR, d))]
        if len(subdirs) >= 10:
            print(f"[✓] GTZAN dataset ready at {config.GTZAN_DIR}")
            return config.GTZAN_DIR

    # 2. Download
    if not os.path.isfile(archive_path):
        print(f"[↓] Downloading GTZAN dataset...")
        urllib.request.urlretrieve(GTZAN_URL, archive_path, _progress_hook)
        print()

    # 3. Extract
    print(f"[⤓] Extracting to {config.DATA_DIR}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=config.DATA_DIR)

    # 4. Correct the "Gotcha": Handle inconsistent folder naming
    # Paths we might find after extraction
    possible_dirs = [
        os.path.join(config.DATA_DIR, "genres"),
        os.path.join(config.DATA_DIR, "genres_original")
    ]
    
    for p in possible_dirs:
        if os.path.isdir(p) and p != config.GTZAN_DIR:
            # If it's named 'genres' but we want 'genres_original', rename it
            os.rename(p, config.GTZAN_DIR)
            break

    print(f"[✓] Dataset prepared at {config.GTZAN_DIR}")
    return config.GTZAN_DIR


# ─────────────────── Mel-spectrogram helpers ────────────────────


def audio_to_mel_spectrogram(filepath: str, offset: float = 0.0,
                              duration: float | None = None) -> np.ndarray:
    """
    Load a wav file and return its Mel-spectrogram (log-scaled).
    Shape: (n_mels, time_frames)
    """
    y, sr = librosa.load(filepath, sr=config.SAMPLE_RATE,
                         offset=offset, duration=duration)
    # Pad to exactly SEGMENT_SAMPLES if shorter
    if len(y) < config.SEGMENT_SAMPLES:
        y = np.pad(y, (0, config.SEGMENT_SAMPLES - len(y)))
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS,
        hop_length=config.HOP_LENGTH, n_fft=config.N_FFT,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


# ─────────────────── PyTorch Dataset ────────────────────────────


class GenreDataset(Dataset):
    """
    Each 30 s clip is split into non-overlapping segments of
    SEGMENT_DURATION seconds, giving ~10 samples per clip.
    """

    def __init__(self, file_list: list[tuple[str, int]]):
        """
        Args:
            file_list: list of (filepath, label_index) tuples.
        """
        self.samples: list[tuple[str, int, float]] = []  # (path, label, offset)
        num_segments = config.DURATION // config.SEGMENT_DURATION

        for fpath, label in file_list:
            for seg_idx in range(num_segments):
                offset = seg_idx * config.SEGMENT_DURATION
                self.samples.append((fpath, label, offset))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        fpath, label, offset = self.samples[idx]
        mel = audio_to_mel_spectrogram(
            fpath, offset=offset, duration=config.SEGMENT_DURATION
        )
        # Shape: (1, n_mels, time_frames)  — single channel
        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        return tensor, label


# ─────────────────── Utility: collect file list ─────────────────


def get_file_list(gtzan_dir: str | None = None) -> list[tuple[str, int]]:
    """
    Walk the genre directories and return a list of (filepath, label_index) pairs.
    Specifically skips the corrupted 'jazz.00054.wav' file to prevent librosa errors.
    """
    gtzan_dir = gtzan_dir or config.GTZAN_DIR
    file_list: list[tuple[str, int]] = []

    for label_idx, genre in enumerate(config.GENRES):
        genre_dir = os.path.join(gtzan_dir, genre)
        
        if not os.path.isdir(genre_dir):
            print(f"[!] Warning: Directory for genre '{genre}' not found.")
            continue

        # Sort to ensure consistent file/label mapping
        for fname in sorted(os.listdir(genre_dir)):
            if fname.endswith(".wav"):
                # Handle the notorious corrupted file in GTZAN
                if fname == "jazz.00054.wav":
                    print(f"[i] Skipping corrupted file: {fname}")
                    continue
                
                fpath = os.path.join(genre_dir, fname)
                file_list.append((fpath, label_idx))

    print(f"Successfully indexed {len(file_list)} audio files.")
    return file_list


if __name__ == "__main__":
    prepare_dataset()
    fl = get_file_list()
    ds = GenreDataset(fl[:2])
    mel, lbl = ds[0]
    print(f"Sample tensor shape: {mel.shape}, label: {config.GENRES[lbl]}")