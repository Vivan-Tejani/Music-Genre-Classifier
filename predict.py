"""
Prediction script — classify a local audio file into one of 10 music genres.

Usage:
    python predict.py --file /path/to/song.wav
"""
import argparse
import os
import sys

import librosa
import numpy as np
import torch

import config
from dataset import audio_to_mel_spectrogram
from model import GenreCNN


def load_model(device: torch.device) -> GenreCNN:
    """Load the best saved model checkpoint."""
    if not os.path.isfile(config.BEST_MODEL_PATH):
        print(f"[✗] No checkpoint found at {config.BEST_MODEL_PATH}")
        print("    Train the model first:  python train.py")
        sys.exit(1)

    model = GenreCNN(num_classes=config.NUM_CLASSES, dropout=0.0)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_file(filepath: str, model: GenreCNN, device: torch.device):
    """
    Run inference on a single audio file.

    The file is split into SEGMENT_DURATION-second segments; per-segment
    predictions are averaged to produce a final genre prediction.
    """
    if not os.path.isfile(filepath):
        print(f"[✗] File not found: {filepath}")
        sys.exit(1)

    # Get total duration of the file
    duration = librosa.get_duration(path=filepath)
    num_segments = max(1, int(duration // config.SEGMENT_DURATION))

    all_probs = []
    for seg_idx in range(num_segments):
        offset = seg_idx * config.SEGMENT_DURATION
        mel = audio_to_mel_spectrogram(
            filepath, offset=offset, duration=config.SEGMENT_DURATION
        )
        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    # Average across segments
    avg_probs = np.mean(np.concatenate(all_probs, axis=0), axis=0)
    top_idx = int(np.argmax(avg_probs))
    confidence = avg_probs[top_idx]

    return top_idx, confidence, avg_probs


def main():
    parser = argparse.ArgumentParser(
        description="Predict the music genre of a local audio file."
    )
    parser.add_argument(
        "--file", "-f", type=str, required=True,
        help="Path to a .wav audio file."
    )
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = load_model(device)
    top_idx, confidence, avg_probs = predict_file(args.file, model, device)

    # ── Pretty output ──
    predicted_genre = config.GENRES[top_idx]
    print()
    print("=" * 50)
    print(f"  🎵  Predicted Genre:  {predicted_genre.upper()}")
    print(f"  📊  Confidence:       {confidence:.2%}")
    print("=" * 50)

    # Show ranking
    print("\n  Genre Probabilities:")
    ranked = sorted(enumerate(avg_probs), key=lambda x: -x[1])
    for rank, (idx, prob) in enumerate(ranked, 1):
        bar = "█" * int(prob * 30)
        marker = " ◀" if idx == top_idx else ""
        print(f"    {rank:2d}. {config.GENRES[idx]:<12s} {prob:.2%} {bar}{marker}")
    print()


if __name__ == "__main__":
    main()
