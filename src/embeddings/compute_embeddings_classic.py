"""Classic audio baseline embeddings on the canonical B2AI (pid, task_group) set.

Reviewer ICASSP expectation : « est-ce que le FM apporte vraiment quelque
chose vs un baseline audio classique ? ». This script computes a sklearn-ready
feature vector per (patient, task_group) using only librosa and `numpy`
no foundation model. The output follows the same npz schema as the speech-FM
embeddings (`compute_embeddings_wavlm.py`) so that `run_mil_stenosis_stridor.py
--fm classic` reuses the exact same MIL pipeline + 5-fold CV.

Features per task (215-d total) :

- log-mel 80 bands × {mean, std}                                    : 160
- MFCC 13 (+ delta-MFCC) × {mean, std}                              : 52
- F0 stats (librosa.yin) : median + std on voiced frames            : 2
- voicing fraction (frames with valid F0)                            : 1

Operates on the same Griffin-Lim waveform variant of the canonical selection
(`compute/b2ai_canonical.py:iter_canonical_bags`) so the cohort exactly
matches WavLM/HuBERT/Whisper.

Usage (run from compute/) :
    python compute_embeddings_classic.py \
        --output ../output/embeddings/embeddings_classic.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import librosa

import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_SRC))

from b2ai_canonical import iter_canonical_bags, SAMPLE_RATE

DATA_PREFIX = os.environ.get("B2AI_ROOT", "")  # PhysioNet b2ai-voice release dir
DEFAULT_SPLIT = str(Path(__file__).resolve().parents[2] / "data" / "binary_stenosis_split.json")

N_MELS  = 80
N_MFCC  = 13
F0_FMIN = 65.0
F0_FMAX = 400.0


def classic_features(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a (215,) feature vector for a single waveform."""
    feats = []

    # log-mel 80 bands × {mean, std}
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=512, hop_length=160,
                                          n_mels=N_MELS, fmax=sr / 2)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    feats.append(log_mel.mean(axis=1))
    feats.append(log_mel.std(axis=1))

    # MFCC 13 × {mean, std} + delta-MFCC × {mean, std}
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc, mode='nearest')
    feats.append(mfcc.mean(axis=1));  feats.append(mfcc.std(axis=1))
    feats.append(delta.mean(axis=1)); feats.append(delta.std(axis=1))

    # F0 stats (librosa.yin pitch estimation, voicing fraction)
    try:
        f0 = librosa.yin(wav, fmin=F0_FMIN, fmax=F0_FMAX, sr=sr,
                          frame_length=1024, hop_length=160)
        voiced = (f0 > F0_FMIN) & (f0 < F0_FMAX) & np.isfinite(f0)
        if voiced.sum() > 0:
            feats.append(np.array([np.median(f0[voiced]),
                                     np.std(f0[voiced]),
                                     float(voiced.mean())], dtype=np.float64))
        else:
            feats.append(np.zeros(3, dtype=np.float64))
    except Exception:
        feats.append(np.zeros(3, dtype=np.float64))

    return np.concatenate(feats).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Canonical classic audio baseline features")
    parser.add_argument("--output", type=str, default="embeddings_classic.npz")
    parser.add_argument("--split_json", type=str, default=DEFAULT_SPLIT)
    args = parser.parse_args()

    parquet_path = os.path.join(DATA_PREFIX, "features",
                                "torchaudio_spectrogram.parquet")

    pids, tasks, embs, durs = [], [], [], []
    n = 0
    for pid, group, wav, dur in iter_canonical_bags(parquet_path, args.split_json):
        feats = classic_features(wav)
        embs.append(feats)
        pids.append(pid); tasks.append(group); durs.append(dur)
        n += 1
        if n % 200 == 0:
            print(f"  [{n}] {pid}/{group}  feats.shape={feats.shape}", flush=True)

    embeddings = np.stack(embs, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        embeddings=embeddings,
        participant_ids=np.array(pids),
        tasks=np.array(tasks),
        original_duration_sec=np.array(durs, dtype=np.float32),
    )
    print(f"\nSaved {args.output} | shape={embeddings.shape} | "
          f"{len(set(pids))} patients × {len(set(tasks))} groups", flush=True)


if __name__ == "__main__":
    main()
