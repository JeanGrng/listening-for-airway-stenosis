"""Compute AST (AudioSet) embeddings on the canonical B2AI (pid, task_group) set.

AST finetuned on AudioSet (768-d CLS). Mel computed directly from the
concatenated linear spectrogram (no Griffin-Lim), shared canonical cohort/task
selection (compute/b2ai_canonical.py).

Usage (run from compute/):
    python compute_embeddings_ast.py --output ../output/embeddings/embeddings_ast.npz --device cuda
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import ASTModel

import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_SRC))

from b2ai_canonical import iter_canonical_bags_spec as _spec_bags

DATA_PREFIX = os.environ.get("B2AI_ROOT", "")  # PhysioNet b2ai-voice release dir
DEFAULT_SPLIT = str(Path(__file__).resolve().parents[2] / "data" / "binary_stenosis_split.json")


def linear_spectrogram_to_mel(spec_db, n_mels=128, sample_rate=16000,
                              target_hop_ms=10, current_hop_ms=20):
    """Linear spectrogram (dB) → mel spectrogram with time upsampling."""
    spec_tensor = torch.tensor(spec_db, dtype=torch.float32)
    spec_power = (torch.pow(10.0, spec_tensor / 20.0)) ** 2
    n_freqs = spec_power.shape[0]
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_freqs, f_min=0.0, f_max=sample_rate / 2.0,
        n_mels=min(n_mels, 80), sample_rate=sample_rate,
    )
    mel_power = mel_fb.T @ spec_power
    mel_db = 10.0 * torch.log10(torch.clamp(mel_power, min=1e-10))
    mel_db = mel_db.clamp(min=mel_db.max() - 95.0)
    target_frames = int(mel_db.shape[1] * (current_hop_ms / target_hop_ms))
    mel_db_resized = torch.nn.functional.interpolate(
        mel_db.unsqueeze(0).unsqueeze(0), size=(n_mels, target_frames),
        mode='bilinear', align_corners=True,
    ).squeeze(0).squeeze(0)
    return mel_db_resized.numpy()


def main():
    parser = argparse.ArgumentParser(description="Canonical AST embeddings")
    parser.add_argument("--output", type=str, default="embeddings_ast.npz")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split_json", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    MAX_LEN = 1024
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
    model.eval()
    print(f"AST on {device}", flush=True)

    parquet_path = os.path.join(DATA_PREFIX, "features", "torchaudio_spectrogram.parquet")
    pids, tasks, embs, durs = [], [], [], []
    mel_buf, meta_buf = [], []

    @torch.no_grad()
    def flush():
        if not mel_buf:
            return
        out = model(input_values=torch.stack(mel_buf).to(device))
        cls = out.last_hidden_state[:, 0].cpu().numpy()
        for e, (pid, grp, dur) in zip(cls, meta_buf):
            embs.append(e); pids.append(pid); tasks.append(grp); durs.append(dur)
        mel_buf.clear(); meta_buf.clear()

    n = 0
    for pid, group, spec, dur in _spec_bags(parquet_path, args.split_json):
        mel = linear_spectrogram_to_mel(spec, n_mels=128)
        mi = mel.T[:MAX_LEN]
        mi = (mi - mi.mean()) / (mi.std() + 1e-6)
        T = mi.shape[0]
        if T < MAX_LEN:                       # AST expects exactly MAX_LEN frames
            mi = np.pad(mi, ((0, MAX_LEN - T), (0, 0)))
        mel_buf.append(torch.tensor(mi, dtype=torch.float32))
        meta_buf.append((pid, group, dur))
        if len(mel_buf) >= args.batch_size:
            flush()
        n += 1
        if n % 100 == 0:
            print(f"  [{n}] {pid}/{group}", flush=True)
    flush()

    embeddings = np.stack(embs).astype(np.float32)
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
