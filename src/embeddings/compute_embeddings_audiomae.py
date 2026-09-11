"""Compute AudioMAE embeddings on the canonical B2AI (pid, task_group) set.

AudioMAE (hance-ai/audiomae) self-supervised masked autoencoder ViT, 768-d.
Mel computed directly from the concatenated linear spectrogram (no Griffin-Lim).
Shared canonical cohort/task selection (compute/b2ai_canonical.py).

Usage (run from compute/):
    python compute_embeddings_audiomae.py --output ../output/embeddings/embeddings_audiomae.npz --device cuda
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from timm.models.vision_transformer import VisionTransformer

import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_SRC))

from b2ai_canonical import iter_canonical_bags_spec as _spec_bags

DATA_PREFIX = os.environ.get("B2AI_ROOT", "")  # PhysioNet b2ai-voice release dir
DEFAULT_SPLIT = str(Path(__file__).resolve().parents[2] / "data" / "binary_stenosis_split.json")
AUDIOMAE_MEAN = -4.2677393
AUDIOMAE_STD = 4.5689974


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
    parser = argparse.ArgumentParser(description="Canonical AudioMAE embeddings")
    parser.add_argument("--output", type=str, default="embeddings_audiomae.npz")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split_json", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    MAX_LEN = 1024  # AudioMAE expects exactly 1024 time frames
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weights_path = hf_hub_download(repo_id="hance-ai/audiomae", filename="model.safetensors")
    encoder = VisionTransformer(
        img_size=(1024, 128), patch_size=(16, 16), in_chans=1,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        num_classes=0, global_pool="",
    )
    sd = load_safetensors(weights_path)
    sd = {k.replace("encoder.", "", 1): v for k, v in sd.items() if k.startswith("encoder.")}
    encoder.load_state_dict(sd, strict=False)
    encoder = encoder.to(device)
    encoder.eval()
    print(f"AudioMAE on {device}", flush=True)

    parquet_path = os.path.join(DATA_PREFIX, "features", "torchaudio_spectrogram.parquet")
    pids, tasks, embs, durs = [], [], [], []
    mel_buf, meta_buf = [], []

    @torch.no_grad()
    def flush():
        if not mel_buf:
            return
        z = encoder.forward_features(torch.stack(mel_buf).to(device))  # (b,1+N,768)
        pooled = z[:, 1:, :].mean(dim=1).cpu().numpy()                 # drop CLS, mean
        for e, (pid, grp, dur) in zip(pooled, meta_buf):
            embs.append(e); pids.append(pid); tasks.append(grp); durs.append(dur)
        mel_buf.clear(); meta_buf.clear()

    n = 0
    for pid, group, spec, dur in _spec_bags(parquet_path, args.split_json):
        mel = linear_spectrogram_to_mel(spec, n_mels=128)
        mi = mel.T                                                     # (T, 128)
        T = mi.shape[0]
        if T > MAX_LEN:
            mi = mi[:MAX_LEN]
        elif T < MAX_LEN:
            mi = np.concatenate([mi, np.zeros((MAX_LEN - T, 128), np.float32)], 0)
        mi = (mi - AUDIOMAE_MEAN) / (AUDIOMAE_STD * 2)
        mel_buf.append(torch.tensor(mi, dtype=torch.float32).unsqueeze(0))
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
