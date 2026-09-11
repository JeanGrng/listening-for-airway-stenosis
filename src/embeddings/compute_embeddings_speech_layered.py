"""Compute speech-SSL layered embeddings on the canonical B2AI set.

For each (patient, task_group) save a stack of `K` mean-pooled vectors, one
per selected transformer layer. Shape: `(N, K, D)`. Used to study whether
shallower layers of WavLM/HuBERT/Whisper carry more information for the
stenosis task (WavLM paper Table 4: best layer varies by downstream task).

Supports `--fm {wavlm, hubert, whisper}`. Layers are passed as a comma-
separated list of transformer-layer indices (1-based; index 0 = conv-encoder
output, excluded by default).

Output npz keys:
    embeddings           (N, K, D) float32
    participant_ids      (N,) <U
    tasks                (N,) <U
    original_duration_sec (N,) float32
    layer_indices        (K,) int32   -- raw transformer-layer indices

Usage (run from compute/):
    python compute_embeddings_speech_layered.py --fm wavlm     --layers 6,10,15,20,24 --output ../output/embeddings/embeddings_wavlm_layered.npz     --device cuda
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoFeatureExtractor, HubertModel, WavLMModel,
                          WhisperFeatureExtractor, WhisperModel)

import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_SRC))

from b2ai_canonical import iter_canonical_bags, prepare_slices, SAMPLE_RATE

DATA_PREFIX = os.environ.get("B2AI_ROOT", "")  # PhysioNet b2ai-voice release dir
DEFAULT_SPLIT = str(Path(__file__).resolve().parents[2] / "data" / "binary_stenosis_split.json")


def parse_layers(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_model(fm: str, device):
    if fm == "wavlm":
        name = "microsoft/wavlm-large"
        return name, AutoFeatureExtractor.from_pretrained(name), WavLMModel.from_pretrained(name).to(device).eval(), False
    if fm == "hubert":
        name = "facebook/hubert-large-ls960-ft"
        return name, AutoFeatureExtractor.from_pretrained(name), HubertModel.from_pretrained(name).to(device).eval(), False
    if fm == "whisper":
        name = "openai/whisper-large-v3"
        return name, WhisperFeatureExtractor.from_pretrained(name), WhisperModel.from_pretrained(name).to(device).eval(), True
    raise ValueError(f"unknown fm: {fm}")


def main():
    p = argparse.ArgumentParser(description="Canonical speech-SSL layered embeddings")
    p.add_argument("--fm", choices=["wavlm", "hubert", "whisper"], required=True)
    p.add_argument("--layers", type=str, required=True,
                   help="Comma-separated transformer-layer indices (1-based)")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--split_json", type=str, default=DEFAULT_SPLIT)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--window_sec", type=float, default=30.0)
    args = p.parse_args()

    layers = parse_layers(args.layers)
    assert all(L > 0 for L in layers), "L0 = conv output, excluded"
    layer_indices = np.array(layers, dtype=np.int32)

    window_size = int(args.window_sec * SAMPLE_RATE)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    name, feat_ext, model, is_whisper = load_model(args.fm, device)
    print(f"{args.fm} ({name}) on {device}  | layers={layers}", flush=True)

    @torch.no_grad()
    def embed_wav(wav: np.ndarray) -> np.ndarray:
        """Return (K, D), mean over slices and time for each requested layer."""
        slices = prepare_slices(wav, window_size)
        per_layer_slice = [[] for _ in layers]

        for start in range(0, len(slices), args.batch_size):
            chunk = slices[start:start + args.batch_size]

            if is_whisper:
                inputs = feat_ext(
                    [s for s in chunk], sampling_rate=SAMPLE_RATE,
                    return_tensors="pt", padding="max_length",
                    max_length=window_size, truncation=True,
                )
                feats = inputs.input_features.to(device=device, dtype=model.dtype)
                out = model.encoder(feats, output_hidden_states=True)
                # hidden_states[0] = post-stem, hidden_states[L] = encoder layer L
                for k, L in enumerate(layers):
                    h = out.hidden_states[L]            # (b, T_enc, D)
                    per_layer_slice[k].append(h.float().mean(dim=1).cpu().numpy())
            else:
                inputs = feat_ext(
                    [s.tolist() for s in chunk], sampling_rate=SAMPLE_RATE,
                    return_tensors="pt", padding=True, return_attention_mask=True,
                )
                iv = inputs.input_values.to(device)
                am = inputs.attention_mask.to(device)
                out = model(iv, attention_mask=am, output_hidden_states=True)
                feat_len = model._get_feat_extract_output_lengths(am.sum(-1))
                # hidden_states[0] = conv-encoder out, hidden_states[L] = transformer layer L
                for k, L in enumerate(layers):
                    h = out.hidden_states[L]            # (b, T', D)
                    T = h.shape[1]
                    m = (torch.arange(T, device=device).unsqueeze(0)
                         < feat_len.unsqueeze(1)).unsqueeze(-1).float()
                    pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)  # (b, D)
                    per_layer_slice[k].append(pooled.cpu().numpy())

        # mean over slices per layer -> (K, D)
        return np.stack([
            np.concatenate(buf, axis=0).mean(axis=0).astype(np.float32)
            for buf in per_layer_slice
        ], axis=0)

    parquet_path = os.path.join(DATA_PREFIX, "features", "torchaudio_spectrogram.parquet")
    pids, tasks, embs, durs = [], [], [], []
    n = 0
    for pid, group, wav, dur in iter_canonical_bags(parquet_path, args.split_json):
        embs.append(embed_wav(wav))
        pids.append(pid); tasks.append(group); durs.append(dur)
        n += 1
        if n % 100 == 0:
            print(f"  [{n}] {pid}/{group}", flush=True)

    embeddings = np.stack(embs, axis=0).astype(np.float32)      # (N, K, D)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        embeddings=embeddings,
        participant_ids=np.array(pids),
        tasks=np.array(tasks),
        original_duration_sec=np.array(durs, dtype=np.float32),
        layer_indices=layer_indices,
    )
    print(f"\nSaved {args.output} | shape={embeddings.shape} | layers={layers} | "
          f"{len(set(pids))} patients × {len(set(tasks))} groups", flush=True)


if __name__ == "__main__":
    main()
