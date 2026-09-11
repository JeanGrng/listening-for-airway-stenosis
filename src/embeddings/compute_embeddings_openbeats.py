"""Compute OpenBEATs embeddings for B2AI Voice recordings and save to disk.

Uses OpenBEATs-Large-i3 (300M params, 24-layer transformer, 1024-dim embeddings),
a general-purpose audio encoder pretrained on multi-domain audio (AudioSet, FreeSound,
FMA, iNaturalist) via masked token prediction, then fine-tuned on AudioSet-2M.
State-of-the-art on environmental sound, bioacoustics, and audio reasoning benchmarks.

Standalone implementation, only requires PyTorch + huggingface_hub (no ESPnet).

Usage:
    python compute_embeddings_openbeats.py --output embeddings_openbeats.npz --device cuda
"""

import argparse
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download, list_repo_files

import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_SRC))

from b2ai_canonical import iter_canonical_bags_spec as _spec_bags

DEFAULT_SPLIT = str(Path(__file__).resolve().parents[2] / "data" / "binary_stenosis_split.json")


# =============================================================================
# Standalone OpenBEATs encoder (no ESPnet / fairseq dependency)
# Architecture based on BEATs (microsoft/unilm) adapted for OpenBEATs-Large
# =============================================================================


def _relative_position_bucket(relative_position, num_buckets=320, max_distance=800):
    """BEATs-style logarithmic relative position bucketing."""
    sign = (relative_position >= 0).long()
    relative_position = relative_position.abs()

    half = num_buckets // 2
    max_exact = half // 2

    is_small = relative_position < max_exact

    val_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (half - max_exact)
    ).long()
    val_if_large = torch.clamp(val_if_large, max=half - 1)

    bucket = torch.where(is_small, relative_position, val_if_large)
    bucket = bucket + sign * half
    return bucket


class SamePad(nn.Module):
    """Remove trailing element for even-kernel Conv1d to maintain length."""
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, :-self.remove]
        return x


class BEATsAttention(nn.Module):
    """Multi-head attention with shared relative position bias."""
    def __init__(self, embed_dim, num_heads, num_buckets=320, max_distance=800,
                 has_relative_attention_bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        # GRU-gated relative position
        self.grep_linear = nn.Linear(self.head_dim, 8)
        self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def compute_bias(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        buckets = _relative_position_bucket(rel_pos, self.num_buckets, self.max_distance)
        bias = self.relative_attention_bias(buckets)  # (T, T, H)
        return bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, T, T)

    def forward(self, x, pos_bias=None):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if pos_bias is not None:
            # GRU gate: content-aware scaling of position bias
            gate = torch.sigmoid(self.grep_linear(q) / 2.0)  # (B, H, T, 8)
            gate = gate.reshape(B, self.num_heads, T, 2, 4).sum(dim=-1)  # (B, H, T, 2)
            gate_a, gate_b = gate.chunk(2, dim=-1)  # each (B, H, T, 1)
            pos_bias = pos_bias * gate_a.transpose(2, 3) * self.grep_a + gate_b.transpose(2, 3)
            attn = attn + pos_bias

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.embed_dim)
        return self.out_proj(out)


class BEATsTransformerLayer(nn.Module):
    """Pre-norm transformer layer with BEATs-style attention."""
    def __init__(self, embed_dim, num_heads, ffn_dim, num_buckets=320,
                 max_distance=800, has_relative_attention_bias=False):
        super().__init__()
        self.self_attn = BEATsAttention(
            embed_dim, num_heads, num_buckets, max_distance,
            has_relative_attention_bias=has_relative_attention_bias,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, pos_bias=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, pos_bias=pos_bias)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x


class TransformerEncoder(nn.Module):
    """Nested transformer encoder matching BEATs state dict key structure.

    Keys nest as: encoder.pos_conv.*, encoder.layers.N.*, encoder.layer_norm.*
    """
    def __init__(self, encoder_embed_dim, num_heads, ffn_dim, num_layers,
                 num_buckets, max_distance):
        super().__init__()
        # Convolutional positional encoding (weight-normalized)
        pos_conv = nn.Conv1d(encoder_embed_dim, encoder_embed_dim,
                             kernel_size=128, padding=128 // 2, groups=16)
        nn.utils.weight_norm(pos_conv, name='weight', dim=2)
        self.pos_conv = pos_conv
        self.pos_conv_pad = SamePad(128)

        # Transformer layers (layer 0 owns the relative position bias)
        self.layers = nn.ModuleList([
            BEATsTransformerLayer(
                encoder_embed_dim, num_heads, ffn_dim,
                num_buckets, max_distance,
                has_relative_attention_bias=(i == 0),
            )
            for i in range(num_layers)
        ])
        # Final layer norm at encoder_embed_dim
        self.layer_norm = nn.LayerNorm(encoder_embed_dim)

    def forward(self, x):
        # Convolutional positional encoding
        x_conv = x.transpose(1, 2)  # (B, D, N)
        x_conv = self.pos_conv(x_conv)
        x_conv = self.pos_conv_pad(x_conv)
        x_conv = F.gelu(x_conv)
        x = x + x_conv.transpose(1, 2)

        # Compute relative position bias from layer 0 (shared across layers)
        pos_bias = self.layers[0].self_attn.compute_bias(x.shape[1], x.device)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, pos_bias=pos_bias)

        x = self.layer_norm(x)
        return x


class OpenBEATsEncoder(nn.Module):
    """Standalone OpenBEATs encoder for embedding extraction.

    Architecture: patch embedding → layer_norm(512) → projection → conv pos enc
    → 24-layer transformer → final layer_norm(1024).
    Loads weights from ESPnet-format checkpoints on HuggingFace.
    """
    def __init__(self, embed_dim=512, encoder_embed_dim=1024, num_layers=24,
                 num_heads=16, ffn_dim=4096, num_buckets=320, max_distance=800,
                 fbank_mean=15.41663, fbank_std=6.55582):
        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.fbank_mean = fbank_mean
        self.fbank_std = fbank_std

        # Patch embedding: spectrogram → patches (checkpoint has bias)
        self.patch_embedding = nn.Conv2d(1, embed_dim, kernel_size=16, stride=16,
                                         bias=True)

        # Layer norm at embed_dim (512), BEFORE projection
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Project patch dim → transformer dim
        self.post_extract_proj = nn.Linear(embed_dim, encoder_embed_dim)

        # Nested transformer encoder (keys: encoder.pos_conv.*, encoder.layers.*, encoder.layer_norm.*)
        self.encoder = TransformerEncoder(
            encoder_embed_dim, num_heads, ffn_dim, num_layers,
            num_buckets, max_distance,
        )

    def extract_features(self, fbank):
        """Extract embeddings from fbank features.

        Args:
            fbank: (B, T, 128) un-normalized log-mel fbank features.
        Returns:
            (B, num_patches, encoder_embed_dim) contextualized embeddings.
        """
        # Normalize (matching BEATs internal preprocessing)
        fbank = (fbank - self.fbank_mean) / (2 * self.fbank_std)

        # Patch embedding
        x = fbank.unsqueeze(1)  # (B, 1, T, 128)
        x = self.patch_embedding(x)  # (B, 512, T//16, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, N, 512)

        # Layer norm at embed_dim=512, then project to 1024
        x = self.layer_norm(x)
        x = self.post_extract_proj(x)  # (B, N, 1024)

        # Transformer encoder (pos_conv → layers → final layer_norm)
        x = self.encoder(x)
        return x


def _find_checkpoint(repo_id):
    """Find the .pth checkpoint file in a HuggingFace repo."""
    files = list_repo_files(repo_id)
    pth_files = [f for f in files if f.endswith('.pth')]
    if not pth_files:
        raise FileNotFoundError(f"No .pth checkpoint found in {repo_id}. Files: {files}")
    # Prefer files with 'best' or 'valid' in name, else take the largest
    for keyword in ['best', 'valid', 'epoch']:
        matches = [f for f in pth_files if keyword in f]
        if matches:
            return matches[0]
    return pth_files[0]


def load_openbeats(model_tag, device):
    """Download and load OpenBEATs encoder from HuggingFace."""
    print(f"Finding checkpoint in {model_tag}...")
    ckpt_path = _find_checkpoint(model_tag)
    print(f"Downloading {ckpt_path}...")
    local_path = hf_hub_download(repo_id=model_tag, filename=ckpt_path)

    print("Loading checkpoint...")
    checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)

    # ESPnet checkpoint format: full model state dict (encoder.* + decoder.*)
    # or BEATs format: {"model": state_dict, "cfg": config}
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip "encoder." prefix if present (ESPnet CLS model nesting)
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            stripped[k[len("encoder."):]] = v
        elif k.startswith("decoder.") or k.startswith("predictor."):
            continue  # Skip classifier head
        else:
            stripped[k] = v

    # Build model
    model = OpenBEATsEncoder()

    # Load weights
    result = model.load_state_dict(stripped, strict=False)
    if result.missing_keys:
        # Filter out expected missing keys (layers 1+ don't have their own rel pos bias)
        unexpected_missing = [k for k in result.missing_keys
                              if 'relative_attention_bias' not in k]
        if unexpected_missing:
            print(f"  Warning, missing keys: {unexpected_missing[:10]}...")
    if result.unexpected_keys:
        print(f"  Skipped {len(result.unexpected_keys)} unexpected keys "
              f"(first 5: {result.unexpected_keys[:5]})")

    model = model.to(device)
    model.eval()
    print(f"OpenBEATs encoder loaded on {device} (1024-dim, 300M params)")
    return model


# =============================================================================
# Spectrogram conversion
# =============================================================================

def linear_spectrogram_to_fbank(spec_db, n_mels=128, sample_rate=16000,
                                 current_hop_ms=20, target_hop_ms=10):
    """Convert a linear spectrogram (dB) to kaldi-style log-mel fbank features.

    BEATs computes 128-bin fbank via torchaudio kaldi.fbank() which produces
    natural-log mel energies. We replicate that: dB → power → mel filterbank → ln.

    Returns:
        numpy array of shape (T, 128), time-first, matching kaldi convention.
    """
    spec_tensor = torch.tensor(spec_db, dtype=torch.float32)
    spec_magnitude = torch.pow(10.0, spec_tensor / 20.0)
    spec_power = spec_magnitude ** 2

    n_freqs = spec_power.shape[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="At least one mel filterbank has all zero")
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs, f_min=0.0, f_max=sample_rate / 2.0,
            n_mels=n_mels, sample_rate=sample_rate,
        )
    mel_power = mel_fb.T @ spec_power  # (n_mels, T)
    fbank = torch.log(torch.clamp(mel_power, min=1e-10))  # (n_mels, T)

    # Upsample time dimension (20ms → 10ms hop)
    target_frames = int(fbank.shape[1] * (current_hop_ms / target_hop_ms))
    fbank_resized = torch.nn.functional.interpolate(
        fbank.unsqueeze(0).unsqueeze(0),
        size=(n_mels, target_frames),
        mode='bilinear', align_corners=True,
    ).squeeze(0).squeeze(0)

    return fbank_resized.T.numpy()  # (T, 128)


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Canonical OpenBEATs embeddings")
    parser.add_argument("--model_tag", type=str, default="espnet/OpenBEATS-Large-i3-as2m")
    parser.add_argument("--output", type=str, default="embeddings_openbeats.npz")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split_json", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    DATA_PREFIX = os.environ.get("B2AI_ROOT", "")  # PhysioNet b2ai-voice release dir
    MAX_LEN = 1024
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = load_openbeats(args.model_tag, device)

    parquet_path = os.path.join(DATA_PREFIX, "features", "torchaudio_spectrogram.parquet")
    pids, tasks, embs, durs = [], [], [], []
    fb_buf, meta_buf = [], []

    @torch.no_grad()
    def flush():
        if not fb_buf:
            return
        feats = encoder.extract_features(torch.stack(fb_buf).to(device))  # (b,N,1024)
        pooled = feats.mean(dim=1).cpu().numpy()
        for e, (pid, grp, dur) in zip(pooled, meta_buf):
            embs.append(e); pids.append(pid); tasks.append(grp); durs.append(dur)
        fb_buf.clear(); meta_buf.clear()

    n = 0
    for pid, group, spec, dur in _spec_bags(parquet_path, args.split_json):
        fbank = linear_spectrogram_to_fbank(spec, n_mels=128)             # (T, 128)
        T = fbank.shape[0]
        if T > MAX_LEN:
            fbank = fbank[:MAX_LEN]
        elif T < MAX_LEN:
            fbank = np.concatenate([fbank, np.zeros((MAX_LEN - T, 128), np.float32)], 0)
        fb_buf.append(torch.tensor(fbank, dtype=torch.float32))
        meta_buf.append((pid, group, dur))
        if len(fb_buf) >= args.batch_size:
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
