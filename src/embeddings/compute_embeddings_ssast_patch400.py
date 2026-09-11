"""Compute SSAST Patch-400 embeddings for B2AI Voice recordings and save to disk.

Usage:
    python compute/compute_embeddings_ssast_patch400.py --output output/embeddings_ssast_patch400.npz
    python compute/compute_embeddings_ssast_patch400.py --device cuda
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import ASTConfig, ASTModel

import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_SRC))

from b2ai_canonical import iter_canonical_bags_spec as _spec_bags

DATA_PREFIX = os.environ.get("B2AI_ROOT", "")  # PhysioNet b2ai-voice release dir
DEFAULT_SPLIT = str(Path(__file__).resolve().parents[2] / "data" / "binary_stenosis_split.json")


def convert_ssast_state_dict_to_astmodel(pretrained_dict, layers: int = 12):
    """Convert SSAST state dict keys to HuggingFace ASTModel format."""
    conversion_dict = {
        "module.v.cls_token": "embeddings.cls_token",
        "module.v.dist_token": "embeddings.distillation_token",
        "module.v.pos_embed": "embeddings.position_embeddings",
        "module.v.patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
        "module.v.patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
        "module.v.norm.weight": "layernorm.weight",
        "module.v.norm.bias": "layernorm.bias",
    }

    for i in range(layers):
        conversion_dict[f"module.v.blocks.{i}.norm1.weight"] = f"encoder.layer.{i}.layernorm_before.weight"
        conversion_dict[f"module.v.blocks.{i}.norm1.bias"] = f"encoder.layer.{i}.layernorm_before.bias"
        conversion_dict[f"module.v.blocks.{i}.attn.qkv.weight"] = [
            f"encoder.layer.{i}.attention.attention.query.weight",
            f"encoder.layer.{i}.attention.attention.key.weight",
            f"encoder.layer.{i}.attention.attention.value.weight",
        ]
        conversion_dict[f"module.v.blocks.{i}.attn.qkv.bias"] = [
            f"encoder.layer.{i}.attention.attention.query.bias",
            f"encoder.layer.{i}.attention.attention.key.bias",
            f"encoder.layer.{i}.attention.attention.value.bias",
        ]
        conversion_dict[f"module.v.blocks.{i}.attn.proj.weight"] = f"encoder.layer.{i}.attention.output.dense.weight"
        conversion_dict[f"module.v.blocks.{i}.attn.proj.bias"] = f"encoder.layer.{i}.attention.output.dense.bias"
        conversion_dict[f"module.v.blocks.{i}.norm2.weight"] = f"encoder.layer.{i}.layernorm_after.weight"
        conversion_dict[f"module.v.blocks.{i}.norm2.bias"] = f"encoder.layer.{i}.layernorm_after.bias"
        conversion_dict[f"module.v.blocks.{i}.mlp.fc1.weight"] = f"encoder.layer.{i}.intermediate.dense.weight"
        conversion_dict[f"module.v.blocks.{i}.mlp.fc1.bias"] = f"encoder.layer.{i}.intermediate.dense.bias"
        conversion_dict[f"module.v.blocks.{i}.mlp.fc2.weight"] = f"encoder.layer.{i}.output.dense.weight"
        conversion_dict[f"module.v.blocks.{i}.mlp.fc2.bias"] = f"encoder.layer.{i}.output.dense.bias"

    converted_dict = {}
    for key, value in pretrained_dict.items():
        mapped_key = conversion_dict.get(key)
        if mapped_key is None:
            continue
        if isinstance(mapped_key, list):
            split_size = value.shape[0] // 3
            converted_dict[mapped_key[0]] = value[:split_size]
            converted_dict[mapped_key[1]] = value[split_size : 2 * split_size]
            converted_dict[mapped_key[2]] = value[2 * split_size :]
        else:
            converted_dict[mapped_key] = value

    return converted_dict


def _require_shape(state_dict, key, expected_shape):
    if key not in state_dict:
        raise KeyError(f"Checkpoint missing required key: {key}")
    found = tuple(state_dict[key].shape)
    if found != expected_shape:
        raise ValueError(f"Shape mismatch for {key}: expected {expected_shape}, found {found}")


def load_ssast_patch400_model(weights_path, device):
    """Load SSAST Patch-400 weights into HuggingFace ASTModel."""
    try:
        ssast_state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        ssast_state_dict = torch.load(weights_path, map_location="cpu")

    _require_shape(ssast_state_dict, "module.v.patch_embed.proj.weight", (768, 1, 16, 16))
    _require_shape(ssast_state_dict, "module.v.patch_embed.proj.bias", (768,))
    _require_shape(ssast_state_dict, "module.v.pos_embed", (1, 514, 768))

    config = ASTConfig(
        architectures=["ASTModel"],
        frequency_stride=16,
        time_stride=16,
        hidden_size=768,
        max_length=1024,
        num_attention_heads=12,
        num_hidden_layers=12,
        num_mel_bins=128,
        qkv_bias=True,
    )
    model = ASTModel(config=config)

    expected_proj = tuple(model.embeddings.patch_embeddings.projection.weight.shape)
    if expected_proj != (768, 1, 16, 16):
        raise ValueError(
            f"AST patch projection shape mismatch. Expected (768, 1, 16, 16), found {expected_proj}"
        )

    expected_pos = tuple(model.embeddings.position_embeddings.shape)
    if expected_pos != (1, 514, 768):
        raise ValueError(f"AST position embedding shape mismatch. Expected (1, 514, 768), found {expected_pos}")

    converted = convert_ssast_state_dict_to_astmodel(ssast_state_dict)
    result = model.load_state_dict(converted, strict=False)
    print(f"Loaded SSAST Patch-400 weights from {weights_path}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {result.unexpected_keys}")
    if result.missing_keys:
        print(f"  Missing keys (newly initialized): {result.missing_keys}")

    model = model.to(device)
    model.eval()
    return model


def linear_spectrogram_to_mel(spec_db, n_mels=128, sample_rate=16000, target_hop_ms=10, current_hop_ms=20):
    """Convert a linear spectrogram (in dB) to a mel spectrogram with time upsampling."""
    spec_tensor = torch.tensor(spec_db, dtype=torch.float32)
    spec_magnitude = torch.pow(10.0, spec_tensor / 20.0)
    spec_power = spec_magnitude**2

    n_freqs = spec_power.shape[0]
    n_mels_native = min(n_mels, 80)

    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_freqs, f_min=0.0, f_max=sample_rate / 2.0, n_mels=n_mels_native, sample_rate=sample_rate
    )
    mel_power = mel_fb.T @ spec_power
    mel_db = 10.0 * torch.log10(torch.clamp(mel_power, min=1e-10))
    mel_db = mel_db.clamp(min=mel_db.max() - 95.0)

    target_frames = int(mel_db.shape[1] * (current_hop_ms / target_hop_ms))
    mel_db_resized = torch.nn.functional.interpolate(
        mel_db.unsqueeze(0).unsqueeze(0), size=(n_mels, target_frames), mode="bilinear", align_corners=True
    ).squeeze(0).squeeze(0)
    return mel_db_resized.numpy()


def main():
    parser = argparse.ArgumentParser(description="Canonical SSAST Patch-400 embeddings")
    parser.add_argument("--weights", type=str, default="weights/SSAST-Base-Patch-400.pth")
    parser.add_argument("--output", type=str, default="embeddings_ssast_patch400.npz")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split_json", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    MAX_LEN = 1024
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_ssast_patch400_model(args.weights, device)
    print(f"SSAST Patch-400 on {device}", flush=True)

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
        if T < MAX_LEN:                       # SSAST expects exactly MAX_LEN frames
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
