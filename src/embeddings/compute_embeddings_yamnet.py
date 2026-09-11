"""YAMNet embeddings on the canonical B2AI (pid, task_group) set.

YAMNet (TF-Hub https://tfhub.dev/google/yamnet/1), 1024-d per 0.96s window,
AudioSet 521 classes. Operates on waveform → uses the Griffin-Lim waveform
variant of the shared canonical selection (compute/b2ai_canonical.py), so YAMNet
appears in the FM comparison on the same 16 groups as every other model.

The Anibal-2025 breathing-only head-to-head is obtained at MIL time via
`run_mil_stenosis_stridor.py --fm yamnet --task_set anibal` (no separate npz).

Usage (run from compute/):
    python compute_embeddings_yamnet.py --output ../output/embeddings/embeddings_yamnet.npz
"""

import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
import tensorflow_hub as hub

import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_SRC))

from b2ai_canonical import iter_canonical_bags

EMB_DIM = 1024
DEFAULT_SPLIT = str(Path(__file__).resolve().parents[2] / "data" / "binary_stenosis_split.json")


def main():
    ap = argparse.ArgumentParser(description="Canonical YAMNet embeddings")
    ap.add_argument('--output', default='embeddings_yamnet.npz')
    ap.add_argument('--data_prefix',
                    default=os.environ.get("B2AI_ROOT", ""))
    ap.add_argument('--split_json', default=DEFAULT_SPLIT)
    ap.add_argument('--yamnet_url', default='https://tfhub.dev/google/yamnet/1')
    args = ap.parse_args()

    print(f'Loading YAMNet from {args.yamnet_url}...', flush=True)
    yamnet = hub.load(args.yamnet_url)
    print('YAMNet ready.', flush=True)

    parquet_path = os.path.join(args.data_prefix, 'features',
                                'torchaudio_spectrogram.parquet')

    def embed_wav(wav: np.ndarray) -> np.ndarray:
        mx = float(np.max(np.abs(wav))) + 1e-9
        x = tf.constant(wav / mx, dtype=tf.float32)        # YAMNet wants [-1,1]
        _, embeddings, _ = yamnet(x)
        emb = embeddings.numpy()
        return (emb.mean(axis=0).astype(np.float32) if len(emb)
                else np.zeros(EMB_DIM, dtype=np.float32))

    pids, tasks, embs, durs = [], [], [], []
    n = 0
    for pid, group, wav, dur in iter_canonical_bags(parquet_path, args.split_json):
        embs.append(embed_wav(wav))
        pids.append(pid); tasks.append(group); durs.append(dur)
        n += 1
        if n % 100 == 0:
            print(f"  [{n}] {pid}/{group}", flush=True)

    embeddings = np.stack(embs, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez(
        args.output,
        embeddings=embeddings,
        participant_ids=np.array(pids),
        tasks=np.array(tasks),
        original_duration_sec=np.array(durs, dtype=np.float32),
    )
    print(f"\nSaved {args.output} | shape={embeddings.shape} | "
          f"{len(set(pids))} patients × {len(set(tasks))} groups", flush=True)


if __name__ == '__main__':
    main()
