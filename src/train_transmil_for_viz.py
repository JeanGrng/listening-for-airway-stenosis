"""Train a single TransMIL on WavLM L15 canonical bags and save the checkpoint.

Used by the explainability pipeline (notebooks/transmil_gradcam.ipynb). No
cross-validation: one train/val split on the patient-level JSON, early stopping
on validation AUC, deterministic seed. Saves the model state, metadata, the
full 748-patient bag tensor (small) and the per-fold threshold picked on
validation by max-F1.

Run on HPC (GPU). Embeddings npz lives on HPC, model needs ~2 GB GPU.

    uv run python training/train_transmil_for_viz.py \
        --npz output/embeddings/embeddings_wavlm_layered.npz \
        --layer_idx 2 \
        --output_dir output/viz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mil_models import TransMIL
from metrics import fold_f1_acc, pick_threshold_f1

# Participants with truncated or unusable recordings, dropped everywhere.
EXCLUDE_PIDS = {'364676', '712389', '821259', '981647'}


def norm_pid(p): return str(p).zfill(6)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', default='output/embeddings/embeddings_wavlm_layered.npz')
    p.add_argument('--layer_idx', type=int, default=2,
                   help='Slice index into the layered npz (ignored for 2D npz)')
    p.add_argument('--tag', default='wavlm_L15',
                   help='Suffix for the output files (transmil_<tag>.pt, ...)')
    p.add_argument('--split_json', default='data/binary_stenosis_split.json')
    p.add_argument('--output_dir', default='output/viz')
    p.add_argument('--inner_val_frac', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_bags(npz_path: str, layer_idx: int, split_json: str):
    """Load layered npz, slice at `layer_idx`, aggregate to 748×16 stenosis bags."""
    with open(split_json) as f:
        bi_split = json.load(f)
    annot = {}
    for s in ('train', 'val', 'test'):
        for e in bi_split[s]:
            annot[norm_pid(e['participant_id'])] = int(e['airway_stenosis'])

    data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
    embs = data['embeddings']        # (N, K_layers, D) layered  OR  (N, D) flat
    pids_arr  = data['participant_ids'].astype(str)
    tasks_arr = data['tasks'].astype(str)
    if embs.ndim == 3:
        assert 0 <= layer_idx < embs.shape[1], f'bad layer_idx {layer_idx} for {embs.shape}'
        slice_fn = lambda i: np.asarray(embs[i, layer_idx], dtype=np.float32)
    elif embs.ndim == 2:
        slice_fn = lambda i: np.asarray(embs[i], dtype=np.float32)
    else:
        raise ValueError(f'unexpected embs shape {embs.shape}')

    acc = defaultdict(list)
    for i in range(len(pids_arr)):
        pid = pids_arr[i]
        if pid not in annot or pid in EXCLUDE_PIDS:
            continue
        acc[(pid, tasks_arr[i])].append(slice_fn(i))
    pid_task = {k: np.mean(v, axis=0) for k, v in acc.items()}
    all_tasks = sorted({t for (_, t) in pid_task})

    X, y, pids = [], [], []
    for pid in sorted(annot):
        if pid in EXCLUDE_PIDS:
            continue
        bag = [pid_task.get((pid, t)) for t in all_tasks]
        if any(b is None for b in bag):
            continue
        X.append(np.stack(bag, axis=0)); y.append(annot[pid]); pids.append(pid)
    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, pids, all_tasks


def train_transmil(X_tr, y_tr, X_va, y_va, args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    model = TransMIL(D=X_tr.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                          device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    Xtr = torch.tensor(X_tr, device=device)
    ytr = torch.tensor(y_tr, device=device, dtype=torch.float32)
    Xva = torch.tensor(X_va, device=device)

    best_auc, best_state, patience = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        for i in torch.randperm(len(Xtr), device=device).split(args.bs):
            logit = model(Xtr[i])
            loss = loss_fn(logit, ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            prob_va = torch.sigmoid(model(Xva)).cpu().numpy()
        if len(np.unique(y_va)) < 2:
            break
        auc = roc_auc_score(y_va, prob_va)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
        if (ep + 1) % 10 == 0:
            print(f'  ep {ep+1:3d} | val AUC {auc:.3f} | best {best_auc:.3f}', flush=True)
    model.load_state_dict(best_state)
    return model, best_auc


def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device={args.device} | layer_idx={args.layer_idx} | out={out_dir}', flush=True)

    X, y, pids, tasks = load_bags(args.npz, args.layer_idx, args.split_json)
    print(f'bags: {X.shape}  pos={int(y.sum())}/{len(y)}  tasks={len(tasks)}', flush=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.inner_val_frac,
                                  random_state=args.seed)
    itr, iva = next(sss.split(np.zeros(len(y)), y))
    print(f'inner split  train={len(itr)} (pos={int(y[itr].sum())})  '
          f'val={len(iva)} (pos={int(y[iva].sum())})', flush=True)

    model, val_auc = train_transmil(X[itr], y[itr], X[iva], y[iva], args)
    model.eval()
    device = torch.device(args.device)
    with torch.no_grad():
        prob_va = torch.sigmoid(model(torch.tensor(X[iva], device=device))).cpu().numpy()
    thr = pick_threshold_f1(y[iva], prob_va)
    val_f1, val_acc = fold_f1_acc(y[iva], prob_va, thr)
    print(f'val AUC={val_auc:.3f} | F1={val_f1:.3f} | Acc={val_acc:.3f} | thr={thr:.3f}',
          flush=True)

    ckpt_path  = out_dir / f'transmil_{args.tag}.pt'
    meta_path  = out_dir / f'transmil_{args.tag}_meta.json'
    bags_path  = out_dir / f'bags_{args.tag}.npz'
    torch.save(model.state_dict(), ckpt_path)
    json.dump({
        'npz': args.npz, 'layer_idx': args.layer_idx, 'tasks': tasks,
        'train_pids': [pids[i] for i in itr.tolist()],
        'val_pids':   [pids[i] for i in iva.tolist()],
        'val_auc': float(val_auc), 'val_f1': float(val_f1),
        'val_acc': float(val_acc), 'threshold': float(thr),
        'arch': {'D': X.shape[-1], 'hidden': 512, 'num_heads': 8, 'dropout': 0.1},
        'seed': args.seed,
    }, open(meta_path, 'w'), indent=2)
    np.savez(bags_path, X=X, y=y, pids=np.array(pids), tasks=np.array(tasks))
    print(f'\nSaved {ckpt_path}\nSaved {meta_path}\nSaved {bags_path}', flush=True)


if __name__ == '__main__':
    main()
