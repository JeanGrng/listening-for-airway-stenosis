"""TransMIL task-level attention + permutation importance for explainability.

Replaces the failed Grad-CAM/IG attempts (see GRADCAM_METHOD.md §8) with a
combination of two complementary attribution methods, both forward-only and
robust :

1. **Attention CLS→instance** from the second (last) TransMIL transformer
   block. Shape `(N, 16)`. Read directly from `nn.MultiheadAttention` with
   `need_weights=True`. Tells us which task-group the CLS token attends to.

2. **Permutation importance** per task-group : for each instance index k,
   replace `bag[k]` with the mean class-0 embedding for task k computed on the
   training fold, forward through TransMIL, measure `Δlogit = logit_full -
   logit_perm`. Shape `(N, 16)`. Forward-only, no gradient.

Inputs (local, already present) :
  - `output/viz/transmil_wavlm_L15.pt`        : trained MIL ckpt (16-task bag)
  - `output/viz/transmil_wavlm_L15_meta.json` : tasks + train/val split
  - `output/viz/bags_wavlm_L15.npz`           : `X` (748, 16, 1024) + `y` + `pids`

Output : `output/viz/transmil_attention.npz` with keys
  - attn_layer2     : (N, 16), CLS→instance attention (softmax-normalised)
  - perm_importance : (N, 16), Δlogit when masking each task with class-0 mean
  - logit_full      : (N,)
  - y               : (N,)
  - pids            : (N,)
  - tasks           : (16,)
  - train_pids / val_pids : split tags

Local CPU run, ~30 s for 748 patients.

    uv run python training/compute_transmil_attention.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mil_models import TransMIL


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',  default='output/viz/transmil_wavlm_L15.pt')
    p.add_argument('--meta',  default='output/viz/transmil_wavlm_L15_meta.json')
    p.add_argument('--bags',  default='output/viz/bags_wavlm_L15.npz')
    p.add_argument('--output', default='output/viz/transmil_attention.npz')
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    meta = json.load(open(args.meta))
    tasks = list(meta['tasks'])
    train_pids = set(meta.get('train_pids', []))
    val_pids   = set(meta.get('val_pids', []))
    print(f'val_auc={meta["val_auc"]:.3f}  tasks={len(tasks)}  '
          f'train_pids={len(train_pids)}  val_pids={len(val_pids)}', flush=True)

    bags = np.load(args.bags, allow_pickle=True)
    X = bags['X'].astype(np.float32)                       # (N, K=16, D=1024)
    y = bags['y'].astype(np.int64)
    pids = bags['pids'].astype(str)
    N, K, D = X.shape
    assert K == len(tasks)
    print(f'bags X={X.shape}  pos={int(y.sum())}/{N}', flush=True)

    # Load TransMIL
    transmil = TransMIL(D=D, n_classes=1).to(device)
    transmil.load_state_dict(torch.load(args.ckpt, map_location=device))
    transmil.eval()

    # ---- 1) full-forward logits + attention ----
    print('\nStep 1, forward pass with attention extraction', flush=True)
    with torch.no_grad():
        Xt = torch.tensor(X, device=device)
        logit_full, attn1, attn2 = transmil(Xt, return_attn=True)
        logit_full = logit_full.cpu().numpy()
        # attn2 : (N, K+1, K+1), row 0 = CLS, cols 1..K = instances
        attn_cls_inst = attn2[:, 0, 1:].cpu().numpy()      # (N, K)
        # Renormalise (the CLS row also attends to itself : drop col 0 then renorm)
        attn_cls_inst = attn_cls_inst / (attn_cls_inst.sum(axis=1, keepdims=True) + 1e-12)
    print(f'attn_layer2 (CLS->inst) shape={attn_cls_inst.shape}  '
          f'mean per task={attn_cls_inst.mean(axis=0).round(3).tolist()}', flush=True)

    # ---- 2) per-task class-0 means (computed on TRAIN PIDs only, no leakage) ----
    print('\nStep 2, class-0 task means on the training split', flush=True)
    train_mask = np.array([p in train_pids for p in pids])
    class0_train_mask = train_mask & (y == 0)
    if class0_train_mask.sum() == 0:
        print('WARNING : no class-0 patient in train split, falling back to all class-0',
              flush=True)
        class0_train_mask = (y == 0)
    class0_mean_per_task = X[class0_train_mask].mean(axis=0)   # (K, D)
    print(f'class-0 train patients used : {int(class0_train_mask.sum())}', flush=True)

    # ---- 3) permutation importance per task per patient ----
    print('\nStep 3, permutation importance per task-group', flush=True)
    perm_importance = np.zeros((N, K), dtype=np.float32)
    with torch.no_grad():
        for k in range(K):
            X_perm = X.copy()
            X_perm[:, k, :] = class0_mean_per_task[k][None, :]
            logit_perm = transmil(torch.tensor(X_perm, device=device)).cpu().numpy()
            perm_importance[:, k] = logit_full - logit_perm
            print(f'  task {k:2d} ({tasks[k]:32s}) : '
                  f'mean Δlogit_+={perm_importance[y == 1, k].mean():+.3f}  '
                  f'mean Δlogit_−={perm_importance[y == 0, k].mean():+.3f}', flush=True)

    # ---- 4) save ----
    np.savez(
        args.output,
        attn_layer2=attn_cls_inst.astype(np.float32),
        perm_importance=perm_importance.astype(np.float32),
        logit_full=logit_full.astype(np.float32),
        y=y,
        pids=np.array(pids),
        tasks=np.array(tasks),
        train_pids=np.array(sorted(train_pids)),
        val_pids=np.array(sorted(val_pids)),
    )
    print(f'\nSaved {args.output}', flush=True)


if __name__ == '__main__':
    main()
