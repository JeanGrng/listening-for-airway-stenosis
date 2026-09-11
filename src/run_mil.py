"""MIL stenosis+stridor, 5-fold CV on HPC GPU.

Standalone version of notebooks/mil_stenosis_stridor.ipynb.
Outputs CSV + JSON to --output_dir.
"""
import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mil_models import DSMIL, SetTransformerMIL, TransMIL
from metrics import (fold_f1_acc, pick_threshold_f1, pooled_f1_acc,
                              stats_dict)
from shared_tasks import (compute_shared_groups, filter_tasks_to_shared,
                                   save_shared_tasks_json, to_group)

warnings.filterwarnings('ignore')

# Breathing tasks ≈ Anibal et al. 2025 (FIMO + deep breath). Canonical group
# names (numbered siblings collapsed), used for the head-to-head subset.
ANIBAL_GROUPS = {'respiration-and-cough-breath', 'respiration-and-cough-fivebreaths'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fm', default=None,
                   help='FM shorthand → embeddings_{fm}.npz, used only when --npz is NOT given')
    p.add_argument('--npz', default=None,
                   help='explicit embeddings npz; takes precedence over --fm')
    p.add_argument('--split_json', default='data/binary_stenosis_split.json')
    p.add_argument('--b2ai_json', default='data/bridge2voice_data_split.json')
    p.add_argument('--output_dir', default='output/mil')
    p.add_argument('--task_set', default='all', choices=['all', 'shared', 'anibal', 'single'])
    p.add_argument('--single_task', default=None, help='raw task name when --task_set single')
    p.add_argument('--shared_tasks_json', default='data/shared_tasks_stenosis.json')
    p.add_argument('--thresh', type=float, default=97.0,
                   help='min %% cohort coverage for a shared task group')
    p.add_argument('--layer', type=int, default=6)
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--inner_val_frac', type=float, default=0.2)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--rs', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--skip_deep', action='store_true', help='skip DSMIL/SetTransformer/TransMIL')
    p.add_argument('--holdout', action='store_true',
                   help='exclude split_json["test"] from CV, refit each method on '
                        'train+val pool, evaluate once on held-out test set')
    return p.parse_args()


def norm_pid(p): return str(p).zfill(6)


# Participants with truncated or unusable recordings, dropped everywhere.
EXCLUDE_PIDS = {'364676', '712389', '821259', '981647'}
N_AUG = 5
STEN_REF_TASK = 'picture-description'
STRID_REF_TASK = 'diadochokinesis-buttercup'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(args):
    with open(args.split_json) as f: bi_split = json.load(f)
    with open(args.b2ai_json) as f: b2ai_split = json.load(f)

    annot = {}
    holdout_pids = set()
    for s in ['train', 'val', 'test']:
        for e in bi_split[s]:
            pid = norm_pid(e['participant_id'])
            annot[pid] = {'stenosis': int(e['airway_stenosis']),
                          'stridor':  0 if e['airway_stenosis'] == 0 else -1}
            if s == 'test':
                holdout_pids.add(pid)
    for s in ['train', 'val', 'test']:
        for e in b2ai_split[s]:
            pid = norm_pid(e['participant_id'])
            if pid in annot:
                raw = e.get('diagnosis_as_as')
                if raw == 'Yes': annot[pid]['stridor'] = 1
                elif raw == 'No': annot[pid]['stridor'] = 0

    # --npz wins when given explicitly; --fm is a shorthand fallback. (Historically
    # --fm silently overrode --npz, so layered npz + --layer were ignored, see git log.)
    if args.npz:
        npz_path = args.npz
        if args.fm:
            print(f'[npz] both --fm and --npz given → using --npz={args.npz} '
                  f'(ignoring --fm={args.fm} for embeddings)', flush=True)
    elif args.fm:
        npz_path = f'output/embeddings/embeddings_{args.fm}.npz'
    else:
        npz_path = 'output/embeddings/embeddings_wavlm_layered.npz'
    data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
    pids_arr = data['participant_ids'].astype(str)
    tasks_arr = data['tasks'].astype(str)
    embs = data['embeddings']
    has_layers = embs.ndim == 3                       # (N, n_layers, D) vs (N, D)
    has_aug = 'aug_id' in data.files
    aug_ids = data['aug_id'] if has_aug else np.zeros(len(pids_arr), dtype=int)
    print(f'npz={npz_path} shape={embs.shape} dtype={embs.dtype} '
          f'layered={has_layers} aug={has_aug}', flush=True)

    acc_orig, acc_aug = defaultdict(list), defaultdict(list)
    for i in range(len(pids_arr)):
        pid = pids_arr[i]
        if pid not in annot or pid in EXCLUDE_PIDS: continue
        vec = (np.asarray(embs[i, args.layer], dtype=np.float32) if has_layers
               else np.asarray(embs[i], dtype=np.float32))
        aid = int(aug_ids[i])
        if aid == 0: acc_orig[(pid, tasks_arr[i])].append(vec)
        else: acc_aug[(pid, tasks_arr[i], aid)].append(vec)

    pid_task_emb = {k: np.mean(v, axis=0) for k, v in acc_orig.items()}
    pid_task_emb_aug = {k: np.mean(v, axis=0) for k, v in acc_aug.items()}
    all_tasks = sorted({t for (_, t) in pid_task_emb})
    print(f'{len(pid_task_emb)} (pid,task) pairs | {len(all_tasks)} tasks | '
          f'{len(pid_task_emb_aug)} aug | holdout pids={len(holdout_pids)}', flush=True)
    return annot, pid_task_emb, pid_task_emb_aug, all_tasks, has_aug, holdout_pids


def build_bags(annot, pid_task_emb, pid_task_emb_aug, all_tasks, label_key,
               stenosis_only=False, has_aug=True):
    X_orig, X_aug_all, y, pids = [], [], [], []
    for pid, d in sorted(annot.items()):
        if pid in EXCLUDE_PIDS: continue
        if stenosis_only and d['stenosis'] != 1: continue
        lab = d[label_key]
        if lab == -1: continue
        bag, ok = [], True
        for t in all_tasks:
            e = pid_task_emb.get((pid, t))
            if e is None: ok = False; break
            bag.append(e)
        if not ok: continue
        X_orig.append(np.stack(bag, axis=0))
        if has_aug:
            aug_bags = []
            for aid in range(1, N_AUG + 1):
                ab = [pid_task_emb_aug.get((pid, t, aid), pid_task_emb[(pid, t)])
                      for t in all_tasks]
                aug_bags.append(np.stack(ab))
            X_aug_all.append(np.stack(aug_bags))
        y.append(lab); pids.append(pid)
    X_orig = np.stack(X_orig).astype(np.float32)
    if has_aug:
        X_aug = np.stack(X_aug_all).astype(np.float32)        # (N, N_AUG, K, D)
    else:
        X_aug = np.empty((len(X_orig), 0, *X_orig.shape[1:]), dtype=np.float32)
    return X_orig, X_aug, np.array(y, dtype=int), pids


def split_holdout(X, X_aug, y, pids, holdout_pids):
    """Partition (X, X_aug, y, pids) into (cv_*, ho_*) by patient ID set."""
    ho_mask = np.array([p in holdout_pids for p in pids])
    cv_mask = ~ho_mask
    X_cv, X_ho = X[cv_mask], X[ho_mask]
    Xa_cv, Xa_ho = X_aug[cv_mask], X_aug[ho_mask]
    y_cv, y_ho = y[cv_mask], y[ho_mask]
    pids_cv = [p for p, k in zip(pids, cv_mask) if k]
    pids_ho = [p for p, k in zip(pids, ho_mask) if k]
    return (X_cv, Xa_cv, y_cv, pids_cv), (X_ho, Xa_ho, y_ho, pids_ho)


# ---------------------------------------------------------------------------
# CV utilities
# ---------------------------------------------------------------------------
def fold_stats(aucs):
    a = np.array(aucs, dtype=float)
    return dict(mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), max=float(a.max()))


def pooled_auc(y_list, p_list):
    return float(roc_auc_score(np.concatenate(y_list), np.concatenate(p_list)))


def cv_splits(y, n_folds, rs):
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs).split(np.zeros(len(y)), y)


def inner_split(y, frac, rs):
    return next(StratifiedShuffleSplit(n_splits=1, test_size=frac,
                                        random_state=rs).split(np.zeros(len(y)), y))


def augmented_train(X_tr_orig, X_tr_aug, y_tr):
    n_aug = X_tr_aug.shape[1]
    X = np.concatenate([X_tr_orig] + [X_tr_aug[:, i] for i in range(n_aug)], axis=0)
    y = np.tile(y_tr, n_aug + 1)
    return X, y


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------
def make_clf(name, rs):
    if name == 'LogReg':
        return Pipeline([('sc', StandardScaler()),
                         ('lr', LogisticRegression(C=0.1, max_iter=1000,
                                                    class_weight='balanced', random_state=rs))])
    return Pipeline([('sc', StandardScaler()),
                     ('rf', RandomForestClassifier(n_estimators=300, max_depth=4,
                                                    class_weight='balanced',
                                                    random_state=rs, n_jobs=-1))])


def pool(X, mode): return X.mean(axis=1) if mode == 'mean' else X.max(axis=1)


def _tune_threshold_sklearn(make_inner_clf, X_tr_feat, X_tr_aug_feat, y_tr,
                             y_tr_aug, itr, iva, X_full_inner_aug, y_full_inner_aug):
    """Generic threshold tuning via inner 80/20 split for sklearn pipelines.

    `X_tr_feat` is the original (un-augmented) feature matrix on the train fold
    (used to index `iva`); `X_full_inner_aug` / `y_full_inner_aug` is the
    augmented inner-train pair to fit on. Returns the picked threshold.
    """
    clf_inner = make_inner_clf()
    clf_inner.fit(X_full_inner_aug, y_full_inner_aug)
    s_iva = clf_inner.predict_proba(X_tr_feat[iva])[:, 1]
    return pick_threshold_f1(y_tr[iva], s_iva)


def run_single_task(X, X_aug, y, all_tasks, task_name, args):
    k = all_tasks.index(task_name)
    aucs, yt, pt, thrs = [], [], [], []
    for tr, te in cv_splits(y, args.n_folds, args.rs):
        # Inner 80/20 on the train fold for threshold tuning.
        itr, iva = inner_split(y[tr], args.inner_val_frac, args.rs)
        X_itr_aug, y_itr_aug = augmented_train(X[tr][itr], X_aug[tr][itr], y[tr][itr])
        thr = _tune_threshold_sklearn(
            lambda: make_clf('LogReg', args.rs),
            X[tr][:, k, :], None, y[tr], None, itr, iva,
            X_itr_aug[:, k, :], y_itr_aug,
        )
        # Refit on full train fold (augmented).
        X_tr, y_tr = augmented_train(X[tr], X_aug[tr], y[tr])
        clf = make_clf('LogReg', args.rs)
        clf.fit(X_tr[:, k, :], y_tr)
        p = clf.predict_proba(X[te][:, k, :])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        yt.append(y[te]); pt.append(p); thrs.append(thr)
    return aucs, yt, pt, thrs


def run_pool(X, X_aug, y, mode, clf_name, args):
    aucs, yt, pt, thrs = [], [], [], []
    for tr, te in cv_splits(y, args.n_folds, args.rs):
        # Inner 80/20 on the train fold for threshold tuning.
        itr, iva = inner_split(y[tr], args.inner_val_frac, args.rs)
        X_itr_aug, y_itr_aug = augmented_train(X[tr][itr], X_aug[tr][itr], y[tr][itr])
        thr = _tune_threshold_sklearn(
            lambda: make_clf(clf_name, args.rs),
            pool(X[tr], mode), None, y[tr], None, itr, iva,
            pool(X_itr_aug, mode), y_itr_aug,
        )
        # Refit on full train fold.
        X_tr, y_tr = augmented_train(X[tr], X_aug[tr], y[tr])
        clf = make_clf(clf_name, args.rs)
        clf.fit(pool(X_tr, mode), y_tr)
        p = clf.predict_proba(pool(X[te], mode))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        yt.append(y[te]); pt.append(p); thrs.append(thr)
    return aucs, yt, pt, thrs


# ---------------------------------------------------------------------------
# mi-SVM
# ---------------------------------------------------------------------------
def mi_svm_train(X_bags, y_bags, rs, C=1.0, max_iter=15):
    N, K, D = X_bags.shape
    X_flat = X_bags.reshape(N * K, D)
    bag_idx = np.repeat(np.arange(N), K)
    y_inst = np.repeat(y_bags, K).astype(int)
    prev, pipe = None, None
    for _ in range(max_iter):
        if len(np.unique(y_inst)) < 2: break
        pipe = Pipeline([('sc', StandardScaler()),
                         ('svm', LinearSVC(C=C, class_weight='balanced',
                                            max_iter=2000, random_state=rs, dual='auto'))])
        pipe.fit(X_flat, y_inst)
        score = pipe.decision_function(X_flat)
        new_inst = y_inst.copy()
        for b in range(N):
            mask = (bag_idx == b)
            if y_bags[b] == 1:
                s = score[mask]; pred = (s > 0).astype(int)
                if pred.sum() == 0: pred[np.argmax(s)] = 1
                new_inst[mask] = pred
            else:
                new_inst[mask] = 0
        if prev is not None and np.array_equal(new_inst, prev): break
        prev, y_inst = y_inst, new_inst
    return pipe


def mi_svm_score(pipe, X_bags):
    N, K, D = X_bags.shape
    return pipe.decision_function(X_bags.reshape(N * K, D)).reshape(N, K).max(axis=1)


def run_misvm(X, X_aug, y, args, C=1.0):
    aucs, yt, pt, thrs = [], [], [], []
    for tr, te in cv_splits(y, args.n_folds, args.rs):
        # Inner 80/20 for threshold tuning.
        itr, iva = inner_split(y[tr], args.inner_val_frac, args.rs)
        X_itr_aug, y_itr_aug = augmented_train(X[tr][itr], X_aug[tr][itr], y[tr][itr])
        pipe_inner = mi_svm_train(X_itr_aug, y_itr_aug, args.rs, C=C)
        s_iva = mi_svm_score(pipe_inner, X[tr][iva])
        thr = pick_threshold_f1(y[tr][iva], s_iva)
        # Refit on full train fold.
        X_tr, y_tr = augmented_train(X[tr], X_aug[tr], y[tr])
        pipe = mi_svm_train(X_tr, y_tr, args.rs, C=C)
        s = mi_svm_score(pipe, X[te])
        aucs.append(roc_auc_score(y[te], s))
        yt.append(y[te]); pt.append(s); thrs.append(thr)
    return aucs, yt, pt, thrs


# ---------------------------------------------------------------------------
# Deep MIL
# ---------------------------------------------------------------------------
class GatedAttentionMIL(nn.Module):
    def __init__(self, D=1024, L=128, dropout=0.3):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(D, L), nn.ReLU(), nn.Dropout(dropout))
        self.V = nn.Linear(L, L); self.U = nn.Linear(L, L); self.w = nn.Linear(L, 1)
        self.head = nn.Linear(L, 1)
    def forward(self, bag):
        h = self.feat(bag)
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        a = torch.softmax(self.w(a).squeeze(-1), dim=1)
        z = (a.unsqueeze(-1) * h).sum(dim=1)
        return self.head(z).squeeze(-1), a


def _model_logits(model, X):
    out = model(X)
    if isinstance(out, tuple): return out[0], out[1] if len(out) > 1 else None
    return out, None


def train_deep(model_fn, X_tr, y_tr, X_va, y_va, seed, args):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_fn().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                          device=args.device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xtr = torch.tensor(X_tr, device=args.device)
    ytr = torch.tensor(y_tr, device=args.device, dtype=torch.float32)
    Xva = torch.tensor(X_va, device=args.device)
    best_auc, best_state, patience = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        for i in torch.randperm(len(Xtr), device=args.device).split(args.bs):
            logit, _ = _model_logits(model, Xtr[i])
            loss = loss_fn(logit, ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logit_va, _ = _model_logits(model, Xva)
            prob_va = torch.sigmoid(logit_va).cpu().numpy()
        if len(np.unique(y_va)) < 2: break
        auc = roc_auc_score(y_va, prob_va)
        if auc > best_auc:
            best_auc, best_state, patience = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= args.patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_deep(model, X, device):
    model.eval()
    Xt = torch.tensor(X, device=device)
    logit, attn = _model_logits(model, Xt)
    p = torch.sigmoid(logit).cpu().numpy()
    a = attn.cpu().numpy() if attn is not None else None
    return p, a


def run_deep(X, X_aug, y, model_fn, args):
    aucs, yt, pt, thrs, atts = [], [], [], [], []
    for tr, te in cv_splits(y, args.n_folds, args.rs):
        itr, iva = inner_split(y[tr], args.inner_val_frac, args.rs)
        X_itr, y_itr = augmented_train(X[tr][itr], X_aug[tr][itr], y[tr][itr])
        X_iva, y_iva = X[tr][iva], y[tr][iva]
        p_seeds, p_iva_seeds, a_seeds = [], [], []
        for s in args.seeds:
            model = train_deep(model_fn, X_itr, y_itr, X_iva, y_iva, seed=s, args=args)
            p, a = predict_deep(model, X[te], args.device)
            p_iva, _ = predict_deep(model, X_iva, args.device)
            p_seeds.append(p); p_iva_seeds.append(p_iva)
            if a is not None: a_seeds.append(a)
        p_mean     = np.mean(p_seeds,     axis=0)
        p_iva_mean = np.mean(p_iva_seeds, axis=0)
        thr = pick_threshold_f1(y_iva, p_iva_mean)
        aucs.append(roc_auc_score(y[te], p_mean))
        yt.append(y[te]); pt.append(p_mean); thrs.append(thr)
        atts.append(np.mean(a_seeds, axis=0) if a_seeds else None)
    return aucs, yt, pt, thrs, atts


# ---------------------------------------------------------------------------
# Hold-out refit helpers (used when --holdout is set)
# Each helper refits a method on the full CV pool (train+val patients) and
# evaluates ONCE on the held-out test patients. Threshold for F1/Acc is
# selected via an inner 80/20 split on the CV pool only ; the held-out set
# is never seen during refit or threshold tuning.
# ---------------------------------------------------------------------------
def holdout_single_task(X, X_aug, y, all_tasks, task_name, X_ho, y_ho, args):
    k = all_tasks.index(task_name)
    itr, iva = inner_split(y, args.inner_val_frac, args.rs)
    X_itr_aug, y_itr_aug = augmented_train(X[itr], X_aug[itr], y[itr])
    thr = _tune_threshold_sklearn(
        lambda: make_clf('LogReg', args.rs),
        X[:, k, :], None, y, None, itr, iva,
        X_itr_aug[:, k, :], y_itr_aug,
    )
    X_full, y_full = augmented_train(X, X_aug, y)
    clf = make_clf('LogReg', args.rs)
    clf.fit(X_full[:, k, :], y_full)
    p_ho = clf.predict_proba(X_ho[:, k, :])[:, 1]
    auc_ho = roc_auc_score(y_ho, p_ho)
    f1_ho, acc_ho = fold_f1_acc(y_ho, p_ho, thr)
    return auc_ho, p_ho, thr, f1_ho, acc_ho


def holdout_pool(X, X_aug, y, mode, clf_name, X_ho, y_ho, args):
    itr, iva = inner_split(y, args.inner_val_frac, args.rs)
    X_itr_aug, y_itr_aug = augmented_train(X[itr], X_aug[itr], y[itr])
    thr = _tune_threshold_sklearn(
        lambda: make_clf(clf_name, args.rs),
        pool(X, mode), None, y, None, itr, iva,
        pool(X_itr_aug, mode), y_itr_aug,
    )
    X_full, y_full = augmented_train(X, X_aug, y)
    clf = make_clf(clf_name, args.rs)
    clf.fit(pool(X_full, mode), y_full)
    p_ho = clf.predict_proba(pool(X_ho, mode))[:, 1]
    auc_ho = roc_auc_score(y_ho, p_ho)
    f1_ho, acc_ho = fold_f1_acc(y_ho, p_ho, thr)
    return auc_ho, p_ho, thr, f1_ho, acc_ho


def holdout_misvm(X, X_aug, y, X_ho, y_ho, args, C=1.0):
    itr, iva = inner_split(y, args.inner_val_frac, args.rs)
    X_itr_aug, y_itr_aug = augmented_train(X[itr], X_aug[itr], y[itr])
    pipe_inner = mi_svm_train(X_itr_aug, y_itr_aug, args.rs, C=C)
    s_iva = mi_svm_score(pipe_inner, X[iva])
    thr = pick_threshold_f1(y[iva], s_iva)
    X_full, y_full = augmented_train(X, X_aug, y)
    pipe = mi_svm_train(X_full, y_full, args.rs, C=C)
    s_ho = mi_svm_score(pipe, X_ho)
    auc_ho = roc_auc_score(y_ho, s_ho)
    f1_ho, acc_ho = fold_f1_acc(y_ho, s_ho, thr)
    return auc_ho, s_ho, thr, f1_ho, acc_ho


def holdout_deep(X, X_aug, y, model_fn, X_ho, y_ho, args):
    itr, iva = inner_split(y, args.inner_val_frac, args.rs)
    X_itr, y_itr = augmented_train(X[itr], X_aug[itr], y[itr])
    X_iva, y_iva = X[iva], y[iva]
    p_ho_seeds, p_iva_seeds = [], []
    for s in args.seeds:
        model = train_deep(model_fn, X_itr, y_itr, X_iva, y_iva, seed=s, args=args)
        p_ho, _ = predict_deep(model, X_ho, args.device)
        p_iva, _ = predict_deep(model, X_iva, args.device)
        p_ho_seeds.append(p_ho); p_iva_seeds.append(p_iva)
    p_ho_mean = np.mean(p_ho_seeds, axis=0)
    p_iva_mean = np.mean(p_iva_seeds, axis=0)
    thr = pick_threshold_f1(y_iva, p_iva_mean)
    auc_ho = roc_auc_score(y_ho, p_ho_mean)
    f1_ho, acc_ho = fold_f1_acc(y_ho, p_ho_mean, thr)
    return auc_ho, p_ho_mean, thr, f1_ho, acc_ho


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def apply_task_set(all_tasks, annot, pid_task_emb, args):
    """Restrict `all_tasks` according to --task_set. Returns filtered list."""
    if args.task_set == 'all':
        return all_tasks
    if args.task_set == 'single':
        if args.single_task not in all_tasks:
            raise SystemExit(f'--single_task {args.single_task!r} not in {len(all_tasks)} tasks')
        return [args.single_task]
    if args.task_set == 'anibal':
        kept = [t for t in all_tasks if to_group(t) in ANIBAL_GROUPS]
        print(f'[task_set=anibal] {len(kept)}/{len(all_tasks)} tasks '
              f'(groups={sorted(ANIBAL_GROUPS)})', flush=True)
        if not kept:
            raise SystemExit('no breathing tasks found for anibal subset in this npz')
        return kept
    # shared
    shared_groups, meta = compute_shared_groups(
        annot, pid_task_emb, thresh=args.thresh, label_key='stenosis')
    save_shared_tasks_json(args.shared_tasks_json, shared_groups, meta)
    kept = filter_tasks_to_shared(all_tasks, set(shared_groups))
    print(f'[task_set=shared] {len(shared_groups)} groups (thresh={args.thresh}%) '
          f'→ {len(kept)}/{len(all_tasks)} raw tasks | saved {args.shared_tasks_json}',
          flush=True)
    return kept


def main():
    args = parse_args()
    if args.fm and args.output_dir == 'output/mil':
        base = 'output/mil_holdout' if args.holdout else 'output/mil'
        args.output_dir = f'{base}/fm_{args.fm}_{args.task_set}'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(args.rs); torch.manual_seed(args.rs)
    print(f'device={args.device} | layer={args.layer} | task_set={args.task_set} '
          f'| out={args.output_dir}', flush=True)

    annot, pid_task_emb, pid_task_emb_aug, all_tasks, has_aug, holdout_pids = load_data(args)
    all_tasks = apply_task_set(all_tasks, annot, pid_task_emb, args)

    X_sten, X_sten_aug, y_sten, pids_sten = build_bags(
        annot, pid_task_emb, pid_task_emb_aug, all_tasks, 'stenosis',
        stenosis_only=False, has_aug=has_aug)
    X_strid, X_strid_aug, y_strid, _ = build_bags(
        annot, pid_task_emb, pid_task_emb_aug, all_tasks, 'stridor',
        stenosis_only=True, has_aug=has_aug)
    print(f'Stenosis : X={X_sten.shape}  pos={int(y_sten.sum())}/{len(y_sten)}', flush=True)
    print(f'Stridor  : X={X_strid.shape} pos={int(y_strid.sum())}/{len(y_strid)}', flush=True)

    # Hold-out mode : partition stenosis bags into CV pool and held-out test set.
    # Stridor head is skipped in hold-out mode (too few positives in test split for
    # a stable single AUC ; CV-only is reported).
    ho_data = {}                      # head → (X_ho, X_aug_ho, y_ho, pids_ho)
    if args.holdout:
        (X_sten_cv, X_sten_aug_cv, y_sten_cv, pids_sten_cv), \
        (X_sten_ho, X_sten_aug_ho, y_sten_ho, pids_sten_ho) = split_holdout(
            X_sten, X_sten_aug, y_sten, pids_sten, holdout_pids)
        print(f'  Stenosis CV : X={X_sten_cv.shape}  pos={int(y_sten_cv.sum())}/{len(y_sten_cv)}', flush=True)
        print(f'  Stenosis HO : X={X_sten_ho.shape}  pos={int(y_sten_ho.sum())}/{len(y_sten_ho)}', flush=True)
        X_sten, X_sten_aug, y_sten = X_sten_cv, X_sten_aug_cv, y_sten_cv
        ho_data['Stenosis'] = (X_sten_ho, X_sten_aug_ho, y_sten_ho, pids_sten_ho)

    HEADS = [('Stenosis', X_sten, X_sten_aug, y_sten, STEN_REF_TASK)]
    if not args.holdout:
        HEADS.append(('Stridor',  X_strid, X_strid_aug, y_strid, STRID_REF_TASK))
    results = []

    def _record(method, head, aucs, yt, pt, thrs, extra=None, ho=None):
        f1s  = [fold_f1_acc(yt[i], pt[i], thrs[i])[0] for i in range(len(thrs))]
        accs = [fold_f1_acc(yt[i], pt[i], thrs[i])[1] for i in range(len(thrs))]
        pf1, pacc = pooled_f1_acc(yt, pt, thrs)
        rec = {'head': head, 'method': method,
               'fold_aucs': aucs,       'stats':     fold_stats(aucs),  'pooled': pooled_auc(yt, pt),
               'fold_f1s':  f1s,        'stats_f1':  stats_dict(f1s),   'pooled_f1':  pf1,
               'fold_accs': accs,       'stats_acc': stats_dict(accs),  'pooled_acc': pacc,
               'fold_thrs': thrs,
               'cv_yt': yt, 'cv_pt': pt}
        if extra: rec.update(extra)
        if ho is not None:
            rec['holdout'] = ho       # dict with auc/f1/acc/thr/p_ho/n/pos
        return rec

    def _ho_pack(auc, p, thr, f1, acc, y_ho):
        return {'auc': float(auc), 'f1': float(f1), 'acc': float(acc),
                'thr': float(thr), 'p': p,
                'n': int(len(y_ho)), 'pos': int(np.sum(y_ho))}

    def _log_ho(head, name, ho):
        print(f"  {head:9s} | {name:18s} | HO AUC={ho['auc']:.3f} | "
              f"HO F1={ho['f1']:.3f} | HO Acc={ho['acc']:.3f}", flush=True)

    # 1. Single-task baseline (skip if reference task filtered out by task_set)
    print('\n=== Single-task baseline ===', flush=True)
    for head, X, X_aug, y, ref in HEADS:
        if ref not in all_tasks:
            print(f"  {head:9s} | ref task {ref!r} not in task_set → skipped", flush=True)
            continue
        aucs, yt, pt, thrs = run_single_task(X, X_aug, y, all_tasks, ref, args)
        ho = None
        if args.holdout and head in ho_data:
            X_ho, _, y_ho, _ = ho_data[head]
            auc_ho, p_ho, thr_ho, f1_ho, acc_ho = holdout_single_task(
                X, X_aug, y, all_tasks, ref, X_ho, y_ho, args)
            ho = _ho_pack(auc_ho, p_ho, thr_ho, f1_ho, acc_ho, y_ho)
        r = _record(f'SingleTask({ref})', head, aucs, yt, pt, thrs, ho=ho)
        results.append(r)
        print(f"  {head:9s} | AUC={r['stats']['mean']:.3f}±{r['stats']['std']:.3f} | "
              f"F1={r['stats_f1']['mean']:.3f}±{r['stats_f1']['std']:.3f} | "
              f"Acc={r['stats_acc']['mean']:.3f}±{r['stats_acc']['std']:.3f}", flush=True)
        if ho: _log_ho(head, f'SingleTask({ref})', ho)

    # 2. Pool baselines
    print('\n=== mean/max-pool x LogReg/RF ===', flush=True)
    for head, X, X_aug, y, _ in HEADS:
        for pm in ['mean', 'max']:
            for cn in ['LogReg', 'RF']:
                aucs, yt, pt, thrs = run_pool(X, X_aug, y, pm, cn, args)
                ho = None
                if args.holdout and head in ho_data:
                    X_ho, _, y_ho, _ = ho_data[head]
                    auc_ho, p_ho, thr_ho, f1_ho, acc_ho = holdout_pool(
                        X, X_aug, y, pm, cn, X_ho, y_ho, args)
                    ho = _ho_pack(auc_ho, p_ho, thr_ho, f1_ho, acc_ho, y_ho)
                method_name = f'{pm}-pool+{cn}'
                r = _record(method_name, head, aucs, yt, pt, thrs, ho=ho)
                results.append(r)
                print(f"  {head:9s} | {method_name:18s} | AUC={r['stats']['mean']:.3f}±{r['stats']['std']:.3f} | "
                      f"F1={r['stats_f1']['mean']:.3f} | Acc={r['stats_acc']['mean']:.3f}", flush=True)
                if ho: _log_ho(head, method_name, ho)

    # 3. mi-SVM
    print('\n=== mi-SVM ===', flush=True)
    for head, X, X_aug, y, _ in HEADS:
        aucs, yt, pt, thrs = run_misvm(X, X_aug, y, args)
        ho = None
        if args.holdout and head in ho_data:
            X_ho, _, y_ho, _ = ho_data[head]
            auc_ho, p_ho, thr_ho, f1_ho, acc_ho = holdout_misvm(
                X, X_aug, y, X_ho, y_ho, args)
            ho = _ho_pack(auc_ho, p_ho, thr_ho, f1_ho, acc_ho, y_ho)
        r = _record('mi-SVM', head, aucs, yt, pt, thrs, ho=ho)
        results.append(r)
        print(f"  {head:9s} | AUC={r['stats']['mean']:.3f}±{r['stats']['std']:.3f} | "
              f"F1={r['stats_f1']['mean']:.3f} | Acc={r['stats_acc']['mean']:.3f}", flush=True)
        if ho: _log_ho(head, 'mi-SVM', ho)

    # 4. Deep MIL
    if not args.skip_deep:
        DEEP_MODELS = [
            ('GatedAttention', lambda D: GatedAttentionMIL(D=D)),
            ('DSMIL',          lambda D: DSMIL(D=D)),
            ('SetTransformer', lambda D: SetTransformerMIL(D=D)),
            ('TransMIL',       lambda D: TransMIL(D=D)),
        ]
        for name, ctor in DEEP_MODELS:
            print(f'\n=== Deep: {name} ===', flush=True)
            for head, X, X_aug, y, _ in HEADS:
                D = X.shape[-1]
                print(f'  -- {head} --', flush=True)
                aucs, yt, pt, thrs, atts = run_deep(X, X_aug, y, lambda D=D: ctor(D), args)
                ho = None
                if args.holdout and head in ho_data:
                    X_ho, _, y_ho, _ = ho_data[head]
                    auc_ho, p_ho, thr_ho, f1_ho, acc_ho = holdout_deep(
                        X, X_aug, y, lambda D=D: ctor(D), X_ho, y_ho, args)
                    ho = _ho_pack(auc_ho, p_ho, thr_ho, f1_ho, acc_ho, y_ho)
                extra = {'attn_per_fold': atts, 'y_per_fold': yt} if name == 'GatedAttention' else None
                r = _record(name, head, aucs, yt, pt, thrs, extra=extra, ho=ho)
                results.append(r)
                print(f"  {head:9s} | AUC={r['stats']['mean']:.3f}±{r['stats']['std']:.3f} | "
                      f"F1={r['stats_f1']['mean']:.3f} | Acc={r['stats_acc']['mean']:.3f}", flush=True)
                if ho: _log_ho(head, name, ho)

    # Save
    rows = []
    for r in results:
        row = {'head': r['head'], 'method': r['method'],
               'mean_auc': r['stats']['mean'], 'std_auc': r['stats']['std'],
               'min_auc':  r['stats']['min'],  'max_auc': r['stats']['max'],
               'pooled_auc': r['pooled'],
               'fold_aucs': ';'.join(f'{a:.3f}' for a in r['fold_aucs'])}
        # F1
        row.update({'mean_f1': r['stats_f1']['mean'], 'std_f1': r['stats_f1']['std'],
                    'pooled_f1': r['pooled_f1'],
                    'fold_f1s': ';'.join(f'{a:.3f}' for a in r['fold_f1s'])})
        # Accuracy
        row.update({'mean_acc': r['stats_acc']['mean'], 'std_acc': r['stats_acc']['std'],
                    'pooled_acc': r['pooled_acc'],
                    'fold_accs': ';'.join(f'{a:.3f}' for a in r['fold_accs'])})
        # Per-fold thresholds (for reproducibility)
        row['fold_thrs'] = ';'.join(f'{t:.4f}' for t in r['fold_thrs'])
        # Hold-out columns (NaN when --holdout not set or head skipped).
        ho = r.get('holdout')
        if ho is not None:
            row.update({'holdout_auc': ho['auc'], 'holdout_f1': ho['f1'],
                        'holdout_acc': ho['acc'], 'holdout_thr': ho['thr'],
                        'holdout_n':   ho['n'],   'holdout_pos': ho['pos']})
        else:
            row.update({'holdout_auc': float('nan'), 'holdout_f1': float('nan'),
                        'holdout_acc': float('nan'), 'holdout_thr': float('nan'),
                        'holdout_n':   0,            'holdout_pos': 0})
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = Path(args.output_dir) / 'mil_results_5fold.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved {csv_path}', flush=True)

    # Save per-patient hold-out predictions for downstream paired analysis.
    if args.holdout and 'Stenosis' in ho_data:
        _, _, y_ho, pids_ho = ho_data['Stenosis']
        preds = {}
        for r in results:
            if r['head'] != 'Stenosis' or r.get('holdout') is None:
                continue
            preds[f"p_{r['method']}"] = r['holdout']['p']
        if preds:
            np.savez(Path(args.output_dir) / 'holdout_predictions.npz',
                     pids=np.array(pids_ho), y=y_ho, **preds)
            print(f"Saved {Path(args.output_dir) / 'holdout_predictions.npz'} "
                  f"with {len(preds)} method predictions", flush=True)

        # Per-patient OOF CV predictions (patient-level stratified bootstrap, paper § 3).
        # One out-of-fold prediction per CV-pool patient, concatenated in fold order.
        oof_order = np.concatenate([te for _, te in cv_splits(y_sten, args.n_folds, args.rs)])
        oof_pids  = np.asarray(pids_sten_cv)[oof_order]
        oof_y     = y_sten[oof_order]
        cv_preds = {}
        for r in results:
            if r['head'] != 'Stenosis':
                continue
            assert np.array_equal(np.concatenate(r['cv_yt']), oof_y), \
                f"OOF fold-order mismatch for {r['method']}"
            cv_preds[f"p_{r['method']}"] = np.concatenate(r['cv_pt'])
        np.savez(Path(args.output_dir) / 'cv_oof_predictions.npz',
                 pids=oof_pids, y=oof_y, **cv_preds)
        print(f"Saved {Path(args.output_dir) / 'cv_oof_predictions.npz'} "
              f"({len(oof_pids)} CV patients, {len(cv_preds)} methods)", flush=True)

    # Save attention weights for interpretability
    ga = next((r for r in results if r['head'] == 'Stenosis' and r['method'] == 'GatedAttention'), None)
    if ga is not None and ga.get('attn_per_fold') and ga['attn_per_fold'][0] is not None:
        all_attn = np.concatenate(ga['attn_per_fold'], axis=0)
        all_y = np.concatenate(ga['y_per_fold'])
        np.savez(Path(args.output_dir) / 'gated_attn_stenosis.npz',
                 attn=all_attn, y=all_y, tasks=np.array(all_tasks))
        print(f'Saved gated_attn_stenosis.npz', flush=True)

    print('\n=== Final ===', flush=True)
    for head in ['Stenosis', 'Stridor']:
        if head not in df['head'].unique(): continue
        print(f'\n--- {head} ---')
        sub = df[df['head'] == head].sort_values('mean_auc', ascending=False)
        for _, r in sub.iterrows():
            base = (f"  {r['method']:22s} | AUC={r['mean_auc']:.3f}±{r['std_auc']:.3f} "
                    f"[{r['min_auc']:.3f}-{r['max_auc']:.3f}] | F1={r['mean_f1']:.3f} | "
                    f"Acc={r['mean_acc']:.3f}")
            if args.holdout and np.isfinite(r.get('holdout_auc', float('nan'))):
                base += (f" | HO AUC={r['holdout_auc']:.3f} "
                         f"F1={r['holdout_f1']:.3f} Acc={r['holdout_acc']:.3f}")
            print(base)


if __name__ == '__main__':
    main()
