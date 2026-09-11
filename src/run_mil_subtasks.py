"""MIL sweep for the 3 sub-tasks : localization, severity, stridor.

Same canonical (pid, task_group) cohort, same 9 FM × 10 methods × 5-fold
patient-level CV protocol as `run_mil_stenosis_stridor.py` (binary headline).
Labels read from `bridge2voice_data_split.json` (stenosis+ subset).

Targets (auto-handled multi-class vs binary) :
    --target localization  → 3-class (Subglottic / Glottic / Other)
    --target severity      → 3-class ordinal (Mild / Moderate / Severe)
    --target stridor       → binary (No / Yes) but with the same CrossEntropy
                              + multiclass_metrics interface for consistency

Output CSV columns :
    target, method, mean_auc, std_auc, mean_f1, std_f1, mean_bacc, std_bacc,
    pooled_*, fold_aucs, fold_f1s, fold_baccs, pooled_confusion[, mean_kappa]

Usage (HPC) :
    uv run python training/run_mil_subtasks.py \
        --target localization --fm wavlm --task_set all \
        --n_folds 5 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mil_models import DSMIL, SetTransformerMIL, TransMIL
from metrics import multiclass_metrics, optimize_multiclass_thresholds

# Participants with truncated or unusable recordings, dropped everywhere.
EXCLUDE_PIDS = {'364676', '712389', '821259', '981647'}
STEN_REF_TASK = 'picture-description'

SUBTASK_DEFS = {
    'localization': {
        'field': 'diagnosis_as_ds',
        'mapping': {
            'Subglottic Stenosis':                                  0,
            'Bilateral Vocal fold immobility or Glottic Stenosis':  1,
            'Tracheal Stenosis':                                    2,
            'Multi-Level Upper Airway Stenosis':                    2,
            'Supraglottic Stenosis':                                2,
        },
        'classes': ['Subglottic', 'Glottic', 'Other'],
        'ordinal': False,
    },
    'severity': {
        'field': 'diagnosis_as_ds_ods',
        'mapping': {'Mild': 0, 'Moderate': 1, 'Severe': 2},
        'classes': ['Mild', 'Moderate', 'Severe'],
        'ordinal': True,
    },
    'stridor': {
        'field': 'diagnosis_as_as',
        'mapping': {'No': 0, 'Yes': 1},
        'classes': ['No', 'Yes'],
        'ordinal': False,
    },
}


def norm_pid(p): return str(p).zfill(6)


# ---------------------------------------------------------------------------
# Args / IO
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--target', required=True, choices=list(SUBTASK_DEFS.keys()))
    p.add_argument('--fm', default=None, help='FM shorthand → embeddings_<fm>.npz, used only when --npz is NOT given')
    p.add_argument('--npz', default=None, help='explicit embeddings npz; takes precedence over --fm')
    p.add_argument('--split_json', default='data/binary_stenosis_split.json')
    p.add_argument('--b2ai_json',  default='data/bridge2voice_data_split.json')
    p.add_argument('--output_dir', default='output/mil')
    p.add_argument('--task_set', default='all', choices=['all', 'shared'])
    p.add_argument('--shared_tasks_json', default='data/shared_tasks_stenosis.json')
    p.add_argument('--single_task', default=STEN_REF_TASK)
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
    p.add_argument('--skip_deep', action='store_true')
    return p.parse_args()


def load_data(args):
    cfg = SUBTASK_DEFS[args.target]
    mapping = cfg['mapping']

    with open(args.split_json) as f: bi_split = json.load(f)
    canonical_pids = set()
    for s in ('train', 'val', 'test'):
        for e in bi_split[s]: canonical_pids.add(norm_pid(e['participant_id']))
    canonical_pids -= EXCLUDE_PIDS

    with open(args.b2ai_json) as f: b2_split = json.load(f)
    annot = {}
    for s in ('train', 'val', 'test'):
        for e in b2_split[s]:
            pid = norm_pid(e['participant_id'])
            if pid not in canonical_pids: continue
            raw = e.get(cfg['field'])
            if raw not in mapping: continue
            annot[pid] = mapping[raw]
    n_classes = len(cfg['classes'])
    print(f'[{args.target}] labelled patients : {len(annot)} | '
          f'dist={np.bincount(list(annot.values()), minlength=n_classes).tolist()}',
          flush=True)

    # --npz wins when given explicitly; --fm is a shorthand fallback.
    if args.npz:
        npz_path = args.npz
        if args.fm:
            print(f'[npz] both --fm and --npz given → using --npz={args.npz}', flush=True)
    elif args.fm:
        npz_path = f'output/embeddings/embeddings_{args.fm}.npz'
    else:
        npz_path = 'output/embeddings/embeddings_wavlm.npz'
    data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
    pids_arr  = data['participant_ids'].astype(str)
    tasks_arr = data['tasks'].astype(str)
    embs = data['embeddings']
    has_layers = embs.ndim == 3
    print(f'npz={npz_path} shape={embs.shape} layered={has_layers}', flush=True)

    acc = defaultdict(list)
    for i in range(len(pids_arr)):
        pid = pids_arr[i]
        if pid not in annot: continue
        vec = (np.asarray(embs[i, args.layer], dtype=np.float32) if has_layers
               else np.asarray(embs[i], dtype=np.float32))
        acc[(pid, tasks_arr[i])].append(vec)
    pid_task_emb = {k: np.mean(v, axis=0) for k, v in acc.items()}
    all_tasks = sorted({t for (_, t) in pid_task_emb})
    print(f'(pid,task) pairs = {len(pid_task_emb)}  | {len(all_tasks)} tasks', flush=True)
    return annot, pid_task_emb, all_tasks, cfg


def build_bags(annot, pid_task_emb, all_tasks):
    X, y, pids = [], [], []
    for pid in sorted(annot):
        bag, ok = [], True
        for t in all_tasks:
            e = pid_task_emb.get((pid, t))
            if e is None: ok = False; break
            bag.append(e)
        if not ok: continue
        X.append(np.stack(bag, axis=0)); y.append(annot[pid]); pids.append(pid)
    if not X:
        raise SystemExit('no bag could be built : embeddings missing for the labelled patients')
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64), pids


# ---------------------------------------------------------------------------
# CV utilities
# ---------------------------------------------------------------------------
def cv_splits(y, n_folds, rs):
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs).split(np.zeros(len(y)), y)


def inner_split(y, frac, rs):
    return next(StratifiedShuffleSplit(n_splits=1, test_size=frac,
                                        random_state=rs).split(np.zeros(len(y)), y))


def pad_proba(scores: np.ndarray, present_classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Pad sklearn predict_proba / decision_function output to (N, n_classes).

    Handles three shapes :
        (N,)             -> binary decision_function -> [-s, s]
        (N, 1) & K==2    -> OneVsRest LinearSVC on binary, same as above
        (N, K_present)   -> generic per-class scores, indexed by present_classes
    """
    out = np.zeros((scores.shape[0], n_classes), dtype=scores.dtype)
    # Binary decision_function (or single-column OneVsRest on binary problem)
    if scores.ndim == 1 or (scores.ndim == 2 and scores.shape[1] == 1 and n_classes == 2):
        s = scores if scores.ndim == 1 else scores[:, 0]
        out[:, 1] =  s
        out[:, 0] = -s
        return out
    for i, c in enumerate(present_classes):
        if i >= scores.shape[1]: break
        out[:, int(c)] = scores[:, i]
    return out


# ---------------------------------------------------------------------------
# Methods (sklearn)
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


def pool_X(X, mode): return X.mean(axis=1) if mode == 'mean' else X.max(axis=1)


def run_single_task(X, y, all_tasks, task_name, n_classes, args):
    if task_name not in all_tasks: return None
    k = all_tasks.index(task_name)
    yt_list, sc_list, off_list = [], [], []
    for tr, te in cv_splits(y, args.n_folds, args.rs):
        # Inner 80/20 on train fold for F1-threshold tuning
        itr, iva = inner_split(y[tr], args.inner_val_frac, args.rs)
        clf_inner = make_clf('LogReg', args.rs)
        clf_inner.fit(X[tr][itr, k, :], y[tr][itr])
        s_iva = clf_inner.predict_proba(X[tr][iva, k, :])
        if s_iva.shape[1] < n_classes:
            s_iva = pad_proba(s_iva, clf_inner.classes_, n_classes)
        b = optimize_multiclass_thresholds(y[tr][iva], s_iva, n_classes)
        # Refit on full train fold, predict on test
        clf = make_clf('LogReg', args.rs)
        clf.fit(X[tr, k, :], y[tr])
        s = clf.predict_proba(X[te, k, :])
        if s.shape[1] < n_classes: s = pad_proba(s, clf.classes_, n_classes)
        yt_list.append(y[te]); sc_list.append(s); off_list.append(b)
    return yt_list, sc_list, off_list


def run_pool(X, y, mode, clf_name, n_classes, args):
    yt_list, sc_list, off_list = [], [], []
    for tr, te in cv_splits(y, args.n_folds, args.rs):
        itr, iva = inner_split(y[tr], args.inner_val_frac, args.rs)
        clf_inner = make_clf(clf_name, args.rs)
        clf_inner.fit(pool_X(X[tr][itr], mode), y[tr][itr])
        s_iva = clf_inner.predict_proba(pool_X(X[tr][iva], mode))
        if s_iva.shape[1] < n_classes:
            s_iva = pad_proba(s_iva, clf_inner.classes_, n_classes)
        b = optimize_multiclass_thresholds(y[tr][iva], s_iva, n_classes)
        # Refit on full train fold
        clf = make_clf(clf_name, args.rs)
        clf.fit(pool_X(X[tr], mode), y[tr])
        s = clf.predict_proba(pool_X(X[te], mode))
        if s.shape[1] < n_classes: s = pad_proba(s, clf.classes_, n_classes)
        yt_list.append(y[te]); sc_list.append(s); off_list.append(b)
    return yt_list, sc_list, off_list


# ---------- mi-SVM (multi-class via OneVsRest) ----------
def mi_svm_train_mc(X_bags, y_bags, n_classes, rs, C=1.0, max_iter=15):
    N, K, D = X_bags.shape
    X_flat = X_bags.reshape(N * K, D)
    bag_idx = np.repeat(np.arange(N), K)
    y_inst = np.repeat(y_bags, K).astype(int)
    pipe, prev = None, None
    for _ in range(max_iter):
        if len(np.unique(y_inst)) < 2: break
        pipe = Pipeline([('sc', StandardScaler()),
                         ('svm', OneVsRestClassifier(
                             LinearSVC(C=C, class_weight='balanced',
                                       max_iter=2000, random_state=rs, dual='auto')))])
        pipe.fit(X_flat, y_inst)
        scores_raw = pipe.decision_function(X_flat)
        if scores_raw.ndim == 1: scores_raw = scores_raw[:, None]
        if scores_raw.shape[1] < n_classes:
            scores_raw = pad_proba(scores_raw, np.array(pipe.classes_), n_classes)
        new_inst = y_inst.copy()
        for b in range(N):
            mask = (bag_idx == b)
            true_class = int(y_bags[b])
            pred = scores_raw[mask].argmax(axis=1)
            if not (pred == true_class).any():
                idx_max = scores_raw[mask, true_class].argmax()
                pred[idx_max] = true_class
            new_inst[mask] = pred
        if prev is not None and np.array_equal(new_inst, prev): break
        prev, y_inst = y_inst, new_inst
    return pipe


def mi_svm_score_mc(pipe, X_bags, n_classes):
    N, K, D = X_bags.shape
    sf = pipe.decision_function(X_bags.reshape(N * K, D))
    if sf.ndim == 1: sf = sf[:, None]
    if sf.shape[1] < n_classes:
        sf = pad_proba(sf, np.array(pipe.classes_), n_classes)
    sb = sf.reshape(N, K, n_classes).max(axis=1)
    e = np.exp(sb - sb.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def run_misvm(X, y, n_classes, args, C=1.0):
    yt_list, sc_list, off_list = [], [], []
    for tr, te in cv_splits(y, args.n_folds, args.rs):
        # Inner 80/20 on train fold for threshold tuning
        itr, iva = inner_split(y[tr], args.inner_val_frac, args.rs)
        pipe_inner = mi_svm_train_mc(X[tr][itr], y[tr][itr], n_classes, args.rs, C=C)
        if pipe_inner is None:
            b = np.zeros(n_classes)
        else:
            s_iva = mi_svm_score_mc(pipe_inner, X[tr][iva], n_classes)
            b = optimize_multiclass_thresholds(y[tr][iva], s_iva, n_classes)
        # Refit on full train fold
        pipe = mi_svm_train_mc(X[tr], y[tr], n_classes, args.rs, C=C)
        if pipe is None:
            yt_list.append(y[te])
            sc_list.append(np.full((len(te), n_classes), 1.0 / n_classes))
            off_list.append(b)
            continue
        sc_list.append(mi_svm_score_mc(pipe, X[te], n_classes))
        yt_list.append(y[te]); off_list.append(b)
    return yt_list, sc_list, off_list


# ---------------------------------------------------------------------------
# Deep MIL (multi-class)
# ---------------------------------------------------------------------------
class GatedAttentionMIL_MC(nn.Module):
    def __init__(self, D=1024, L=128, dropout=0.3, n_classes=2):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(D, L), nn.ReLU(), nn.Dropout(dropout))
        self.V = nn.Linear(L, L); self.U = nn.Linear(L, L); self.w = nn.Linear(L, 1)
        self.n_classes = n_classes
        self.head = nn.Linear(L, n_classes)

    def forward(self, bag):
        h = self.feat(bag)
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        a = torch.softmax(self.w(a).squeeze(-1), dim=1)
        z = (a.unsqueeze(-1) * h).sum(dim=1)
        return self.head(z), a                                # (B, n_classes), (B, K)


def _logits(model, X):
    out = model(X)
    if isinstance(out, tuple): return out[0]
    return out


def train_deep(model_fn, X_tr, y_tr, X_va, y_va, n_classes, seed, args):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_fn().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    w = (len(y_tr) / (n_classes * np.maximum(counts, 1.0))).astype(np.float32)
    class_w = torch.tensor(w, device=args.device, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    Xtr = torch.tensor(X_tr, device=args.device)
    ytr = torch.tensor(y_tr, device=args.device, dtype=torch.long)
    Xva = torch.tensor(X_va, device=args.device)
    best_score, best_state, patience = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        for i in torch.randperm(len(Xtr), device=args.device).split(args.bs):
            logits = _logits(model, Xtr[i])
            if logits.ndim == 1: logits = logits.unsqueeze(-1)
            loss = loss_fn(logits, ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits_va = _logits(model, Xva)
            if logits_va.ndim == 1: logits_va = logits_va.unsqueeze(-1)
            prob_va = torch.softmax(logits_va, dim=-1).cpu().numpy()
        if len(np.unique(y_va)) < 2: break
        try:
            score = multiclass_metrics(y_va, prob_va, n_classes, ordinal=False)['macro_auc']
        except Exception:
            break
        if not np.isfinite(score): break
        if score > best_score:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_deep(model, X, n_classes, device):
    model.eval()
    logits = _logits(model, torch.tensor(X, device=device))
    if logits.ndim == 1: logits = logits.unsqueeze(-1)
    return torch.softmax(logits, dim=-1).cpu().numpy()


def run_deep(X, y, model_fn, n_classes, args):
    yt_list, sc_list, off_list = [], [], []
    for tr, te in cv_splits(y, args.n_folds, args.rs):
        itr, iva = inner_split(y[tr], args.inner_val_frac, args.rs)
        X_itr, y_itr = X[tr][itr], y[tr][itr]
        X_iva, y_iva = X[tr][iva], y[tr][iva]
        p_seeds, p_iva_seeds = [], []
        for s in args.seeds:
            model = train_deep(model_fn, X_itr, y_itr, X_iva, y_iva, n_classes, s, args)
            p_seeds.append(predict_deep(model, X[te], n_classes, args.device))
            p_iva_seeds.append(predict_deep(model, X_iva, n_classes, args.device))
        p_mean = np.mean(p_seeds, axis=0)
        p_iva_mean = np.mean(p_iva_seeds, axis=0)
        b = optimize_multiclass_thresholds(y_iva, p_iva_mean, n_classes)
        yt_list.append(y[te]); sc_list.append(p_mean); off_list.append(b)
    return yt_list, sc_list, off_list


# ---------------------------------------------------------------------------
# Metrics aggregation + CSV
# ---------------------------------------------------------------------------
def fold_aggregate_metrics(yt_list, sc_list, n_classes, ordinal, off_list=None):
    """Per-fold + pooled metrics. If `off_list` is provided, F1/bAcc/kappa
    use per-fold tuned argmax ; the pooled metric uses an averaged offset.
    """
    if off_list is None:
        off_list = [None] * len(yt_list)
    per_fold = [multiclass_metrics(yt, sc, n_classes, ordinal, offsets=off)
                for yt, sc, off in zip(yt_list, sc_list, off_list)]
    avg_off = (np.mean(off_list, axis=0)
               if all(o is not None for o in off_list) else None)
    pooled = multiclass_metrics(np.concatenate(yt_list),
                                 np.concatenate(sc_list, axis=0), n_classes, ordinal,
                                 offsets=avg_off)
    return per_fold, pooled


def apply_task_set(all_tasks, args):
    if args.task_set == 'shared' and Path(args.shared_tasks_json).exists():
        with open(args.shared_tasks_json) as f:
            shared = set(json.load(f).get('shared_groups', []))
        filtered = [t for t in all_tasks if t in shared]
        print(f'[task_set=shared] {len(filtered)}/{len(all_tasks)} tasks kept', flush=True)
        return filtered if filtered else all_tasks
    return all_tasks


def main():
    args = parse_args()
    annot, pid_task_emb_full, all_tasks_full, cfg = load_data(args)
    all_tasks = apply_task_set(all_tasks_full, args)
    pid_task_emb = {(p, t): v for (p, t), v in pid_task_emb_full.items() if t in all_tasks}
    X, y, pids = build_bags(annot, pid_task_emb, all_tasks)
    n_classes = len(cfg['classes'])
    ordinal   = cfg['ordinal']
    print(f'X={X.shape}  y dist={np.bincount(y, minlength=n_classes).tolist()}', flush=True)

    if args.fm and args.output_dir == 'output/mil':
        args.output_dir = f'output/mil/fm_{args.fm}_{args.target}_{args.task_set}'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f'device={args.device}  layer={args.layer}  target={args.target}  '
          f'out={args.output_dir}', flush=True)

    # Patient-level OOF setup: one prediction per CV patient, concatenated in fold order.
    _oof_order = np.concatenate([te for _, te in cv_splits(y, args.n_folds, args.rs)])
    _oof_y     = y[_oof_order]
    _oof_pids  = np.asarray(pids)[_oof_order]
    oof_scores = {}

    records = []

    def record(method, yt_list, sc_list, off_list=None):
        per_fold, pooled = fold_aggregate_metrics(yt_list, sc_list, n_classes, ordinal,
                                                    off_list=off_list)
        aucs  = [m['macro_auc']    for m in per_fold]
        f1s   = [m['macro_f1']     for m in per_fold]
        baccs = [m['balanced_acc'] for m in per_fold]
        rec = {
            'target':       args.target,
            'method':       method,
            'mean_auc':     float(np.nanmean(aucs)),
            'std_auc':      float(np.nanstd(aucs)),
            'mean_f1':      float(np.nanmean(f1s)),
            'std_f1':       float(np.nanstd(f1s)),
            'mean_bacc':    float(np.nanmean(baccs)),
            'std_bacc':     float(np.nanstd(baccs)),
            'pooled_auc':   pooled['macro_auc'],
            'pooled_f1':    pooled['macro_f1'],
            'pooled_bacc':  pooled['balanced_acc'],
            'pooled_confusion': json.dumps(pooled['confusion']),
            'fold_aucs':    ';'.join(f'{a:.3f}' for a in aucs),
            'fold_f1s':     ';'.join(f'{a:.3f}' for a in f1s),
            'fold_baccs':   ';'.join(f'{a:.3f}' for a in baccs),
        }
        if ordinal:
            kappas = [m.get('kappa', float('nan')) for m in per_fold]
            rec.update({
                'mean_kappa':   float(np.nanmean(kappas)),
                'std_kappa':    float(np.nanstd(kappas)),
                'pooled_kappa': pooled.get('kappa', float('nan')),
                'fold_kappas':  ';'.join(f'{a:.3f}' for a in kappas),
            })
        records.append(rec)
        assert np.array_equal(np.concatenate(yt_list), _oof_y), f'OOF order mismatch for {method}'
        oof_scores[method] = np.concatenate(sc_list, axis=0)   # (N, n_classes) multi-class scores
        kappa_str = f' | kappa={rec.get("mean_kappa", float("nan")):.3f}' if ordinal else ''
        print(f'  {method:24s} | macro_AUC={rec["mean_auc"]:.3f}±{rec["std_auc"]:.3f} '
              f'| macro_F1={rec["mean_f1"]:.3f} | bAcc={rec["mean_bacc"]:.3f}{kappa_str}',
              flush=True)

    print(f'\n=== SingleTask ({args.single_task}) ===', flush=True)
    sub = run_single_task(X, y, all_tasks, args.single_task, n_classes, args)
    if sub is not None: record(f'SingleTask({args.single_task})', *sub)
    else: print(f'  {args.single_task!r} not in {all_tasks} -> skip', flush=True)

    print('\n=== mean/max-pool × LogReg/RF ===', flush=True)
    for pm in ('mean', 'max'):
        for cn in ('LogReg', 'RF'):
            record(f'{pm}-pool+{cn}', *run_pool(X, y, pm, cn, n_classes, args))

    print('\n=== mi-SVM ===', flush=True)
    record('mi-SVM', *run_misvm(X, y, n_classes, args))

    if not args.skip_deep:
        DEEP_MODELS = [
            ('GatedAttention', lambda D: GatedAttentionMIL_MC(D=D, n_classes=n_classes)),
            ('DSMIL',          lambda D: DSMIL(D=D, n_classes=n_classes)),
            ('SetTransformer', lambda D: SetTransformerMIL(D=D, n_classes=n_classes)),
            ('TransMIL',       lambda D: TransMIL(D=D, n_classes=n_classes)),
        ]
        for name, ctor in DEEP_MODELS:
            print(f'\n=== Deep: {name} ===', flush=True)
            D = X.shape[-1]
            yt_list, sc_list, off_list = run_deep(X, y, lambda D=D: ctor(D), n_classes, args)
            record(name, yt_list, sc_list, off_list)

    df = pd.DataFrame(records)
    csv_path = Path(args.output_dir) / 'mil_subtask_results_5fold.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved {csv_path}', flush=True)

    # Per-patient OOF CV predictions for patient-level stratified bootstrap (paper § S2).
    # Multi-class: scores are (N, n_classes); bootstrap resamples patients stratified by class.
    np.savez(Path(args.output_dir) / 'cv_oof_predictions.npz',
             pids=_oof_pids, y=_oof_y,
             classes=np.array(cfg['classes'], dtype=object),
             **{f'p_{m}': v for m, v in oof_scores.items()})
    print(f'Saved cv_oof_predictions.npz ({len(_oof_pids)} patients, '
          f'{len(oof_scores)} methods, {n_classes} classes)', flush=True)


if __name__ == '__main__':
    main()
