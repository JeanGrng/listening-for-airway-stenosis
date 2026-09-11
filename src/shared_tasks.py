"""Shared-task computation for leakage-controlled stenosis evaluation.

A "shared task group" is a task (numeric suffix stripped, e.g.
`harvard-sentences-50` -> `harvard-sentences`) covered by >= `thresh` percent
of patients in BOTH the stenosis and control cohorts. Restricting MIL bags to
shared tasks removes the task-identity leakage documented in the project
(class-exclusive tasks let a model shortcut via task identity rather than
pathology).

Logic ported from notebooks/binary_stenosis_probe.ipynb (cell 4) and
centralized here so the MIL pipeline and notebooks share one definition.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

_NUM = re.compile(r"^(.+)-(\d+)$")


def to_group(task: str) -> str:
    """Strip a trailing `-<digits>` to collapse numbered variants into a group."""
    m = _NUM.match(task)
    return m.group(1) if m else task


def compute_shared_groups(
    annot: dict[str, dict],
    pid_task_emb: dict[tuple[str, str], object],
    thresh: float = 97.0,
    label_key: str = "stenosis",
) -> tuple[list[str], dict]:
    """Return (shared_task_groups, coverage_table).

    Args:
        annot:        pid -> {label_key: 0/1/-1, ...}
        pid_task_emb: keys are (pid, raw_task), only the keys are used here
        thresh:       min coverage percent required in BOTH cohorts
        label_key:    which binary label defines the two cohorts

    A patient counts toward a group if it has >=1 raw task in that group.
    """
    pid_groups: dict[str, set[str]] = defaultdict(set)
    for (pid, task) in pid_task_emb:
        pid_groups[pid].add(to_group(task))

    pos_pids = [p for p, d in annot.items() if d.get(label_key) == 1 and p in pid_groups]
    neg_pids = [p for p, d in annot.items() if d.get(label_key) == 0 and p in pid_groups]
    n_pos, n_neg = len(pos_pids), len(neg_pids)
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(f"empty cohort: n_pos={n_pos} n_neg={n_neg}")

    def coverage(pids: list[str]) -> dict[str, int]:
        c: dict[str, set[str]] = defaultdict(set)
        for pid in pids:
            for g in pid_groups[pid]:
                c[g].add(pid)
        return {g: len(s) for g, s in c.items()}

    cov_pos = coverage(pos_pids)
    cov_neg = coverage(neg_pids)
    all_groups = sorted(set(cov_pos) | set(cov_neg))

    table = []
    shared: list[str] = []
    for g in all_groups:
        pos_pct = cov_pos.get(g, 0) / n_pos * 100
        neg_pct = cov_neg.get(g, 0) / n_neg * 100
        min_pct = min(pos_pct, neg_pct)
        table.append({"group": g, "pos_pct": round(pos_pct, 1),
                      "neg_pct": round(neg_pct, 1), "min_pct": round(min_pct, 1)})
        if min_pct >= thresh:
            shared.append(g)

    table.sort(key=lambda r: r["min_pct"], reverse=True)
    meta = {"n_pos": n_pos, "n_neg": n_neg, "thresh": thresh,
            "n_groups_total": len(all_groups), "n_shared": len(shared),
            "coverage": table}
    return sorted(shared), meta


def filter_tasks_to_shared(all_tasks: list[str], shared_groups: set[str]) -> list[str]:
    """Keep raw task names whose group is in `shared_groups`."""
    return [t for t in all_tasks if to_group(t) in shared_groups]


def save_shared_tasks_json(
    path: str | Path,
    shared_groups: list[str],
    meta: dict,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"shared_groups": shared_groups, "meta": meta}, f, indent=2)


def load_shared_groups(path: str | Path) -> set[str]:
    with open(path) as f:
        return set(json.load(f)["shared_groups"])
