#!/usr/bin/env python3
"""Figure 2 - which recording task the classifier relies on.

Leave-one-task-out permutation importance: for each of the 16 task groups the
corresponding embedding is replaced by the mean embedding of stenosis-negative
training patients, and the drop in the pre-sigmoid logit is recorded. The figure
averages that drop over stenosis-positive patients.

    python figures/make_fig2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from tasks import FAMILY, FAMILY_ORDER, PRETTY  # noqa: E402

ACCENT = (0.70, 0.15, 0.15)
FAMILY_COLOR = {
    "Complex speech":       ACCENT,
    "Complex articulation": (0.85, 0.40, 0.20),
    "Simple DDK":           (0.25, 0.35, 0.55),
    "Sustained acoustic":   (0.60, 0.60, 0.60),
    "Respiration":          (0.75, 0.75, 0.75),
}


def main() -> None:
    npz = HERE.parent / "results" / "viz" / "transmil_attention.npz"
    d = np.load(npz, allow_pickle=True)
    tasks = d["tasks"].astype(str)
    importance = d["perm_importance"][d["y"] == 1].mean(0)

    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.barh(range(len(order)), importance[order],
            color=[FAMILY_COLOR[FAMILY[tasks[i]]] for i in order])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([PRETTY[tasks[i]] for i in order], fontsize=11)
    ax.set_xlabel("Permutation importance on stenosis+ (drop in logit)")
    ax.legend(handles=[Patch(color=FAMILY_COLOR[f], label=f) for f in FAMILY_ORDER],
              fontsize=10, loc="lower right", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = HERE / f"task_group_importance.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out.relative_to(HERE.parent)}")

    print("\ntop task groups:")
    for i in order[::-1][:4]:
        print(f"  {PRETTY[tasks[i]]:22s} {importance[i]:+.3f}  ({FAMILY[tasks[i]]})")


if __name__ == "__main__":
    main()
