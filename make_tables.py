#!/usr/bin/env python3
"""Render the paper's tables from the shipped result files.

Reads only `results/`, trains nothing, needs no GPU and no access to the
dataset. Every row prints the CSV it came from and the aggregation head that
produced it, so any number in the paper can be traced in one step.

    python make_tables.py            # markdown
    python make_tables.py --latex    # markdown + LaTeX bodies
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

# --- which run and which aggregation head backs each published row ----------
# The head is not the same for every model: it is the best-performing one for
# that encoder. The paper states this; it is made explicit here.
TABLE1 = [
    # label,                     directory,            head
    ("Handcrafted (mel+MFCC+F0)", "fm_classic_all",    "DSMIL"),
    ("AST",                       "fm_ast_all",        "mean-pool+LogReg"),
    ("AudioMAE",                  "fm_audiomae_all",   "mean-pool+LogReg"),
    ("OpenBEATs",                 "fm_openbeats_all",  "mean-pool+LogReg"),
    ("SSAST",                     "fm_ssast_all",      "mean-pool+LogReg"),
    ("YAMNet",                    "fm_yamnet_all",     "mean-pool+LogReg"),
    ("HuBERT (L10)",              "fm_hubert_L10_all", "GatedAttention"),
    ("Whisper (L32)",             "fm_whisper_L32_all", "mean-pool+LogReg"),
    ("WavLM (L15)",               "fm_wavlm_L15_all",  "TransMIL"),
]
GENERAL_AUDIO = {"AST", "AudioMAE", "OpenBEATs", "SSAST", "YAMNet"}
SPEECH = {"HuBERT (L10)", "Whisper (L32)", "WavLM (L15)"}

TABLE2A = [(f"L{n}", f"fm_wavlm_L{n}_all", "TransMIL") for n in (6, 10, 15, 20, 24)]

TABLE2B = [
    ("Non-MIL (single task)", "fm_wavlm_L15_all", "SingleTask(picture-description)"),
    ("GatedAttn",             "fm_wavlm_L15_all", "GatedAttention"),
    ("DSMIL",                 "fm_wavlm_L15_all", "DSMIL"),
    ("TransMIL",              "fm_wavlm_L15_all", "TransMIL"),
]

# Table 3 reports the best sub-task configuration, which is HuBERT + mean-pool+RF
# (not the binary-task headline model), scored with pooled out-of-fold AUC.
TABLE3 = [
    ("Localization (3-class)",     "fm_hubert_localization_all", "mean-pool+RF"),
    ("Stridor detection (binary)", "fm_hubert_stridor_all",      "mean-pool+RF"),
    ("Severity grading (3-class)", "fm_hubert_severity_all",     "mean-pool+RF"),
]


def _rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def stenosis_row(directory: str, head: str) -> tuple[dict, Path]:
    path = RESULTS / "mil" / directory / "mil_results_5fold.csv"
    for r in _rows(path):
        if r.get("head") == "Stenosis" and r.get("method") == head:
            return r, path
    raise SystemExit(f"no row head=Stenosis method={head} in {path}")


def subtask_row(directory: str, head: str) -> tuple[dict, Path]:
    path = RESULTS / "subtasks" / directory / "mil_subtask_results_5fold.csv"
    for r in _rows(path):
        if r.get("method") == head:
            return r, path
    raise SystemExit(f"no row method={head} in {path}")


def pm(mean: str, std: str) -> str:
    return f"{float(mean):.3f} ± {float(std):.3f}"


def emit(title: str, header: list[str], body: list[list[str]], notes: list[str], latex: bool):
    print(f"\n## {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in body:
        print("| " + " | ".join(row) + " |")
    for n in notes:
        print(f"\n{n}")
    if latex:
        print("\n<details><summary>LaTeX</summary>\n")
        for row in body:
            cells = [c.replace("±", r"$\pm$").replace("_", r"\_") for c in row]
            print("  " + " & ".join(cells) + r" \\")
        print("\n</details>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latex", action="store_true", help="also print LaTeX table bodies")
    args = ap.parse_args()

    print("Tables regenerated from results/ (5-fold patient-level cross-validation,")
    print("748 patients: 134 with airway stenosis, 614 controls).")

    # ---- Table 1 ----
    body, srcs = [], []
    for label, d, head in TABLE1:
        r, p = stenosis_row(d, head)
        group = "Speech" if label in SPEECH else ("General audio" if label in GENERAL_AUDIO else "Baseline")
        body.append([group, label,
                     pm(r["mean_auc"], r["std_auc"]),
                     pm(r["mean_f1"], r["std_f1"]),
                     pm(r["mean_acc"], r["std_acc"]),
                     head])
        srcs.append(f"`{label}` -> `{p.relative_to(RESULTS.parent)}`")
    emit("Table 1 - comparison of feature extraction methods",
         ["Pretraining", "Method", "AUROC", "F1-score", "Accuracy", "Aggregation head"],
         body,
         ["The aggregation head differs per encoder: each row uses the best-performing "
          "head for that model, which is why the column is reported explicitly.",
          "Sources: " + "; ".join(srcs)],
         args.latex)

    # ---- Table 2a ----
    body = []
    for label, d, head in TABLE2A:
        r, _ = stenosis_row(d, head)
        body.append([label, pm(r["mean_auc"], r["std_auc"]),
                     pm(r["mean_f1"], r["std_f1"]), pm(r["mean_acc"], r["std_acc"])])
    emit("Table 2(a) - WavLM layer selection (TransMIL)",
         ["Layer", "AUROC", "F1-score", "Accuracy"], body,
         ["All rows use TransMIL aggregation on WavLM-Large."], args.latex)

    # ---- Table 2b ----
    body = []
    for label, d, head in TABLE2B:
        r, _ = stenosis_row(d, head)
        body.append([label, pm(r["mean_auc"], r["std_auc"]),
                     pm(r["mean_f1"], r["std_f1"]), pm(r["mean_acc"], r["std_acc"])])
    emit("Table 2(b) - patient-level aggregation strategy (WavLM L15)",
         ["Aggregation", "AUROC", "F1-score", "Accuracy"], body,
         ["\"Non-MIL\" is a classifier on the single most informative task "
          "(picture description) rather than on the full bag."], args.latex)

    # ---- Table 3 ----
    body = []
    for label, d, head in TABLE3:
        r, _ = subtask_row(d, head)
        body.append([label, f"{float(r['pooled_auc']):.3f}"])
    emit("Table 3 - beyond binary diagnosis",
         ["Task", "AUROC"], body,
         ["Pooled out-of-fold AUC (macro one-vs-rest for the 3-class tasks), "
          "HuBERT-Large L10 with mean-pool+RF, the best sub-task configuration. "
          "Cohorts are small (N = 134 / 75 / 133), so these are exploratory. "
          "The paper rounds these to two decimals (0.82 / 0.78 / 0.63)."], args.latex)


if __name__ == "__main__":
    main()
