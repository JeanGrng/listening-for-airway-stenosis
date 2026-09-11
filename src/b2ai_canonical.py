"""Canonical B2AI cohort + task selection for FM embedding extraction.

Single source of truth so every `compute_embeddings_*.py` script produces
embeddings over the *exact same* (patient, task_group) set, making the
foundation-model comparison fair and `task_set=all` well-defined in the MIL
pipeline (`training/run_mil_stenosis_stridor.py`).

Extracted verbatim (logic-preserving) from
`compute/compute_embeddings_wavlm_layers_v2.py`, the reference that produced
`embeddings_wavlm_layers_v4.npz` (16 task groups, EXCLUDE_PIDS applied).

Usage in an FM script:

    from b2ai_canonical import iter_canonical_bags, TASK_GROUPS

    rows = []
    for pid, group, wav, dur in iter_canonical_bags(parquet_path, split_json):
        emb = my_model_forward(wav)          # (D,) or (n_layers, D)
        rows.append((pid, group, emb, dur))
    # save: participant_ids, tasks (=group), embeddings, original_duration_sec
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

from griffin_lim import linear_spectrogram_to_waveform

SAMPLE_RATE = 16000
# Linear spectrogram hop (torchaudio parquet): 20 ms. Used to derive an
# approximate clip duration for the spectrogram variant.
LINEAR_HOP_SEC = 0.020

# Participants with truncated or unusable recordings, dropped everywhere.
EXCLUDE_PIDS = {'364676', '712389', '821259', '981647'}

ALL_SHARED_TASKS = {
    'respiration-and-cough-breath-1',     'respiration-and-cough-breath-2',
    'respiration-and-cough-fivebreaths-1', 'respiration-and-cough-fivebreaths-2',
    'respiration-and-cough-fivebreaths-3', 'respiration-and-cough-fivebreaths-4',
    'respiration-and-cough-cough-1',      'respiration-and-cough-cough-2',
    'prolonged-vowel',
    'maximum-phonation-time-1', 'maximum-phonation-time-2', 'maximum-phonation-time-3',
    'rainbow-passage',
    'picture-description', 'story-recall',
    'diadochokinesis-pa', 'diadochokinesis-ta', 'diadochokinesis-ka',
    'diadochokinesis-pataka', 'diadochokinesis-buttercup',
    'glides-high-to-low', 'glides-low-to-high', 'loudness',
}

_NUMBERED = re.compile(r'^(.+)-(\d+)$')


def task_to_group(task: str) -> str:
    """Strip trailing -N to get the group name if numbered siblings exist."""
    m = _NUMBERED.match(task)
    if m:
        base = m.group(1)
        if f"{base}-1" in ALL_SHARED_TASKS:
            return base
    return task


# group_name → sorted list of member tasks
TASK_GROUPS: dict[str, list[str]] = defaultdict(list)
for _t in sorted(ALL_SHARED_TASKS):
    TASK_GROUPS[task_to_group(_t)].append(_t)
TASK_GROUPS = dict(sorted(TASK_GROUPS.items()))


def load_target_pids(split_json: str) -> set[str]:
    """train+val+test participant_ids from the binary stenosis split, zero-padded,
    minus EXCLUDE_PIDS."""
    with open(split_json) as f:
        split_info = json.load(f)
    target = set()
    for split in ('train', 'val', 'test'):
        for s in split_info[split]:
            pid = str(s['participant_id']).zfill(6)
            if pid not in EXCLUDE_PIDS:
                target.add(pid)
    return target


def prepare_slices(wav: np.ndarray, window_size: int) -> list[np.ndarray]:
    """Split wav into non-overlapping chunks of at most window_size samples.

    Short recordings (< window_size) returned as-is, NO tiling (tiling would
    create artificial periodicity unseen during pretraining).
    """
    if len(wav) <= window_size:
        return [wav]
    slices, offset = [], 0
    while offset < len(wav):
        slices.append(wav[offset:offset + window_size])
        offset += window_size
    return slices


def _iter_pid_group_specs(
    parquet_path: str,
    split_json: str,
    verbose: bool = True,
) -> Iterator[tuple[str, str, list[np.ndarray]]]:
    """Core: yield (pid, task_group, [row_linear_spectrograms]).

    Two-pass over the spectrogram parquet (lightweight index, then load only
    needed rows). Member tasks collected in sorted member order, the single
    place cohort/task selection is defined.
    """
    target_pids = load_target_pids(split_json)
    if verbose:
        print(f"Target patients (excl. truncated): {len(target_pids)}", flush=True)
        print(f"Task groups ({len(TASK_GROUPS)}): {list(TASK_GROUPS)}", flush=True)

    # Pass 1, index: (pid, task) → [global_row_idx, ...]
    if verbose:
        print("Pass 1: building index...", flush=True)
    pid_task_rows: dict[tuple, list[int]] = defaultdict(list)
    global_row = 0
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=500, columns=['participant_id', 'task_name']):
        for pid, task in zip(batch['participant_id'].to_pylist(),
                             batch['task_name'].to_pylist()):
            pid = str(pid).zfill(6)
            if pid in target_pids and task in ALL_SHARED_TASKS:
                pid_task_rows[(pid, task)].append(global_row)
            global_row += 1

    needed_rows = set(ri for ris in pid_task_rows.values() for ri in ris)
    if verbose:
        print(f"  {len(pid_task_rows)} (pid, task) pairs | "
              f"{len(needed_rows)} rows to load", flush=True)

    # Pass 2, load spectrogram for needed rows only
    if verbose:
        print("Pass 2: loading spectrograms...", flush=True)
    row_specs: dict[int, np.ndarray] = {}
    global_row = 0
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=500):
        bs = len(batch)
        local_needed = [i for i in range(bs) if global_row + i in needed_rows]
        if local_needed:
            df = batch.to_pydict()
            for local_i in local_needed:
                abs_i = global_row + local_i
                spec = np.array([np.array(v, dtype=np.float32)
                                 for v in df['spectrogram'][local_i]])
                row_specs[abs_i] = spec
        global_row += bs
        if len(row_specs) == len(needed_rows):
            break
    if verbose:
        print(f"  Loaded {len(row_specs)} spectrograms.", flush=True)

    # Per (pid, group): collect sibling row spectrograms in sorted member order
    for pid in sorted(target_pids):
        for group, members in TASK_GROUPS.items():
            specs = []
            for task in members:
                for ri in pid_task_rows.get((pid, task), []):
                    if ri in row_specs:
                        specs.append(row_specs[ri])
            if not specs:
                continue
            yield pid, group, specs


def iter_canonical_bags(
    parquet_path: str,
    split_json: str,
    verbose: bool = True,
) -> Iterator[tuple[str, str, np.ndarray, float]]:
    """Waveform variant: (pid, group, concat_griffinlim_waveform, dur_sec).

    For speech FMs (WavLM/HuBERT/Whisper) and YAMNet. Sibling recordings'
    Griffin-Lim waveforms are concatenated in sorted member order.
    """
    for pid, group, specs in _iter_pid_group_specs(parquet_path, split_json, verbose):
        wav = np.concatenate(
            [linear_spectrogram_to_waveform(s) for s in specs]
        ).astype(np.float32)
        yield pid, group, wav, len(wav) / SAMPLE_RATE


def iter_canonical_bags_spec(
    parquet_path: str,
    split_json: str,
    verbose: bool = True,
) -> Iterator[tuple[str, str, np.ndarray, float]]:
    """Spectrogram variant: (pid, group, concat_linear_spectrogram, dur_sec).

    For AudioSet FMs (SSAST/AST/AudioMAE/BEATs) that compute mel directly from
    the linear spectrogram, avoids the spec→GL→wav→mel double degradation.
    Sibling spectrograms concatenated along the time axis (axis=1; spec is
    (n_freq, n_time)). Duration ≈ n_time × LINEAR_HOP_SEC.
    """
    for pid, group, specs in _iter_pid_group_specs(parquet_path, split_json, verbose):
        spec = np.concatenate(specs, axis=1).astype(np.float32)
        yield pid, group, spec, spec.shape[1] * LINEAR_HOP_SEC
