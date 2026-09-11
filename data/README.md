# Data

**No dataset content is distributed with this repository.**

Bridge2AI-Voice is de-identified (HIPAA Safe Harbor identifiers removed), but it is released under
**credentialed access** and the Bridge2AI Voice Registered Access License, with a data use agreement
signed by each user. Redistributing any part of it, including files derived from it such as a table
of participant identifiers and their labels, is outside what that agreement allows. That, rather than
any re-identification risk, is why the split files below are described here instead of shipped.

## Obtaining the dataset

Complete the PhysioNet credentialing process, sign the agreement, download a release, then:

```bash
export B2AI_ROOT=/path/to/physionet.org/files/b2ai-voice/<version>
```

The extraction scripts read `$B2AI_ROOT/features/torchaudio_spectrogram.parquet`, which holds one
linear magnitude spectrogram per recording, with `participant_id` and `task_name` columns.

## Split files you need to provide

Recreate the two files below from your own credentialed copy and place them in this directory.

### `binary_stenosis_split.json`

Patient-level partition used for the main experiments.

```json
{
  "train": [ { "participant_id": "<6-digit id>", "airway_stenosis": 0 }, ... ],
  "val":   [ ... ],
  "test":  [ ... ]
}
```

`airway_stenosis` is 1 for a patient with airway stenosis and 0 for a control. The three keys are a
historical artefact: the main protocol pools all of them and runs patient-level 5-fold
cross-validation over the union.

Cohort actually used in the paper, after four participants with truncated or unusable recordings are
dropped (`EXCLUDE_PIDS` in `src/b2ai_canonical.py`):

| | patients |
|---|---|
| airway stenosis | 134 |
| controls | 614 |
| **total** | **748** |

Note the class balance: roughly 18 % positive. Accuracy is therefore inflated by the control
majority, which is why AUROC and F1 are the informative metrics.

### `bridge2voice_data_split.json`

Sub-labels for the stenosis-positive patients only, used for Table 3.

```json
{
  "train": [ {
      "participant_id":      "<6-digit id>",
      "diagnosis_as_ds":     "<anatomical subtype, e.g. Subglottic Stenosis>",
      "diagnosis_as_ds_ods": "<severity: Mild | Moderate | Severe>",
      "diagnosis_as_as":     "<stridor: Yes | No>"
  }, ... ],
  "val": [ ... ], "test": [ ... ]
}
```

Sub-task cohorts: localization N = 134, severity N = 133, stridor N = 75.

## Recording tasks

`src/b2ai_canonical.py` defines the 23 raw task names that are kept and collapses numbered repeats
(for example `respiration-and-cough-breath-1` and `-2`) into **16 task groups**. Those groups, and
what each one asks the patient to do, are described in the main `README.md`.
