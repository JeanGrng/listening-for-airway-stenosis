# Third-party code, models and references

This repository is MIT licensed. It does not vendor third-party source trees, but parts of it are
adapted from, or load weights published by, the projects below. Their terms apply to their own code
and weights.

## Adapted code

| File | Adapted from | Licence |
|---|---|---|
| `src/embeddings/compute_embeddings_ssast.py`, `..._ssast_patch400.py` | SSAST (YuanGongND), state-dict remapping onto Hugging Face `ASTModel` | BSD-3-Clause |
| `src/embeddings/compute_embeddings_openbeats.py` | standalone BEATs / OpenBEATs encoder, reimplemented on PyTorch + huggingface_hub | see BEATs (MIT) and OpenBEATs |

## Pretrained models

Loaded at runtime from Hugging Face or TensorFlow Hub, never redistributed here: WavLM-Large,
HuBERT-Large, Whisper-Large-v3, AST, AudioMAE, SSAST, OpenBEATs, YAMNet. The SSAST checkpoints must
be downloaded manually from the SSAST repository.

## MIL architectures

Implemented in `src/mil_models.py` and `src/run_mil.py` from their papers:

- TransMIL, Shao et al., NeurIPS 2021. Simplified here for bags of 16 instances: PPEG and Nystrom
  attention are removed in favour of standard multi-head attention over 17 tokens (CLS + 16 tasks).
- DSMIL, Li et al., CVPR 2021.
- Gated-attention MIL, Ilse et al., ICML 2018.
- Set Transformer, Lee et al., ICML 2019.

## Dataset

Bridge2AI-Voice (Bensoussan et al., PhysioNet). De-identified, released under credentialed access and the Bridge2AI Voice Registered Access License.
No dataset content is redistributed in this repository; the shipped result files contain aggregate metrics only.
