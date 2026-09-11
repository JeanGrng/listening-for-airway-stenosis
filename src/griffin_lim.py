"""Griffin-Lim waveform reconstruction from linear spectrograms.

The B2AI Voice dataset stores linear spectrograms (201 freq bins, dB, hop=20ms, sr=16000).
Speech foundation models (WavLM, HuBERT, Whisper) require raw waveforms as input.
This module reconstructs approximate waveforms using the Griffin-Lim algorithm.

Usage:
    from griffin_lim import linear_spectrogram_to_waveform
    waveform = linear_spectrogram_to_waveform(spec_db)  # (n_samples,) numpy array
"""

import numpy as np
import torch
import torchaudio


def linear_spectrogram_to_waveform(
    spec_db: np.ndarray,
    n_fft: int = 400,
    hop_length: int = 320,
    sample_rate: int = 16000,
    n_iter: int = 32,
) -> np.ndarray:
    """Reconstruct a waveform from a linear spectrogram (dB) via Griffin-Lim.

    Args:
        spec_db: Linear spectrogram in dB, shape (n_freqs, T) where n_freqs = n_fft//2 + 1 = 201.
                 Stored with hop=20ms (320 samples at 16kHz).
        n_fft: FFT size used to compute the spectrogram (default 400 for 201 bins).
        hop_length: Hop length in samples (default 320 = 20ms at 16kHz).
        sample_rate: Sample rate (default 16000).
        n_iter: Number of Griffin-Lim iterations (default 32).

    Returns:
        Waveform as numpy array of shape (n_samples,), float32, normalized to [-1, 1].
    """
    spec_tensor = torch.tensor(spec_db, dtype=torch.float32)

    # dB → magnitude
    magnitude = torch.pow(10.0, spec_tensor / 20.0)

    # Griffin-Lim: magnitude → waveform (iterative phase estimation)
    waveform = torchaudio.functional.griffinlim(
        magnitude,
        window=torch.hann_window(n_fft),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        power=1.0,
        n_iter=n_iter,
        momentum=0.99,
        length=None,
        rand_init=True,
    )

    # Normalize to [-1, 1]
    waveform_np = waveform.numpy()
    peak = np.abs(waveform_np).max()
    if peak > 0:
        waveform_np = waveform_np / peak

    return waveform_np
