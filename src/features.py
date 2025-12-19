import numpy as np
import pandas as pd
import mne


BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}


def bandpower_features(epochs, bands=BANDS, method="welch"):
    """
    Returns a dict of bandpower features aggregated across epochs.

    Output keys look like: bp_alpha_EEG17
    Values are average power in that band (linear units) across epochs.
    """
    psd = epochs.compute_psd(
        method=method,
        fmin=min(b[0] for b in bands.values()),
        fmax=max(b[1] for b in bands.values()),
        verbose=False,
    )
    data = psd.get_data()         # (n_epochs, n_ch, n_freqs)
    freqs = psd.freqs
    ch_names = epochs.ch_names

    feats = {}
    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs < fmax)

        # Integrate PSD over band (approx by mean * bandwidth)
        band_power = data[:, :, idx].mean(axis=2)  # (n_epochs, n_ch)
        band_power = band_power.mean(axis=0)       # average across epochs -> (n_ch,)

        for ci, ch in enumerate(ch_names):
            feats[f"bp_{band_name}_{ch}"] = float(band_power[ci])

    return feats


def relative_bandpower_features(epochs, bands=BANDS, method="welch"):
    """
    Relative bandpower = band_power / total_power(1-45).
    """
    psd = epochs.compute_psd(method=method, fmin=1, fmax=45, verbose=False)
    data = psd.get_data()   # (n_epochs, n_ch, n_freqs)
    freqs = psd.freqs
    ch_names = epochs.ch_names

    total = data.mean(axis=2).mean(axis=0)  # avg over freqs then epochs -> (n_ch,)

    feats = {}
    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs < fmax)
        band = data[:, :, idx].mean(axis=2).mean(axis=0)  # (n_ch,)
        rel = band / (total + 1e-12)

        for ci, ch in enumerate(ch_names):
            feats[f"rbp_{band_name}_{ch}"] = float(rel[ci])

    return feats


def alpha_asymmetry(epochs, left=None, right=None, use_relative=True):
    """
    Simple alpha asymmetry feature:
    log(alpha_right) - log(alpha_left)

    If you don't have channel locations mapped yet, we approximate:
    - left = EEG1..EEG64
    - right = EEG65..EEG128
    (You can replace later with real left/right frontal sets.)

    Returns a small dict of asymmetry features.
    """
    ch_names = epochs.ch_names

    if left is None:
        left = ch_names[:64]
    if right is None:
        right = ch_names[64:]

    # get alpha power per channel
    if use_relative:
        feats = relative_bandpower_features(epochs, bands={"alpha": (8, 12)})
        alpha = {k.split("_")[-1]: v for k, v in feats.items()}  # ch -> val
    else:
        feats = bandpower_features(epochs, bands={"alpha": (8, 12)})
        alpha = {k.split("_")[-1]: v for k, v in feats.items()}

    left_vals = np.array([alpha[ch] for ch in left if ch in alpha])
    right_vals = np.array([alpha[ch] for ch in right if ch in alpha])

    left_mean = float(np.mean(left_vals))
    right_mean = float(np.mean(right_vals))

    asym = float(np.log(right_mean + 1e-12) - np.log(left_mean + 1e-12))

    return {
        "alpha_left_mean": left_mean,
        "alpha_right_mean": right_mean,
        "alpha_asym_log_right_minus_left": asym,
    }


def featurize_subject(epochs_clean):
    """
    Combine multiple feature families into one flat dict.
    """
    feats = {}
    feats.update(relative_bandpower_features(epochs_clean))
    feats.update(alpha_asymmetry(epochs_clean))
    return feats


def dict_to_row(feats, subject_id=None, label=None):
    row = dict(feats)
    if subject_id is not None:
        row["subject_id"] = subject_id
    if label is not None:
        row["label"] = label
    return row
