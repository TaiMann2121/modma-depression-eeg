import mne

def preprocess_raw(
    raw,
    *,
    l_freq=1.0,
    h_freq=45.0,
    notch_freq=50.0,
    ref="average",
):
    """
    Basic preprocessing for resting-state EEG.

    Steps:
    - Notch filter (line noise)
    - Band-pass filter
    - Re-reference

    Returns a NEW Raw object.
    """
    raw = raw.copy()

    # Notch filter (EU power line = 50 Hz; harmless if absent)
    if notch_freq is not None:
        raw.notch_filter(notch_freq, verbose=False)

    # Band-pass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw = raw.copy().crop(tmin=1.0, tmax=raw.times[-1] - 1.0) # remove edge artifacts
    # Re-reference
    if ref == "average":
        raw.set_eeg_reference("average", verbose=False)
    
    return raw

def epoch_raw(
    raw,
    *,
    epoch_len=2.0,
    overlap=0.0,
):
    """
    Split continuous raw EEG into fixed-length epochs.

    Parameters
    ----------
    epoch_len : float
        Length of each epoch in seconds (2–4s typical)
    overlap : float
        Overlap between epochs in seconds
    """
    events = mne.make_fixed_length_events(
        raw,
        id=1,
        duration=epoch_len,
        overlap=overlap,
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"rest": 1},
        tmin=0.0,
        tmax=epoch_len - 1/raw.info["sfreq"],
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs
