from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import re
import numpy as np
import scipy.io as sio
import mne

def load_metadata(xlsx_path):
    import pandas as pd

    meta = pd.read_excel(xlsx_path, dtype={"subject id": str})
    meta.columns = [c.strip() for c in meta.columns]

    # Drop useless "Unnamed" columns that come from side notes in the sheet
    meta = meta.loc[:, ~meta.columns.str.startswith("Unnamed")]

    # Create normalized ID (8 digits, preserves leading zeros)
    meta["subject_id"] = meta["subject id"].astype(str).str.strip().str.zfill(8)

    # Optional: drop the original 'subject id' column to avoid duplication
    # (I usually keep it for traceability, but you can drop it.)
    meta = meta.drop(columns=["subject id"])

    # Normalize diagnosis column
    if "type" in meta.columns:
        meta["type"] = meta["type"].astype(str).str.strip().str.upper()

    return meta


def extract_subject_id_from_filename(filename: str) -> str:
    m = re.match(r"(\d{8})", filename)
    if not m:
        raise ValueError(f"Could not parse 8-digit subject id from: {filename}")
    return m.group(1)

def pick_eeg_key(mat: dict) -> str:
    ignore = {"samplingRate", "Impedances_0"}
    candidates = [k for k in mat.keys() if not k.startswith("__") and k not in ignore]
    if not candidates:
        raise ValueError("Could not find EEG data key in .mat file.")
    return candidates[0]

def load_subject(mat_path, meta=None, montage="GSN-HydroCel-128"):
    """
    Load one MODMA subject .mat into MNE Raw.
    Returns: raw, label, info_dict
    label: 0=HC, 1=MDD, None if no meta match
    """
    mat_path = Path(mat_path)
    subject_id = extract_subject_id_from_filename(mat_path.name)

    mat = sio.loadmat(mat_path)

    # Sampling rate
    sfreq = float(np.asarray(mat["samplingRate"]).squeeze()) if "samplingRate" in mat else 250.0

    # EEG data array key
    eeg_key = pick_eeg_key(mat)
    eeg = np.asarray(mat[eeg_key], dtype=np.float64)

    # Ensure 2D
    if eeg.ndim != 2:
        raise ValueError(f"Expected 2D EEG array, got shape {eeg.shape}")

    # Make (channels, samples)
    if eeg.shape[0] not in (128, 129) and eeg.shape[1] in (128, 129):
        eeg = eeg.T

    # Drop 129th channel if it's all zeros
    if eeg.shape[0] == 129 and np.allclose(eeg[-1, :], 0.0):
        eeg = eeg[:-1, :]

    if eeg.shape[0] != 128:
        raise ValueError(f"Expected 128 channels after cleanup, got {eeg.shape[0]}")
    
    eeg = eeg * 1e-6  # microvolts -> volts
    
    # Create Raw
    ch_names = [f"EEG{i+1}" for i in range(128)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg, info, verbose=False)
    raw.set_montage(montage, on_missing="ignore")

    # Label + metadata
    label = None
    extra = {"subject_id": subject_id, "sfreq": sfreq, "mat_key": eeg_key, "file": mat_path.name}

    if meta is not None:
        row = meta.loc[meta["subject_id"] == subject_id]
        if not row.empty:
            t = str(row.iloc[0]["type"]).strip().upper()
            extra["type"] = t
            extra["PHQ-9"] = row.iloc[0].get("PHQ-9", None)
            label = 0 if t == "HC" else 1 if t == "MDD" else None

    return raw, label, extra

