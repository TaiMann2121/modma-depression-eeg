from pathlib import Path
import pandas as pd

from src.load_data import load_subject, load_metadata
from src.preprocess import preprocess_raw, epoch_raw
from src.features import featurize_subject, dict_to_row


def build_dataset(data_dir, meta_path, epoch_len=2.0):
    meta = load_metadata(meta_path)
    rows = []

    mat_files = sorted(Path(data_dir).glob("*.mat"))

    for mat_path in mat_files:
        try:
            raw, label, info = load_subject(mat_path, meta)

            raw_clean = preprocess_raw(raw)
            epochs = epoch_raw(raw_clean, epoch_len=epoch_len)

            if len(epochs) < 20:
                continue  # skip very short or broken recordings

            feats = featurize_subject(epochs)
            row = dict_to_row(
                feats,
                subject_id=info["subject_id"],
                label=label,
            )
            rows.append(row)

        except Exception as e:
            print(f"Skipping {mat_path.name}: {e}")

    df = pd.DataFrame(rows)
    return df
