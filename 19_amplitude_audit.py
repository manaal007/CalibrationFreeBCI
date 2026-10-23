
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz"
]

# Read the existing scripts without changing them.
def import_script(filename, name):
    path = HERE / filename
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = import_script(
    "15_external_transfer_pilot.py", "pilot_audit"
)

validation = import_script(
    "15B_full_external_validation.py", "validation_audit"
)


def summarize_epochs(dataset, subject, epochs):
    """One summary per participant, in microvolts."""
    x = np.asarray(epochs, dtype=np.float64)

    if x.ndim != 3 or x.shape[1:] != (22, 501):
        raise ValueError(
            f"Unexpected shape for {dataset} {subject}: {x.shape}"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite EEG values detected")

    uv = x * 1e6
    absolute = np.abs(uv)

    return {
        "dataset": dataset,
        "subject": subject,
        "n_trials": len(x),
        "rms_uv": float(np.sqrt(np.mean(uv ** 2))),
        "median_abs_uv": float(np.median(absolute)),
        "p99_abs_uv": float(np.percentile(absolute, 99)),
        "max_abs_uv": float(np.max(absolute))
    }


# --------------------------------------
# SOURCE: BNCI2014-001
# --------------------------------------
print("Loading BNCI2014-001 source epochs...")

source_result = pilot.load_source()

if not isinstance(source_result, (tuple, list)):
    raise ValueError("Unexpected load_source() return type")

source_x = np.asarray(source_result[0])
source_y = np.asarray(source_result[1])

if source_x.shape[0] != len(source_y):
    raise ValueError("Source labels and epochs do not match")

if source_x.shape[0] != 2592:
    raise ValueError(
        f"Expected 2592 source epochs, got {len(source_x)}"
    )

rows = []

# Existing source loader concatenates subjects 1–9
# in order, 288 left/right epochs per subject.
for subject in range(1, 10):
    start = (subject - 1) * 288
    stop = start + 288

    rows.append(
        summarize_epochs(
            "BNCI2014-001",
            subject,
            source_x[start:stop]
        )
    )


# --------------------------------------
# TARGET: Cho2017
# --------------------------------------
print("Loading Cho2017 target epochs...")

from moabb.datasets import Cho2017

cho_names = pilot.get_cho_eeg_names()

if len(cho_names) != 64:
    raise ValueError("Expected 64 Cho EEG channel names")

indices = [cho_names.index(ch) for ch in CHANNELS]

folder = (
    Path.home() / "mne_data" / "MNE-gigadb-data"
    / "gigadb-datasets" / "live" / "pub"
    / "10.5524" / "100001_101000"
    / "100295" / "mat_data"
)

# Representative diagnostic participants,
# including those with unusually high exclusions.
target_subjects = [7, 12, 19, 20, 29, 34]

for subject in target_subjects:
    path = folder / f"s{subject:02d}.mat"

    if not path.exists():
        raise FileNotFoundError(path)

    result = validation.read_subject(path, indices)

    target_x = np.asarray(result[0])

    rows.append(
        summarize_epochs(
            "Cho2017",
            subject,
            target_x
        )
    )

    print(
        f"Cho subject {subject}: "
        f"{len(target_x)} retained trials"
    )


# --------------------------------------
# RESULTS
# --------------------------------------
df = pd.DataFrame(rows)

output = HERE / "experiment19_amplitude_audit"
output.mkdir(exist_ok=True)

df.to_csv(
    output / "participant_amplitudes.csv",
    index=False
)

print("\nPARTICIPANT-LEVEL RESULTS")
print(df.round(3).to_string(index=False))

summary = df.groupby("dataset")[
    ["rms_uv", "median_abs_uv", "p99_abs_uv"]
].agg(["median", "min", "max"])

print("\nDATASET SUMMARY")
print(summary.round(3).to_string())

source_median = df.loc[
    df.dataset == "BNCI2014-001", "rms_uv"
].median()

target_median = df.loc[
    df.dataset == "Cho2017", "rms_uv"
].median()

print("\nRMS COMPARISON")
print("BNCI median participant RMS:", round(source_median, 3))
print("Cho median participant RMS:", round(target_median, 3))
print("Cho / BNCI ratio:", round(target_median / source_median, 3))

print("\nAudit complete. Existing experiments unchanged.")
