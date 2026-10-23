
from pathlib import Path
import json

import numpy as np
from scipy.io import loadmat


# ============================================================
# EXPERIMENT 13D — CHO2017 MATLAB PROTOCOL AUDIT
# ============================================================
# Research purpose:
# Inspect original MATLAB metadata, trial structure,
# event markers, and bad-trial information.
#
# No preprocessing.
# No model fitting.
# No classification.
# No target-label tuning.
# ============================================================


# STEP 1: File paths

MAT_FILE = Path(
    r"C:\Users\laptopwala\mne_data\MNE-gigadb-data"
    r"\gigadb-datasets\live\pub"
    r"\10.5524\100001_101000\100295"
    r"\mat_data\s01.mat"
)

OUTPUT_DIR = Path("experiment13d_protocol")
OUTPUT_DIR.mkdir(exist_ok=True)


# STEP 2: Helper functions

def describe_value(value):
    """Summarize a MATLAB value without printing huge arrays."""

    if hasattr(value, "_fieldnames"):
        return {
            "type": "MATLAB structure",
            "fields": list(value._fieldnames or [])
        }

    arr = np.asarray(value)

    result = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size)
    }

    if arr.size <= 30 and arr.dtype != object:
        result["values"] = arr.tolist()

    return result


def inspect_matlab_structure(obj, indent=0):
    """Recursively inspect nested MATLAB structures."""

    prefix = " " * indent

    if hasattr(obj, "_fieldnames"):
        fields = obj._fieldnames or []

        print(f"{prefix}Structure fields: {fields}")

        result = {}

        for field in fields:
            value = getattr(obj, field)

            print(f"\n{prefix}Field: {field}")

            result[field] = inspect_matlab_structure(
                value, indent + 4
            )

        return result

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        print(f"{prefix}Object array shape: {obj.shape}")

        return {
            "shape": list(obj.shape),
            "elements": [
                inspect_matlab_structure(item, indent + 4)
                for item in obj.flat
            ]
        }

    arr = np.asarray(obj)

    print(f"{prefix}Shape: {arr.shape}")
    print(f"{prefix}Dtype: {arr.dtype}")

    if arr.size <= 100:
        print(f"{prefix}Values: {arr}")
    else:
        print(
            f"{prefix}First 20 values: "
            f"{arr.ravel()[:20]}"
        )

    return describe_value(obj)


# STEP 3: Load original MATLAB file

def main():

    print("=" * 65)
    print("EXPERIMENT 13D — CHO2017 MATLAB PROTOCOL AUDIT")
    print("Original MATLAB data inspection")
    print("No preprocessing, model fitting, or target-label tuning.")
    print("=" * 65)

    if not MAT_FILE.exists():
        raise FileNotFoundError(
            f"Cho2017 Subject 1 file not found:\n{MAT_FILE}"
        )

    print(f"\nReading:\n{MAT_FILE}")

    data = loadmat(
        MAT_FILE,
        struct_as_record=False,
        squeeze_me=True
    )

    if "eeg" not in data:
        raise KeyError(
            "Expected 'eeg' structure not found. "
            f"Available keys: {list(data.keys())}"
        )

    eeg = data["eeg"]

    fields = getattr(eeg, "_fieldnames", [])

    print("\nMATLAB EEG FIELDS:")
    print(fields)

    report = {
        "dataset": "Cho2017",
        "subject": 1,
        "matlab_file": str(MAT_FILE),
        "fields": fields,
        "inspection_only": True
    }

    # STEP 4: Sampling frequency

    sampling_rate = float(
        np.asarray(eeg.srate).squeeze()
    )

    print(f"\nSampling frequency: {sampling_rate} Hz")

    report["sampling_rate_hz"] = sampling_rate

    # STEP 5: Imagery arrays

    left = np.asarray(eeg.imagery_left)
    right = np.asarray(eeg.imagery_right)

    print("\nIMAGERY ARRAYS")
    print("Left imagery shape:", left.shape)
    print("Right imagery shape:", right.shape)

    report["imagery_left_shape"] = list(left.shape)
    report["imagery_right_shape"] = list(right.shape)

    # STEP 6: Number of trials

    n_trials = int(
        np.asarray(eeg.n_imagery_trials).squeeze()
    )

    print("\nReported imagery trials per class:", n_trials)

    report["reported_trials_per_class"] = n_trials

    if left.ndim == 2 and n_trials > 0:
        samples_per_trial = left.shape[1] / n_trials
        seconds_per_trial = samples_per_trial / sampling_rate

        print("Samples per trial:", samples_per_trial)
        print("Seconds per trial:", seconds_per_trial)

        report["samples_per_trial"] = samples_per_trial
        report["seconds_per_trial"] = seconds_per_trial

    # STEP 7: Imagery event markers

    events = np.asarray(eeg.imagery_event).ravel()

    event_indices = np.flatnonzero(events)

    event_values = events[event_indices]

    print("\nIMAGERY EVENT MARKERS")
    print("Event vector length:", len(events))
    print("Nonzero event count:", len(event_indices))
    print("First 12 event indices:", event_indices[:12])
    print("First 12 event values:", event_values[:12])

    event_times = event_indices / sampling_rate

    print("First 12 event times (seconds):")
    print(event_times[:12])

    if len(event_indices) > 1:
        event_intervals = np.diff(event_indices) / sampling_rate

        print("First 12 event intervals (seconds):")
        print(event_intervals[:12])

        report["first_event_intervals_s"] = (
            event_intervals[:12].tolist()
        )

    report["event_count"] = int(len(event_indices))
    report["first_event_indices"] = event_indices[:12].tolist()
    report["first_event_times_s"] = event_times[:12].tolist()
    report["first_event_values"] = event_values[:12].tolist()

    # STEP 8: Inspect bad-trial metadata

    print("\n" + "=" * 65)
    print("BAD TRIAL METADATA — DETAILED INSPECTION")
    print("=" * 65)

    if hasattr(eeg, "bad_trial_indices"):

        bad = eeg.bad_trial_indices

        print("\nPython type:", type(bad))

        print(
            "Top-level fields:",
            getattr(bad, "_fieldnames", [])
        )

        bad_report = inspect_matlab_structure(bad)

        report["bad_trial_indices"] = bad_report

    else:
        print("No bad_trial_indices field found.")

        report["bad_trial_indices"] = {
            "status": "field_not_found"
        }

    # STEP 9: Save JSON report

    output_file = (
        OUTPUT_DIR / "13d_matlab_protocol_report.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=4,
            default=str
        )

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print("=" * 65)

    print("\nSaved report:")
    print(output_file.resolve())

    print("\nResearch status:")
    print("- Original trial arrays inspected.")
    print("- Event structure inspected.")
    print("- Bad-trial metadata structure inspected.")
    print("- No preprocessing choices finalized.")
    print("- No classifier training performed.")


if __name__ == "__main__":
    main()
