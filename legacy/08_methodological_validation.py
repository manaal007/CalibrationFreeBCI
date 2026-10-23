
"""
Calibration-Free BCI
Experiment 08: Methodological Validation

Objectives:
1. Verify subject-level evaluation records.
2. Check EEG event labels and trial counts.
3. Examine sensorimotor covariance structure.
4. Compare covariance variability with decoding performance.

This is an exploratory quality-control analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import moabb

from huggingface_hub import snapshot_download
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

SUBJECTS = list(range(1, 10))
CHANNELS = ["C3", "Cz", "C4"]

moabb.set_download_provider("upstream")
moabb.set_log_level("WARNING")

print("=" * 55)
print("EXPERIMENT 08: METHODOLOGICAL VALIDATION")
print("=" * 55)


# --------------------------------------------------
# 2. CHECK EXISTING RESULTS
# --------------------------------------------------

results = pd.read_csv(
    "full_cross_subject_results.csv"
)

required = {"subject", "session", "score"}

assert required.issubset(results.columns)
assert results["subject"].nunique() == 9
assert len(results) == 18
assert results["score"].between(0, 1).all()

print("\nRESULTS INTEGRITY")
print("Participants:", results["subject"].nunique())
print("Evaluation rows:", len(results))
print("Sessions per subject:")
print(results.groupby("subject")["session"].nunique())

print("\nAll result-format checks passed.")


# --------------------------------------------------
# 3. LOAD CACHED EEG RECORDINGS
# --------------------------------------------------

print("\nLoading EEG recordings...")

path = snapshot_download(
    repo_id="EEGDash/eeg-mi-datasets",
    repo_type="dataset",
    allow_patterns=[
        f"BNCI2014_001/*/A{i:02d}*"
        for i in SUBJECTS
    ],
)

os.environ["MNE_DATASETS_BNCI_PATH"] = os.path.join(
    path, "BNCI2014_001"
)

dataset = BNCI2014_001()
dataset.subject_list = SUBJECTS

paradigm = LeftRightImagery(
    fmin=8,
    fmax=30,
    tmin=0.5,
    tmax=3.5
)

X, labels, metadata = paradigm.get_data(
    dataset=dataset,
    subjects=SUBJECTS
)

print("\nEEG ARRAY INFORMATION")
print("Shape:", X.shape)
print("Labels:", np.unique(labels))
print("All values finite:", np.isfinite(X).all())

assert np.isfinite(X).all()
assert X.shape[1] == 22

print("\nTRIAL COUNTS")
print(
    metadata.groupby(
        ["subject", "session"]
    ).size().to_string()
)


# --------------------------------------------------
# 4. EXAMINE SENSORIMOTOR COVARIANCE
# --------------------------------------------------

channel_names = [
    ch.upper() for ch in dataset.channels
] if getattr(dataset, "channels", None) else []

# The BNCI2014-001 EEG montage uses these indices
# for C3, Cz, and C4.
sensor_indices = [7, 9, 11]

covariance_summaries = []

for subject in SUBJECTS:

    mask = metadata["subject"].to_numpy() == subject
    subject_epochs = X[mask]

    # Shape: trials x 3 channels x samples
    signals = subject_epochs[:, sensor_indices, :]

    # Average trial covariance matrices
    matrices = np.array([
        np.cov(epoch)
        for epoch in signals
    ])

    mean_cov = matrices.mean(axis=0)

    # Scale-invariant normalized covariance
    trace = np.trace(mean_cov)
    normalized_cov = mean_cov / trace

    covariance_summaries.append({
        "subject": subject,
        "covariance": normalized_cov,
        "trials": len(signals)
    })


# --------------------------------------------------
# 5. COMPARE EACH SUBJECT WITH THE OTHERS
# --------------------------------------------------

records = []

for item in covariance_summaries:

    subject = item["subject"]
    matrix = item["covariance"]

    other_matrices = [
        x["covariance"]
        for x in covariance_summaries
        if x["subject"] != subject
    ]

    reference = np.mean(other_matrices, axis=0)

    # Frobenius distance between matrices
    distance = np.linalg.norm(
        matrix - reference,
        ord="fro"
    )

    records.append({
        "subject": subject,
        "covariance_distance": distance,
        "n_trials": item["trials"]
    })

cov_df = pd.DataFrame(records)


# --------------------------------------------------
# 6. MERGE WITH DECODING PERFORMANCE
# --------------------------------------------------

subject_auc = (
    results.groupby("subject")["score"]
    .mean()
    .reset_index()
    .rename(columns={"score": "mean_auc"})
)

analysis = cov_df.merge(
    subject_auc,
    on="subject",
    validate="one_to_one"
)

print("\nCOVARIANCE AND DECODING PERFORMANCE")
print(analysis.to_string(index=False))

analysis.to_csv(
    "08_covariance_analysis.csv",
    index=False
)


# --------------------------------------------------
# 7. EXPLORATORY ASSOCIATION
# --------------------------------------------------

correlation = analysis[
    ["covariance_distance", "mean_auc"]
].corr().iloc[0, 1]

print("\nExploratory Pearson correlation:", correlation)

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    analysis["covariance_distance"],
    analysis["mean_auc"],
    s=80
)

for _, row in analysis.iterrows():
    ax.annotate(
        f"S{int(row['subject'])}",
        (
            row["covariance_distance"],
            row["mean_auc"]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

ax.axhline(
    0.5,
    color="gray",
    linestyle="--",
    label="Chance-level ROC-AUC"
)

ax.set_xlabel(
    "Normalized sensorimotor covariance distance"
)
ax.set_ylabel("Cross-subject ROC-AUC")
ax.set_title(
    "EEG Covariance Variability and BCI Generalization"
)

ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()
fig.savefig(
    "08_covariance_vs_performance.png",
    dpi=200
)
plt.close(fig)

print("\nSaved:")
print("08_covariance_analysis.csv")
print("08_covariance_vs_performance.png")
print("\nExperiment 08 completed!")
