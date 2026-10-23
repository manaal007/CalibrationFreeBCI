
"""
PROJECT: Calibration-Free Brain-Computer Interface

Experiment 07:
Full Cross-Subject Motor Imagery Classification

Dataset: BNCI2014-001
Participants: 9
Task: Left-hand vs Right-hand Motor Imagery
Model: CSP + Linear Discriminant Analysis
Evaluation: Leave-One-Subject-Out
Metric: ROC-AUC

Research objective:
Evaluate whether an EEG classifier trained on
other participants can generalize to a previously
unseen participant without individual calibration.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import moabb

from huggingface_hub import snapshot_download
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import CrossSubjectEvaluation

from mne.decoding import CSP
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis
)
from sklearn.pipeline import make_pipeline


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

moabb.set_log_level("WARNING")

SUBJECTS = list(range(1, 10))

print("=" * 55)
print("CALIBRATION-FREE BCI")
print("FULL CROSS-SUBJECT EVALUATION")
print("=" * 55)

print("Participants:", SUBJECTS)
print("Classification: Left hand vs Right hand")
print("Model: CSP + LDA")
print("Evaluation: Leave-One-Subject-Out")


# --------------------------------------------------
# 2. DOWNLOAD / LOAD EEG DATA
# --------------------------------------------------

print("\nPreparing EEG dataset...")

path = snapshot_download(
    repo_id="EEGDash/eeg-mi-datasets",
    repo_type="dataset",
    allow_patterns=[
        f"BNCI2014_001/*/A{i:02d}*"
        for i in SUBJECTS
    ]
)

os.environ["MNE_DATASETS_BNCI_PATH"] = os.path.join(
    path,
    "BNCI2014_001"
)

moabb.set_download_provider("upstream")

dataset = BNCI2014_001()
dataset.subject_list = SUBJECTS

print("\nDataset ready!")
print("Number of participants:", len(SUBJECTS))


# --------------------------------------------------
# 3. MOTOR IMAGERY PARADIGM
# --------------------------------------------------

paradigm = LeftRightImagery(
    fmin=8,
    fmax=30,
    tmin=0.5,
    tmax=3.5
)

print("\nEEG preprocessing configuration:")
print("Frequency band: 8-30 Hz")
print("Epoch window: 0.5-3.5 seconds")
print("Classes: Left hand and Right hand")


# --------------------------------------------------
# 4. CSP + LDA CLASSIFIER
# --------------------------------------------------

pipeline = make_pipeline(
    CSP(
        n_components=4,
        reg="ledoit_wolf",
        log=True
    ),
    LinearDiscriminantAnalysis()
)

pipelines = {
    "CSP_LDA": pipeline
}


# --------------------------------------------------
# 5. CROSS-SUBJECT EVALUATION
# --------------------------------------------------

print("\nStarting leave-one-subject-out evaluation...")
print("This may take several minutes.\n")

evaluation = CrossSubjectEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
    overwrite=True,
    suffix="cross_subject_full"
)

results = evaluation.process(pipelines)


# --------------------------------------------------
# 6. DISPLAY AND SAVE RAW RESULTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("FULL CROSS-SUBJECT RESULTS")
print("=" * 55)

print(
    results[
        ["subject", "session", "score", "pipeline"]
    ].to_string(index=False)
)

results.to_csv(
    "full_cross_subject_results.csv",
    index=False
)

print("\nRaw results saved.")


# --------------------------------------------------
# 7. SUBJECT-LEVEL PERFORMANCE
# --------------------------------------------------

subject_results = (
    results.groupby("subject")["score"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

subject_results.columns = [
    "Subject",
    "Mean_ROC_AUC",
    "Session_SD",
    "Number_of_Scores"
]

print("\n" + "=" * 55)
print("SUBJECT-LEVEL PERFORMANCE")
print("=" * 55)

print(subject_results.to_string(index=False))

subject_results.to_csv(
    "subject_level_performance.csv",
    index=False
)


# --------------------------------------------------
# 8. OVERALL PERFORMANCE
# --------------------------------------------------

scores = subject_results["Mean_ROC_AUC"]

mean_auc = scores.mean()
std_auc = scores.std(ddof=1)
min_auc = scores.min()
max_auc = scores.max()

print("\n" + "=" * 55)
print("OVERALL PERFORMANCE")
print("=" * 55)

print(f"Mean ROC-AUC: {mean_auc:.4f}")
print(f"Subject-level SD: {std_auc:.4f}")
print(f"Minimum ROC-AUC: {min_auc:.4f}")
print(f"Maximum ROC-AUC: {max_auc:.4f}")

print(
    f"Participants evaluated: {len(subject_results)}"
)


# --------------------------------------------------
# 9. VISUALIZE CROSS-SUBJECT PERFORMANCE
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    subject_results["Subject"].astype(str),
    subject_results["Mean_ROC_AUC"]
)

ax.axhline(
    y=0.5,
    linestyle="--",
    color="red",
    label="Chance-level ROC-AUC (0.50)"
)

ax.axhline(
    y=mean_auc,
    linestyle="--",
    color="green",
    label=f"Mean ROC-AUC ({mean_auc:.3f})"
)

ax.set_xlabel("Held-Out Participant")
ax.set_ylabel("ROC-AUC")

ax.set_title(
    "Calibration-Free BCI: "
    "Cross-Subject Generalization"
)

ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    "full_cross_subject_performance.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\nPerformance visualization saved.")


# --------------------------------------------------
# 10. EXPERIMENT COMPLETED
# --------------------------------------------------

print("\n" + "=" * 55)
print("EXPERIMENT 07 COMPLETED SUCCESSFULLY")
print("=" * 55)

print("\nGenerated files:")
print("1. full_cross_subject_results.csv")
print("2. subject_level_performance.csv")
print("3. full_cross_subject_performance.png")
