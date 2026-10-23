
"""
Calibration-Free BCI
Experiment 09: Class-Conditional Sensorimotor Covariance

Research question:
Do individual differences in left/right motor-imagery
covariance patterns relate to cross-subject decoding?

Outputs:
09_class_conditional_results.csv
09_class_conditional_covariance.png
09_class_covariance_matrices.png

This is an exploratory, retrospective neuroscience
analysis, not a predictive validation experiment.
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
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=FutureWarning)

SUBJECTS = list(range(1, 10))
SENSORS = ["C3", "Cz", "C4"]

moabb.set_download_provider("upstream")
moabb.set_log_level("WARNING")

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("EXPERIMENT 09: CLASS-CONDITIONAL COVARIANCE")
print("=" * 60)

path = snapshot_download(
    repo_id="EEGDash/eeg-mi-datasets",
    repo_type="dataset",
    allow_patterns=[
        f"BNCI2014_001/*/A{i:02d}*"
        for i in SUBJECTS
    ]
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

X, y, metadata = paradigm.get_data(
    dataset=dataset,
    subjects=SUBJECTS
)

metadata = metadata.reset_index(drop=True)

assert X.shape[1] == 22
assert np.isfinite(X).all()

print("EEG shape:", X.shape)
print("Classes:", np.unique(y))

# BNCI2014-001 channel order verified in Experiment 02.
SENSOR_INDICES = [7, 9, 11]

# --------------------------------------------------
# 2. CALCULATE CLASS-CONDITIONAL COVARIANCE
# --------------------------------------------------

def mean_normalized_covariance(epochs):
    """Average trial covariance, then normalize by trace."""
    matrices = np.array([
        np.cov(epoch)
        for epoch in epochs
    ])

    mean_cov = matrices.mean(axis=0)
    return mean_cov / np.trace(mean_cov)


class_covariances = {}
records = []

for subject in SUBJECTS:

    subject_mask = (
        metadata["subject"].to_numpy() == subject
    )

    subject_X = X[subject_mask][:, SENSOR_INDICES, :]
    subject_y = y[subject_mask]

    left = subject_X[subject_y == "left_hand"]
    right = subject_X[subject_y == "right_hand"]

    assert len(left) > 0 and len(right) > 0

    cov_left = mean_normalized_covariance(left)
    cov_right = mean_normalized_covariance(right)

    difference = cov_left - cov_right

    # Frobenius norm measures magnitude of
    # class-conditional covariance separation.
    separation = np.linalg.norm(
        difference,
        ord="fro"
    )

    class_covariances[subject] = {
        "left": cov_left,
        "right": cov_right,
        "difference": difference
    }

    records.append({
        "subject": subject,
        "left_trials": len(left),
        "right_trials": len(right),
        "class_covariance_separation": separation
    })

analysis = pd.DataFrame(records)

# --------------------------------------------------
# 3. ALIGNMENT WITH OTHER PARTICIPANTS
# --------------------------------------------------

alignment_records = []

for subject in SUBJECTS:

    target = class_covariances[subject]["difference"]

    source_patterns = [
        class_covariances[s]["difference"]
        for s in SUBJECTS
        if s != subject
    ]

    source_mean = np.mean(source_patterns, axis=0)

    target_vector = target.ravel()
    source_vector = source_mean.ravel()

    denominator = (
        np.linalg.norm(target_vector)
        * np.linalg.norm(source_vector)
    )

    alignment = (
        np.dot(target_vector, source_vector) / denominator
        if denominator > 1e-12
        else np.nan
    )

    alignment_records.append({
        "subject": subject,
        "source_pattern_alignment": alignment
    })

alignment_df = pd.DataFrame(alignment_records)

analysis = analysis.merge(
    alignment_df,
    on="subject",
    validate="one_to_one"
)

# --------------------------------------------------
# 4. MERGE WITH CROSS-SUBJECT ROC-AUC
# --------------------------------------------------

baseline = pd.read_csv(
    "full_cross_subject_results.csv"
)

performance = (
    baseline.groupby("subject")["score"]
    .mean()
    .reset_index()
    .rename(columns={"score": "mean_auc"})
)

analysis = analysis.merge(
    performance,
    on="subject",
    validate="one_to_one"
)

analysis.to_csv(
    "09_class_conditional_results.csv",
    index=False
)

print("\nCLASS-CONDITIONAL RESULTS")
print(analysis.to_string(index=False))

# --------------------------------------------------
# 5. EXPLORATORY CORRELATIONS
# --------------------------------------------------

print("\nEXPLORATORY ASSOCIATIONS")

for variable in [
    "class_covariance_separation",
    "source_pattern_alignment"
]:
    valid = analysis[
        [variable, "mean_auc"]
    ].dropna()

    if len(valid) >= 3:
        r, p = pearsonr(
            valid[variable],
            valid["mean_auc"]
        )

        print(
            f"{variable}: r={r:.4f}, "
            f"p={p:.4f}, n={len(valid)}"
        )

# --------------------------------------------------
# 6. VISUALIZE ASSOCIATIONS
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

variables = [
    "class_covariance_separation",
    "source_pattern_alignment"
]

titles = [
    "Class-Specific Covariance Separation",
    "Alignment With Source Participants"
]

for ax, variable, title in zip(
    axes, variables, titles
):
    ax.scatter(
        analysis[variable],
        analysis["mean_auc"],
        s=70
    )

    for _, row in analysis.iterrows():
        if np.isfinite(row[variable]):
            ax.annotate(
                f"S{int(row['subject'])}",
                (row[variable], row["mean_auc"]),
                xytext=(5, 5),
                textcoords="offset points"
            )

    ax.axhline(
        0.5,
        color="gray",
        linestyle="--"
    )

    ax.set_xlabel(variable.replace("_", " "))
    ax.set_ylabel("Cross-subject ROC-AUC")
    ax.set_title(title)
    ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    "09_class_conditional_covariance.png",
    dpi=200
)
plt.close(fig)

# --------------------------------------------------
# 7. VISUALIZE COVARIANCE MATRICES
# --------------------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(12, 11))

for ax, subject in zip(axes.flat, SUBJECTS):

    matrix = class_covariances[
        subject
    ]["difference"]

    im = ax.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-0.3,
        vmax=0.3
    )

    ax.set_title(f"Subject {subject}")
    ax.set_xticks(range(3), SENSORS)
    ax.set_yticks(range(3), SENSORS)

fig.suptitle(
    "Left Minus Right Motor-Imagery Covariance",
    fontsize=14
)

fig.colorbar(
    im,
    ax=axes.ravel().tolist(),
    shrink=0.65,
    label="Normalized covariance difference"
)

fig.savefig(
    "09_class_covariance_matrices.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

print("\nGenerated files:")
print("09_class_conditional_results.csv")
print("09_class_conditional_covariance.png")
print("09_class_covariance_matrices.png")

print("\nExperiment 09 completed successfully.")
