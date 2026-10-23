
"""
Calibration-Free BCI
Experiment 10: Riemannian Cross-Subject Decoding

Dataset: BNCI2014-001
Task: Left vs Right Motor Imagery

Models:
    1. CSP + LDA (reference baseline)
    2. Covariance + Riemannian MDM

Evaluation:
    Leave-one-subject-out (9 folds)
    Two held-out sessions per participant

Methodological safeguards:
- No held-out participant used in model fitting
- Fresh model in every fold
- Identical EEG epochs for both classifiers
- Training-only covariance regularization
- No test-label-based model selection
- Participant-level paired comparisons
- No hyperparameter tuning on held-out subjects

IMPORTANT:
This is an exploratory benchmark on a previously
examined dataset. Independent external validation
is still required before confirmatory claims.
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

from mne.decoding import CSP

from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis
)
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

from pyriemann.estimation import Covariances
from pyriemann.classification import MDM

warnings.filterwarnings("ignore", category=FutureWarning)

SUBJECTS = list(range(1, 10))
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

moabb.set_download_provider("upstream")
moabb.set_log_level("WARNING")

print("=" * 65)
print("EXPERIMENT 10: RIEMANNIAN CROSS-SUBJECT DECODING")
print("=" * 65)

# --------------------------------------------------
# 1. LOAD CACHED EEG DATA
# --------------------------------------------------

print("\nLoading cached BNCI2014-001 recordings...")

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

assert len(X) == len(y) == len(metadata)
assert X.shape[1] == 22
assert np.isfinite(X).all()

subject_ids = metadata["subject"].to_numpy()

assert set(subject_ids) == set(SUBJECTS)

classes = np.unique(y)
assert len(classes) == 2

positive_class = classes[1]

print("EEG array shape:", X.shape)
print("Participants:", len(SUBJECTS))
print("EEG channels:", X.shape[1])
print("Classes:", classes)

# --------------------------------------------------
# 2. DEFINE FIXED CLASSIFIERS
# --------------------------------------------------

def create_models():
    """
    Return fresh, unfitted pipelines.

    CSP + LDA:
        Learns spatial filters maximizing class
        variance differences.

    Riemannian MDM:
        Represents each trial by an SPD covariance
        matrix and classifies using distance to
        class-specific Riemannian means.
    """

    csp_lda = make_pipeline(
        CSP(
            n_components=4,
            reg="ledoit_wolf",
            log=True
        ),
        LinearDiscriminantAnalysis()
    )

    riemann_mdm = make_pipeline(
        Covariances(estimator="oas"),
        MDM(metric="riemann")
    )

    return {
        "CSP_LDA": csp_lda,
        "Riemannian_MDM": riemann_mdm
    }

# --------------------------------------------------
# 3. LEAVE-ONE-SUBJECT-OUT EVALUATION
# --------------------------------------------------

records = []

for held_out in SUBJECTS:

    print("\n" + "-" * 60)
    print(f"HELD-OUT PARTICIPANT: {held_out}")
    print("-" * 60)

    train_mask = subject_ids != held_out
    test_mask = subject_ids == held_out

    train_subjects = set(subject_ids[train_mask])
    test_subjects = set(subject_ids[test_mask])

    assert train_subjects.isdisjoint(test_subjects)
    assert test_subjects == {held_out}

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    test_metadata = (
        metadata.loc[test_mask]
        .reset_index(drop=True)
    )

    print("Training subjects:", sorted(train_subjects))
    print("Training trials:", len(X_train))
    print("Testing trials:", len(X_test))

    models = create_models()

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        # All learned model parameters are fitted
        # exclusively using training participants.
        model.fit(X_train, y_train)

        # Decision scores allow ROC-AUC calculation.
        # MDM decision_function uses class distances
        # transformed into class-discrimination scores.
        model_classes = model.classes_
        positive_index = list(model_classes).index(
            positive_class
        )

        for session in sorted(
            test_metadata["session"].unique()
        ):

            session_mask = (
                test_metadata["session"].to_numpy()
                == session
            )

            X_session = X_test[session_mask]
            y_session = y_test[session_mask]

            assert len(np.unique(y_session)) == 2

            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(
                    X_session
                )[:, positive_index]
            else:
                decision = model.decision_function(
                    X_session
                )

                if decision.ndim == 2:
                    scores = decision[:, positive_index]
                else:
                    scores = decision

                    # sklearn-style binary decision
                    # scores correspond to classes_[1].
                    assert positive_index == 1

            auc = roc_auc_score(
                (y_session == positive_class).astype(int),
                scores
            )

            records.append({
                "subject": held_out,
                "session": session,
                "model": model_name,
                "roc_auc": auc,
                "n_train_trials": len(y_train),
                "n_test_trials": len(y_session)
            })

            print(
                f"  {session}: ROC-AUC = {auc:.4f}"
            )

# --------------------------------------------------
# 4. SAVE SESSION-LEVEL RESULTS
# --------------------------------------------------

results = pd.DataFrame(records)

results.to_csv(
    "10_riemannian_session_results.csv",
    index=False
)

print("\nSession-level results saved.")

# --------------------------------------------------
# 5. PARTICIPANT-LEVEL COMPARISON
# --------------------------------------------------

participant_results = (
    results.groupby(["subject", "model"])["roc_auc"]
    .mean()
    .unstack()
)

participant_results["MDM_minus_CSP"] = (
    participant_results["Riemannian_MDM"]
    - participant_results["CSP_LDA"]
)

participant_results.to_csv(
    "10_riemannian_subject_comparison.csv"
)

print("\n" + "=" * 65)
print("PARTICIPANT-LEVEL RESULTS")
print("=" * 65)

print(participant_results.round(4).to_string())

# --------------------------------------------------
# 6. CHECK BASELINE REPRODUCIBILITY
# --------------------------------------------------

baseline = pd.read_csv(
    "full_cross_subject_results.csv"
)

reference = (
    baseline.groupby("subject")["score"]
    .mean()
    .sort_index()
)

current = participant_results["CSP_LDA"].sort_index()

max_difference = np.max(
    np.abs(reference.to_numpy() - current.to_numpy())
)

print("\nBaseline reproduction difference:")
print(f"{max_difference:.8f}")

assert max_difference < 1e-5, (
    "CSP baseline does not reproduce Experiment 07."
)

print("Baseline reproduction check passed.")

# --------------------------------------------------
# 7. DESCRIPTIVE STATISTICS
# --------------------------------------------------

print("\n" + "=" * 65)
print("OVERALL MODEL COMPARISON")
print("=" * 65)

for model_name in ["CSP_LDA", "Riemannian_MDM"]:

    values = participant_results[model_name]

    print(f"\n{model_name}")
    print(f"Mean ROC-AUC: {values.mean():.4f}")
    print(f"Subject-level SD: {values.std(ddof=1):.4f}")
    print(f"Minimum: {values.min():.4f}")
    print(f"Maximum: {values.max():.4f}")

differences = participant_results["MDM_minus_CSP"]

print("\nMean paired improvement:", round(
    differences.mean(), 4
))

print("Participants improved:", int(
    (differences > 0).sum()
), "out of", len(differences))

# --------------------------------------------------
# 8. EXPLORATORY PAIRED STATISTICAL TEST
# --------------------------------------------------

if not np.allclose(differences, 0):

    statistic, p_value = wilcoxon(
        differences,
        alternative="two-sided"
    )

    print("\nExploratory Wilcoxon signed-rank test")
    print(f"Statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.4f}")

else:
    print("\nNo paired differences detected.")

# --------------------------------------------------
# 9. VISUALIZE PARTICIPANT PERFORMANCE
# --------------------------------------------------

subjects = participant_results.index.to_numpy()

x = np.arange(len(subjects))
width = 0.38

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    x - width / 2,
    participant_results["CSP_LDA"],
    width,
    label="CSP + LDA"
)

ax.bar(
    x + width / 2,
    participant_results["Riemannian_MDM"],
    width,
    label="Riemannian MDM"
)

ax.axhline(
    0.5,
    color="gray",
    linestyle="--",
    label="Chance-level ROC-AUC"
)

ax.set_xticks(x)
ax.set_xticklabels(subjects)

ax.set_xlabel("Held-out Participant")
ax.set_ylabel("ROC-AUC")

ax.set_title(
    "Cross-Subject Motor Imagery Decoding: "
    "CSP vs Riemannian MDM"
)

ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(
    "10_riemannian_model_comparison.png",
    dpi=200
)
plt.close(fig)

print("\nGenerated files:")
print("10_riemannian_session_results.csv")
print("10_riemannian_subject_comparison.csv")
print("10_riemannian_model_comparison.png")

print("\nExperiment 10 completed successfully.")
