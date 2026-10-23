
"""
Calibration-Free BCI
Experiment 08B: Independent Pipeline Audit

Dataset: BNCI2014-001
Task: Left-hand vs right-hand motor imagery

Objectives:
1. Verify participant-level train/test separation.
2. Fit CSP and LDA only on training participants.
3. Evaluate both sessions of each unseen participant.
4. Check class balance and non-finite EEG values.
5. Compare independent results with Experiment 07.

IMPORTANT:
This independently audits the model-fitting procedure.
It does not prove that all preprocessing is leakage-free.
"""

import os
import warnings

import numpy as np
import pandas as pd
import moabb

from huggingface_hub import snapshot_download
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

SUBJECTS = list(range(1, 10))

moabb.set_download_provider("upstream")
moabb.set_log_level("WARNING")

print("=" * 60)
print("EXPERIMENT 08B: INDEPENDENT PIPELINE AUDIT")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD CACHED DATA
# --------------------------------------------------

print("\nLoading cached EEG dataset...")

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

print("EEG shape:", X.shape)
print("Total trials:", len(y))
print("Participants:", sorted(metadata["subject"].unique()))

assert len(X) == len(y) == len(metadata)
assert X.shape[1] == 22
assert np.isfinite(X).all()

# --------------------------------------------------
# 2. VERIFY CLASS LABELS
# --------------------------------------------------

classes = np.unique(y)

assert len(classes) == 2, (
    f"Expected 2 classes, found {classes}"
)

positive_class = classes[1]

print("\nClass labels:", classes)
print("Positive class for ROC-AUC:", positive_class)

# --------------------------------------------------
# 3. INDEPENDENT LOSO EVALUATION
# --------------------------------------------------

records = []

for held_out in SUBJECTS:

    print(f"\nEvaluating held-out subject {held_out}...")

    subject_ids = metadata["subject"].to_numpy()

    train_mask = subject_ids != held_out
    test_mask = subject_ids == held_out

    train_subjects = set(subject_ids[train_mask])
    test_subjects = set(subject_ids[test_mask])

    assert train_subjects.isdisjoint(test_subjects)

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    assert len(np.unique(y_train)) == 2
    assert len(np.unique(y_test)) == 2

    print("Training participants:", sorted(train_subjects))
    print("Test participant:", held_out)
    print("Training trials:", len(X_train))
    print("Testing trials:", len(X_test))

    # Fresh model for every held-out participant.
    # CSP is fitted ONLY on training participants.

    model = make_pipeline(
        CSP(
            n_components=4,
            reg="ledoit_wolf",
            log=True
        ),
        LinearDiscriminantAnalysis()
    )

    model.fit(X_train, y_train)

    assert list(model.classes_) == list(classes)

    # Evaluate each test session separately.
    test_metadata = metadata.loc[test_mask].reset_index(
        drop=True
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

        probabilities = model.predict_proba(X_session)

        positive_index = list(
            model.classes_
        ).index(positive_class)

        auc = roc_auc_score(
            (y_session == positive_class).astype(int),
            probabilities[:, positive_index]
        )

        records.append({
            "subject": held_out,
            "session": session,
            "score": auc,
            "n_test_trials": len(y_session),
            "n_train_trials": len(y_train)
        })

        print(
            f"  {session}: ROC-AUC = {auc:.4f}"
        )

# --------------------------------------------------
# 4. SAVE INDEPENDENT RESULTS
# --------------------------------------------------

audit = pd.DataFrame(records)

audit.to_csv(
    "08b_independent_loso_results.csv",
    index=False
)

print("\nIndependent evaluation completed.")

# --------------------------------------------------
# 5. COMPARE WITH EXPERIMENT 07
# --------------------------------------------------

reference = pd.read_csv(
    "full_cross_subject_results.csv"
)

reference = reference[
    ["subject", "session", "score"]
].copy()

reference["subject"] = reference["subject"].astype(int)
audit["subject"] = audit["subject"].astype(int)

comparison = audit.merge(
    reference,
    on=["subject", "session"],
    how="outer",
    suffixes=("_audit", "_experiment07"),
    validate="one_to_one",
    indicator=True
)

assert (comparison["_merge"] == "both").all(), (
    "Some subject/session rows do not match."
)

comparison["absolute_difference"] = (
    comparison["score_audit"]
    - comparison["score_experiment07"]
).abs()

comparison.to_csv(
    "08b_pipeline_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("PIPELINE COMPARISON")
print("=" * 60)

print(
    comparison[
        [
            "subject",
            "session",
            "score_audit",
            "score_experiment07",
            "absolute_difference"
        ]
    ].to_string(index=False)
)

max_difference = comparison[
    "absolute_difference"
].max()

print(
    f"\nMaximum absolute ROC-AUC difference: "
    f"{max_difference:.6f}"
)

if max_difference < 0.01:
    print(
        "Independent results closely reproduce "
        "Experiment 07."
    )
else:
    print(
        "Differences detected. Investigate before "
        "claiming methodological equivalence."
    )

# --------------------------------------------------
# 6. SUMMARY
# --------------------------------------------------

subject_means = audit.groupby("subject")["score"].mean()

print("\n" + "=" * 60)
print("INDEPENDENT AUDIT SUMMARY")
print("=" * 60)

print(f"Mean ROC-AUC: {subject_means.mean():.4f}")
print(f"Subject-level SD: {subject_means.std(ddof=1):.4f}")
print(f"Subjects evaluated: {len(subject_means)}")

print("\nGenerated files:")
print("08b_independent_loso_results.csv")
print("08b_pipeline_comparison.csv")

print("\nExperiment 08B completed.")
