
import os

import moabb
from huggingface_hub import snapshot_download
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import CrossSubjectEvaluation

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

# --------------------------------------------------
# 1. Download/cache the first three participants
# --------------------------------------------------

path = snapshot_download(
    repo_id="EEGDash/eeg-mi-datasets",
    repo_type="dataset",
    allow_patterns=[
        "BNCI2014_001/*/A01*",
        "BNCI2014_001/*/A02*",
        "BNCI2014_001/*/A03*",
    ],
)

os.environ["MNE_DATASETS_BNCI_PATH"] = os.path.join(
    path, "BNCI2014_001"
)

moabb.set_download_provider("upstream")
moabb.set_log_level("WARNING")

# --------------------------------------------------
# 2. Define dataset and paradigm
# --------------------------------------------------

dataset = BNCI2014_001()
dataset.subject_list = [1, 2, 3]

paradigm = LeftRightImagery(
    fmin=8,
    fmax=30,
    tmin=0.5,
    tmax=3.5
)

# --------------------------------------------------
# 3. CSP + LDA
# --------------------------------------------------

pipeline = make_pipeline(
    CSP(
        n_components=4,
        reg="ledoit_wolf",
        log=True
    ),
    LinearDiscriminantAnalysis()
)

# --------------------------------------------------
# 4. Leave-one-subject-out evaluation
# --------------------------------------------------

evaluation = CrossSubjectEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
    overwrite=True,
    suffix="cross_subject_pilot"
)

print("Running cross-subject evaluation...")

results = evaluation.process({
    "CSP_LDA": pipeline
})

print("\nCROSS-SUBJECT RESULTS")
print(results[["subject", "score", "pipeline"]])

results.to_csv(
    "cross_subject_pilot_results.csv",
    index=False
)

print("\nMean ROC-AUC:", results["score"].mean())
print("Saved: cross_subject_pilot_results.csv")
