
import os

import moabb
from huggingface_hub import snapshot_download
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import CrossSessionEvaluation

from mne.decoding import CSP
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis
)
from sklearn.pipeline import make_pipeline

# 1. Reuse the downloaded Subject 1 files
path = snapshot_download(
    repo_id="EEGDash/eeg-mi-datasets",
    repo_type="dataset",
    allow_patterns="BNCI2014_001/*/A01*",
)

os.environ["MNE_DATASETS_BNCI_PATH"] = os.path.join(
    path, "BNCI2014_001"
)

moabb.set_download_provider("upstream")
moabb.set_log_level("WARNING")

# 2. Select Subject 1
dataset = BNCI2014_001()
dataset.subject_list = [1]

# 3. Define the motor-imagery task
paradigm = LeftRightImagery(
    fmin=8,
    fmax=30,
    tmin=0.5,
    tmax=3.5
)

# 4. Define CSP + LDA
pipeline = make_pipeline(
    CSP(
        n_components=4,
        reg="ledoit_wolf",
        log=True
    ),
    LinearDiscriminantAnalysis()
)

# 5. Evaluate on a separate recording session
evaluation = CrossSessionEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
    overwrite=True
)

print("Running cross-session CSP + LDA...")

results = evaluation.process({
    "CSP_LDA": pipeline
})

# 6. Save results
print("\nRESULTS")
print(results[
    ["subject", "session", "score", "pipeline"]
])

results.to_csv(
    "subject01_cross_session_results.csv",
    index=False
)

print("\nSaved results successfully!")
