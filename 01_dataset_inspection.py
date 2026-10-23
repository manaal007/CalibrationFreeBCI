import os

from huggingface_hub import snapshot_download

# Download both sessions for participant 1
print("Downloading subject 1 from Hugging Face...")

path = snapshot_download(
    repo_id="EEGDash/eeg-mi-datasets",
    repo_type="dataset",
    allow_patterns="BNCI2014_001/*/A01*",
)

# Tell MOABB where to find the downloaded files
os.environ["MNE_DATASETS_BNCI_PATH"] = os.path.join(
    path, "BNCI2014_001"
)

import moabb
from moabb.datasets import BNCI2014_001

moabb.set_download_provider("upstream")

dataset = BNCI2014_001()

print("Dataset:", dataset.code)
print("Participants:", dataset.subject_list)
print("Classes:", dataset.event_id)

data = dataset.get_data(subjects=[1])

for subject, sessions in data.items():
    print(f"\nSubject {subject}")

    for session_name, runs in sessions.items():
        print("Session:", session_name)

        for run_name, raw in runs.items():
            print(
                "Run:", run_name,
                "Channels:", len(raw.ch_names),
                "Sampling frequency:", raw.info["sfreq"],
                "Duration:", round(raw.times[-1], 2), "seconds"
            )

print("\nDataset inspection completed successfully!")
