import os
import matplotlib.pyplot as plt
import mne
import moabb

from moabb.datasets import BNCI2014_001
from huggingface_hub import snapshot_download

# --------------------------------------------------
# 1. Load the previously downloaded dataset
# --------------------------------------------------

path = snapshot_download(
    repo_id="EEGDash/eeg-mi-datasets",
    repo_type="dataset",
    allow_patterns="BNCI2014_001/*/A01*",
)

os.environ["MNE_DATASETS_BNCI_PATH"] = os.path.join(
    path, "BNCI2014_001"
)

moabb.set_download_provider("upstream")

dataset = BNCI2014_001()
data = dataset.get_data(subjects=[1])

# Select the first run of the training session
raw = data[1]["0train"]["0"]

# --------------------------------------------------
# 2. Inspect recording properties
# --------------------------------------------------

print("\nEEG RECORDING INFORMATION")
print("-" * 40)

print(raw.info)
print("\nChannel names:")
print(raw.ch_names)

print("\nChannel types:")
for channel_type in set(raw.get_channel_types()):
    print(
        channel_type,
        len(mne.pick_types(
            raw.info,
            eeg=(channel_type == "eeg"),
            eog=(channel_type == "eog"),
            stim=(channel_type == "stim")
        ))
    )

# --------------------------------------------------
# 3. Inspect motor-imagery event annotations
# --------------------------------------------------

print("\nANNOTATIONS")
print("-" * 40)

for annotation in raw.annotations[:15]:
    print(annotation)

print("\nAnnotation descriptions:")
print(set(raw.annotations.description))

# --------------------------------------------------

# --------------------------------------------------
# 4. Inspect actual EEG values and create a plot
# --------------------------------------------------

import numpy as np

channels = ["C3", "Cz", "C4"]
sfreq = raw.info["sfreq"]

# Extract 10 seconds from the first recording
signals, times = raw.get_data(
    picks=channels,
    start=0,
    stop=int(10 * sfreq),
    return_times=True
)

# Convert volts to microvolts for visualization only
signals_uv = signals * 1e6

print("\nEEG SIGNAL QUALITY CHECK")
print("-" * 40)
print("Array shape:", signals.shape)
print("All values finite:", np.isfinite(signals).all())

for name, signal in zip(channels, signals_uv):
    print(
        f"{name}: min={signal.min():.3f} uV, "
        f"max={signal.max():.3f} uV, "
        f"std={signal.std():.3f} uV"
    )

# Plot each channel separately
fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 8),
    sharex=True
)

for ax, name, signal in zip(axes, channels, signals_uv):
    ax.plot(times, signal, linewidth=0.7)
    ax.set_ylabel(f"{name} (uV)")
    ax.grid(alpha=0.25)

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "Subject 1 - Raw Sensorimotor EEG",
    fontsize=14
)

fig.tight_layout()

fig.savefig(
    "subject01_raw_eeg_fixed.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\nSaved: subject01_raw_eeg_fixed.png")
