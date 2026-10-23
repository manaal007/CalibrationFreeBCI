
import os
import numpy as np
import matplotlib.pyplot as plt
import moabb

from huggingface_hub import snapshot_download
from moabb.datasets import BNCI2014_001

# Reuse cached subject 1 data
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
raw = dataset.get_data(subjects=[1])[1]["0train"]["0"]

# Analyze three sensorimotor electrodes
eeg = raw.copy().pick(["C3", "Cz", "C4"])

# Estimate power spectral density using Welch's method
spectrum = eeg.compute_psd(
    method="welch",
    fmin=1,
    fmax=40,
    n_fft=1000,
    n_overlap=500
)

psd, freqs = spectrum.get_data(return_freqs=True)

fig, ax = plt.subplots(figsize=(11, 6))

for name, power in zip(eeg.ch_names, psd):
    power_db = 10 * np.log10(
        np.maximum(power * 1e12, 1e-20)
    )
    ax.plot(freqs, power_db, label=name)

ax.axvspan(8, 13, alpha=0.12, label="Mu band")
ax.axvspan(13, 30, alpha=0.08, label="Beta band")

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power spectral density (dB re 1 uV²/Hz)")
ax.set_title("Subject 1 - Sensorimotor EEG Power Spectrum")
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("subject01_frequency_spectrum.png", dpi=200)
plt.close(fig)

print("Frequency spectrum saved successfully!")
