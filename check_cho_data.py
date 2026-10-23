
from pathlib import Path
import mne

data_root = Path(mne.get_config("MNE_DATA", str(Path.home() / "mne_data")))
print("MNE data root:", data_root)

files = list(data_root.rglob("s*.mat")) if data_root.exists() else []

print("MATLAB files found:", len(files))
for file in files[:60]:
    print(file)
