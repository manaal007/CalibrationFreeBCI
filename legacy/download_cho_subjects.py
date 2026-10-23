
from moabb.datasets import Cho2017

dataset = Cho2017()

# Download a small initial batch.
# Subject 1 is excluded because it was used in the pilot.
subjects = [2, 3, 4, 5, 6]

for subject in subjects:
    print(f"\nDownloading Cho2017 Subject {subject}...", flush=True)

    try:
        dataset.data_path(subject)
        print(f"Subject {subject}: ready!", flush=True)
    except Exception as error:
        print(f"Subject {subject}: FAILED — {error}", flush=True)

print("\nDownload attempt finished.")
