
from pathlib import Path
import importlib.util
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, balanced_accuracy_score

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "experiment15A_source_optimization"
OUT = HERE / "experiment21_amplitude_sensitivity"

# Fixed before running this experiment.
# These are diagnostic scaling factors, NOT optimized values.
SCALES = [1.0, 0.5, 0.1, 0.05, 0.025]


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(
        name, HERE / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    source = load_module(
        "15A_source_only_optimization.py", "source21"
    )
    external = load_module(
        "15B_full_external_validation.py", "external21"
    )
    pilot = load_module(
        "15_external_transfer_pilot.py", "pilot21"
    )

    names = pilot.get_cho_eeg_names()
    indices = [names.index(c) for c in pilot.CHANNELS]

    models = {}

    for family in source.GRID:
        for role in ("untuned_baseline", "source_tuned"):
            key = f"{family}_{role}"
            models[key] = (
                family,
                joblib.load(MODEL_DIR / f"{key}.joblib")
            )

    rows = []

    for subject in external.CONFIRMATORY_SUBJECTS:
        path = external.MAT_DIR / f"s{subject:02d}.mat"

        if not path.is_file():
            raise FileNotFoundError(path)

        x, y, ids, quality = external.read_subject(
            path, indices
        )

        print(
            f"Subject {subject}: {len(y)} retained trials",
            flush=True
        )

        for scale in SCALES:
            scaled_x = x * scale

            for name, (family, model) in models.items():
                scores = np.asarray(
                    source.score_right(
                        model, family, scaled_x
                    ),
                    dtype=float
                )

                predictions = np.asarray(
                    model.predict(scaled_x), dtype=int
                )

                rows.append({
                    "subject": subject,
                    "scale": scale,
                    "model": name,
                    "n_trials": len(y),
                    "auc": roc_auc_score(y == 2, scores),
                    "balanced_accuracy":
                        balanced_accuracy_score(
                            y, predictions
                        ),
                    "predicted_right_fraction":
                        np.mean(predictions == 2),
                    "score_min": np.min(scores),
                    "score_median": np.median(scores),
                    "score_max": np.max(scores)
                })

    OUT.mkdir(exist_ok=True)

    results = pd.DataFrame(rows)
    results.to_csv(
        OUT / "21_subject_metrics.csv",
        index=False
    )

    summary = (
        results.groupby(["scale", "model"])
        .agg(
            n_subjects=("subject", "nunique"),
            mean_auc=("auc", "mean"),
            mean_balanced_accuracy=(
                "balanced_accuracy", "mean"
            ),
            mean_predicted_right_fraction=(
                "predicted_right_fraction", "mean"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        OUT / "21_summary.csv", index=False
    )

    print("\n=== EXPERIMENT 21 SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nExploratory analysis complete.")


if __name__ == "__main__":
    main()
