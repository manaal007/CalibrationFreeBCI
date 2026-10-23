
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "experiment15B_confirmatory_external"
OUT = HERE / "experiment22_class_balance_robustness"

trials = pd.read_csv(SOURCE / "15B_trial_predictions.csv")

# Count trials once per subject, not once per model.
reference = trials[
    trials["model"] == "CSP_LDA_source_tuned"
]

counts = (
    reference.groupby(["subject", "true_label"])
    .size()
    .unstack(fill_value=0)
)

counts["minimum_class"] = counts[[1, 2]].min(axis=1)
eligible = set(
    counts.index[counts["minimum_class"] >= 20]
)

rows = []

for model, group in trials.groupby("model"):
    for cohort, selected in [
        ("all_46", group),
        ("min_20_per_class",
         group[group["subject"].isin(eligible)])
    ]:
        # Use saved per-subject AUC and balanced accuracy.
        from sklearn.metrics import (
            roc_auc_score, balanced_accuracy_score
        )

        metrics = []

        for subject, data in selected.groupby("subject"):
            metrics.append({
                "auc": roc_auc_score(
                    data["true_label"] == 2,
                    data["score_right"]
                ),
                "balanced_accuracy":
                    balanced_accuracy_score(
                        data["true_label"],
                        data["predicted_label"]
                    )
            })

        metrics = pd.DataFrame(metrics)

        rows.append({
            "model": model,
            "cohort": cohort,
            "n_subjects": len(metrics),
            "mean_auc": metrics["auc"].mean(),
            "mean_balanced_accuracy":
                metrics["balanced_accuracy"].mean()
        })

OUT.mkdir(exist_ok=True)

summary = pd.DataFrame(rows)
summary.to_csv(
    OUT / "22_class_balance_summary.csv",
    index=False
)

print("\n=== EXPERIMENT 22 SUMMARY ===")
print(summary.to_string(index=False))
print("\nExcluded subjects:",
      sorted(set(counts.index) - eligible))
print("\nDONE")
