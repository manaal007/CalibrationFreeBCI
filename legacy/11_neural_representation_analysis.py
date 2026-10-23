
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, rankdata

# ============================================================
# EXPERIMENT 11
# Neural Representations and BCI Generalization
# ============================================================

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "experiment11_results"
OUTPUT.mkdir(exist_ok=True)

# Results previously obtained from Experiment 10.
# Verify these against your original CSV before publication.
model_results = pd.DataFrame({
    "subject": np.arange(1, 10),
    "CSP_LDA": [
        0.6652199074, 0.5566165123, 0.9842785494,
        0.6938657407, 0.5320216049, 0.7408371914,
        0.7640817901, 0.9889081790, 0.7188464506
    ],
    "Riemannian_MDM": [
        0.7505787037, 0.5204475309, 0.8372878086,
        0.7478780864, 0.6182484568, 0.6983024691,
        0.7087191358, 0.9232253086, 0.8533950617
    ]
})

# Results previously obtained from your covariance analysis.
# These are descriptive, label-dependent measures.
neural_results = pd.DataFrame({
    "subject": np.arange(1, 10),
    "class_covariance_separation": [
        0.0258488961, 0.0231597131, 0.1038274035,
        0.0210873523, 0.0113353071, 0.0442541500,
        0.0182515035, 0.0452140284, 0.1830474916
    ],
    "source_pattern_alignment": [
        0.6485604564, 0.6579169951, 0.2472785151,
        0.9695749624, 0.8172833211, 0.7488391760,
        0.8645775277, 0.3705808880, 0.3642945171
    ]
})

df = model_results.merge(
    neural_results,
    on="subject",
    validate="one_to_one"
)

df["delta_auc"] = (
    df["Riemannian_MDM"] - df["CSP_LDA"]
)

assert len(df) == 9
assert df["subject"].is_unique
assert df.notna().all().all()

# ============================================================
# PERMUTATION TEST FOR SPEARMAN CORRELATION
# ============================================================

def permutation_spearman(x, y, n_permutations=20000, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    observed = spearmanr(x, y).statistic

    rng = np.random.default_rng(seed)
    count = 0

    for _ in range(n_permutations):
        permuted_y = rng.permutation(y)
        statistic = spearmanr(x, permuted_y).statistic

        if abs(statistic) >= abs(observed) - 1e-12:
            count += 1

    # Plus-one Monte Carlo correction
    p_value = (count + 1) / (n_permutations + 1)

    return observed, p_value


# ============================================================
# ANALYSIS
# ============================================================

predictors = [
    "class_covariance_separation",
    "source_pattern_alignment"
]

results = []

print("\nEXPERIMENT 11: NEURAL REPRESENTATION ANALYSIS")
print("=" * 55)

for predictor in predictors:

    rho, p_perm = permutation_spearman(
        df[predictor],
        df["delta_auc"]
    )

    # Leave-one-participant-out sensitivity analysis
    sensitivity = []

    for subject in df["subject"]:
        subset = df[df["subject"] != subject]

        rho_subset = spearmanr(
            subset[predictor],
            subset["delta_auc"]
        ).statistic

        sensitivity.append(rho_subset)

    results.append({
        "predictor": predictor,
        "spearman_rho": rho,
        "permutation_p": p_perm,
        "min_leave_one_out_rho": min(sensitivity),
        "max_leave_one_out_rho": max(sensitivity)
    })

    print(f"\nPredictor: {predictor}")
    print(f"Spearman rho: {rho:.4f}")
    print(f"Permutation p-value: {p_perm:.4f}")
    print(
        "Leave-one-out rho range: "
        f"{min(sensitivity):.4f} to "
        f"{max(sensitivity):.4f}"
    )


# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, predictor in zip(axes, predictors):

    ax.scatter(
        df[predictor],
        df["delta_auc"],
        s=80
    )

    for _, row in df.iterrows():
        ax.annotate(
            f"S{int(row['subject'])}",
            (row[predictor], row["delta_auc"]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.axhline(
        0,
        linestyle="--",
        color="gray"
    )

    ax.set_xlabel(predictor.replace("_", " ").title())
    ax.set_ylabel("MDM minus CSP ROC-AUC")
    ax.grid(alpha=0.25)

fig.suptitle(
    "Neural Covariance Structure and Decoder Advantage"
)

plt.tight_layout()

plt.savefig(
    OUTPUT / "neural_representation_analysis.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT / "subject_level_neural_features.csv",
    index=False
)

pd.DataFrame(results).to_csv(
    OUTPUT / "neural_representation_statistics.csv",
    index=False
)

print("\nFiles saved in:", OUTPUT)
print("\nExperiment 11 completed successfully.")
