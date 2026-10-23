"""Read-only inventory and independent checks of historical saved predictions."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root):
    """Hash original scripts and outputs, without importing or executing scripts."""
    rows = []
    paths = list(root.glob("*.py")) + list(root.glob("*.csv")) + list(root.glob("*.png"))
    for directory in sorted(root.glob("experiment*")):
        if directory.is_dir():
            paths.extend(p for p in directory.rglob("*") if p.is_file())
    for path in sorted(set(paths)):
        row = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
               "sha256": sha256(path)}
        if path.suffix == ".py":
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                row["syntax"] = "valid"
                row["functions"] = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            except SyntaxError as exc:
                row["syntax"] = str(exc)
        rows.append(row)
    return rows


def recompute_predictions(trials):
    required = ["subject", "model", "imagery_class", "matlab_trial", "true_label",
                "predicted_label", "score_right"]
    if set(required) - set(trials):
        raise ValueError("Missing trial columns")
    if trials[required].isna().any().any():
        raise ValueError("Missing prediction data")
    keys = ["subject", "model", "imagery_class", "matlab_trial"]
    if trials.duplicated(keys).any():
        raise ValueError("Duplicate trial predictions")
    if not np.isfinite(trials.score_right.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite scores")
    if set(trials.true_label) != {1, 2} or not set(trials.predicted_label) <= {1, 2}:
        raise ValueError("Labels must encode left=1 and right=2")
    expected_labels = trials.imagery_class.map({"left": 1, "right": 2})
    if not np.array_equal(expected_labels, trials.true_label):
        raise ValueError("Class names and labels disagree")
    # Every model must be evaluated on exactly the same trials and labels.
    reference = None
    rows = []
    for model, frame in trials.groupby("model", sort=True):
        identity = frame.sort_values(["subject", "imagery_class", "matlab_trial"])[
            ["subject", "imagery_class", "matlab_trial", "true_label"]].reset_index(drop=True)
        if reference is None:
            reference = identity
        elif not reference.equals(identity):
            raise ValueError("Model trial cohorts differ")
        for subject, group in frame.groupby("subject", sort=True):
            if set(group.true_label) != {1, 2}:
                raise ValueError("Participant has only one retained class")
            rows.append(dict(subject=subject, model=model, n_trials=len(group),
                             auc=roc_auc_score(group.true_label == 2, group.score_right),
                             balanced_accuracy=balanced_accuracy_score(group.true_label, group.predicted_label),
                             accuracy=float(np.mean(group.true_label == group.predicted_label)),
                             predicted_right_fraction=float(np.mean(group.predicted_label == 2)),
                             n_left=int(sum(group.true_label == 1)), n_right=int(sum(group.true_label == 2))))
    return pd.DataFrame(rows)


def bootstrap_mean(values, seed=20261023, draws=10000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or draws < 1:
        raise ValueError('Bootstrap requires a nonempty finite participant vector and positive draws')
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    records = inventory(root)
    (out / "original_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    script_rows = [{k: v for k, v in row.items() if k != "functions"}
                   for row in records if row["path"].endswith(".py")]
    pd.DataFrame(script_rows).to_csv(out / "script_inventory.csv", index=False)
    source = root / "experiment15B_confirmatory_external"
    metrics = recompute_predictions(pd.read_csv(source / "15B_trial_predictions.csv"))
    if set(metrics.subject) != set(range(7, 53)) or metrics.model.nunique() != 4:
        raise ValueError("Historical cohort is incomplete")
    saved = pd.read_csv(source / "15B_subject_metrics.csv")
    if set(map(tuple, saved[["subject", "model"]].values)) != set(map(tuple, metrics[["subject", "model"]].values)):
        raise ValueError("Saved and recomputed metric keys differ")
    merged = metrics.merge(saved, on=["subject", "model"], validate="one_to_one", suffixes=("_new", "_saved"))
    errors = {}
    for column in ["auc", "balanced_accuracy", "accuracy", "n_trials", "predicted_right_fraction"]:
        errors[column] = float(np.max(np.abs(merged[column + "_new"] - merged[column + "_saved"])))
        if errors[column] > 1e-10:
            raise ValueError(f"Saved {column} disagrees with predictions")
    metrics.to_csv(out / "participant_metrics.csv", index=False)
    summary = []
    for model, group in metrics.groupby("model"):
        for column in ["auc", "balanced_accuracy"]:
            lo, hi = bootstrap_mean(group[column])
            summary.append(dict(model=model, metric=column, mean=float(group[column].mean()),
                                ci_low=lo, ci_high=hi, n_subjects=len(group),
                                interpretation="historical exploratory; exclusion policy provisional"))
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    environment = {p: importlib.metadata.version(p) for p in
                   ["numpy", "scipy", "scikit-learn", "mne", "moabb", "pyriemann", "pandas", "matplotlib"]}
    report = dict(status="completed_saved_prediction_audit", original_files=len(records),
                  scripts=len(script_rows), trial_rows=int(metrics.n_trials.sum()),
                  metric_max_errors=errors, environment=environment,
                  raw_data_independently_verified=False,
                  untouched_confirmatory_cohort_available=False)
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
