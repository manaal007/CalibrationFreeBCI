"""Independently verify saved predictions and create a shareable results bundle."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .experiment import metrics
from .audit import bootstrap_mean, sha256


def publish_local(source, destination):
    source, destination = Path(source), Path(destination)
    predictions = pd.read_csv(source/'predictions.csv')
    saved = pd.read_csv(source/'participant_metrics.csv')
    if predictions.duplicated(['model','subject','session','run','trial']).any():
        raise ValueError('Duplicate prediction identities')
    reference = None
    for name, group in predictions.groupby('model'):
        identity = group[['subject','session','run','trial','label']].reset_index(drop=True)
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            raise ValueError('Candidate trial cohorts differ')
        for subject, frame in group.groupby('subject'):
            actual = metrics(frame.label.to_numpy(), frame.prediction.to_numpy(), frame.score_right.to_numpy())
            expected = saved[(saved.model == name) & (saved.subject == subject)].iloc[0]
            for metric, value in actual.items():
                np.testing.assert_allclose(value, expected[metric], atol=1e-12)
    selection = json.loads((source/'selection.json').read_text())
    winner = selection['model']
    differences = []
    pivot = saved.pivot(index='subject', columns='model', values='balanced_accuracy')
    for baseline in ('csp_lda','oas_mdm'):
        values = pivot[winner]-pivot[baseline]
        lo, hi = bootstrap_mean(values)
        differences.append(dict(comparison=f'{winner} minus {baseline}',
                                mean_difference=float(values.mean()), ci_low=lo, ci_high=hi,
                                interpretation='exploratory; selected on same source folds'))
    pd.DataFrame(differences).to_csv(source/'paired_differences.csv', index=False)
    sensitivity = pd.read_csv(source/'sensitivity.csv')
    gains = sensitivity[sensitivity.gain.notna()]
    (source/'verification.json').write_text(json.dumps(dict(
        status='saved predictions independently recomputed', rows=len(predictions),
        participants=int(saved.subject.nunique()),
        gain_max_score_shifts=gains.groupby('model').max_score_shift.max().to_dict(),
        model_sha256=sha256(source/'frozen_model.joblib')), indent=2))
    destination.mkdir(parents=True, exist_ok=False)
    for file in source.iterdir():
        if file.suffix in ('.csv','.json','.png','.svg'):
            shutil.copy2(file, destination/file.name)
    print(json.dumps(dict(verified_prediction_rows=len(predictions), paired_differences=differences), indent=2))
