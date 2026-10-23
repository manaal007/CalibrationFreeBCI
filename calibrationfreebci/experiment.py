"""Small frozen candidate comparison; every training split is participant separated."""
import json
import importlib.metadata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score
from threadpoolctl import threadpool_limits
from .data import load_bnci, CHANNELS
from .models import build, right_score
from .audit import bootstrap_mean, sha256

CANDIDATES = ['csp_lda', 'oas_mdm', 'rms_csp_lda', 'trace_oas_mdm', 'trace_tangent_lda']


def metrics(y, predicted, scores):
    y,predicted,scores = np.asarray(y),np.asarray(predicted),np.asarray(scores)
    if (y.ndim != 1 or predicted.shape != y.shape or scores.shape != y.shape or
        set(y) != {1, 2} or not set(predicted) <= {1,2} or not np.isfinite(scores).all()):
        raise ValueError('Evaluation requires both classes and finite scores')
    recalls = recall_score(y, predicted, labels=[1, 2], average=None)
    return dict(balanced_accuracy=balanced_accuracy_score(y, predicted),
                auc=roc_auc_score(y == 2, scores), recall_left=recalls[0], recall_right=recalls[1],
                predicted_right_fraction=float(np.mean(predicted == 2)), n_trials=len(y))


def run(out, subjects, data_root=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if len(set(subjects)) < 3 or len(subjects) != len(set(subjects)):
        raise ValueError('Need at least three distinct participants')
    protocol = dict(subjects=subjects, candidates=CANDIDATES, selection='participant-mean LOSO balanced accuracy',
                    tie_break='candidate order', external_data_accessed=False,
                    preprocessing=dict(channels=CHANNELS, units='volts', band=[8,30],
                                       reference='22-channel average', window_from_trial_start=[2.5,4.5]),
                    artifact_policy='retain supplied flags; report sensitivity, no label-dependent exclusions',
                    interpretation='source development; selection scores are optimistic for the selected winner',
                    threshold=0, target_batch_adaptation=False)
    protocol['data_root'] = str(Path(data_root).resolve()) if data_root else None
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    x, metadata, provenance = load_bnci(subjects, data_root)
    metadata.to_csv(out/'epochs.csv', index=False)
    (out/'data_provenance.json').write_text(json.dumps(provenance, indent=2))
    y = metadata.label.to_numpy()
    groups = metadata.subject.to_numpy()
    records, predictions, sensitivity = [], [], []
    with threadpool_limits(limits=2):
        for name in CANDIDATES:
            for subject in subjects:
                train, test = groups != subject, groups == subject
                model = build(name).fit(x[train], y[train])
                scores = right_score(model, x[test])
                pred = np.where(scores > 0, 2, 1)
                result = dict(model=name, subject=subject, **metrics(y[test], pred, scores))
                records.append(result)
                frame = metadata.loc[test].copy()
                frame['model'], frame['prediction'], frame['score_right'] = name, pred, scores
                predictions.append(frame)
                clean = frame.artifact_flag.to_numpy() == 0
                if set(y[test][clean]) == {1, 2}:
                    sensitivity.append(dict(model=name, subject=subject, **metrics(y[test][clean], pred[clean], scores[clean])))
                # Diagnose score displacement under physically equivalent global gain changes.
                for gain in (.1, 10.):
                    shifted = right_score(model, x[test]*gain)
                    sensitivity.append(dict(model=name, subject=subject, gain=gain,
                                            max_score_shift=float(np.max(np.abs(shifted-scores))),
                                            **metrics(y[test], np.where(shifted > 0, 2, 1), shifted)))
                print(f'{name} participant {subject}: BA={result["balanced_accuracy"]:.3f} AUC={result["auc"]:.3f}', flush=True)
                pd.DataFrame(records).to_csv(out/'participant_metrics.csv', index=False)
            pd.concat(predictions).to_csv(out/'predictions.csv', index=False)
        summary = []
        results = pd.DataFrame(records)
        for name in CANDIDATES:
            group = results[results.model == name]
            row = dict(model=name)
            for metric in ('balanced_accuracy','auc','recall_left','recall_right'):
                row[metric] = float(group[metric].mean())
                lo, hi = bootstrap_mean(group[metric])
                row[metric+'_ci_low'], row[metric+'_ci_high'] = lo, hi
            summary.append(row)
        summary = pd.DataFrame(summary)
        summary.to_csv(out/'summary.csv', index=False)
        pd.DataFrame(sensitivity).to_csv(out/'sensitivity.csv', index=False)
        winner = summary.sort_values('balanced_accuracy', ascending=False, kind='stable').iloc[0].model
        fitted = build(winner).fit(x, y)
        joblib.dump(fitted, out/'frozen_model.joblib')
    environment = {p: importlib.metadata.version(p) for p in
                   ['numpy','scipy','scikit-learn','mne','moabb','pyriemann','pandas','matplotlib']}
    (out/'environment.json').write_text(json.dumps(environment, indent=2))
    (out/'selection.json').write_text(json.dumps(dict(model=winner, threshold=0,
        model_sha256=sha256(out/'frozen_model.joblib'), external_confirmation='pending approval and independent cohort'), indent=2))
    plot(results, out)
    print(summary.to_string(index=False), flush=True)
    return summary


def plot(results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, metric in zip(axes, ['balanced_accuracy','auc']):
        pivot = results.pivot(index='subject', columns='model', values=metric)[CANDIDATES]
        ax.plot(np.arange(len(CANDIDATES)), pivot.T, color='grey', alpha=.35, linewidth=.8)
        ax.scatter(np.arange(len(CANDIDATES)), pivot.mean(), color='navy', zorder=3)
        ax.axhline(.5, color='black', linestyle='--', linewidth=1)
        ax.set(xticks=np.arange(len(CANDIDATES)), xticklabels=CANDIDATES, ylim=(0,1), ylabel=metric)
        ax.tick_params(axis='x', labelrotation=35)
    fig.suptitle('Source participant-held-out development (each line = participant)')
    fig.savefig(out/'source_comparison.png', dpi=220)
    fig.savefig(out/'source_comparison.svg')
    plt.close(fig)
