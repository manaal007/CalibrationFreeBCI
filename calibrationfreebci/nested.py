"""Nested participant validation for a frozen, small frequency-bank candidate set."""
import json
from pathlib import Path
import importlib.metadata
import joblib
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from .filterbank import BANDS, build_bank
from .data import load_bnci
from .experiment import metrics
from .audit import sha256, bootstrap_mean
from .models import build as build_epoch_model


def participant_folds(groups, splits, seed):
    participants = np.unique(groups)
    for train, test in KFold(splits, shuffle=True, random_state=seed).split(participants):
        yield np.isin(groups, participants[train]), np.isin(groups, participants[test])


def fit_selection(x, y, groups, config):
    rows = []
    for name in config['candidates']:
        for fold, (train, valid) in enumerate(participant_folds(groups, config['inner_splits'], config['seed'])):
            model = build_bank(name).fit(x[train], y[train])
            scores = model.decision_function(x[valid])
            frame = pd.DataFrame(dict(subject=groups[valid], label=y[valid], score=scores))
            for subject,g in frame.groupby('subject'):
                rows.append(dict(candidate=name, inner_fold=fold, subject=subject,
                                 **metrics(g.label.to_numpy(), np.where(g.score>0,2,1),g.score.to_numpy())))
    results = pd.DataFrame(rows)
    means = results.groupby('candidate').balanced_accuracy.mean()
    # max preserves predeclared ordering on exact ties.
    winner = max(config['candidates'], key=lambda name: means[name])
    return winner, results


def run_nested(out, config_path, data_root=None):
    config = json.loads(Path(config_path).read_text())
    if not set(config) == {'subjects','candidates','outer_splits','inner_splits','seed','threshold_policy','artifact_policy'}:
        raise ValueError('Configuration keys mismatch')
    if config['threshold_policy'] != 'zero' or not config['candidates']:
        raise ValueError('Only fixed zero threshold is supported in this protocol')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(**config, bands=BANDS, input='volts, continuous-run filtered before epoching',
                    window_after_cue=[.5,2.5], type='nested participant development', external_data_accessed=False)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    mne.set_log_level('WARNING')
    x, metadata, provenance = load_bnci(config['subjects'], data_root, BANDS, config['artifact_policy'])
    metadata.to_csv(out/'epochs.csv', index=False)
    (out/'data_provenance.json').write_text(json.dumps(provenance, indent=2))
    y, groups = metadata.label.to_numpy(), metadata.subject.to_numpy()
    records, selections, prediction_frames, inner_frames = [], [], [], []
    with threadpool_limits(limits=2):
        for outer, (train,test) in enumerate(participant_folds(groups, config['outer_splits'], config['seed'])):
            chosen, inner = fit_selection(x[train], y[train], groups[train], config)
            inner['outer_fold'] = outer
            inner_frames.append(inner)
            selections.append(dict(outer_fold=outer, candidate=chosen, train_subjects=np.unique(groups[train]).tolist(),
                                   test_subjects=np.unique(groups[test]).tolist()))
            for role,name in [('nested_selected',chosen), ('fixed_rms_csp4','rms_csp4')]:
                model = build_bank(name).fit(x[train], y[train])
                scores = model.decision_function(x[test])
                frame = metadata.loc[test].copy()
                frame['score_right'], frame['prediction'] = scores, np.where(scores>0,2,1)
                frame['model'], frame['candidate'], frame['outer_fold'] = role, name, outer
                prediction_frames.append(frame)
                for subject,g in frame.groupby('subject'):
                    records.append(dict(model=role, subject=subject, candidate=name, outer_fold=outer,
                        **metrics(g.label.to_numpy(),g.prediction.to_numpy(),g.score_right.to_numpy())))
            pd.DataFrame(records).to_csv(out/'participant_metrics.csv', index=False)
            pd.concat(prediction_frames).to_csv(out/'predictions.csv', index=False)
            pd.concat(inner_frames).to_csv(out/'inner_metrics.csv', index=False)
            print(f'Outer fold {outer}: selected {chosen}; held-out participants {np.unique(groups[test]).tolist()}',flush=True)
        chosen, final_cv = fit_selection(x, y, groups, config)
        final_cv.to_csv(out/'final_selection_cv.csv', index=False)
        manifest = {}
        for name in dict.fromkeys([chosen, 'rms_csp4']):
            model = build_bank(name).fit(x,y)
            path = out/f'comparison_{name}.joblib'
            before = model.decision_function(x[:8])
            joblib.dump(model,path)
            np.testing.assert_allclose(joblib.load(path).decision_function(x[:8]),before,atol=1e-12)
            manifest[name] = dict(file=path.name, sha256=sha256(path), threshold=0, input='filterbank',bands=BANDS)
        broad=x[:,0].astype(float)
        broad-=broad.mean(axis=1,keepdims=True)
        for name in ('csp_lda','oas_mdm'):
            model = build_epoch_model(name).fit(broad,y)
            path = out/f'comparison_{name}.joblib'
            joblib.dump(model,path)
            manifest[name] = dict(file=path.name,sha256=sha256(path),threshold=0,input='epochs')
        joblib.dump(joblib.load(out/manifest[chosen]['file']),out/'frozen_model.joblib')
    (out/'comparison_manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'selection.json').write_text(json.dumps(dict(model=chosen, threshold=0,input='filterbank',bands=BANDS,
         model_sha256=sha256(out/'frozen_model.joblib')),indent=2))
    (out/'outer_selections.json').write_text(json.dumps(selections,indent=2))
    summary = []
    for name,g in pd.DataFrame(records).groupby('model'):
        for metric in ('balanced_accuracy','auc','recall_left','recall_right'):
            lo,hi = bootstrap_mean(g[metric])
            summary.append(dict(model=name,metric=metric,mean=float(g[metric].mean()),ci_low=lo,ci_high=hi))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    environment = {p:importlib.metadata.version(p) for p in ['numpy','scipy','mne','scikit-learn','pyriemann']}
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
