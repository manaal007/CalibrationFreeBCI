"""Experiment 15A: nested subject-wise CV on BNCI only; no Cho labels accessed.
Place beside 15_external_transfer_pilot.py and run in the same PyCharm environment.
Computationally intensive: nested 9x8 subject-held-out fits per candidate.
"""
import importlib.util
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM

HERE = Path(__file__).resolve().parent
OUT = HERE / 'experiment15A_source_optimization'
PILOT = HERE / '15_external_transfer_pilot.py'
# Fixed BEFORE source cross-validation; no target performance is used to select candidates.
GRID = {
    'CSP_LDA': [
        {'n_components': 4, 'reg': 'ledoit_wolf', 'lda': 'svd'},
        {'n_components': 6, 'reg': 'ledoit_wolf', 'lda': 'svd'},
        {'n_components': 8, 'reg': 'ledoit_wolf', 'lda': 'svd'},
        {'n_components': 4, 'reg': 'oas', 'lda': 'svd'},
    ],
    'Riemann_MDM': [
        {'covariance': 'oas', 'metric': 'riemann'},
        {'covariance': 'lwf', 'metric': 'riemann'},
    ],
}
# Baseline = candidate index 0 for each family.


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def load_pilot():
    require(PILOT.is_file(), f'Missing prerequisite {PILOT.name} in project directory')
    spec = importlib.util.spec_from_file_location('calibrationfree_pilot', PILOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_for(family, cfg):
    if family == 'CSP_LDA':
        return make_pipeline(
            CSP(n_components=cfg['n_components'], reg=cfg['reg'], log=True,
                norm_trace=False),
            LinearDiscriminantAnalysis(solver=cfg['lda']))
    return make_pipeline(Covariances(estimator=cfg['covariance']),
                         MDM(metric=cfg['metric']))


def score_right(model, family, x):
    if family == 'CSP_LDA':
        return model.predict_proba(x)[:, list(model.classes_).index(2)]
    cov = model.named_steps['covariances'].transform(x)
    mdm = model.named_steps['mdm']
    distances = mdm.transform(cov)
    p = list(mdm.classes_).index(2)
    n = list(mdm.classes_).index(1)
    return distances[:, n] - distances[:, p]


def evaluate(model, family, x, y):
    scores = score_right(model, family, x)
    predictions = model.predict(x)
    return {'auc': float(roc_auc_score(y == 2, scores)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predictions))}


def loso_candidates(x, y, subjects, candidate_grid, family, heldout_subjects):
    rows = []
    for idx, cfg in enumerate(candidate_grid):
        fold_scores = []
        for heldout in heldout_subjects:
            train = subjects != heldout
            valid = subjects == heldout
            require(train.any() and valid.any(), 'Empty subject CV fold')
            clf = model_for(family, cfg)
            clf.fit(x[train], y[train])
            result = evaluate(clf, family, x[valid], y[valid])
            fold_scores.append(result['auc'])
        rows.append({'candidate': idx, 'mean_subject_auc': float(np.mean(fold_scores)),
                     'fold_auc': fold_scores, 'parameters': cfg})
    # Prespecified tie-breaking: lowest index in GRID.
    winner = sorted(rows, key=lambda r: (-r['mean_subject_auc'], r['candidate']))[0]
    return winner, rows


def main():
    OUT.mkdir(exist_ok=True)
    report = {'experiment': '15A', 'status': 'NOT RUN',
              'target_data_accessed': False,
              'source_subjects': list(range(1, 10)),
              'outer_cv': 'leave one source subject out',
              'inner_cv': 'leave one of eight training subjects out',
              'selection_metric': 'mean per-subject ROC-AUC',
              'tie_break': 'earliest candidate in frozen grid',
              'candidate_grid': GRID,
              'warning': 'Source CV performance is not an external Cho estimate.'}
    try:
        pilot = load_pilot()
        print('EXPERIMENT 15A — NESTED SOURCE-ONLY CV', flush=True)
        pilot.configure_source_cache()
        x, y, runs = pilot.load_source()
        require(x.shape == (2592, 22, 501), f'Unexpected BNCI shape {x.shape}')
        # The loader has exactly 288 epochs per participant in ascending order.
        subjects = np.repeat(np.arange(1, 10), 288)
        require(len(subjects) == len(y) and all(Counter(y[subjects == s]) == {1: 144, 2: 144}
                    for s in range(1, 10)), 'Unexpected subject alignment')
        report['source_shape'] = list(x.shape)
        outer_rows, inner_rows, selection_rows = [], [], []
        for family, grid in GRID.items():
            print('\nMODEL FAMILY:', family, flush=True)
            for outer in range(1, 10):
                tr = subjects != outer
                te = subjects == outer
                inner_subjects = [s for s in range(1, 10) if s != outer]
                print(f'  Outer subject {outer}/9 — inner CV...', flush=True)
                winner, candidates = loso_candidates(x[tr], y[tr], subjects[tr],
                                                      grid, family, inner_subjects)
                for row in candidates:
                    inner_rows.append({'family': family, 'outer_subject': outer,
                                       'candidate': row['candidate'],
                                       'mean_inner_auc': row['mean_subject_auc'],
                                       'fold_auc_json': json.dumps(row['fold_auc'])})
                for role, idx in [('untuned_baseline', 0), ('nested_tuned', winner['candidate'])]:
                    clf = model_for(family, grid[idx])
                    clf.fit(x[tr], y[tr])
                    result = evaluate(clf, family, x[te], y[te])
                    outer_rows.append({'family': family, 'outer_subject': outer,
                                       'role': role, 'candidate': idx, **result})
                print(f'    selected #{winner["candidate"]} '
                      f'(inner mean AUC={winner["mean_subject_auc"]:.4f})', flush=True)
            print('  Final selection via source-only 9-fold LOSO...', flush=True)
            final, candidates = loso_candidates(x, y, subjects, grid, family, list(range(1, 10)))
            for row in candidates:
                selection_rows.append({'family': family, 'candidate': row['candidate'],
                                       'mean_source_auc': row['mean_subject_auc'],
                                       'fold_auc_json': json.dumps(row['fold_auc']),
                                       'selected': row['candidate'] == final['candidate']})
            for role, idx in [('untuned_baseline', 0), ('source_tuned', final['candidate'])]:
                clf = model_for(family, grid[idx])
                clf.fit(x, y)
                joblib.dump(clf, OUT / f'{family}_{role}.joblib', compress=3)
            report.setdefault('selected', {})[family] = {
                'candidate_index': final['candidate'], 'parameters': final['parameters'],
                'source_cv_mean_auc': final['mean_subject_auc']}
            print('  Frozen selection:', report['selected'][family], flush=True)
        pd.DataFrame(outer_rows).to_csv(OUT / '15A_nested_outer_results.csv', index=False)
        pd.DataFrame(inner_rows).to_csv(OUT / '15A_inner_search.csv', index=False)
        pd.DataFrame(selection_rows).to_csv(OUT / '15A_final_source_selection.csv', index=False)
        report['status'] = 'COMPLETE'
        print('15A COMPLETE. Models saved for 15B.', flush=True)
    except Exception as exc:
        report['status'] = 'STOPPED'
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc(limit=10)
        print('STOPPED:', report['error'], flush=True)
    finally:
        (OUT / '15A_protocol.json').write_text(json.dumps(report, indent=2, default=str),
                                               encoding='utf-8')
    if report['status'] != 'COMPLETE':
        sys.exit(1)


if __name__ == '__main__':
    main()
