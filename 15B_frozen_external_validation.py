"""Experiment 15B: frozen BNCI models evaluated on locally cached Cho2017 subjects 2..52.
No target fitting or downloads. MATLAB bad-trial indices are interpreted as 1-based,
class-specific trial positions; ambiguous/out-of-range metadata stops evaluation.
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

HERE = Path(__file__).resolve().parent
A = HERE / '15A_source_only_optimization.py'
PILOT = HERE / '15_external_transfer_pilot.py'
MODELS = HERE / 'experiment15A_source_optimization'
OUT = HERE / 'experiment15B_external_validation'
SUBJECTS = list(range(2, 53))
MAT_DIR = (Path.home() / 'mne_data/MNE-gigadb-data/gigadb-datasets/live/pub/'
           '10.5524/100001_101000/100295/mat_data')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load_module(path, name):
    require(path.is_file(), f'Missing required file: {path.name}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bad_trial_sets(eeg, mat_name, n):
    """Return class-specific, zero-based exclusions converted from MATLAB 1-based IDs.

    Class ordering assumed to match imagery_left, imagery_right in the Cho MATLAB
    structure. Do not silently guess if the metadata structure is unexpected.
    """
    exclusions = [set(), set()]
    audit = {}
    for field in ('bad_trial_idx_voltage', 'bad_trial_idx_mi'):
        raw = getattr(eeg.bad_trial_indices, field)
        entries = np.asarray(raw, dtype=object).ravel()
        require(len(entries) == 2,
                f'{mat_name}: expected two class-specific entries in {field}; got {len(entries)}')
        audit[field] = {}
        for class_idx, item in enumerate(entries):
            vals = np.asarray(item).ravel()
            if vals.size == 0:
                ids = np.empty(0, dtype=int)
            else:
                require(np.issubdtype(vals.dtype, np.number),
                        f'{mat_name}: nonnumeric {field} class {class_idx + 1}')
                numeric = np.asarray(vals, dtype=float)
                require(np.isfinite(numeric).all() and np.equal(numeric, np.floor(numeric)).all(),
                        f'{mat_name}: invalid/noninteger indices in {field}')
                ids = numeric.astype(int)
                require(np.all((ids >= 1) & (ids <= n)),
                        f'{mat_name}: {field} has indices outside MATLAB 1..{n}: {ids.tolist()}')
            audit[field][('left', 'right')[class_idx]] = ids.tolist()
            exclusions[class_idx].update((ids - 1).tolist())
    return exclusions, audit


def read_subject(mat_path, channels, indices):
    eeg = loadmat(mat_path, squeeze_me=True, struct_as_record=False)['eeg']
    rate = float(np.asarray(eeg.srate).item())
    n = int(np.asarray(eeg.n_imagery_trials).item())
    require(rate == 512 and n == 100, f'{mat_path.name}: unexpected rate/trial count')
    marker = np.asarray(eeg.imagery_event).reshape(n, 3584)
    require(all(np.array_equal(np.flatnonzero(row), [1023]) for row in marker),
            f'{mat_path.name}: event marker mismatch')

    exclusions, audit = bad_trial_sets(eeg, mat_path.name, n)
    start = int(round((1023 / 512 + 0.5) * 250))
    x, y, trial_ids = [], [], []
    for class_idx, (label, field) in enumerate(((1, 'imagery_left'), (2, 'imagery_right'))):
        values = np.asarray(getattr(eeg, field), dtype=np.float64)
        require(values.shape == (68, n * 3584),
                f'{mat_path.name}: unexpected {field} dimensions {values.shape}')
        trials = values[:64].reshape(64, n, 3584).transpose(1, 0, 2)
        for trial_idx, trial in enumerate(trials):
            if trial_idx in exclusions[class_idx]:
                continue
            signal = trial[indices].copy() * 1e-6
            signal -= signal.mean(axis=0, keepdims=True)
            signal = mne.filter.filter_data(signal, sfreq=512, l_freq=8.0,
                                            h_freq=30.0, verbose=False)
            down = resample_poly(signal, 125, 256, axis=-1)
            epoch = down[:, start:start + 501]
            require(epoch.shape == (22, 501) and np.isfinite(epoch).all(),
                    f'{mat_path.name}: invalid epoch')
            x.append(epoch)
            y.append(label)
            trial_ids.append({'class': ('left', 'right')[class_idx],
                              'original_matlab_trial': trial_idx + 1})
    y = np.asarray(y, dtype=int)
    require(len(np.unique(y)) == 2, f'{mat_path.name}: missing class after exclusions')
    quality = {'subject': int(mat_path.stem[1:]),
               'n_left_excluded': len(exclusions[0]),
               'n_right_excluded': len(exclusions[1]),
               'n_left_retained': int(np.sum(y == 1)),
               'n_right_retained': int(np.sum(y == 2)),
               'bad_trial_metadata': audit}
    return np.stack(x), y, trial_ids, quality


def main():
    OUT.mkdir(exist_ok=True)
    report = {'experiment': '15B', 'status': 'NOT RUN',
              'external_test_subjects_requested': SUBJECTS,
              'excluded_exploratory_subject': 1,
              'target_model_fitting': False, 'no_downloads': True,
              'bad_trial_policy': ('Union of class-specific voltage/MI bad-trial IDs; '
                                   'MATLAB 1-based positions converted to Python 0-based; '
                                   'filter each original trial separately'),
              'caveats': ['Cross-dataset acquisition/reference/cue differences.',
                          'Uncached Cho files are not counted as evaluated.',
                          'Frozen BNCI hyperparameters and thresholds.',
                          'Bad-trial index semantics should be confirmed against original '
                          'dataset documentation before publication.']}
    try:
        print('EXPERIMENT 15B — FROZEN CHO EXTERNAL VALIDATION', flush=True)
        a = load_module(A, 'source_optimizer')
        pilot = load_module(PILOT, 'pilot_preprocessing')
        protocol_path = MODELS / '15A_protocol.json'
        require(protocol_path.is_file(), 'Run Experiment 15A first (protocol missing)')
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        require(protocol.get('status') == 'COMPLETE', '15A did not complete successfully')
        models = {}
        for family in a.GRID:
            for role in ('untuned_baseline', 'source_tuned'):
                name = f'{family}_{role}'
                path = MODELS / f'{name}.joblib'
                require(path.is_file(), f'Missing fitted model: {path}')
                models[name] = (family, joblib.load(path))
        available = [s for s in SUBJECTS if (MAT_DIR / f's{s:02d}.mat').is_file()]
        missing = sorted(set(SUBJECTS) - set(available))
        report['available_subjects'] = available
        report['missing_subjects'] = missing
        print(f'Cached Cho subjects available: {available}', flush=True)
        require(len(available) >= 2, 'Need at least two cached untouched Cho subjects.')
        names = pilot.get_cho_eeg_names()
        indices = [names.index(c) for c in pilot.CHANNELS]
        results, trial_rows, quality_rows = [], [], []
        for subject in available:
            print(f'Cho subject {subject}: processing...', flush=True)
            x, y, trial_ids, quality = read_subject(
                MAT_DIR / f's{subject:02d}.mat', pilot.CHANNELS, indices)
            quality_rows.append(quality)
            print(f'  Retained {len(y)}; excluded left={quality["n_left_excluded"]}, '
                  f'right={quality["n_right_excluded"]}', flush=True)
            for name, (family, model) in models.items():
                score = a.score_right(model, family, x)
                pred = model.predict(x)
                results.append({'subject': subject, 'model': name,
                                'auc': float(roc_auc_score(y == 2, score)),
                                'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
                                'accuracy': float(accuracy_score(y, pred)),
                                'n_trials': len(y)})
                trial_rows.extend({'subject': subject, 'trial': i,
                                   'class': trial_ids[i]['class'],
                                   'original_matlab_trial': trial_ids[i]['original_matlab_trial'],
                                   'model': name, 'true_label': int(y[i]),
                                   'predicted_label': int(pred[i]),
                                   'score_right': float(score[i])}
                                  for i in range(len(y)))
            print(f'  Finished subject {subject}', flush=True)
        pd.DataFrame(results).to_csv(OUT / '15B_subject_metrics.csv', index=False)
        pd.DataFrame(trial_rows).to_csv(OUT / '15B_trial_predictions.csv', index=False)
        summary = (pd.DataFrame(results).groupby('model')
                   .agg(n_subjects=('subject', 'nunique'), mean_auc=('auc', 'mean'),
                        sd_auc=('auc', 'std'),
                        mean_balanced_accuracy=('balanced_accuracy', 'mean'))
                   .reset_index())
        summary.to_csv(OUT / '15B_subject_summary.csv', index=False)
        report['quality_by_subject'] = quality_rows
        report['evaluated_subjects'] = available
        report['status'] = 'COMPLETE_PARTIAL_COHORT' if missing else 'COMPLETE_ALL_REQUESTED'
        print(summary.to_string(index=False), flush=True)
        print('15B COMPLETE — review participant-level variability before conclusions.', flush=True)
    except Exception as exc:
        report['status'] = 'STOPPED'
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc(limit=10)
        print('STOPPED:', report['error'], flush=True)
    finally:
        (OUT / '15B_protocol.json').write_text(json.dumps(report, indent=2, default=str),
                                               encoding='utf-8')
    if report['status'] == 'STOPPED':
        sys.exit(1)


if __name__ == '__main__':
    main()
