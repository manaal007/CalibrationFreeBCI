"""CalibrationFreeBCI: frozen external Cho2017 evaluation, participants 7..52.

Run from the same PyCharm project folder as the original Experiment 15A and 15 scripts.
Never fits or updates a model. Never downloads. Previously inspected subjects 1..6
are excluded from the confirmatory cohort. Missing files are reported, not imputed.

IMPORTANT: bad-trial metadata interpretation (1-based, left/right class-specific)
remains an ASSUMPTION pending independent verification against original dataset
materials. Therefore this script labels all outputs PROVISIONAL, not confirmed.
"""
import csv
import hashlib
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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / 'experiment15A_source_optimization'
OUT = HERE / 'experiment15B_confirmatory_external'
SOURCE_SCRIPT = HERE / '15A_source_only_optimization.py'
PILOT_SCRIPT = HERE / '15_external_transfer_pilot.py'
MAT_DIR = (Path.home() / 'mne_data/MNE-gigadb-data/gigadb-datasets/live/pub/'
           '10.5524/100001_101000/100295/mat_data')
CONFIRMATORY_SUBJECTS = tuple(range(7, 53))
INDEX_POLICY_VERIFIED = False  # Do not change without documented independent verification.


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_module(path, name):
    require(path.is_file(), f'Missing dependency: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def get_bad_trials(eeg, filename, n):
    """Provisional interpretation of two class-specific MATLAB index arrays."""
    excluded = [set(), set()]
    audit = {}
    require(hasattr(eeg, 'bad_trial_indices'), f'{filename}: no bad_trial_indices')
    for field in ('bad_trial_idx_voltage', 'bad_trial_idx_mi'):
        entries = np.asarray(getattr(eeg.bad_trial_indices, field), dtype=object).ravel()
        require(len(entries) == 2, f'{filename}: {field} does not have 2 class entries')
        audit[field] = {}
        for cls, entry in enumerate(entries):
            vals = np.asarray(entry).ravel()
            if not vals.size:
                ids = np.array([], dtype=int)
            else:
                try:
                    nums = vals.astype(float)
                except (ValueError, TypeError):
                    raise RuntimeError(f'{filename}: nonnumeric {field}')
                require(np.isfinite(nums).all() and np.equal(nums, np.floor(nums)).all(),
                        f'{filename}: invalid/noninteger {field}')
                ids = nums.astype(int)
                require(((ids >= 1) & (ids <= n)).all(),
                        f'{filename}: {field} indices outside 1..{n}: {ids.tolist()}')
            label = ('left', 'right')[cls]
            audit[field][label] = ids.tolist()
            excluded[cls].update((ids - 1).tolist())
    return excluded, audit


def read_subject(path, indices):
    eeg = loadmat(path, squeeze_me=True, struct_as_record=False)['eeg']
    sfreq = float(np.asarray(eeg.srate).item())
    n = int(np.asarray(eeg.n_imagery_trials).item())
    require(sfreq == 512 and n > 0,
            f'{path.name}: unexpected sample rate/trials (sfreq={sfreq}, n={n})')
    require(np.asarray(eeg.imagery_event).size == n * 3584,
            f'{path.name}: imagery_event size inconsistent with n_imagery_trials={n}')
    marker = np.asarray(eeg.imagery_event).reshape(n, 3584)
    require(all(np.array_equal(np.flatnonzero(row), [1023]) for row in marker),
            f'{path.name}: unexpected cue markers')
    exclusions, metadata = get_bad_trials(eeg, path.name, n)
    start = int(round((1023 / 512 + 0.5) * 250))
    xs, ys, trial_ids = [], [], []
    for cls, (label, field) in enumerate(((1, 'imagery_left'), (2, 'imagery_right'))):
        arr = np.asarray(getattr(eeg, field), dtype=np.float64)
        require(arr.shape == (68, n * 3584), f'{path.name}: invalid {field} shape')
        trials = arr[:64].reshape(64, n, 3584).transpose(1, 0, 2)
        for trial_index, trial in enumerate(trials):
            if trial_index in exclusions[cls]:
                continue
            signal = trial[indices].copy() * 1e-6
            signal -= signal.mean(axis=0, keepdims=True)
            # Filter EACH original trial; never filter zero-buffer concatenated MOABB raw.
            signal = mne.filter.filter_data(signal, sfreq=512, l_freq=8., h_freq=30., verbose=False)
            down = resample_poly(signal, 125, 256, axis=-1)
            epoch = down[:, start:start + 501]
            require(epoch.shape == (22, 501) and np.isfinite(epoch).all(),
                    f'{path.name}: invalid epoch')
            xs.append(epoch)
            ys.append(label)
            trial_ids.append((('left', 'right')[cls], trial_index + 1))
    require(len(set(ys)) == 2, f'{path.name}: missing class after exclusions')
    quality = dict(subject=int(path.stem[1:]), n_imagery_trials_per_class=n, n_retained=len(ys),
                   n_left_excluded=len(exclusions[0]), n_right_excluded=len(exclusions[1]),
                   n_left_retained=ys.count(1), n_right_retained=ys.count(2),
                   bad_trial_metadata=metadata)
    return np.stack(xs), np.asarray(ys), trial_ids, quality


def main():
    OUT.mkdir(exist_ok=True)
    report = dict(experiment='15B_confirmatory', status='NOT_RUN',
                  interpretation='PROVISIONAL_BAD_TRIAL_SEMANTICS_UNVERIFIED',
                  bad_trial_index_policy_verified=INDEX_POLICY_VERIFIED,
                  bad_trial_assumption='MATLAB 1-based, first entry left, second entry right; union voltage and MI flags',
                  frozen_models=True, target_fitting=False, target_threshold_tuning=False,
                  downloads=False, exploratory_subjects_excluded=list(range(1, 7)),
                  requested_subjects=list(CONFIRMATORY_SUBJECTS),
                  caveats=['External cohort may be incomplete if files are not cached.',
                           'Bad-trial field semantics require independent documentation verification.',
                           'Cho and BNCI use different acquisition protocols.'])
    try:
        print('15B — FROZEN CONFIRMATORY COHORT (CHO SUBJECTS 7..52)')
        require((MODEL_DIR / '15A_protocol.json').is_file(), 'Missing 15A protocol')
        source_protocol = json.loads((MODEL_DIR / '15A_protocol.json').read_text(encoding='utf-8'))
        require(source_protocol.get('status') == 'COMPLETE', '15A protocol not complete')
        source = load_module(SOURCE_SCRIPT, 'frozen_source_models')
        pilot = load_module(PILOT_SCRIPT, 'frozen_cho_channels')
        names = pilot.get_cho_eeg_names()
        indices = [names.index(c) for c in pilot.CHANNELS]
        require(len(indices) == 22 and len(set(indices)) == 22, 'Invalid channel mapping')
        models = {}
        for family in source.GRID:
            for role in ('untuned_baseline', 'source_tuned'):
                key = f'{family}_{role}'
                path = MODEL_DIR / f'{key}.joblib'
                require(path.is_file(), f'Missing model: {path}')
                models[key] = (family, joblib.load(path))
        report['model_sha256'] = {name: sha256(MODEL_DIR / f'{name}.joblib') for name in models}
        available = [s for s in CONFIRMATORY_SUBJECTS if (MAT_DIR / f's{s:02d}.mat').is_file()]
        missing = sorted(set(CONFIRMATORY_SUBJECTS) - set(available))
        report.update(available_subjects=available, missing_subjects=missing)
        print('Cached confirmatory subjects:', available)
        print('Missing:', len(missing), 'of', len(CONFIRMATORY_SUBJECTS))
        require(not missing, f'Missing Cho subjects: {missing}. Download these first to preserve full-cohort evaluation.')
        subject_rows, trial_rows, quality_rows = [], [], []
        for s in available:
            print(f'Subject {s}: processing...', flush=True)
            x, y, ids, quality = read_subject(MAT_DIR / f's{s:02d}.mat', indices)
            quality_rows.append(quality)
            for name, (family, model) in models.items():
                score = np.asarray(source.score_right(model, family, x), dtype=float)
                pred = np.asarray(model.predict(x), dtype=int)
                require(len(score) == len(y) and len(pred) == len(y) and np.isfinite(score).all(),
                        f'Subject {s}: invalid model outputs')
                subject_rows.append(dict(subject=s, cohort='confirmatory', model=name,
                                         auc=float(roc_auc_score(y == 2, score)),
                                         balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                                         accuracy=float(accuracy_score(y, pred)),
                                         predicted_right_fraction=float(np.mean(pred == 2)),
                                         n_trials=len(y)))
                trial_rows.extend(dict(subject=s, cohort='confirmatory', model=name,
                                       imagery_class=ids[i][0], matlab_trial=ids[i][1],
                                       true_label=int(y[i]), predicted_label=int(pred[i]),
                                       score_right=float(score[i])) for i in range(len(y)))
            print(f'  Retained {len(y)}; excluded L={quality["n_left_excluded"]}, R={quality["n_right_excluded"]}', flush=True)
        metrics = pd.DataFrame(subject_rows)
        summary = metrics.groupby('model', as_index=False).agg(
            n_subjects=('subject', 'nunique'), mean_auc=('auc', 'mean'),
            sd_auc=('auc', 'std'), mean_balanced_accuracy=('balanced_accuracy', 'mean'))
        # Write only once every subject has been processed successfully.
        metrics.to_csv(OUT / '15B_subject_metrics.csv', index=False)
        pd.DataFrame(trial_rows).to_csv(OUT / '15B_trial_predictions.csv', index=False)
        summary.to_csv(OUT / '15B_subject_summary.csv', index=False)
        (OUT / '15B_quality_by_subject.json').write_text(json.dumps(quality_rows, indent=2), encoding='utf-8')
        report['evaluated_subjects'] = available
        report['status'] = 'COMPLETE_PARTIAL_COHORT' if missing else 'COMPLETE_FULL_COHORT'
        print(summary.to_string(index=False))
        print('15B finished. RESULTS PROVISIONAL until bad-trial index policy is verified.')
    except Exception as exc:
        report['status'] = 'STOPPED'
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc(limit=8)
        print('STOPPED:', report['error'])
    finally:
        (OUT / '15B_protocol.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if report['status'] == 'STOPPED':
        sys.exit(1)

if __name__ == '__main__':
    main()
