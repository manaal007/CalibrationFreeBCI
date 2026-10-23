"""Experiment 15: prespecified BNCI2014-001 -> Cho2017 Subject 1 external pilot.
No target calibration, hyperparameter search, or download requests.
Requires locally cached BNCI A01..A09 training AND evaluation files, Cho s01.mat.
This is a PILOT, not a final cohort-level generalization claim.
"""
import json
import os
import sys
import traceback
from pathlib import Path
from collections import Counter

import mne
import moabb
import numpy as np
import pandas as pd
from huggingface_hub import scan_cache_dir
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from moabb.datasets import BNCI2014_001, Cho2017

CHANNELS = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6',
            'CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
SOURCE_SUBJECTS = list(range(1, 10))
TARGET_SUBJECT = 1
SFREQ = 250
BAND = (8.0, 30.0)
WINDOW = (0.5, 2.5)  # seconds after class marker, inclusive endpoints
N_SAMPLES = 501
CHO_RATE = 512
CHO_TRIAL_SAMPLES = 3584
CHO_MARKER_SAMPLE = 1023
CHO_FILE = (Path.home() / 'mne_data/MNE-gigadb-data/gigadb-datasets/live/pub/'
            '10.5524/100001_101000/100295/mat_data/s01.mat')
OUT = Path(__file__).resolve().parent / 'experiment15_external_transfer'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def configure_source_cache():
    for repo in scan_cache_dir().repos:
        if repo.repo_type == 'dataset' and repo.repo_id == 'EEGDash/eeg-mi-datasets':
            for rev in repo.revisions:
                folder = Path(rev.snapshot_path) / 'BNCI2014_001'
                if all(list(folder.rglob(f'A{s:02d}T.mat')) and
                       list(folder.rglob(f'A{s:02d}E.mat')) for s in SOURCE_SUBJECTS):
                    os.environ['MNE_DATASETS_BNCI_PATH'] = str(folder)
                    moabb.set_download_provider('upstream')
                    return folder
    raise FileNotFoundError('All nine BNCI subjects (A01T/E through A09T/E) are not '
                            'present in the verified Hugging Face cache. STOPPED before '
                            'model fitting to avoid unexpected downloads. Do not train on '
                            'only Subject 1 as a substitute.')


def source_event_mapping(raw, dataset):
    descriptions = set(raw.annotations.description)
    if {'left_hand', 'right_hand'}.issubset(descriptions):
        return {'left_hand': 1, 'right_hand': 2}
    codes = {name: str(dataset.event_id[name]) for name in ('left_hand', 'right_hand')}
    require(set(codes.values()).issubset(descriptions),
            f'BNCI annotations not recognized: {sorted(descriptions)}')
    return {codes['left_hand']: 1, codes['right_hand']: 2}


def load_source():
    ds = BNCI2014_001()
    arrays, labels, run_log = [], [], []
    for subject in SOURCE_SUBJECTS:
        # Load one participant at a time; do not concatenate raw recordings across runs.
        sessions = ds.get_data(subjects=[subject])[subject]
        subject_count = 0
        for session, runs in sessions.items():
            for run_name, raw_original in runs.items():
                require(isinstance(raw_original, mne.io.BaseRaw), 'BNCI returned non-Raw object')
                require(all(c in raw_original.ch_names for c in CHANNELS),
                        f'BNCI subject {subject}: missing channels')
                require(np.isclose(raw_original.info['sfreq'], SFREQ),
                        f'BNCI subject {subject}: unexpected sampling frequency')
                require(all(raw_original.get_channel_types(picks=[c])[0] == 'eeg' for c in CHANNELS),
                        f'BNCI subject {subject}: channel type mismatch')
                mapping = source_event_mapping(raw_original, ds)
                raw = raw_original.copy().load_data()
                raw.pick(CHANNELS)
                raw.reorder_channels(CHANNELS)
                raw.set_eeg_reference('average', projection=False, verbose=False)
                raw.filter(l_freq=BAND[0], h_freq=BAND[1], picks='eeg', verbose=False)
                events, _ = mne.events_from_annotations(raw, event_id=mapping, verbose=False)
                epochs = mne.Epochs(raw, events,
                                    event_id={'left_hand': 1, 'right_hand': 2},
                                    tmin=WINDOW[0], tmax=WINDOW[1], baseline=None,
                                    preload=True, reject_by_annotation=True, verbose=False)
                x = epochs.get_data(copy=True)
                y = epochs.events[:, 2].astype(int)
                require(x.ndim == 3 and x.shape[1:] == (22, N_SAMPLES),
                        f'BNCI {subject}/{session}/{run_name}: unexpected {x.shape}')
                require(np.isfinite(x).all(), 'Nonfinite BNCI samples')
                arrays.append(x)
                labels.append(y)
                subject_count += len(y)
                run_log.append({'subject': subject, 'session': str(session),
                                'run': str(run_name), 'trials': len(y)})
        print(f'BNCI subject {subject}: {subject_count} binary epochs', flush=True)
        require(subject_count == 288,
                f'BNCI subject {subject}: expected 288 binary epochs, found {subject_count}')
    x = np.concatenate(arrays, axis=0)
    y = np.concatenate(labels, axis=0)
    require(x.shape == (2592, 22, 501), f'Unexpected source array shape: {x.shape}')
    require(Counter(y) == {1: 1296, 2: 1296}, f'Source imbalance: {Counter(y)}')
    return x, y, run_log


def get_cho_eeg_names():
    # MOABB is used for its installed electrode row naming, NOT its zero-buffered
    # continuous signal for filtering. Cho trial filtering uses original MATLAB rows.
    sessions = Cho2017().get_data(subjects=[TARGET_SUBJECT])[TARGET_SUBJECT]
    raw = next(raw for runs in sessions.values() for raw in runs.values())
    require(isinstance(raw, mne.io.BaseRaw), 'Cho loader did not return Raw')
    names = raw.copy().pick(picks='eeg').ch_names
    require(len(names) == 64 and all(c in names for c in CHANNELS),
            'Cho channel layout does not match the validated 64-channel montage')
    return names


def load_target():
    require(CHO_FILE.is_file(), f'Cached Cho Subject 1 file missing: {CHO_FILE}')
    names = get_cho_eeg_names()
    indices = [names.index(c) for c in CHANNELS]
    eeg = loadmat(CHO_FILE, squeeze_me=True, struct_as_record=False)['eeg']
    require(float(np.asarray(eeg.srate).item()) == CHO_RATE, 'Cho rate changed')
    require(int(np.asarray(eeg.n_imagery_trials).item()) == 100, 'Cho trial count changed')
    markers = np.asarray(eeg.imagery_event).reshape(100, CHO_TRIAL_SAMPLES)
    require(all(np.array_equal(np.flatnonzero(row), [CHO_MARKER_SAMPLE]) for row in markers),
            'Cho event markers changed; stop and inspect')
    # Pilot only: Subject 1 had no flagged trials in either metadata field.
    for field in ('bad_trial_idx_voltage', 'bad_trial_idx_mi'):
        entries = np.asarray(getattr(eeg.bad_trial_indices, field), dtype=object).ravel()
        require(len(entries) == 2 and all(np.asarray(v).size == 0 for v in entries),
                f'Cho {field} contains flagged trials; stop for prespecified exclusions')
    start = int(round((CHO_MARKER_SAMPLE / CHO_RATE + WINDOW[0]) * SFREQ))
    stop = start + N_SAMPLES
    out, labels = [], []
    for label, field in ((1, 'imagery_left'), (2, 'imagery_right')):
        values = np.asarray(getattr(eeg, field), dtype=np.float64)
        require(values.shape == (68, 358400), f'Cho {field}: shape {values.shape}')
        segments = values[:64].reshape(64, 100, CHO_TRIAL_SAMPLES).transpose(1, 0, 2)
        for trial in segments:
            # MATLAB EEG is microvolts; installed MOABB loader applies 1e-6.
            signal = trial[indices].copy() * 1e-6
            signal -= signal.mean(axis=0, keepdims=True)  # 22-channel CAR
            signal = mne.filter.filter_data(signal, sfreq=CHO_RATE,
                                            l_freq=BAND[0], h_freq=BAND[1], verbose=False)
            down = resample_poly(signal, 125, 256, axis=-1)
            epoch = down[:, start:stop]
            require(epoch.shape == (22, N_SAMPLES) and np.isfinite(epoch).all(),
                    'Cho epoch shape/finite check failed')
            out.append(epoch)
            labels.append(label)
    x = np.stack(out)
    y = np.asarray(labels, dtype=int)
    require(x.shape == (200, 22, 501) and Counter(y) == {1: 100, 2: 100},
            'Unexpected Cho test shape/class counts')
    return x, y, indices


def make_models():
    # Parameters selected before inspecting target prediction scores.
    return {
        'CSP_LDA': make_pipeline(CSP(n_components=4, reg='ledoit_wolf', log=True,
                                     norm_trace=False), LinearDiscriminantAnalysis()),
        'Riemann_MDM': make_pipeline(Covariances(estimator='oas'), MDM(metric='riemann')),
    }


def main():
    OUT.mkdir(exist_ok=True)
    report = {
        'experiment': '15', 'scope': 'single-target-subject external pilot',
        'source_dataset': 'BNCI2014-001', 'source_subjects': SOURCE_SUBJECTS,
        'target_dataset': 'Cho2017', 'target_subjects': [TARGET_SUBJECT],
        'target_training_or_calibration': False,
        'channels': CHANNELS, 'bandpass_hz': BAND, 'sample_rate_hz': SFREQ,
        'window_after_class_marker_seconds': WINDOW,
        'source_reference': 'average of 22 common electrodes',
        'target_reference': 'average of 22 common electrodes',
        'target_units_conversion': 'MATLAB microvolts multiplied by 1e-6 to volts',
        'source_filtering': 'continuous run, before epoching',
        'target_filtering': 'each original 7-second trial, before epoching',
        'models': {'CSP_LDA': 'CSP 4 components, Ledoit-Wolf regularization, LDA',
                   'Riemann_MDM': 'OAS covariance, Riemannian MDM'},
        'status': 'NOT RUN', 'caveats': [
            'Source and target acquisition references, cue semantics and temporal preprocessing are not fully equivalent.',
            'A single Cho participant cannot support a population-level generalization claim.',
            'No target-driven tuning is permitted; subsequent full-cohort analysis needs prespecified trial exclusions.',
            'Filtering and resampling choices should be reported transparently.'
        ]
    }
    try:
        print('EXPERIMENT 15 — BNCI (9 subjects) -> CHO (Subject 1) PILOT', flush=True)
        report['bnci_cache'] = str(configure_source_cache())
        require(CHO_FILE.is_file(), 'Cho Subject 1 local file missing')
        print('Loading harmonized source training data...', flush=True)
        xs, ys, runs = load_source()
        report['source_shape'] = list(xs.shape)
        report['source_runs'] = runs
        print('Training models on BNCI ONLY...', flush=True)
        fitted = {}
        for name, model in make_models().items():
            print('Fitting:', name, flush=True)
            model.fit(xs, ys)
            fitted[name] = model
        del xs
        # Target labels are not loaded until AFTER all model fits are complete.
        print('Models fitted. Loading held-out Cho Subject 1...', flush=True)
        xt, yt, row_indices = load_target()
        report['target_shape'] = list(xt.shape)
        report['cho_matlab_row_indices'] = row_indices
        records, predictions = [], pd.DataFrame({'trial_index': np.arange(len(yt)),
                                                   'true_label': yt})
        for name, model in fitted.items():
            pred = model.predict(xt)
            # MDM uses distances rather than predict_proba; convert two class
            # distances to a monotone score for class 2 ROC-AUC.
            if name == 'Riemann_MDM':
                distances = model.named_steps['mdm'].transform(
                    model.named_steps['covariances'].transform(xt))
                classes = model.named_steps['mdm'].classes_
                pos = int(np.where(classes == 2)[0][0])
                neg = int(np.where(classes == 1)[0][0])
                score = distances[:, neg] - distances[:, pos]
            else:
                score = model.predict_proba(xt)[:, list(model.classes_).index(2)]
            result = {'model': name,
                      'roc_auc': float(roc_auc_score(yt == 2, score)),
                      'balanced_accuracy': float(balanced_accuracy_score(yt, pred)),
                      'accuracy': float(accuracy_score(yt, pred)),
                      'n_target_trials': int(len(yt))}
            records.append(result)
            predictions[f'{name}_predicted'] = pred
            predictions[f'{name}_score_right'] = score
            print(name, result, flush=True)
        pd.DataFrame(records).to_csv(OUT / '15_pilot_metrics.csv', index=False)
        predictions.to_csv(OUT / '15_pilot_predictions.csv', index=False)
        report['results'] = records
        report['status'] = 'PILOT COMPLETED; FULL COHORT NOT EVALUATED'
        print('PILOT COMPLETE. Do not generalize from one target participant.', flush=True)
    except Exception as exc:
        report['status'] = 'STOPPED'
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc(limit=8)
        print('EXPERIMENT STOPPED:', report['error'], flush=True)
    finally:
        path = OUT / '15_pilot_protocol_and_report.json'
        path.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        print('Saved:', path, flush=True)
    if report['status'] == 'STOPPED':
        sys.exit(1)


if __name__ == '__main__':
    main()
