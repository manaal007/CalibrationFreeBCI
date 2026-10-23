"""Experiment 14: harmonization feasibility audit, no classifier training.
Reads existing BNCI2014-001 and Cho2017 Subject 1 files only.
Fails explicitly if event semantics or signal units cannot be verified.
"""
from pathlib import Path
from collections import Counter
import json
import os
import sys
import traceback
import numpy as np
import mne
from scipy.io import loadmat
from scipy.signal import resample_poly
from huggingface_hub import scan_cache_dir
import moabb
from moabb.datasets import BNCI2014_001, Cho2017

CHANNELS = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
SFREQ = 250
BAND = (8., 30.)
TMIN, TMAX = .5, 2.5
N_SAMPLES = 501
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiment14_harmonization'
CHO = Path.home() / 'mne_data/MNE-gigadb-data/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data/s01.mat'


def cached_bnci():
    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_type != 'dataset' or repo.repo_id != 'EEGDash/eeg-mi-datasets':
            continue
        for revision in repo.revisions:
            base = Path(revision.snapshot_path) / 'BNCI2014_001'
            if list(base.rglob('A01T.mat')) and list(base.rglob('A01E.mat')):
                os.environ['MNE_DATASETS_BNCI_PATH'] = str(base)
                moabb.set_download_provider('upstream')
                print('Using cached BNCI:', base)
                return str(base)
    raise FileNotFoundError('No complete cached BNCI Subject 1 files found; no download attempted')


def validate(label, x):
    if x.ndim != 3 or x.shape[1:] != (len(CHANNELS), N_SAMPLES):
        raise ValueError(f'{label}: shape {x.shape}, expected (trials, 22, 501)')
    if not np.isfinite(x).all():
        raise ValueError(f'{label}: NaN/Inf detected')
    return {'shape': list(x.shape), 'mean_absolute_raw_value': float(np.abs(x).mean()),
            'max_absolute_raw_value': float(np.abs(x).max())}


def event_codes(raw, dataset):
    """Accept descriptive labels or exact numeric IDs from MOABB; never guess codes."""
    desc = set(raw.annotations.description)
    names = ('left_hand', 'right_hand')
    if set(names).issubset(desc):
        return {'left_hand': 'left_hand', 'right_hand': 'right_hand'}
    # MOABB BNCI raw annotations can be numeric strings from dataset.event_id.
    codes = {name: str(dataset.event_id[name]) for name in names}
    if set(codes.values()).issubset(desc):
        return codes
    raise RuntimeError(f'Cannot verify event mapping. Annotation descriptions: {sorted(desc)}; '
                       f'MOABB event_id: {dataset.event_id}. No labels guessed.')


def source_data():
    dataset = BNCI2014_001()
    runs = dataset.get_data(subjects=[1])[1]
    arrays, labels, details = [], [], []
    for session, session_runs in runs.items():
        for run_id, raw_orig in session_runs.items():
            if not isinstance(raw_orig, mne.io.BaseRaw):
                raise TypeError(f'Expected MNE Raw for {session}/{run_id}, got {type(raw_orig)}')
            raw = raw_orig.copy().load_data()
            missing = [c for c in CHANNELS if c not in raw.ch_names]
            if missing:
                raise ValueError(f'BNCI missing channels: {missing}')
            if any(raw.get_channel_types(picks=[c])[0] != 'eeg' for c in CHANNELS):
                raise ValueError('BNCI channel type mismatch')
            codes = event_codes(raw, dataset)
            raw.pick(CHANNELS)
            raw.reorder_channels(CHANNELS)
            raw.set_eeg_reference('average', projection=False, verbose=False)
            # Original BNCI recording is a continuous run; filter only within this run.
            raw.filter(l_freq=BAND[0], h_freq=BAND[1], picks='eeg', verbose=False)
            # BNCI is already 250 Hz, but guard against unexpected rates.
            if not np.isclose(raw.info['sfreq'], SFREQ):
                raise RuntimeError(f'Unexpected BNCI sampling rate: {raw.info["sfreq"]}')
            mapping = {codes['left_hand']: 1, codes['right_hand']: 2}
            events, _ = mne.events_from_annotations(raw, event_id=mapping, verbose=False)
            if len(events) == 0:
                raise RuntimeError('No BNCI motor imagery events extracted')
            epochs = mne.Epochs(raw, events, event_id={'left_hand': 1, 'right_hand': 2},
                                tmin=TMIN, tmax=TMAX, baseline=None, preload=True,
                                reject_by_annotation=True, verbose=False)
            x = epochs.get_data(copy=True)
            arrays.append(x)
            labels.extend(['left_hand' if k == 1 else 'right_hand' for k in epochs.events[:, 2]])
            details.append({'session': str(session), 'run': str(run_id),
                            'event_mapping': codes, 'epochs': len(x),
                            'raw_annotation_descriptions': sorted(set(raw.annotations.description))})
            print(f'BNCI {session}/{run_id}: {len(x)} epochs; mapping={codes}')
    if not arrays:
        raise RuntimeError('No BNCI runs')
    return np.concatenate(arrays), labels, details


def target_data():
    if not CHO.is_file():
        raise FileNotFoundError(f'Cho2017 not found: {CHO}')
    eeg = loadmat(CHO, squeeze_me=True, struct_as_record=False)['eeg']
    rate = float(np.asarray(eeg.srate).item())
    n_trials = int(np.asarray(eeg.n_imagery_trials).item())
    if rate != 512 or n_trials != 100:
        raise ValueError(f'Unexpected Cho metadata: rate={rate}, n_trials={n_trials}')
    for field in ('bad_trial_idx_voltage', 'bad_trial_idx_mi'):
        entries = np.asarray(getattr(eeg.bad_trial_indices, field), dtype=object).ravel()
        if len(entries) != 2 or any(np.asarray(v).size for v in entries):
            raise RuntimeError(f'Cho {field} is nonempty or unexpected; exclusion protocol required')

    # Use MOABB to obtain the actual channel names, not the outer OrderedDict.
    sessions = Cho2017().get_data(subjects=[1])[1]
    raw = next(raw for runs in sessions.values() for raw in runs.values())
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError('Cho loader did not return MNE Raw')
    names = raw.copy().pick_types(eeg=True).ch_names
    if len(names) != 64 or any(c not in names for c in CHANNELS):
        raise RuntimeError(f'Unexpected Cho montage: {names}')
    indices = [names.index(c) for c in CHANNELS]
    print('Cho channel mapping indices (zero-based):', indices)

    # Check cue marker location in each of the 100 original 7-second segments.
    markers = np.asarray(eeg.imagery_event).ravel()
    if markers.shape != (n_trials * 3584,):
        raise RuntimeError(f'Unexpected imagery_event length: {markers.shape}')
    per_trial = markers.reshape(n_trials, 3584)
    positions = [np.flatnonzero(v) for v in per_trial]
    if any(len(p) != 1 or p[0] != 1023 for p in positions):
        raise RuntimeError('Imagery cue markers differ from expected sample 1023; stop for review')

    result, labels = [], []
    for label, field in [('left_hand', 'imagery_left'), ('right_hand', 'imagery_right')]:
        arr = np.asarray(getattr(eeg, field))
        if arr.shape != (68, n_trials * 3584):
            raise RuntimeError(f'Unexpected {field} shape: {arr.shape}')
        # Original MATLAB values are not converted to volts here: units remain unverified.
        trials = arr[:64].reshape(64, n_trials, 3584).transpose(1, 0, 2)
        for trial in trials:
            x = np.asarray(trial[indices], dtype=np.float64)
            x -= x.mean(axis=0, keepdims=True)  # CAR on selected 22 electrodes
            x = mne.filter.filter_data(x, sfreq=rate, l_freq=BAND[0],
                                       h_freq=BAND[1], verbose=False)
            x = resample_poly(x, 125, 256, axis=-1)
            # Source event sample is at 1023/512 s; target 250 Hz sample nearest it.
            start = int(round((1023 / rate + TMIN) * SFREQ))
            epoch = x[:, start:start + N_SAMPLES]
            if epoch.shape != (22, N_SAMPLES):
                raise RuntimeError(f'Cho epoch shape: {epoch.shape}')
            result.append(epoch)
            labels.append(label)
    return np.stack(result), labels, {'units': 'UNVERIFIED original MATLAB units',
                                      'row_mapping': 'MOABB 64 EEG channel order; requires source-code confirmation',
                                      'cue_sample_zero_based': 1023}


def main():
    print('EXPERIMENT 14 — HARMONIZATION IMPLEMENTATION AUDIT')
    print('No training. No arrays saved for modeling. Protocol NOT frozen.')
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'status': 'NOT FROZEN', 'channels': CHANNELS,
              'bandpass_hz': BAND, 'sfreq_hz': SFREQ, 'epoch_s': [TMIN, TMAX],
              'outstanding': ['Confirm Cho MATLAB physical units',
                              'Confirm Cho MATLAB row-to-electrode mapping against loader',
                              'Confirm common-reference scientific suitability',
                              'Validate full subject coverage and final leakage safeguards']}
    try:
        report['bnci_cache'] = cached_bnci()
        xs, ys, runs = source_data()
        report['source'] = {'array': validate('BNCI', xs),
                            'class_counts': dict(Counter(ys)), 'runs': runs}
        print('BNCI:', report['source']['array'], dict(Counter(ys)))
        xt, yt, meta = target_data()
        report['target'] = {'array': validate('Cho2017', xt),
                            'class_counts': dict(Counter(yt)), 'metadata': meta}
        print('Cho2017:', report['target']['array'], dict(Counter(yt)))
        report['status'] = 'SHAPE AND NUMERICAL CHECKS PASSED; NOT FROZEN'
        print('SHAPE AND NUMERICAL CHECKS PASSED (not scientific validation)')
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc(limit=5)
        print('AUDIT STOPPED:', report['error'])
    dest = OUT / '14_harmonization_audit.json'
    dest.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print('Saved:', dest)
    print('Protocol frozen: NO')
    if 'error' in report:
        sys.exit(1)


if __name__ == '__main__':
    main()
