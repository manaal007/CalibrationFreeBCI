"""Experiment 13C: Cho2017 Subject 1 trial/boundary integrity audit.

Read-only audit; no filtering, epoching, model fitting or target-label tuning.
Uses MOABB 1.7.x Cho2017 loader and the cached s01.mat file.
"""
import json
import platform
from pathlib import Path
from collections import Counter

import mne
import moabb
import numpy as np
from scipy.io import whosmat, loadmat
from moabb.datasets import Cho2017

OUT = Path(__file__).resolve().parent / 'experiment13c_integrity'
OUT.mkdir(exist_ok=True)
CHANNELS = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz',
            'C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']


def convert(obj, depth=0):
    """Describe MATLAB metadata structures without dumping large EEG arrays."""
    if depth > 4:
        return {'type': type(obj).__name__, 'note': 'maximum inspection depth'}
    if isinstance(obj, np.ndarray):
        result = {'type': 'ndarray', 'shape': list(obj.shape), 'dtype': str(obj.dtype)}
        if obj.dtype.names:
            result['fields'] = {name: convert(obj[name], depth+1) for name in obj.dtype.names}
        elif obj.size <= 30 and obj.dtype.kind in 'biufUS':
            result['values'] = obj.tolist()
        elif obj.dtype == object and obj.size <= 8:
            result['items'] = [convert(x, depth+1) for x in obj.flat]
        return result
    if hasattr(obj, '_fieldnames'):
        return {'type': 'mat_struct', 'fields': {
            name: convert(getattr(obj, name), depth+1) for name in obj._fieldnames}}
    if isinstance(obj, (str, int, float, np.number)):
        return str(obj)[:200]
    return {'type': type(obj).__name__}


def run_audit():
    print('EXPERIMENT 13C — CHO2017 TRIAL AND PREPROCESSING INTEGRITY')
    print('Read-only inspection: no preprocessing or classification.')
    ds = Cho2017(subjects=[1])
    paths = ds.data_path(1)
    mat_files = [Path(p) for p in paths if str(p).lower().endswith('.mat')]
    report = {
        'experiment': '13c_preprocessing_integrity_audit',
        'versions': {'python': platform.python_version(), 'moabb': getattr(moabb, '__version__', 'unknown'), 'mne': mne.__version__},
        'dataset_event_id': ds.event_id, 'dataset_interval_s': list(ds.interval),
        'matlab_files': [], 'recordings': [],
        'important_limits': [
            'Metadata and raw-data inspection only; no causal proof of event/cue timing.',
            'MOABB reconstructed Raw may contain artificial trial boundaries and zero buffers.',
            'Original acquisition reference cannot be established from MNE info alone.',
            'MATLAB bad-trial fields are only inventoried; no trial is excluded.',
            'No evidence of cross-dataset transfer is produced.'
        ]
    }
    for path in mat_files:
        print('\nMATLAB file:', path, '| exists:', path.exists())
        if not path.exists():
            continue
        info = {'path': str(path), 'bytes': path.stat().st_size}
        try:
            variables = whosmat(str(path))
            info['variables'] = [{'name': n, 'shape': list(s), 'class': c} for n, s, c in variables]
            print('MATLAB top-level variables:', info['variables'])
            # Only inspect small likely metadata variables, never duplicate EEG arrays.
            names = [n for n, s, _ in variables if ('bad' in n.lower() or 'event' in n.lower() or 'trial' in n.lower()) and np.prod(s) < 100000]
            if names:
                subset = loadmat(str(path), variable_names=names, squeeze_me=True, struct_as_record=False)
                info['candidate_metadata'] = {n: convert(subset[n]) for n in names if n in subset}
        except Exception as exc:
            info['inspection_error'] = repr(exc)
            print('MATLAB inventory error:', exc)
        report['matlab_files'].append(info)

    recordings = ds.get_data(subjects=[1])
    for session, runs in recordings[1].items():
        for run, raw in runs.items():
            fs = float(raw.info['sfreq'])
            anns = raw.annotations
            starts = np.asarray(anns.onset, dtype=float)
            durations = np.asarray(anns.duration, dtype=float)
            labels = list(map(str, anns.description))
            eeg = [ch for ch in CHANNELS if ch in raw.ch_names and raw.get_channel_types(picks=[ch])[0] == 'eeg']
            # Check only small windows around boundaries, not the entire ~200MB signal.
            probe_channels = eeg[:min(3, len(eeg))]
            probes = []
            for onset, duration, label in list(zip(starts, durations, labels))[:12]:
                center = int(round((onset - raw.first_time) * fs))
                half = int(round(.25 * fs))
                lo, hi = max(0, center-half), min(raw.n_times, center+half)
                segment = raw.get_data(picks=probe_channels, start=lo, stop=hi) if hi > lo else np.empty((0,0))
                probes.append({
                    'annotation': label, 'onset_s': float(onset), 'duration_s': float(duration),
                    'sample_index_approx': center, 'window_s': [lo/fs, hi/fs],
                    'n_samples_examined': int(segment.shape[-1]) if segment.ndim else 0,
                    'zero_fraction_all_samples': float(np.mean(segment == 0)) if segment.size else None,
                    'channel_std_volts': np.std(segment, axis=1).tolist() if segment.size else []
                })
            rec = {
                'session': str(session), 'run': str(run), 'sampling_hz': fs,
                'n_samples': int(raw.n_times), 'n_channels': len(raw.ch_names),
                'required_channel_count': len(eeg), 'missing_required_channels': sorted(set(CHANNELS)-set(eeg)),
                'annotation_counts': dict(Counter(labels)),
                'unique_annotation_durations_s': sorted(set(np.round(durations, 6).tolist())),
                'first_annotation_onsets_s': starts[:10].tolist(),
                'first_inter_annotation_intervals_s': np.diff(starts[:11]).tolist(),
                'raw_first_time_s': float(raw.first_time),
                'raw_info_filter_bounds_hz': [float(raw.info['highpass']), float(raw.info['lowpass'])],
                'raw_info_filter_bounds_are_not_acquisition_proof': True,
                'annotation_boundary_probes': probes,
                'bad_channels_in_mne': list(raw.info['bads']),
                'raw_annotations_do_not_necessarily_include_matlab_bad_trial_flags': True,
            }
            report['recordings'].append(rec)
            print(f'\nSession {session} run {run}: {fs} Hz, {len(eeg)}/22 required channels')
            print('Annotation counts:', rec['annotation_counts'])
            print('Annotation durations:', rec['unique_annotation_durations_s'])
            print('First onset times:', rec['first_annotation_onsets_s'][:5])
            print('First onset intervals:', rec['first_inter_annotation_intervals_s'][:5])
            print('Boundary probe zero fractions:', [p['zero_fraction_all_samples'] for p in probes[:5]])

    output = OUT / '13c_integrity_report.json'
    output.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print('\nSaved:', output)
    print('Audit complete. No preprocessing choices finalized.')


if __name__ == '__main__':
    run_audit()
