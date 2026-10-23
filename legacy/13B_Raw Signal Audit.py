"""Experiment 13B: one-subject raw signal compatibility audit. No model fitting.
Requires: moabb, mne, numpy. Cho2017 Subject 1 may download from GigaDB.
"""
import json
import platform
from pathlib import Path
from collections import Counter

import mne
import moabb
import numpy as np
from moabb.datasets import Cho2017, BNCI2014_001

SOURCE_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
    'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
]
OUT = Path(__file__).resolve().parent / 'experiment13b_signal_audit'
OUT.mkdir(exist_ok=True)


def finite_xyz(raw, channels):
    """Check that MNE has valid channel positions; no signal modifications."""
    return {ch: bool(np.isfinite(raw.info['chs'][raw.ch_names.index(ch)]['loc'][:3]).all()
                     and np.linalg.norm(raw.info['chs'][raw.ch_names.index(ch)]['loc'][:3]) > 0)
            for ch in channels}


def main():
    print('EXPERIMENT 13B — CHO2017 RAW SIGNAL AUDIT')
    print('No training, no classification, no target-label tuning.')
    print('This may download the original Cho2017 Subject 1 MATLAB file.')
    print('MOABB:', getattr(moabb, '__version__', 'unknown'), '| MNE:', mne.__version__)
    dataset = Cho2017(subjects=[1])
    print('Dataset events:', dataset.event_id)
    print('Dataset interval:', dataset.interval)

    # Dataset.get_data returns {subject: {session: {run: Raw}}}.
    # This invokes MOABB's original Cho2017 loader, including its event mapping.
    recordings = dataset.get_data(subjects=[1])
    report = {
        'experiment': '13b_cho2017_signal_audit',
        'purpose': 'Raw signal and event compatibility only; no trained models',
        'subject': 1,
        'software': {'python': platform.python_version(), 'mne': mne.__version__,
                     'moabb': getattr(moabb, '__version__', 'unknown')},
        'source_channel_order': SOURCE_CHANNELS,
        'dataset_events': dict(dataset.event_id),
        'dataset_interval_seconds': list(dataset.interval),
        'recordings': []
    }
    for session_id, runs in recordings[1].items():
        for run_id, raw in runs.items():
            eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
            common = [ch for ch in SOURCE_CHANNELS if ch in eeg_names]
            missing = [ch for ch in SOURCE_CHANNELS if ch not in eeg_names]
            annotations = raw.annotations
            annotation_counts = dict(Counter(str(d) for d in annotations.description))
            # Read metadata only: no epoching, fitting, or model evaluation.
            record = {
                'session': str(session_id), 'run': str(run_id),
                'sfreq_hz': float(raw.info['sfreq']),
                'duration_seconds': float(raw.n_times / raw.info['sfreq']),
                'n_samples': int(raw.n_times),
                'n_channels_total': len(raw.ch_names),
                'n_eeg_channels': len(eeg_names),
                'eeg_channel_names': eeg_names,
                'channel_types': dict(Counter(raw.get_channel_types())),
                'common_channels_in_source_order': common,
                'missing_source_channels': missing,
                'all_22_labels_present': len(missing) == 0,
                'common_channel_positions_valid': finite_xyz(raw, common),
                'annotations_counts': annotation_counts,
                'first_12_annotations': [
                    {'onset_s': float(o), 'duration_s': float(d), 'description': str(t)}
                    for o, d, t in zip(annotations.onset[:12], annotations.duration[:12], annotations.description[:12])
                ],
                'highpass_hz': float(raw.info['highpass']),
                'lowpass_hz': float(raw.info['lowpass']),
                'line_freq_hz': raw.info.get('line_freq'),
                'custom_ref_applied': raw.info.get('custom_ref_applied'),
                'bad_channels': list(raw.info['bads']),
            }
            report['recordings'].append(record)
            print(f'\nSession {session_id}, run {run_id}')
            print('Sampling:', record['sfreq_hz'], 'Hz; EEG channels:', len(eeg_names))
            print('Common source labels:', len(common), '/ 22; missing:', missing)
            print('Annotation counts:', annotation_counts)
            print('First annotations:', record['first_12_annotations'][:5])
            print('Stored frequency bounds:', record['highpass_hz'], '-', record['lowpass_hz'])
            print('Bad channels:', record['bad_channels'])
            print('Position validity:', sum(record['common_channel_positions_valid'].values()), '/', len(common))

    report['limitations'] = [
        'Only Subject 1 inspected; does not establish generalization.',
        'Dataset interval and event annotations alone do not prove cue timing.',
        'MNE channel positions may reflect a standard montage rather than measured locations.',
        'Reference comparability and actual signal quality require additional verification.',
        'No data-driven choices, classifiers, or accuracy metrics were used.'
    ]
    path = OUT / '13b_raw_signal_report.json'
    path.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print('\nSaved:', path)
    print('Audit complete. Review report before deciding on harmonization.')


if __name__ == '__main__':
    main()
