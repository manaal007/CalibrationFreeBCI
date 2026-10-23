"""Read-only diagnostic for the existing BNCI2014-001 / MOABB decoding protocol.
Does not train a classifier or modify prior experiment outputs.
Run in the same PyCharm virtual environment as Experiments 07 and 10.
"""
from __future__ import annotations

import inspect
import os
import json
import platform
from pathlib import Path
from importlib import metadata as im

import numpy as np
import pandas as pd

import moabb
from huggingface_hub import snapshot_download
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiment12b_protocol'
OUT.mkdir(exist_ok=True)


def version(name):
    try:
        return im.version(name)
    except im.PackageNotFoundError:
        return 'not installed'


def safe_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


report = {
    'purpose': 'Inspect actual installed MOABB protocol, not certify all leakage risks',
    'versions': {k: version(k) for k in ['moabb', 'mne', 'numpy', 'scikit-learn', 'pyriemann', 'scipy', 'pandas', 'huggingface-hub']},
    'python': platform.python_version(),
}

print('EXPERIMENT 12b — MOABB PROTOCOL VERIFICATION')
print('=' * 65)
print('Versions:', report['versions'])

# Reproduce the EEGDash/Hugging Face data route from Experiments 07/10.
# Prefer the already-cached snapshot; only contact Hugging Face if absent.
print('\nResolving the EEGDash snapshot used by Experiments 07/10...')
patterns = ['BNCI2014_001/*/A01*']
try:
    snapshot_path = snapshot_download(
        repo_id='EEGDash/eeg-mi-datasets', repo_type='dataset',
        allow_patterns=patterns, local_files_only=True
    )
    print('Found locally cached Hugging Face snapshot.')
except Exception as cache_error:
    print('Cached snapshot lookup failed:', type(cache_error).__name__)
    print('Trying Hugging Face download (not the Graz server)...')
    snapshot_path = snapshot_download(
        repo_id='EEGDash/eeg-mi-datasets', repo_type='dataset',
        allow_patterns=patterns
    )

bnci_path = Path(snapshot_path) / 'BNCI2014_001'
if not bnci_path.exists():
    raise FileNotFoundError(f'BNCI2014_001 missing from snapshot: {bnci_path}')
os.environ['MNE_DATASETS_BNCI_PATH'] = str(bnci_path)
moabb.set_download_provider('upstream')
report['data_source'] = {
    'snapshot_path': str(snapshot_path),
    'mne_datasets_bnci_path': str(bnci_path),
    'provider': 'upstream',
    'matching_subject1_files': [str(p.relative_to(bnci_path)) for p in bnci_path.rglob('A01*')][:30],
}
print('MNE_DATASETS_BNCI_PATH:', bnci_path)
print('Subject 1 files found:', report['data_source']['matching_subject1_files'])

# The same dataset and paradigm used in Experiments 07/10.
dataset = BNCI2014_001()
paradigm = LeftRightImagery(fmin=8, fmax=30, tmin=0.5, tmax=3.5)

report['dataset'] = {
    'class': type(dataset).__name__,
    'subject_list': list(dataset.subject_list),
    'event_id': getattr(dataset, 'event_id', None),
    'interval': getattr(dataset, 'interval', None),
    'unit_factor': getattr(dataset, 'unit_factor', None),
}
report['paradigm'] = {
    'class': type(paradigm).__name__,
    'fmin': 8, 'fmax': 30, 'tmin': 0.5, 'tmax': 3.5,
    'events': getattr(paradigm, 'events', None),
    'resample': getattr(paradigm, 'resample', None),
    'baseline': getattr(paradigm, 'baseline', None),
}

print('Dataset event_id:', report['dataset']['event_id'])
print('Dataset interval (seconds):', report['dataset']['interval'])
print('Paradigm epoch offsets (seconds): 0.5 to 3.5')

# Source snippets let us review how the installed MOABB version handles
# event offsets, filters and epochs without guessing from online docs.
for label, obj in [('dataset_class', BNCI2014_001), ('paradigm_class', LeftRightImagery)]:
    try:
        path = inspect.getfile(obj)
        report[label + '_source_path'] = path
        print(f'{label} source: {path}')
    except (TypeError, OSError) as exc:
        report[label + '_source_path'] = f'unavailable: {exc}'

# Load only one subject. If MOABB requires a download, it may download this
# participant. Data are NOT altered or exported.
print('\nLoading subject 1 using the EXACT experiment paradigm...')
try:
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1])
    report['processed'] = {
        'shape': list(X.shape),
        'labels': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        'metadata_columns': list(meta.columns),
        'sessions': {str(k): int(v) for k, v in meta['session'].value_counts().items()} if 'session' in meta else None,
        'finite': bool(np.isfinite(X).all()),
    }
    print('Processed X shape:', X.shape)
    print('Label counts:', report['processed']['labels'])
    print('Session counts:', report['processed']['sessions'])
    print('Finite data:', report['processed']['finite'])
    pd.DataFrame(meta).to_csv(OUT / '12b_subject1_epoch_metadata.csv', index=False)
except Exception as exc:
    report['processed_error'] = f'{type(exc).__name__}: {exc}'
    print('Could not load processed epochs:', report['processed_error'])

# get_data(return_epochs=True) is supported by many MOABB versions, but the
# installed API must be checked rather than assumed.
try:
    sig = inspect.signature(paradigm.get_data)
    report['get_data_signature'] = str(sig)
    if 'return_epochs' in sig.parameters:
        print('\nLoading MNE Epochs to inspect actual time axis and channel names...')
        epochs, labels, metadata = paradigm.get_data(
            dataset=dataset, subjects=[1], return_epochs=True
        )
        report['epochs'] = {
            'tmin': float(epochs.tmin),
            'tmax': float(epochs.tmax),
            'sfreq': float(epochs.info['sfreq']),
            'n_epochs': int(len(epochs)),
            'n_channels': int(len(epochs.ch_names)),
            'channel_names': list(epochs.ch_names),
            'event_id': dict(epochs.event_id),
            'baseline': epochs.baseline,
            'highpass': float(epochs.info['highpass']),
            'lowpass': float(epochs.info['lowpass']),
        }
        print('Actual epochs:', report['epochs'])
    else:
        report['epochs_note'] = 'Installed paradigm.get_data lacks return_epochs; manual API inspection needed'
        print(report['epochs_note'])
except Exception as exc:
    report['epochs_error'] = f'{type(exc).__name__}: {exc}'
    print('MNE Epochs inspection unavailable:', report['epochs_error'])

report['unverified'] = [
    'Whether EEGDash snapshot matches the canonical MOABB dataset byte-for-byte',
    'Exact filter/reference implementation and event-cue alignment without reviewing installed source and raw annotations',
    'Whether all preprocessing is independent across participants and sessions',
    'Whether the decoding pipeline generalizes to another dataset',
]

(OUT / '12b_protocol_report.json').write_text(
    json.dumps(report, indent=2, default=safe_json), encoding='utf-8'
)
print('\nSaved:', OUT / '12b_protocol_report.json')
print('IMPORTANT: This diagnostic does not certify the entire pipeline leakage-free.')
