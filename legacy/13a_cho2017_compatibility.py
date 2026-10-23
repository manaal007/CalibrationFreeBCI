"""Experiment 13A: metadata-first BNCI2014-001 vs Cho2017 compatibility audit.

Does not download EEG, fit models, or inspect target labels/performance.
Run from the existing CalibrationFreeBCI PyCharm environment.
"""
from pathlib import Path
import json
import platform
from importlib.metadata import version, PackageNotFoundError

from moabb.datasets import BNCI2014_001, Cho2017

OUT = Path(__file__).resolve().parent / 'experiment13a_compatibility'
OUT.mkdir(exist_ok=True)

# Verified Subject 1 BNCI2014-001 channels from Experiment 12b.
SOURCE_CHANNELS = [
    'Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4',
    'C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz'
]
# Public Cho2017 metadata channel names (NEMAR / MOABB documentation).
TARGET_CHANNELS = [
    'AF3','AF4','AF7','AF8','AFz','C1','C2','C3','C4','C5','C6',
    'CP1','CP2','CP3','CP4','CP5','CP6','CPz','Cz',
    'F1','F2','F3','F4','F5','F6','F7','F8',
    'FC1','FC2','FC3','FC4','FC5','FC6','FCz',
    'FT7','FT8','Fp1','Fp2','Fpz','Fz','Iz',
    'O1','O2','Oz','P1','P10','P2','P3','P4','P5','P6','P7','P8','P9',
    'PO3','PO4','PO7','PO8','POz','Pz','T7','T8','TP7','TP8'
]


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return 'not installed'


def get_metadata(dataset):
    return {
        'code': str(dataset.code),
        'subjects': len(dataset.subject_list),
        'events': dict(dataset.event_id),
        'interval_seconds': list(dataset.interval),
    }


def main():
    source = BNCI2014_001()
    target = Cho2017()
    source_set = set(SOURCE_CHANNELS)
    target_set = set(TARGET_CHANNELS)
    common = [ch for ch in SOURCE_CHANNELS if ch in target_set]

    assert len(SOURCE_CHANNELS) == 22 and len(source_set) == 22
    assert len(TARGET_CHANNELS) == 64 and len(target_set) == 64
    assert len(common) >= 3

    report = {
        'experiment': '13A — metadata-only compatibility audit',
        'python': platform.python_version(),
        'versions': {name: pkg_version(name) for name in ['moabb','mne','numpy','scikit-learn','pyriemann']},
        'source_dataset': get_metadata(source),
        'target_dataset': get_metadata(target),
        'source_eeg_channels': SOURCE_CHANNELS,
        'target_eeg_channels_from_published_metadata': TARGET_CHANNELS,
        'common_channels_in_source_order': common,
        'n_common_channels': len(common),
        'source_only_channels': sorted(source_set - target_set),
        'target_only_channels': sorted(target_set - source_set),
        'sampling_rates_hz_from_documentation': {'BNCI2014_001': 250, 'Cho2017': 512},
        'acquisition_references_from_documentation': {
            'BNCI2014_001': 'left mastoid',
            'Cho2017': 'BioSemi CMS/DRL (verify offline rereferencing in raw metadata)'
        },
        'proposed_shared_cue_relative_window_seconds': [0.5, 2.5],
        'status': 'METADATA_ONLY — requires raw target validation before final protocol',
        'limitations': [
            'Channel names are from documentation, not a downloaded Cho2017 recording.',
            'Electrode name overlap does not prove equivalent reference or sensor geometry.',
            'Do not use target labels or AUC to choose preprocessing parameters.',
            'Cross-dataset transfer requires new source-only training on harmonized channels.',
            'A common cue-relative time window needs validation against actual event markers.',
            'Filtering/resampling/referencing must be implemented and verified on raw signals.'
        ]
    }

    output = OUT / '13a_metadata_compatibility_report.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('EXPERIMENT 13A — DATASET COMPATIBILITY (NO DOWNLOAD)')
    print('=' * 62)
    print('Source:', report['source_dataset'])
    print('Target:', report['target_dataset'])
    print(f'Common EEG channel labels ({len(common)}):', ', '.join(common))
    print('Source-only channels:', report['source_only_channels'])
    print('Proposed cue-relative analysis window: 0.5–2.5 s (NOT YET FINALIZED)')
    print('WARNING: Different acquisition references; signal harmonization is required.')
    print('WARNING: This is metadata compatibility, NOT proof of transfer validity.')
    print('Saved:', output)


if __name__ == '__main__':
    main()
