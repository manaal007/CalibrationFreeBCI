"""Experiment 14B: consolidated evidence-based preprocessing audit, no training.
Uses locally cached Cho2017 Subject 1 and BNCI2014-001 Subject 1.
It does NOT freeze the full 52-subject external evaluation protocol.
"""
from pathlib import Path
from collections import Counter
import inspect, json, os, sys, traceback
import numpy as np
import mne
import moabb
from scipy.io import loadmat
from scipy.signal import resample_poly
from huggingface_hub import scan_cache_dir
from moabb.datasets import BNCI2014_001, Cho2017

CHANNELS = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
ROOT = Path(__file__).resolve().parent
CHO = Path.home() / 'mne_data/MNE-gigadb-data/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data/s01.mat'
OUT = ROOT / 'experiment14b_final_validation'
SFREQ, N_SAMPLES, CUE = 250, 501, 1023


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def source_cache():
    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_type == 'dataset' and repo.repo_id == 'EEGDash/eeg-mi-datasets':
            for revision in repo.revisions:
                folder = Path(revision.snapshot_path) / 'BNCI2014_001'
                if list(folder.rglob('A01T.mat')) and list(folder.rglob('A01E.mat')):
                    os.environ['MNE_DATASETS_BNCI_PATH'] = str(folder)
                    moabb.set_download_provider('upstream')
                    return str(folder)
    raise FileNotFoundError('Existing BNCI2014-001 Subject 1 cache not found; no download requested')


def get_raw(dataset):
    sessions = dataset.get_data(subjects=[1])[1]
    for session, runs in sessions.items():
        for run, raw in runs.items():
            require(isinstance(raw, mne.io.BaseRaw), f'Not MNE Raw: {session}/{run}')
            return raw, str(session), str(run)
    raise RuntimeError('No recording returned')


def main():
    OUT.mkdir(exist_ok=True)
    report = {'experiment': '14B', 'training_performed': False,
              'external_evaluation_performed': False, 'full_protocol_frozen': False,
              'checks': {}, 'limitations': []}
    def check(name, condition, detail):
        report['checks'][name] = {'pass': bool(condition), 'detail': detail}
        print(('PASS' if condition else 'FAIL'), name, '-', detail)
        require(condition, name + ': ' + str(detail))
    try:
        print('EXPERIMENT 14B — CONSOLIDATED PREPROCESSING AUDIT')
        print('No classification, no target-label model tuning, no new downloads intended.')
        require(CHO.is_file(), f'Cho2017 cached MATLAB file not found: {CHO}')
        report['bnci_cache'] = source_cache()
        raw_s, session, run = get_raw(BNCI2014_001())
        check('BNCI channels', all(c in raw_s.ch_names for c in CHANNELS), '22 requested electrodes found')
        check('BNCI sampling', np.isclose(raw_s.info['sfreq'], SFREQ), str(raw_s.info['sfreq']))
        check('BNCI channel types', all(raw_s.get_channel_types(picks=[c])[0] == 'eeg' for c in CHANNELS), '22 EEG types')
        ds = BNCI2014_001()
        descriptions = set(raw_s.annotations.description)
        codes = {k: str(ds.event_id[k]) for k in ('left_hand','right_hand')}
        check('BNCI event labels', set(codes.values()).issubset(descriptions) or {'left_hand','right_hand'}.issubset(descriptions),
              f'Observed descriptions: {sorted(descriptions)}; MOABB IDs: {codes}')

        # Verify the *installed* loader, not a guessed unit conversion.
        loader_source = inspect.getsource(Cho2017._get_single_subject_data)
        check('Installed Cho loader volts conversion', 'imagery_left * 1e-6' in loader_source and 'imagery_right * 1e-6' in loader_source,
              'Installed MOABB loader explicitly multiplies both imagery arrays by 1e-6')
        check('Installed Cho loader row mapping', 'ch_names = eeg_ch_names + emg_ch_names + ["Stim"]' in loader_source and 'imagery_left = data.imagery_left' in loader_source,
              'Installed MOABB loader labels first 64 rows as EEG, next 4 as EMG')
        report['cho_loader_unit_conversion'] = 1e-6

        eeg = loadmat(CHO, squeeze_me=True, struct_as_record=False)['eeg']
        rate = float(np.asarray(eeg.srate).item())
        n = int(np.asarray(eeg.n_imagery_trials).item())
        check('Cho MATLAB dimensions', rate == 512 and n == 100 and
              np.asarray(eeg.imagery_left).shape == (68,358400) and np.asarray(eeg.imagery_right).shape == (68,358400),
              f'{rate} Hz; {n} trials/class; left={np.asarray(eeg.imagery_left).shape}')
        markers = np.asarray(eeg.imagery_event).reshape(n,3584)
        marker_positions = [np.flatnonzero(row).tolist() for row in markers]
        check('Cho marker position', all(p == [CUE] for p in marker_positions), 'One marker at sample 1023 in every 7-second trial')
        bad_summary = {}
        for field in ('bad_trial_idx_voltage','bad_trial_idx_mi'):
            entries = np.asarray(getattr(eeg.bad_trial_indices, field), dtype=object).ravel()
            sizes = [int(np.asarray(item).size) for item in entries]
            bad_summary[field] = sizes
            check('Cho '+field, len(sizes) == 2 and sizes == [0,0], f'Subject 1 two class arrays: {sizes} flagged indices')
        report['bad_trial_summary'] = bad_summary

        raw_t, _, _ = get_raw(Cho2017())
        cho_names = raw_t.copy().pick(picks='eeg').ch_names
        check('Cho channels', len(cho_names) == 64 and all(c in cho_names for c in CHANNELS),
              f'64 EEG names; all {len(CHANNELS)} common electrodes present')
        indices = [cho_names.index(c) for c in CHANNELS]
        report['channel_order'] = CHANNELS
        report['cho_matlab_row_indices'] = indices
        check('Cho raw scaling check', np.isclose(raw_t.get_data(picks=[cho_names[0]], start=0, stop=1)[0,0],
              (float(eeg.imagery_left[0,0])-float(np.mean(eeg.imagery_left[0,:])))*1e-6, rtol=1e-3, atol=1e-9),
              'First sample matches installed loader temporal demeaning and microvolt-to-volt conversion')

        # A single trial numerical smoke test, not a whole-dataset preprocessing result.
        trial = np.asarray(eeg.imagery_left[:64,:3584], dtype=np.float64)[indices] * 1e-6
        trial -= trial.mean(axis=0, keepdims=True)
        trial = mne.filter.filter_data(trial, sfreq=rate, l_freq=8, h_freq=30, verbose=False)
        trial = resample_poly(trial, 125, 256, axis=-1)
        start = int(round((CUE/rate + 0.5)*SFREQ))
        epoch = trial[:, start:start+N_SAMPLES]
        check('Cho processed example shape', epoch.shape == (22,501), str(epoch.shape))
        check('Cho processed example finite', bool(np.isfinite(epoch).all()), 'No NaN/Inf')
        report['cho_example_mean_absolute_volts'] = float(np.abs(epoch).mean())
        report['cho_example_max_absolute_volts'] = float(np.abs(epoch).max())
        print('Cho example mean |EEG|:', report['cho_example_mean_absolute_volts'], 'V')

        report['limitations'] = [
            'Only Cho Subject 1 and the first BNCI recording were inspected here; not all subjects.',
            'Common-average reference and 0.5-2.5 s post-cue window are prespecified analysis choices, not proven equivalent across acquisition systems.',
            'The MATLAB trial event marker and source cue meaning must be reconciled with published acquisition documentation.',
            'Filtering was demonstrated within a single original Cho trial; full target cohort preprocessing and exclusions remain to be implemented.',
            'Train/test separation and no target-label fitting must be enforced and audited in Experiment 15.',
            'Model hyperparameters and reporting plan must be frozen before examining target performance.'
        ]
        report['status'] = 'SUBJECT-1 AUDIT PASS; FULL PROTOCOL NOT FROZEN'
        print('\nSubject-1 consolidated checks PASSED. Full protocol frozen: NO')
    except Exception as exc:
        report['status'] = 'STOPPED — CHECK FAILED'
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc(limit=5)
        print('AUDIT STOPPED:', report['error'])
    finally:
        dest = OUT / '14b_validation_report.json'
        dest.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        print('Saved:', dest)
    if report['status'].startswith('STOPPED'):
        sys.exit(1)

if __name__ == '__main__':
    main()
