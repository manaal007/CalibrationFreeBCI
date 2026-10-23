"""Offline BNCI MATLAB loader with explicit trial-start versus cue semantics."""
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .audit import sha256

CHANNELS = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6',
            'CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
CUE_DELAY = 2.0
WINDOW = (.5, 2.5)


def find_bnci(root=None):
    roots = [Path(root)] if root else [
        Path.home()/'.cache/huggingface/hub/datasets--EEGDash--eeg-mi-datasets/snapshots',
        Path.home()/'mne_data/MNE-bnci-data']
    files = {}
    for folder in roots:
        for path in sorted(folder.rglob('A??[TE].mat')):
            files.setdefault(path.name, path)
    return files


def epoch_raw(raw, events, band=(8., 30.)):
    """Events are zero-based trial starts; cue occurs exactly two seconds later."""
    raw = raw.copy().pick(CHANNELS).reorder_channels(CHANNELS).load_data()
    raw.set_eeg_reference('average', projection=False, verbose=False)
    raw.filter(*band, verbose=False)
    return mne.Epochs(raw, events, event_id={'left': 1, 'right': 2},
                      tmin=CUE_DELAY+WINDOW[0], tmax=CUE_DELAY+WINDOW[1],
                      baseline=None, preload=True, reject_by_annotation=True,
                      on_missing='ignore', verbose=False)


def load_bnci(subjects, root=None, bands=None, artifact_policy='retain'):
    if not subjects or len(set(subjects)) != len(subjects) or any(s not in range(1,10) for s in subjects):
        raise ValueError('BNCI participants must be distinct integers 1..9')
    if artifact_policy not in ('retain', 'reject_flagged'):
        raise ValueError('Unknown artifact policy')
    files = find_bnci(root)
    arrays, rows, provenance = [], [], []
    for subject in subjects:
        count = 0
        for session in ('T', 'E'):
            name = f'A{subject:02d}{session}.mat'
            if name not in files:
                raise FileNotFoundError(f'{name} unavailable locally; no automatic downloads')
            path = files[name]
            provenance.append(dict(subject=subject, session=session, path=str(path), sha256=sha256(path)))
            runs = np.atleast_1d(loadmat(path, squeeze_me=True, struct_as_record=False)['data'])
            for run_index, run in enumerate(runs):
                labels = np.asarray(run.y).reshape(-1).astype(int)
                if not len(labels):
                    continue
                if float(run.fs) != 250 or run.X.shape[1] != 25:
                    raise ValueError('Unexpected BNCI sampling rate or channel layout')
                starts = np.asarray(run.trial).reshape(-1).astype(int)-1
                flags = np.asarray(run.artifacts).reshape(-1)
                if len(starts) != len(labels) or len(flags) != len(labels):
                    raise ValueError('Trial metadata misalignment')
                selected = np.flatnonzero(np.isin(labels, [1, 2]))
                if artifact_policy == 'reject_flagged':
                    selected = selected[flags[selected] == 0]
                if len(selected) == 0:
                    continue
                raw = mne.io.RawArray(run.X[:, :22].T*1e-6,
                                     mne.create_info(CHANNELS, 250, 'eeg'), verbose=False)
                events = np.column_stack([starts[selected], np.zeros(len(selected), int), labels[selected]])
                epochs = epoch_raw(raw, events)
                if bands is None:
                    values = epochs.get_data(copy=True)
                else:
                    bank = []
                    for band in bands:
                        band_epochs = epoch_raw(raw, events, band)
                        if not np.array_equal(epochs.selection, band_epochs.selection):
                            raise ValueError('Frequency bands retained different trial identities')
                        bank.append(band_epochs.get_data(copy=True))
                    values = np.stack(bank, axis=1)
                if values.shape[-2:] != (22, 501) or not np.isfinite(values).all():
                    raise ValueError('Invalid epoch data')
                arrays.append(values)
                for index in epochs.selection:
                    trial = selected[index]
                    rows.append(dict(subject=subject, session=session, run=run_index,
                                     trial=int(trial), label=int(labels[trial]),
                                     artifact_flag=int(flags[trial]), trial_start_sample=int(starts[trial])))
                count += len(values)
        print(f'Loaded source participant {subject}: {count} epochs', flush=True)
    if not arrays:
        raise ValueError('No retained source trials')
    x, metadata = np.concatenate(arrays), pd.DataFrame(rows)
    if len(x) != len(metadata):
        raise ValueError('Epoch identity mismatch')
    return x, metadata, provenance
