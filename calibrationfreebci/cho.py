"""Exploratory Cho2017 loader: original trial blocks, never zero-buffered concatenations."""
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from .data import CHANNELS
from .audit import sha256

CHO_CHANNELS = [
    'Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1',
    'C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7',
    'P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2',
    'AF8','AF4','AFz','Fz','F2','F4','F6','F8','FT8','FC6','FC4',
    'FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2',
    'P2','P4','P6','P8','P10','PO8','PO4','O2']


def load_cho(subjects, root=None, bands=None):
    root = Path(root) if root else Path.home()/'mne_data/MNE-gigadb-data/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data'
    arrays, rows, files = [], [], []
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError('Need distinct target participants')
    indices = [CHO_CHANNELS.index(c) for c in CHANNELS]
    for subject in subjects:
        path = root/f's{subject:02d}.mat'
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(dict(path=str(path), sha256=sha256(path)))
        eeg = loadmat(path, squeeze_me=True, struct_as_record=False)['eeg']
        if float(eeg.srate) != 512 or int(eeg.n_imagery_trials) != 100:
            raise ValueError('Unexpected Cho sample rate or trial count')
        markers = np.asarray(eeg.imagery_event).reshape(100,3584)
        positions = [np.flatnonzero(row) for row in markers]
        if any(len(p) != 1 for p in positions) or len(set(int(p[0]) for p in positions)) != 1:
            raise ValueError('Cho task marker pattern changed')
        flags = np.zeros((2,100),bool)
        for field in ('bad_trial_idx_voltage','bad_trial_idx_mi'):
            entries = np.asarray(getattr(eeg.bad_trial_indices,field),dtype=object).ravel()
            if len(entries) != 2:
                raise ValueError('Invalid Cho artifact metadata')
            for label,entry in enumerate(entries):
                bad = np.asarray(entry).reshape(-1).astype(int)
                if np.any((bad<1)|(bad>100)):
                    raise ValueError('Cho artifact indices out of bounds')
                flags[label,bad-1] = True
        for label,field in ((1,'imagery_left'),(2,'imagery_right')):
            values = np.asarray(getattr(eeg,field),dtype=float)
            if values.shape != (68,358400):
                raise ValueError('Cho row/trial layout changed')
            trials = values[:64].reshape(64,100,3584).transpose(1,0,2)
            for trial,signal in enumerate(trials):
                signal = signal[indices]*1e-6
                signal -= signal.mean(axis=0,keepdims=True)
                bank = []
                for band in (bands or [(8.,30.)]):
                    filtered = mne.filter.filter_data(signal,512,*band,verbose=False)
                    down = resample_poly(filtered,125,256,axis=-1)
                    start = round((positions[trial][0]/512+.5)*250)
                    epoch = down[:,start:start+501]
                    if epoch.shape != (22,501) or not np.isfinite(epoch).all():
                        raise ValueError('Invalid Cho epoch')
                    bank.append(epoch)
                arrays.append(np.stack(bank) if bands else bank[0])
                rows.append(dict(subject=subject,label=label,trial=trial,run=0,
                                 artifact_flag=int(flags[label-1,trial]),cue_sample=int(positions[trial][0])))
    return np.stack(arrays),pd.DataFrame(rows),files
