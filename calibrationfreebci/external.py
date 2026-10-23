"""Independent PhysioNet evaluation. Invoke only after explicit user approval."""
import json
from pathlib import Path
import joblib
import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from .data import CHANNELS
from .models import right_score
from .experiment import metrics
from .audit import sha256, bootstrap_mean


def load_physionet(subjects, root=None, bands=None):
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError('Target participants must be distinct and nonempty')
    root = Path(root) if root else Path.home()/'mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0'
    arrays, rows, files = [], [], []
    for subject in subjects:
        for run in (4, 8, 12):
            path = root/f'S{subject:03d}'/f'S{subject:03d}R{run:02d}.edf'
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(dict(path=str(path), sha256=sha256(path)))
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            mne.datasets.eegbci.standardize(raw)
            if raw.info['sfreq'] != 160:
                raise ValueError('Unexpected PhysioNet sample rate')
            raw.pick(CHANNELS).reorder_channels(CHANNELS)
            raw.set_eeg_reference('average', projection=False, verbose=False)
            events, _ = mne.events_from_annotations(raw, event_id={'T1':1, 'T2':2}, verbose=False)
            filtered = [raw.copy().filter(*band, verbose=False) for band in (bands or [(8.,30.)])]
            # Here the marker IS the task cue. Filter runs, then resample each
            # padded epoch before cropping; no mixing across recording boundaries.
            for index, event in enumerate(events):
                start = int(event[0]-raw.first_samp)
                bank = []
                for band_raw in filtered:
                    signal = band_raw.get_data(start=start, stop=start+4*160)
                    if signal.shape[1] != 640:
                        raise ValueError('Incomplete task segment')
                    epoch = resample_poly(signal, 25, 16, axis=-1)[:, 125:626]
                    if epoch.shape != (22,501) or not np.isfinite(epoch).all():
                        raise ValueError('Invalid target epoch')
                    bank.append(epoch)
                arrays.append(np.stack(bank) if bands else bank[0])
                rows.append(dict(subject=subject, run=run, trial=index, label=int(event[2]),cue_sample=start))
    return np.stack(arrays), pd.DataFrame(rows), files


def validate_exposure(exposure, subjects, evaluation_kind):
    if evaluation_kind not in ('confirmatory','provisional'):
        raise ValueError('Unknown evaluation kind')
    if exposure.get('dataset') != 'PhysioNet EEGMMIDB 1.0.0' or exposure.get('subjects') != subjects:
        raise ValueError('Exposure record dataset or cohort mismatch')
    if evaluation_kind == 'confirmatory':
        if exposure.get('owner_confirmed') is not True or exposure.get('previously_used') is not False:
            raise ValueError('Exposure record does not establish an untouched approved cohort')
    elif exposure.get('evaluation_approved') is not True:
        raise ValueError('Provisional evaluation requires recorded approval')


def evaluate(source, out, subjects, root=None, exposure_record=None, evaluation_kind='confirmatory'):
    if exposure_record is None:
        raise ValueError('Confirmatory evaluation requires an owner-confirmed exposure record')
    exposure = json.loads(Path(exposure_record).read_text())
    validate_exposure(exposure,subjects,evaluation_kind)
    source, out = Path(source), Path(out)
    from .verification import load_comparisons
    models,manifest,selection=load_comparisons(source)
    out.mkdir(parents=True, exist_ok=False)
    protocol=dict(dataset='PhysioNet EEGMMIDB 1.0.0',subjects=subjects,
        evaluation_kind=evaluation_kind,exposure_record=exposure,primary_model=selection,
        comparison_manifest=manifest,primary_endpoint='participant-mean balanced accuracy',
        secondary_endpoints=['auc','recall_left','recall_right'],
        threshold_policy='frozen source thresholds',target_label_tuning=False,
        bootstrap=dict(unit='participant',draws=10000,seed=20261023),
        participant_exclusions='none based on performance',
        source_model_refitting=False,
        independence='owner-confirmed unused' if evaluation_kind=='confirmatory' else 'not established')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    band_spec = selection.get('bands') if selection.get('input') == 'filterbank' else None
    x, frame, provenance = load_physionet(subjects, root, band_spec)
    frames = []
    for name, model in models.items():
        from .verification import model_input
        current = frame.copy()
        current['model'] = name
        values = model_input(x,manifest[name])
        current['score_right'] = right_score(model, values,frame.subject.to_numpy())
        np.testing.assert_allclose(current.score_right,
            right_score(joblib.load(source/manifest[name]['file']),values,frame.subject.to_numpy()),atol=1e-12)
        if name == selection['model']:
            np.testing.assert_allclose(current.score_right,
                right_score(joblib.load(source/'frozen_model.joblib'),values,frame.subject.to_numpy()),atol=1e-12)
        current['prediction'] = np.where(current.score_right > manifest[name]['threshold'], 2, 1)
        frames.append(current)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(out/'predictions.csv', index=False)
    rows = [dict(model=name, subject=s, **metrics(g.label.to_numpy(), g.prediction.to_numpy(), g.score_right.to_numpy()))
            for (name,s),g in frame.groupby(['model','subject'])]
    results = pd.DataFrame(rows)
    results.to_csv(out/'participant_metrics.csv', index=False)
    report = dict(model=selection, subjects=subjects, provenance=provenance, exposure_record=exposure,
                  evaluation_kind=evaluation_kind,independence=protocol['independence'],
                  warning='Confirmatory status depends on absence of prior use, not software flags')
    report['summary'] = {name: {metric: dict(mean=float(group[metric].mean()),
                          participant_bootstrap_ci=bootstrap_mean(group[metric]))
                         for metric in ['balanced_accuracy','auc','recall_left','recall_right']}
                         for name, group in results.groupby('model')}
    pivot=results.pivot(index='subject',columns='model',values='balanced_accuracy')
    report['paired_balanced_accuracy_differences']={name:dict(
        selected_minus_comparator=float((pivot[selection['model']]-pivot[name]).mean()),
        participant_bootstrap_ci=bootstrap_mean(pivot[selection['model']]-pivot[name]))
        for name in pivot if name!=selection['model']}
    (out/'report.json').write_text(json.dumps(report, indent=2))
    from .verification import verify_run
    verify_run(out)
    from .verification import plot_external
    plot_external(frame,out,title=f'PhysioNet: {evaluation_kind} frozen external evaluation, per-class recall')
    from .external_report import generate_external_report
    generate_external_report(out)
    print(results.to_string(index=False))
