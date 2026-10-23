"""Experiment 13E: protocol feasibility checks, not validation or model training.
Uses cached Cho2017 Subject 1. Does not download other subjects.
"""
from pathlib import Path
import json
import numpy as np
import mne
from moabb.datasets import Cho2017

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiment13e_protocol'
OUT.mkdir(exist_ok=True)
CHANNELS = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
SFREQ_TARGET = 250
TMIN, TMAX = 0.5, 2.5


def main():
    print('EXPERIMENT 13E — HARMONIZATION FEASIBILITY')
    print('No classifier training; one Cho2017 subject only.')
    ds = Cho2017(subjects=[1])
    runs = ds.get_data(subjects=[1])[1]
    report = {'dataset':'Cho2017', 'subject':1, 'proposed':{
        'channel_order': CHANNELS, 'reference':'average over 22 selected EEG channels',
        'filter_hz':[8,30], 'resample_hz':SFREQ_TARGET,
        'epoch_relative_to_annotation_s':[TMIN,TMAX]}, 'runs':[],
        'status':'FEASIBILITY ONLY — NOT A FROZEN OR VALIDATED PROTOCOL',
        'unresolved':[
            'Confirm acquisition references and the effect of re-referencing in both datasets.',
            'Confirm precise equivalence of imagery cue/event semantics between datasets.',
            'Inspect original MATLAB trial boundaries and avoid filtering across artificial concatenation boundaries.',
            'Predefine artifact and bad-trial rules for every target participant.',
            'Verify source BNCI2014-001 transformations and final sample counts independently.',
            'Determine whether source/target acquisition differences prevent meaningful transfer.'
        ]}
    for session, run_dict in runs.items():
        for run, raw in run_dict.items():
            print(f'\nSession {session}, run {run}')
            missing = [ch for ch in CHANNELS if ch not in raw.ch_names]
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            eeg_names = {raw.ch_names[i] for i in eeg_picks}
            non_eeg = [ch for ch in CHANNELS if ch not in eeg_names]
            anns = raw.annotations
            descriptions = [str(x) for x in anns.description]
            left = descriptions.count('left_hand')
            right = descriptions.count('right_hand')
            event_onsets = [float(o) for o,d in zip(anns.onset, descriptions) if d in ('left_hand','right_hand')]
            # These checks use annotations only, not labels to optimize a model.
            recording_end = raw.n_times / raw.info['sfreq']
            valid_bounds = all((o+TMIN >= 0) and (o+TMAX < recording_end) for o in event_onsets)
            sample_count_250 = int(round((TMAX-TMIN)*SFREQ_TARGET))+1
            record = {
                'session':str(session),'run':str(run), 'sfreq':float(raw.info['sfreq']),
                'eeg_channel_count':len(eeg_picks), 'missing_channels':missing,
                'required_channels_not_eeg':non_eeg,
                'annotations':{'left_hand':left,'right_hand':right},
                'annotation_duration_s':sorted(set(float(x) for x,d in zip(anns.duration, descriptions) if d in ('left_hand','right_hand'))),
                'proposed_epoch_inside_recording':valid_bounds,
                'expected_samples_at_250_hz_inclusive':sample_count_250,
                'valid_channel_labels_and_types':not missing and not non_eeg,
                'warning':'Raw is assembled from trials with artificial buffers. No filtering applied here.'
            }
            report['runs'].append(record)
            print('Required channels:', len(CHANNELS)-len(missing), '/', len(CHANNELS))
            print('Required channels typed EEG:', len(CHANNELS)-len(non_eeg), '/', len(CHANNELS))
            print('Annotation counts: left',left,'right',right)
            print('Candidate epoch inside reconstructed recording:',valid_bounds)
            print('Expected 250-Hz samples for 0.5–2.5 s:',sample_count_250)
    report['metadata_feasibility_pass'] = bool(report['runs']) and all(
        r['valid_channel_labels_and_types'] and r['proposed_epoch_inside_recording']
        for r in report['runs'])
    path = OUT / '13e_feasibility_report.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nMetadata feasibility:', 'PASS' if report['metadata_feasibility_pass'] else 'REVIEW')
    print('Preprocessing protocol frozen: NO')
    print('Saved:',path.resolve())

if __name__ == '__main__':
    main()
