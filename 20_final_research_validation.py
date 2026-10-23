"""CalibrationFreeBCI: combined post-hoc integrity, robustness and statistics audit.
Run in PyCharm from the existing project directory. Original experiments are read-only.
"""
from __future__ import annotations
import csv
import importlib.util
import json
import re
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

HERE = Path(__file__).resolve().parent
OUT = HERE / 'experiment20_final_research_validation'
OUT.mkdir(exist_ok=True)
SUBJECTS = list(range(7, 53))
CHANNELS = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
REPORT = []

def note(s):
    s = str(s)
    print(s, flush=True)
    REPORT.append(s)

def save_csv(path, rows):
    if not rows: return
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

def import_file(filename, name):
    p = HERE / filename
    if not p.exists(): raise FileNotFoundError(f'Missing required original script: {p}')
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (22,501):
        raise ValueError(f'Unexpected EEG shape: {x.shape}; expected (trials,22,501)')
    if not np.isfinite(x).all(): raise ValueError('Non-finite EEG values')
    # Participant median across per-trial root mean square, in microvolts.
    rms = np.sqrt(np.mean(x*x, axis=(1,2))) * 1e6
    return dict(n_trials=int(len(x)), median_trial_rms_uv=float(np.median(rms)),
                p25_trial_rms_uv=float(np.percentile(rms,25)),
                p75_trial_rms_uv=float(np.percentile(rms,75)),
                median_abs_uv=float(np.median(np.abs(x))*1e6),
                p99_abs_uv=float(np.percentile(np.abs(x),99)*1e6),
                max_abs_uv=float(np.max(np.abs(x))*1e6))

def subject_mat_files():
    # Exact MAT_DIR and sXX.mat naming from 15B_full_external_validation.py.
    validation = import_file('15B_full_external_validation.py', 'validation_paths_audit')
    root = Path(validation.MAT_DIR)
    note('Using original 15B MAT_DIR: ' + str(root))
    return {s: [root / f's{s:02d}.mat'] for s in SUBJECTS
            if (root / f's{s:02d}.mat').is_file()}

def source_audit():
    note('\n=== A. SOURCE EEG (BNCI2014-001) ===')
    pilot = import_file('15_external_transfer_pilot.py','pilot_final_audit')
    source = pilot.load_source()
    x = np.asarray(source[0]); y = np.asarray(source[1]).ravel()
    if len(x) != 2592 or len(y) != 2592:
        raise ValueError(f'Source epoch count changed: x={len(x)}, y={len(y)}; cannot assume 288/subject')
    rows = []
    for i in range(9):
        a = x[i*288:(i+1)*288]
        row = dict(dataset='BNCI2014-001',subject=i+1,**stats(a))
        rows.append(row)
        note(f"BNCI {i+1}: {row['median_trial_rms_uv']:.3f} µV median trial RMS")
    save_csv(OUT/'source_amplitudes.csv', rows)
    return rows, pilot

def target_audit(pilot):
    note('\n=== B. ALL 46 TARGET PARTICIPANTS (Cho2017) ===')
    validation = import_file('15B_full_external_validation.py','validation_final_audit')
    eeg_names = pilot.get_cho_eeg_names()
    indices = [eeg_names.index(ch) for ch in CHANNELS]
    matches = subject_mat_files()
    if any(len(v)>1 for v in matches.values()):
        raise RuntimeError('Ambiguous Cho MATLAB subject files: '+str({k:[str(p) for p in v] for k,v in matches.items() if len(v)>1}))
    missing = [s for s in SUBJECTS if s not in matches]
    if missing:
        note('MISSING Cho files: '+str(missing))
        note('Put original subject files in data/cho2017_mat with names such as subject7.mat, or adjust subject_mat_files() to match your existing folder and naming.')
        note('NO full-cohort claim will be made until all 46 files are found.')
        return [], missing
    rows = []
    for s in SUBJECTS:
        path = matches[s][0]
        try:
            result = validation.read_subject(path,indices)
            x,y,trial_ids,quality = result
            x=np.asarray(x); y=np.asarray(y)
            row=dict(dataset='Cho2017',subject=s,source_file=str(path),**stats(x),
                     n_left=int(np.sum(y==1)),n_right=int(np.sum(y==2)))
            if len(np.unique(y))<2: row['warning']='ONE_CLASS_RETAINED'
            if len(y)<30: row['warning']='FEWER_THAN_30_TRIALS'
            rows.append(row)
            note(f"Cho {s}: {len(y)} trials, RMS {row['median_trial_rms_uv']:.3f} µV")
        except Exception as e:
            note(f'ERROR Cho {s}: {type(e).__name__}: {e}')
            rows.append(dict(dataset='Cho2017',subject=s,source_file=str(path),error=str(e)))
    save_csv(OUT/'target_amplitudes.csv',rows)
    return rows, []

def prediction_files():
    roots=[HERE/'experiment15B_confirmatory_external', HERE/'experiment16', HERE/'experiment17', HERE/'experiment18']
    found=[]
    for root in roots:
        if root.exists(): found += list(root.rglob('*.csv'))
    return sorted(set(found))

def identify(df_cols, candidates):
    lower={c.lower().strip():c for c in df_cols}
    for name in candidates:
        if name in lower: return lower[name]
    return None

def metrics_audit():
    note('\n=== C. SAVED PREDICTION AUDIT (READ ONLY) ===')
    import pandas as pd
    path = HERE/'experiment15B_confirmatory_external'/'15B_trial_predictions.csv'
    saved = HERE/'experiment15B_confirmatory_external'/'15B_subject_metrics.csv'
    if not path.is_file():
        note('MISSING original 15B trial predictions: '+str(path)); return None
    df = pd.read_csv(path)
    required={'subject','model','true_label','predicted_label','score_right','imagery_class','matlab_trial'}
    if not required.issubset(df.columns):
        note('SCHEMA ERROR: missing '+str(sorted(required-set(df.columns))));return None
    if df[list(required)].isna().any().any():
        note('ERROR: missing values in trial predictions');return None
    if not set(df.true_label.unique()).issubset({1,2}) or not set(df.predicted_label.unique()).issubset({1,2}):
        note('ERROR: unexpected label encoding');return None
    if not np.isfinite(df.score_right.to_numpy(dtype=float)).all():
        note('ERROR: nonfinite scores');return None
    keys=['subject','model','imagery_class','matlab_trial']
    if df.duplicated(keys).any():
        note('ERROR: duplicate subject/model/class/trial predictions');return None
    if df.subject.nunique()!=46 or set(df.subject.unique())!=set(SUBJECTS):
        note('ERROR: expected all 46 subjects; found '+str(df.subject.nunique()));return None
    rows=[]
    for (subject,model),g in df.groupby(['subject','model'],sort=True):
        y=g.true_label.to_numpy(dtype=int)
        pred=g.predicted_label.to_numpy(dtype=int)
        scores=g.score_right.to_numpy(dtype=float)
        if set(y)!={1,2}:
            note(f'ERROR: only one true class for subject {subject} model {model}');return None
        rows.append(dict(subject=int(subject),model=str(model),n_trials=len(g),
                         auc=float(roc_auc_score(y==2,scores)),
                         balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                         accuracy=float(np.mean(y==pred)),
                         predicted_right_fraction=float(np.mean(pred==2))))
    calc=pd.DataFrame(rows)
    model_set=set(df.model.unique())
    if len(model_set)!=4 or any(set(g.model)!=model_set for _,g in calc.groupby('subject')):
        note('ERROR: incomplete model/subject combinations');return None
    save_csv(OUT/'recomputed_participant_metrics.csv',rows)
    note(f'Validated {len(df):,} trial-level prediction rows, {len(calc)} subject/model groups, {len(model_set)} models.')
    if saved.is_file():
        old=pd.read_csv(saved)
        merged=calc.merge(old,on=['subject','model'],suffixes=('_recomputed','_saved'),validate='one_to_one',indicator=True)
        if len(merged)!=len(calc) or not (merged._merge=='both').all():
            note('ERROR: saved metrics incomplete or mismatched');return None
        for col in ['auc','balanced_accuracy','accuracy','predicted_right_fraction','n_trials']:
            err=np.max(np.abs(merged[col+'_recomputed']-merged[col+'_saved']))
            note(f'Max absolute difference vs saved {col}: {err:.12g}')
            if err>1e-8: note('WARNING: discrepancy in '+col)
    else: note('Saved subject metrics file missing; independent cross-check unavailable.')
    for model,g in calc.groupby('model'):
        note(f'{model}: n={len(g)}, mean AUC={g.auc.mean():.6f}, '
             f'mean balanced accuracy={g.balanced_accuracy.mean():.6f}, '
             f'right predictions={np.average(g.predicted_right_fraction,weights=g.n_trials):.4%}')
    return calc

def sensitivity(rows, metrics=None):
    note('\n=== D. POST-HOC PARTICIPANT RETENTION SENSITIVITY ===')
    clean=[r for r in rows if 'error' not in r]
    if len(clean)!=46:
        note(f'Incomplete target cohort: {len(clean)}/46; full-cohort summaries withheld.'); return
    output=[]
    for k in (1,20,30,50,100):
        included={r['subject'] for r in clean if r['n_trials']>=k}
        line=dict(minimum_retained_trials=k,participants_retained=len(included),
                  participants_excluded=46-len(included),
                  median_rms_uv=float(np.median([r['median_trial_rms_uv'] for r in clean if r['subject'] in included])) if included else '')
        if metrics is not None:
            for model,g in metrics.groupby('model'):
                subset=g[g.subject.isin(included)]
                line['mean_auc__'+model]=float(subset.auc.mean()) if len(subset) else ''
                line['mean_balanced_accuracy__'+model]=float(subset.balanced_accuracy.mean()) if len(subset) else ''
        output.append(line)
        note(f'Post-hoc minimum {k}: {len(included)}/46 subjects retained')
    save_csv(OUT/'trial_retention_sensitivity.csv',output)
    note('Subjects with <30 trials: '+str([r['subject'] for r in clean if r['n_trials']<30]))

def main():
    note('CalibrationFreeBCI FINAL RESEARCH AUDIT')
    note('Started: '+datetime.now().isoformat(timespec='seconds'))
    note('READ-ONLY originals. All outputs in '+str(OUT))
    source_rows=[];target_rows=[]
    try:
        source_rows,pilot=source_audit()
        target_rows,missing=target_audit(pilot)
        if len(target_rows)==46 and all('error' not in r for r in target_rows):
            s=np.median([r['median_trial_rms_uv'] for r in source_rows]);
            t=np.median([r['median_trial_rms_uv'] for r in target_rows]);
            note(f'FULL-COHORT MEDIAN PARTICIPANT TRIAL RMS: BNCI={s:.4f} µV; Cho={t:.4f} µV; ratio={t/s:.4f}x')
    except Exception as e:
        note('AMPLITUDE AUDIT FAILED: '+repr(e)); note(traceback.format_exc())
    calculated=None
    try: calculated=metrics_audit()
    except Exception as e:
        note('PREDICTION AUDIT FAILED: '+repr(e));note(traceback.format_exc())
    try: sensitivity(target_rows,calculated)
    except Exception as e: note('SENSITIVITY AUDIT FAILED: '+repr(e)); note(traceback.format_exc())
    note('\n=== SCIENTIFIC LIMITATIONS ===')
    note('1. Amplitude mismatch is descriptive; raw-unit correctness requires independent documentation/source inspection.')
    note('2. Target cohort was already inspected: all sensitivity analyses are post-hoc.')
    note('3. No target-label-based model fitting, recalibration or threshold optimization performed.')
    note('4. MDM signed distance differences are NOT probabilities; no 0.5 threshold assumed.')
    note('5. Preprocessing and bad-trial metadata semantics must be independently verified.')
    note('6. Saved prediction metrics require correct score orientation and no duplicated trials.')
    note('Completed: '+datetime.now().isoformat(timespec='seconds'))
    (OUT/'audit_report.txt').write_text('\n'.join(REPORT)+'\n',encoding='utf-8')
    note('Report saved: '+str(OUT/'audit_report.txt'))

if __name__=='__main__': main()
