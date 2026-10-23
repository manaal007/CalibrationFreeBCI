"""Experiment 12: retrospective integrity audit of CalibrationFreeBCI.

Run from anywhere: python 12_pipeline_integrity_audit.py --project "C:/.../CalibrationFreeBCI"
Requires: numpy, pandas, scipy, matplotlib. Does NOT download EEG or retrain models.
An automated audit cannot certify absence of leakage from source code alone.
"""
from __future__ import annotations
import argparse
import ast
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

EXPECTED_SUBJECTS = set(range(1, 10))
EXPECTED_SESSIONS = {'0train', '1test'}
EXPECTED_MODELS = {'CSP_LDA', 'Riemannian_MDM'}


def log(rows, area, status, message, evidence=''):
    rows.append(dict(area=area, status=status, message=message, evidence=str(evidence)))


def read_csv(path, rows, label):
    try:
        df = pd.read_csv(path)
        log(rows, label, 'PASS', 'CSV read successfully', f'{path.name}: {len(df)} rows')
        return df
    except Exception as exc:
        log(rows, label, 'FAIL', 'Cannot read CSV', f'{path}: {exc}')
        return None


def source_review(project, rows):
    records = []
    patterns = {
        'fit_or_fit_transform': r'\.fit(?:_transform)?\s*\(',
        'cross_validation': r'cross_val_score|cross_validate|GridSearchCV|RandomizedSearchCV',
        'subject_splitting': r'LeaveOneGroupOut|LeaveOneOut|subject|train_mask|test_mask',
        'preprocessing': r'StandardScaler|CSP\s*\(|Covariances\s*\(|PCA\s*\(|SelectKBest|filter_data|resample',
        'target_label_usage': r'\by_test\b|\by_train\b|\by\s*\[|get_data\s*\(',
    }
    scripts = sorted(p for p in project.glob('*.py') if re.match(r'^\d\d_', p.name))
    if not scripts:
        log(rows, 'Source audit', 'WARN', 'No numbered experiment scripts found', project)
        return records
    for path in scripts:
        try:
            text = path.read_text(encoding='utf-8-sig')
            ast.parse(text, filename=str(path))
            parsed = 'PASS'
        except (SyntaxError, UnicodeError) as exc:
            parsed = 'FAIL'
            log(rows, 'Source audit', 'FAIL', 'Source cannot be parsed', f'{path.name}: {exc}')
            try:
                text = path.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
        for category, regex in patterns.items():
            hits = [i for i, line in enumerate(text.splitlines(), 1) if re.search(regex, line)]
            if hits:
                records.append({'script': path.name, 'category': category,
                                'line_numbers': ','.join(map(str, hits[:35])), 'count': len(hits)})
        if parsed == 'PASS':
            log(rows, 'Source audit', 'PASS', 'Python source parses', path.name)
    log(rows, 'Leakage certification', 'REVIEW',
        'Source keyword/AST inspection cannot prove fitting order or absence of leakage. '
        'Inspect preprocessing, event extraction, train/test masks, and all fitted transforms manually.',
        'See source_review_locations.csv')
    return records


def analyze_sessions(df, rows, out):
    required = {'subject','session','model','roc_auc','n_train_trials','n_test_trials'}
    if not required.issubset(df.columns):
        log(rows, 'Session results', 'FAIL', 'Required columns missing', sorted(required-set(df.columns)))
        return None
    d = df.copy()
    for c in ('subject','roc_auc','n_train_trials','n_test_trials'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    if d[list(required)].isna().any().any():
        log(rows, 'Session results', 'FAIL', 'Null/non-numeric required fields')
        return None
    if not d['roc_auc'].between(0,1).all():
        log(rows, 'Session results', 'FAIL', 'AUC outside [0,1]')
    else:
        log(rows, 'Session results', 'PASS', 'All ROC-AUC scores in [0,1]')
    duplicates = d.duplicated(['subject','session','model']).sum()
    log(rows, 'Session results', 'PASS' if duplicates == 0 else 'FAIL',
        'Unique participant/session/model rows' if duplicates == 0 else 'Duplicate rows', duplicates)
    observed = set(zip(d.subject.astype(int), d.session.astype(str), d.model.astype(str)))
    expected = {(s, sess, model) for s in EXPECTED_SUBJECTS for sess in EXPECTED_SESSIONS for model in EXPECTED_MODELS}
    missing, extra = expected-observed, observed-expected
    log(rows, 'Session results', 'PASS' if not missing and not extra else 'FAIL',
        'Expected complete 9 × 2 × 2 result grid', f'missing={sorted(missing)}, extra={sorted(extra)}')
    count_ok = (d.n_train_trials.eq(2304) & d.n_test_trials.eq(144)).all()
    log(rows, 'Trial counts', 'PASS' if count_ok else 'WARN',
        'Reported counts consistent with 8 × 288 training and 144 session-test trials',
        'Reported counts are NOT independent verification of actual indices')
    if missing or duplicates or extra:
        return None
    means = d.groupby(['subject','model'], as_index=False)['roc_auc'].mean()
    wide = means.pivot(index='subject', columns='model', values='roc_auc').sort_index()
    wide['delta_auc'] = wide['Riemannian_MDM'] - wide['CSP_LDA']
    wide.to_csv(out/'12_recomputed_subject_auc.csv', float_format='%.12f')
    d.to_csv(out/'12_verified_session_rows.csv', index=False)
    return wide


def compare_subject_file(project, wide, rows):
    path = project/'10_riemannian_subject_comparison.csv'
    if not path.exists():
        log(rows, 'Cross-file agreement', 'WARN', 'Original subject summary not found', path.name)
        return
    other = read_csv(path, rows, 'Cross-file agreement')
    if other is None or 'subject' not in other:
        return
    other = other.set_index('subject')
    for col in ['CSP_LDA','Riemannian_MDM','MDM_minus_CSP']:
        corresponding = 'delta_auc' if col == 'MDM_minus_CSP' else col
        if col not in other.columns:
            log(rows, 'Cross-file agreement', 'WARN', f'Missing column {col}')
            continue
        vals = pd.to_numeric(other[col], errors='coerce').reindex(wide.index)
        diff = (vals-wide[corresponding]).abs()
        good = diff.notna().all() and (diff <= 1e-6).all()
        log(rows, 'Cross-file agreement', 'PASS' if good else 'FAIL',
            f'{col} agrees with session-level recomputation', f'max_abs_diff={diff.max()}')


def stats_and_figures(wide, rows, out):
    delta = wide['delta_auc'].to_numpy()
    rng = np.random.default_rng(20261021)
    # Resample whole participants, NOT sessions/trials (cluster independence).
    idx = rng.integers(0, len(delta), size=(20000, len(delta)))
    boot = delta[idx].mean(axis=1)
    ci = np.quantile(boot, [0.025,0.975])
    try:
        w = wilcoxon(delta, zero_method='wilcox', alternative='two-sided', method='exact')
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    except ValueError:
        w_stat, w_p = float('nan'), float('nan')
    metrics = {
        'n_independent_participants': len(delta),
        'mean_CSP_LDA': float(wide['CSP_LDA'].mean()),
        'mean_Riemannian_MDM': float(wide['Riemannian_MDM'].mean()),
        'mean_paired_delta_MDM_minus_CSP': float(delta.mean()),
        'participant_bootstrap_95pct_CI_delta': [float(x) for x in ci],
        'wilcoxon_statistic': w_stat, 'wilcoxon_two_sided_p': w_p,
        'participants_MDM_better': int((delta>0).sum()),
        'participants_CSP_better': int((delta<0).sum()),
        'minimum_subject_CSP_AUC': float(wide['CSP_LDA'].min()),
        'maximum_subject_CSP_AUC': float(wide['CSP_LDA'].max()),
        'caution': 'Bootstrap is descriptive and unstable with n=9; no prospective external validation.'
    }
    (out/'12_statistical_summary.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(9,5.5))
    x = np.arange(len(wide))
    ax.plot(x, wide['CSP_LDA'], marker='o', label='CSP + LDA')
    ax.plot(x, wide['Riemannian_MDM'], marker='s', label='Riemannian MDM')
    ax.axhline(0.5, linestyle='--', color='gray', linewidth=1, label='Chance AUC')
    ax.set_xticks(x, [f'S{int(s)}' for s in wide.index])
    ax.set_ylim(0.4,1.02)
    ax.set_ylabel('Mean session ROC-AUC')
    ax.set_title('Experiment 12: held-out participant performance')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'12_subject_generalization.png', dpi=250)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(x, delta)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x, [f'S{int(s)}' for s in wide.index])
    ax.set_ylabel('MDM − CSP (ROC-AUC)')
    ax.set_title('Experiment 12: paired decoder differences')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'12_paired_decoder_differences.png', dpi=250)
    plt.close(fig)
    log(rows, 'Statistical analysis', 'PASS', 'Participant-level bootstrap and paired Wilcoxon computed',
        f"mean_delta={delta.mean():+.6f}; CI=[{ci[0]:+.6f},{ci[1]:+.6f}]; p={w_p:.6f}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    project = args.project.expanduser().resolve()
    if not project.is_dir():
        ap.error(f'Project folder not found: {project}')
    out = project/'experiment12_audit'
    out.mkdir(exist_ok=True)
    rows = []
    print('\nEXPERIMENT 12 — PIPELINE INTEGRITY & GENERALIZATION AUDIT')
    print('='*65)
    print('Project:', project)
    source_records = source_review(project, rows)
    pd.DataFrame(source_records, columns=['script','category','line_numbers','count']).to_csv(
        out/'12_source_review_locations.csv', index=False)
    session_path = project/'10_riemannian_session_results.csv'
    if session_path.exists():
        sessions = read_csv(session_path, rows, 'Session results')
        wide = analyze_sessions(sessions, rows, out) if sessions is not None else None
    else:
        wide = None
        log(rows, 'Session results', 'FAIL', 'Missing required session-level results', session_path.name)
    metrics = None
    if wide is not None:
        compare_subject_file(project, wide, rows)
        metrics = stats_and_figures(wide, rows, out)
    log(rows, 'Dataset protocol', 'REVIEW',
        'Confirm MOABB paradigm epoch boundaries, event labels, filtering, referencing, and subject/session IDs '
        'against raw dataset and library documentation. Not independently checked by this script.')
    log(rows, 'Data independence', 'REVIEW',
        'CSV n_train/n_test columns cannot verify unique sample IDs or preprocessing order. '
        'Inspect or instrument actual folds before claiming leakage-free evaluation.')
    log(rows, 'Generalization', 'REVIEW',
        'All 9 participants are from one benchmark dataset; external-dataset generalization untested.')
    report = pd.DataFrame(rows)
    report.to_csv(out/'12_audit_checklist.csv', index=False)
    lines = ['# Experiment 12 — Integrity audit', '',
             'Automated checks are limited: PASS does not certify absence of leakage.', '',
             '## Findings', '']
    for r in rows:
        lines.append(f"- **{r['status']} — {r['area']}:** {r['message']}. {r['evidence']}")
    if metrics:
        lines += ['', '## Paired model comparison',
                  f"- CSP mean AUC: {metrics['mean_CSP_LDA']:.4f}",
                  f"- MDM mean AUC: {metrics['mean_Riemannian_MDM']:.4f}",
                  f"- MDM − CSP: {metrics['mean_paired_delta_MDM_minus_CSP']:+.4f}",
                  f"- Subject bootstrap 95% CI: {metrics['participant_bootstrap_95pct_CI_delta']}",
                  f"- Exact Wilcoxon p: {metrics['wilcoxon_two_sided_p']:.4f}"]
    lines += ['', '## Mandatory follow-up',
              '1. Review the actual source lines listed in 12_source_review_locations.csv.',
              '2. Verify fit/transform operations happen inside each training fold only.',
              '3. Confirm subject masks are disjoint and trial/session IDs match source data.',
              '4. Freeze preprocessing and model selection before independent dataset evaluation.',
              '5. Do not treat the 9 participants as 18 independent observations.']
    (out/'12_audit_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for r in rows:
        print(f"[{r['status']}] {r['area']}: {r['message']}")
    print('\nCounts:', report.status.value_counts().to_dict())
    if metrics:
        print('CSP mean AUC:', round(metrics['mean_CSP_LDA'], 4))
        print('MDM mean AUC:', round(metrics['mean_Riemannian_MDM'], 4))
        print('Mean difference:', round(metrics['mean_paired_delta_MDM_minus_CSP'], 4))
        print('Wilcoxon p:', round(metrics['wilcoxon_two_sided_p'], 4))
    print('\nOutput folder:', out)
    print('Note: REVIEW findings require actual code/data inspection.')


if __name__ == '__main__':
    main()
