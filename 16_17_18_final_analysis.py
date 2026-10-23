"""CalibrationFreeBCI combined Experiments 16–18: statistics, robustness, figures.

Requires Experiment 15B confirmatory subject metrics and trial predictions.
No training, threshold fitting, or use of exploratory subjects for model selection.
Produces DESCRIPTIVE results only when n<10 and flags provisional metadata policy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

HERE = Path(__file__).resolve().parent
IN = HERE / 'experiment15B_confirmatory_external'
OUT = HERE / 'experiment16_17_18_final_analysis'
SEED = 20261023
BOOTSTRAPS = 5000


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def ci_subject_bootstrap(values, rng):
    """Resample participants, never individual trials."""
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return (np.nan, np.nan)
    sample_indices = rng.integers(0, len(vals), size=(BOOTSTRAPS, len(vals)))
    draws = vals[sample_indices].mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]).tolist())


def main():
    OUT.mkdir(exist_ok=True)
    require((IN / '15B_protocol.json').is_file(), 'Run 15B_full_external_validation.py first.')
    protocol = json.loads((IN / '15B_protocol.json').read_text(encoding='utf-8'))
    require(protocol.get('status', '').startswith('COMPLETE'), '15B not complete')
    require((IN / '15B_subject_metrics.csv').is_file() and
            (IN / '15B_trial_predictions.csv').is_file(), 'Missing 15B CSV files')
    df = pd.read_csv(IN / '15B_subject_metrics.csv')
    trials = pd.read_csv(IN / '15B_trial_predictions.csv')
    require(len(df) and len(trials), 'Empty 15B output')
    require(set(df.subject.astype(int)).issubset(set(range(7, 53))),
            'Confirmatory metrics include previously inspected subjects 1..6')
    require(set(df.subject.astype(int)) == set(range(7, 53)),
            'Expected all 46 confirmatory participants (7..52)')
    require(set(trials.subject.astype(int)) == set(df.subject.astype(int)),
            'Participant mismatch between metrics and trial predictions')
    require(not df.duplicated(['subject', 'model']).any(), 'Duplicate subject/model rows')
    require(not trials.duplicated(['subject', 'model', 'imagery_class', 'matlab_trial']).any(),
            'Duplicate trial prediction rows')
    require(set(trials.true_label.unique()).issubset({1, 2}) and
            set(trials.predicted_label.unique()).issubset({1, 2}), 'Unexpected class labels')
    require(set(map(tuple, df[['subject', 'model']].itertuples(index=False, name=None))) ==
            set(map(tuple, trials[['subject', 'model']].drop_duplicates().itertuples(index=False, name=None))),
            'Model/participant pairs differ between metrics and predictions')
    require(np.isfinite(df[['auc', 'balanced_accuracy']].to_numpy(dtype=float)).all() and
            np.isfinite(trials['score_right'].to_numpy(dtype=float)).all(),
            'Non-finite performance values or decision scores')
    for (s, model), g in trials.groupby(['subject', 'model']):
        require(len(g) == int(df.loc[(df.subject == s) & (df.model == model), 'n_trials'].iloc[0]),
                f'Trial count mismatch for {s}/{model}')
        computed_auc = roc_auc_score(g.true_label == 2, g.score_right)
        computed_ba = balanced_accuracy_score(g.true_label, g.predicted_label)
        row = df.loc[(df.subject == s) & (df.model == model)].iloc[0]
        require(abs(computed_auc - row.auc) < 1e-8 and
                abs(computed_ba - row.balanced_accuracy) < 1e-8,
                f'Saved metrics do not match predictions for {s}/{model}')

    require(protocol.get('status') == 'COMPLETE_FULL_COHORT',
            '15B protocol must indicate COMPLETE_FULL_COHORT')
    require(not protocol.get('missing_subjects'), '15B protocol still lists missing participants')
    rng = np.random.default_rng(SEED)
    n_subjects = df.subject.nunique()
    # Experiment 16: subject-level estimates and paired model comparisons.
    summary_rows = []
    for model, group in df.groupby('model', sort=True):
        for metric in ('auc', 'balanced_accuracy'):
            vals = group.sort_values('subject')[metric].to_numpy()
            lo, hi = ci_subject_bootstrap(vals, rng)
            summary_rows.append(dict(model=model, metric=metric, n_subjects=len(vals),
                                     mean=float(vals.mean()), sd=float(vals.std(ddof=1)) if len(vals)>1 else np.nan,
                                     ci95_low=lo, ci95_high=hi,
                                     fraction_above_chance=float(np.mean(vals > .5))))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / '16_subject_bootstrap_summary.csv', index=False)

    comparisons = []
    for family in ('CSP_LDA', 'Riemann_MDM'):
        base, tuned = family + '_untuned_baseline', family + '_source_tuned'
        for metric in ('auc', 'balanced_accuracy'):
            pivot = df.pivot(index='subject', columns='model', values=metric)
            if not {base, tuned}.issubset(pivot.columns):
                continue
            paired = pivot[[base, tuned]].dropna()
            differences = (paired[tuned] - paired[base]).to_numpy()
            lo, hi = ci_subject_bootstrap(differences, rng)
            # Wilcoxon is descriptive/exploratory for small cohorts and tied differences.
            p = np.nan
            if len(differences) >= 6 and np.any(differences != 0):
                try:
                    p = float(wilcoxon(differences, zero_method='wilcox').pvalue)
                except ValueError:
                    pass
            comparisons.append(dict(family=family, metric=metric, n_paired=len(differences),
                                    mean_tuned_minus_baseline=float(differences.mean()),
                                    ci95_low=lo, ci95_high=hi, wilcoxon_p_exploratory=p))
    pd.DataFrame(comparisons).to_csv(OUT / '16_paired_model_comparisons.csv', index=False)

    # Experiment 17: descriptive robustness diagnostics without target tuning.
    diagnostics = []
    for (subject, model), group in trials.groupby(['subject', 'model']):
        predictions = group.predicted_label.to_numpy()
        diagnostics.append(dict(subject=subject, model=model, n_trials=len(group),
                                fraction_predicted_right=float(np.mean(predictions == 2)),
                                fraction_actual_right=float(np.mean(group.true_label == 2)),
                                score_min=float(group.score_right.min()),
                                score_max=float(group.score_right.max()),
                                score_median=float(group.score_right.median())))
    pd.DataFrame(diagnostics).to_csv(OUT / '17_threshold_and_prediction_diagnostics.csv', index=False)
    sensitivity = []
    for model, group in df.groupby('model'):
        for metric in ('auc', 'balanced_accuracy'):
            vals = group[metric].to_numpy(dtype=float)
            for i, (_, row) in enumerate(group.iterrows()):
                if len(vals) < 2:
                    continue
                sensitivity.append(dict(model=model, metric=metric,
                                        omitted_subject=int(row.subject),
                                        mean_after_omission=float(np.delete(vals, i).mean())))
    pd.DataFrame(sensitivity).to_csv(OUT / '17_leave_one_subject_out_sensitivity.csv', index=False)

    # Experiment 18: publication-oriented figures (subject units, not pooled trials).
    fig, ax = plt.subplots(figsize=(10, 5.5))
    models = sorted(df.model.unique())
    data = [df.loc[df.model == model, 'auc'].to_numpy() for model in models]
    ax.boxplot(data, tick_labels=models, showmeans=True)
    ax.axhline(.5, linestyle='--', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('ROC-AUC per participant')
    ax.set_title('Frozen BNCI-to-Cho external transfer (participants 7–52)')
    ax.tick_params(axis='x', labelrotation=18)
    fig.tight_layout()
    fig.savefig(OUT / '18_participant_auc_boxplot.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model in models:
        g = df[df.model == model].sort_values('subject')
        ax.plot(g.subject, g.auc, marker='o', linewidth=1, label=model)
    ax.axhline(.5, linestyle='--', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Cho2017 participant')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Between-participant external-validation variability')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / '18_subject_auc_variability.png', dpi=220)
    plt.close(fig)

    # Model equivalence audit: identical summary scores do not prove identical predictions.
    equivalence = []
    for family in ('CSP_LDA', 'Riemann_MDM'):
        base, tuned = family + '_untuned_baseline', family + '_source_tuned'
        a = trials[trials.model == base].sort_values(['subject', 'imagery_class', 'matlab_trial'])
        b = trials[trials.model == tuned].sort_values(['subject', 'imagery_class', 'matlab_trial'])
        if len(a) and len(a) == len(b):
            require(np.array_equal(a[['subject', 'imagery_class', 'matlab_trial']].to_numpy(),
                                   b[['subject', 'imagery_class', 'matlab_trial']].to_numpy()),
                    f'Mismatched trial identities for {family}')
            equivalence.append(dict(family=family, n_trials=len(a),
                                    identical_predictions=bool(np.array_equal(a.predicted_label.to_numpy(), b.predicted_label.to_numpy())),
                                    identical_scores=bool(np.array_equal(a.score_right.to_numpy(), b.score_right.to_numpy())),
                                    maximum_absolute_score_difference=float(np.max(np.abs(a.score_right.to_numpy()-b.score_right.to_numpy())))))
    pd.DataFrame(equivalence).to_csv(OUT / '17_tuned_vs_baseline_equivalence.csv', index=False)

    report = dict(experiments=['16', '17', '18'], status='COMPLETE',
                  confirmatory_participants=int(n_subjects),
                  cohort_complete=(len(protocol.get('missing_subjects', [])) == 0),
                  bad_trial_index_policy_verified=bool(protocol.get('bad_trial_index_policy_verified', False)),
                  scientific_status='PROVISIONAL' if not protocol.get('bad_trial_index_policy_verified') else 'REVIEW_REQUIRED',
                  bootstrap_seed=SEED, bootstrap_replicates=BOOTSTRAPS,
                  bootstrap_unit='participant',
                  model_training=False, threshold_tuning=False,
                  inference_notes=['CIs reflect participant sampling only; do not resolve preprocessing uncertainty.',
                                   'Wilcoxon p-values are exploratory and unadjusted for multiple comparisons.',
                                   'No statistical claims of above-chance generalization from AUC point estimates alone.',
                                   'Previously inspected subjects 1..6 are excluded from confirmatory analysis.',
                                   'The source-trained fixed threshold is never recalibrated on Cho labels.'])
    (OUT / '16_17_18_analysis_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('EXPERIMENTS 16–18 COMPLETE')
    print('Participants:', n_subjects)
    print(summary.to_string(index=False))
    print('Results:', OUT)
    print('Model equivalence audit:')
    print(pd.DataFrame(equivalence).to_string(index=False))
    if not report['bad_trial_index_policy_verified']:
        print('WARNING: PROVISIONAL results; independently verify bad-trial index semantics.')
    if n_subjects < 10:
        print('WARNING: small confirmatory cohort; interpret estimates cautiously.')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'STOPPED: {type(e).__name__}: {e}', file=sys.stderr)
        sys.exit(1)
