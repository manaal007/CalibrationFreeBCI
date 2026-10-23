"""Generate a participant-level external report without fitting or selecting models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .verification import verify_run


def generate_external_report(directory):
    directory=Path(directory)
    verify_run(directory)
    report=json.loads((directory/'report.json').read_text())
    results=pd.read_csv(directory/'participant_metrics.csv')
    rows=[]
    for model,values in report['summary'].items():
        for metric,entry in values.items():
            np.testing.assert_allclose(results.loc[results.model==model,metric].mean(),entry['mean'],atol=1e-12)
            rows.append(dict(model=model,metric=metric,mean=entry['mean'],
                ci_low=entry['participant_bootstrap_ci'][0],ci_high=entry['participant_bootstrap_ci'][1]))
    pd.DataFrame(rows).to_csv(directory/'summary.csv',index=False)
    selected=report['model']['model']
    participants=results.subject.nunique()
    trial_counts=results[results.model==selected].n_trials
    lines=['# Frozen BNCI-to-PhysioNet external evaluation','',
           f"Evaluation kind: **{report['evaluation_kind']}**. Cohort independence: **{report['independence']}**.",'',
           f'Primary model fixed before predictions: `{selected}`. {participants} participants, {int(trial_counts.sum())} trials total.',
           'No model refitting, threshold adjustment, participant exclusions, label flipping or target-label selection was performed.',
           'Alignment estimates a separate reference from each participant\'s unlabeled batch; it is transductive adaptation.','',
           '| Model | Participant-mean balanced accuracy (95% CI) | ROC-AUC (95% CI) | Left recall | Right recall |',
           '|---|---:|---:|---:|---:|']
    for name,values in report['summary'].items():
        ba,auc=values['balanced_accuracy'],values['auc']
        lo,hi=ba['participant_bootstrap_ci']
        al,ah=auc['participant_bootstrap_ci']
        lines.append(f"| {name} | {ba['mean']:.4f} ({lo:.4f}–{hi:.4f}) | {auc['mean']:.4f} ({al:.4f}–{ah:.4f}) | {values['recall_left']['mean']:.4f} | {values['recall_right']['mean']:.4f} |")
    lines+=['','## Paired comparison','']
    for name,entry in report['paired_balanced_accuracy_differences'].items():
        lo,hi=entry['participant_bootstrap_ci']
        lines.append(f"Selected model minus `{name}`: {entry['selected_minus_comparator']:.4f} balanced accuracy; participant-bootstrap 95% interval [{lo:.4f}, {hi:.4f}].")
    lines+=['','## Participant outcomes','',
            '| Participant | Method | Balanced accuracy | ROC-AUC | Left recall | Right recall | Predicted right |',
            '|---|---|---:|---:|---:|---:|---:|']
    for row in results.sort_values(['subject','model']).itertuples():
        lines.append(f'| S{row.subject:03d} | {row.model} | {row.balanced_accuracy:.4f} | {row.auc:.4f} | {row.recall_left:.4f} | {row.recall_right:.4f} | {row.predicted_right_fraction:.4f} |')
    ci=report['summary'][selected]['balanced_accuracy']['participant_bootstrap_ci']
    chance='includes' if ci[0]<=.5<=ci[1] else 'excludes'
    lines+=['','## Interpretation','',
            f'The selected model\'s balanced-accuracy interval {chance} chance. Paired comparison intervals are reported above. A small cohort cannot establish broad population reliability. All below-chance participant outcomes were retained without flipping labels or predictions.',
            f"Evaluation kind is {report['evaluation_kind']}; cohort independence is {report['independence']}. This cohort is now exposed and must not guide model changes presented as fresh confirmation on the same cohort.",
            'All intervals resample participants (10,000 draws; seed 20261023). Small-cohort bootstrap intervals do not provide strong population guarantees.','',
            '![Participant results](participant_comparison.png)','',
            '![Class recall](confusion.png)','']
    (directory/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    plot_participants(results,directory,report)


def plot_participants(results,directory,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,metric in zip(axes,['balanced_accuracy','auc']):
        for name,group in results.groupby('model'):
            group=group.sort_values('subject')
            ax.plot(group.subject,group[metric],marker='o',label=name)
        ax.axhline(.5,color='black',linestyle='--',linewidth=.8)
        ax.set(ylim=(0,1),xlabel='External participant',ylabel=metric.replace('_',' ').upper() if metric=='auc' else 'Balanced accuracy',
               xticks=sorted(results.subject.unique()))
        ax.legend(fontsize=8)
    fig.suptitle(f"Frozen PhysioNet evaluation ({report['evaluation_kind']}; independence {report['independence']})")
    fig.savefig(directory/'participant_comparison.png',dpi=240)
    fig.savefig(directory/'participant_comparison.svg')
    plt.close(fig)
