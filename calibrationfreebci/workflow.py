"""Single entry point: nested source development, verification, exploratory inference, reports."""
import json
from pathlib import Path
import shutil
import pandas as pd
from .nested import run_nested
from .verification import verify_run,exploratory_smoke


def workflow(out,config,data_root=None,cho_root=None):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    run_nested(out/'source',config,data_root)
    verify_run(out/'source')
    exploratory_smoke(out/'source',out/'cho_exploratory',[1],cho_root)
    generate_report(out/'source',out/'cho_exploratory',out/'report')
    from .alignment import run_alignment
    run_alignment(out/'alignment_source',data_root)
    exploratory_smoke(out/'alignment_source',out/'alignment_cho_exploratory',[1],cho_root)
    generate_report(out/'alignment_source',out/'alignment_cho_exploratory',out/'alignment_report')
    selection=json.loads((out/'alignment_source/selection.json').read_text())
    (out/'deployment.json').write_text(json.dumps(dict(source='alignment_source',selection=selection,
        rule='fixed aligned CSP versus RMS/CSP, selected on matched source LOSO; no target labels used',
        confirmation='pending owner approval and prior-use confirmation'),indent=2))


def generate_report(source,external,out):
    from .audit import bootstrap_mean
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source,external,out = Path(source),Path(external),Path(out)
    verify_run(source)
    verify_run(external)
    out.mkdir(parents=True,exist_ok=False)
    source_metrics = pd.read_csv(source/'participant_metrics.csv')
    external_metrics = pd.read_csv(external/'participant_metrics.csv')
    pivot = source_metrics.pivot(index='subject',columns='model',values='balanced_accuracy')
    selected,baseline=('nested_selected','fixed_rms_csp4') if 'nested_selected' in pivot else ('aligned_csp','rms_csp_lda')
    difference = pivot[selected]-pivot[baseline]
    paired = dict(mean_difference=float(difference.mean()),participant_ci=bootstrap_mean(difference),
                  limitation='source development; only nine participants; no independent confirmation')
    (out/'paired_comparison.json').write_text(json.dumps(paired,indent=2))
    fig,axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,metric in zip(axes,['balanced_accuracy','auc']):
        for name,g in source_metrics.groupby('model'):
            ax.plot(g.subject,g[metric],marker='o',label=name)
        ax.axhline(.5,color='black',linestyle='--',linewidth=.8)
        ax.set(xlabel='Held-out source participant',ylabel=metric,ylim=(0,1))
        ax.legend(fontsize=8)
    fig.suptitle('Source validation: methods compared on identical held-out participants')
    fig.savefig(out/'nested_comparison.png',dpi=240)
    fig.savefig(out/'nested_comparison.svg')
    plt.close(fig)
    for label,folder in [('source',source),('cho_exploratory',external)]:
        target = out/label
        target.mkdir()
        for path in folder.iterdir():
            if path.suffix in ('.csv','.json','.png','.svg'):
                shutil.copy2(path,target/path.name)
    # Avoid optional tabulate dependency for a deterministic Markdown report.
    lines = ['# Executed validation report','',
             'External results below are historically exposed Cho2017, not confirmatory.','',
             '## Source participant means ('+('nested validation' if selected=='nested_selected' else 'LOSO development')+')','',
             '| Method | Balanced accuracy | ROC-AUC | Left recall | Right recall |',
             '|---|---:|---:|---:|---:|']
    for name,g in source_metrics.groupby('model'):
        lines.append(f'| {name} | {g.balanced_accuracy.mean():.4f} | {g.auc.mean():.4f} | {g.recall_left.mean():.4f} | {g.recall_right.mean():.4f} |')
    lines += ['', '## Exploratory external participant means','',
              '| Method | Balanced accuracy | ROC-AUC | Left recall | Right recall |',
              '|---|---:|---:|---:|---:|']
    for name,g in external_metrics.groupby('model'):
        lines.append(f'| {name} | {g.balanced_accuracy.mean():.4f} | {g.auc.mean():.4f} | {g.recall_left.mean():.4f} | {g.recall_right.mean():.4f} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
