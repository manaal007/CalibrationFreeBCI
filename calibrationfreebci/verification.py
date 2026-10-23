"""Independent saved-output checks shared by source, nested and external runs."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .audit import sha256
from .experiment import metrics
from .models import right_score


def load_comparisons(source):
    source = Path(source)
    selection = json.loads((source/'selection.json').read_text())
    if sha256(source/'frozen_model.joblib') != selection['model_sha256']:
        raise ValueError('Frozen selected model checksum mismatch')
    manifest = json.loads((source/'comparison_manifest.json').read_text())
    if selection['model'] not in manifest:
        raise ValueError('Selected method missing from comparison manifest')
    models = {}
    for name, entry in manifest.items():
        path = source/entry['file']
        if path.parent.resolve() != source.resolve() or sha256(path) != entry['sha256']:
            raise ValueError('Comparison model path or checksum mismatch')
        models[name] = joblib.load(path)
    return models,manifest,selection


def model_input(x,entry):
    if x.ndim == 4 and entry.get('input') != 'filterbank':
        return x[:,0].astype(float)
    return x


def verify_run(directory):
    directory = Path(directory)
    predictions = pd.read_csv(directory/'predictions.csv')
    saved = pd.read_csv(directory/'participant_metrics.csv')
    keys = [k for k in ['model','subject','session','run','trial','label'] if k in predictions]
    if predictions.duplicated(keys).any() or predictions[keys+['prediction','score_right']].isna().any().any():
        raise ValueError('Duplicate or missing predictions')
    if not set(predictions.prediction) <= {1,2}:
        raise ValueError('Invalid predicted labels')
    if saved.duplicated(['model','subject']).any():
        raise ValueError('Duplicate saved metrics')
    expected_keys = set(map(tuple,saved[['model','subject']].to_numpy()))
    actual_keys = set()
    identity = None
    trial_keys = [k for k in keys if k != 'model']
    for model,group in predictions.groupby('model'):
        current = group[trial_keys].sort_values(trial_keys).reset_index(drop=True)
        if identity is None:
            identity = current
        elif not identity.equals(current):
            raise ValueError('Different evaluation cohorts')
        for subject,frame in group.groupby('subject'):
            actual_keys.add((model,subject))
            result = metrics(frame.label.to_numpy(),frame.prediction.to_numpy(),frame.score_right.to_numpy())
            row = saved[(saved.model==model)&(saved.subject==subject)]
            if len(row) != 1:
                raise ValueError('Missing saved metric row')
            for metric,value in result.items():
                np.testing.assert_allclose(value,row.iloc[0][metric],atol=1e-12,rtol=1e-12)
    if actual_keys != expected_keys:
        raise ValueError('Metric and prediction keys differ')
    report = dict(status='passed',prediction_rows=len(predictions),participant_model_rows=len(saved),
                  predictions_sha256=sha256(directory/'predictions.csv'))
    (directory/'metric_verification.json').write_text(json.dumps(report,indent=2))
    return report


def exploratory_smoke(source,out,subjects=(1,),root=None):
    from .cho import load_cho
    source,out = Path(source),Path(out)
    models,manifest,selection = load_comparisons(source)
    out.mkdir(parents=True,exist_ok=False)
    protocol = dict(dataset='Cho2017',subjects=list(subjects),status='historically exposed exploratory smoke',
                    source_selection=selection,target_tuning=False,artifact_policy='retain and report flags',
                    model_manifest=manifest)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    x,metadata,files = load_cho(subjects,root,selection.get('bands') if selection.get('input')=='filterbank' else None)
    frames,rows = [],[]
    for name,model in models.items():
        values = model_input(x,manifest[name])
        scores = right_score(model,values,metadata.subject.to_numpy())
        again = right_score(joblib.load(source/manifest[name]['file']),values,metadata.subject.to_numpy())
        np.testing.assert_allclose(scores,again,atol=1e-12,rtol=1e-12)
        if name == selection['model']:
            np.testing.assert_allclose(scores,right_score(joblib.load(source/'frozen_model.joblib'),values,metadata.subject.to_numpy()),atol=1e-12)
        frame = metadata.copy()
        frame['model'],frame['score_right'] = name,scores
        frame['prediction'] = np.where(scores>manifest[name]['threshold'],2,1)
        frames.append(frame)
        for subject,g in frame.groupby('subject'):
            rows.append(dict(model=name,subject=subject,**metrics(g.label.to_numpy(),g.prediction.to_numpy(),g.score_right.to_numpy())))
    pd.concat(frames).to_csv(out/'predictions.csv',index=False)
    results = pd.DataFrame(rows)
    results.to_csv(out/'participant_metrics.csv',index=False)
    (out/'data_provenance.json').write_text(json.dumps(files,indent=2))
    print(json.dumps(verify_run(out),indent=2))
    print(results.to_string(index=False))
    plot_external(pd.concat(frames),out)


def plot_external(frame,out,title='Exploratory Cho2017: per-class recall (not independent confirmation)'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    names = list(frame.model.unique())
    fig,axes = plt.subplots(1,len(names),figsize=(4*len(names),3.8),squeeze=False,constrained_layout=True)
    for ax,name in zip(axes[0],names):
        group = frame[frame.model==name]
        ConfusionMatrixDisplay.from_predictions(group.label,group.prediction,labels=[1,2],
            display_labels=['Left','Right'],normalize='true',ax=ax,colorbar=False,values_format='.2f',
            im_kw={'vmin':0,'vmax':1})
        ax.set_title(name)
    fig.suptitle(title)
    fig.savefig(Path(out)/'confusion.png',dpi=220)
    fig.savefig(Path(out)/'confusion.svg')
    plt.close(fig)
