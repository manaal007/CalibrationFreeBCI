"""Label-free participant-batch Euclidean alignment; explicitly transductive."""
import json
from pathlib import Path
import joblib
import mne
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,ClassifierMixin
from threadpoolctl import threadpool_limits
from .models import TrialRMS,build
from .data import load_bnci
from .experiment import metrics
from .audit import sha256,bootstrap_mean


def align_participant(x):
    """Whiten mean trace-normalized trial covariance with fixed diagonal loading.

    No labels. Each participant is aligned independently, including held-out
    participants. Predictions depend on which unlabeled trials form the batch.
    """
    x = TrialRMS().transform(x)
    x = x-x.mean(axis=-1,keepdims=True)
    covariance = np.einsum('nct,ndt->ncd',x,x)/(x.shape[-1]-1)
    covariance /= np.trace(covariance,axis1=1,axis2=2)[:,None,None]
    reference = covariance.mean(axis=0)
    reference = .99*reference+.01*np.eye(x.shape[1])/x.shape[1]
    eigenvalues,eigenvectors = np.linalg.eigh(reference)
    whitening = (eigenvectors*(1/np.sqrt(eigenvalues)))@eigenvectors.T
    return np.einsum('cd,ndt->nct',whitening,x)


class AlignedCSP(ClassifierMixin,BaseEstimator):
    def fit(self,x,y,groups=None):
        if groups is None:
            raise ValueError('Source participant groups are required for alignment')
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(x):
            raise ValueError('Source participant groups must match trials')
        aligned = np.empty_like(x,dtype=float)
        for subject in np.unique(groups):
            selected = groups==subject
            aligned[selected] = align_participant(x[selected])
        self.model_ = build('rms_csp_lda').fit(aligned,y)
        self.classes_ = self.model_.classes_
        return self

    def decision_by_subject(self,x,groups=None):
        groups = np.ones(len(x),int) if groups is None else np.asarray(groups)
        if len(groups) != len(x):
            raise ValueError('Participant groups must match trials')
        scores = np.empty(len(x))
        for subject in np.unique(groups):
            selected = groups==subject
            scores[selected] = self.model_.decision_function(align_participant(x[selected]))
        return scores

    def decision_function(self,x):
        return self.decision_by_subject(x)

    def predict(self,x):
        return np.where(self.decision_function(x)>0,2,1)


def run_alignment(out,data_root=None):
    from .verification import verify_run
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    protocol = dict(subjects=list(range(1,10)),candidates=['rms_csp_lda','aligned_csp'],
        alignment='mean trace covariance whitening, 1 percent diagonal loading',
        target_batch_adaptation=True,target_labels_used=False,
        interpretation='source LOSO development; alignment is unlabeled participant-batch adaptation, not inductive inference',
        artifact_policy='retain flags',window_after_cue=[.5,2.5],threshold=0)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    mne.set_log_level('WARNING')
    x,frame,files = load_bnci(protocol['subjects'],data_root)
    y,groups = frame.label.to_numpy(),frame.subject.to_numpy()
    predictions,rows = [],[]
    with threadpool_limits(limits=2):
        for name in protocol['candidates']:
            for subject in protocol['subjects']:
                train,test = groups!=subject,groups==subject
                if name=='aligned_csp':
                    model = AlignedCSP().fit(x[train],y[train],groups[train])
                    scores = model.decision_by_subject(x[test],groups[test])
                else:
                    model = build(name).fit(x[train],y[train])
                    scores = model.decision_function(x[test])
                current = frame.loc[test].copy()
                current['model'],current['score_right'],current['prediction'] = name,scores,np.where(scores>0,2,1)
                predictions.append(current)
                result = dict(model=name,subject=subject,**metrics(y[test],current.prediction.to_numpy(),scores))
                rows.append(result)
                print(f'{name} participant {subject}: BA={result["balanced_accuracy"]:.3f} AUC={result["auc"]:.3f}',flush=True)
        results = pd.DataFrame(rows)
        results.to_csv(out/'participant_metrics.csv',index=False)
        pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
        summary = []
        for name,g in results.groupby('model',sort=False):
            row = dict(model=name)
            for metric in ('balanced_accuracy','auc','recall_left','recall_right'):
                lo,hi=bootstrap_mean(g[metric])
                row[metric],row[metric+'_ci_low'],row[metric+'_ci_high']=float(g[metric].mean()),lo,hi
            summary.append(row)
        summary=pd.DataFrame(summary)
        summary.to_csv(out/'summary.csv',index=False)
        winner=summary.sort_values('balanced_accuracy',ascending=False,kind='stable').iloc[0].model
        manifest={}
        for name in protocol['candidates']:
            model=AlignedCSP().fit(x,y,groups) if name=='aligned_csp' else build(name).fit(x,y)
            path=out/f'comparison_{name}.joblib'
            joblib.dump(model,path)
            manifest[name]=dict(file=path.name,sha256=sha256(path),threshold=0,input='epochs',
                                participant_batch_adaptation=(name=='aligned_csp'))
        joblib.dump(joblib.load(out/manifest[winner]['file']),out/'frozen_model.joblib')
    (out/'comparison_manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'selection.json').write_text(json.dumps(dict(model=winner,threshold=0,input='epochs',
        participant_batch_adaptation=(winner=='aligned_csp'),model_sha256=sha256(out/'frozen_model.joblib')),indent=2))
    (out/'data_provenance.json').write_text(json.dumps(files,indent=2))
    verify_run(out)
    print(summary.to_string(index=False),flush=True)
