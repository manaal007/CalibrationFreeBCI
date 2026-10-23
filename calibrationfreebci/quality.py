"""Real target loading checks without model predictions or decoding metrics."""
import json
from pathlib import Path
import numpy as np
from .data import CHANNELS
from .external import load_physionet


def check_physionet(out,subjects,root=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    x,frame,files = load_physionet(subjects,root)
    if not np.isfinite(x).all() or np.any(np.std(x,axis=(1,2))==0):
        raise ValueError('Invalid target signals')
    np.testing.assert_allclose(x.mean(axis=1),0,atol=1e-15)
    counts = []
    for subject,g in frame.groupby('subject'):
        if set(g.label) != {1,2} or set(g.run) != {4,8,12}:
            raise ValueError('Missing target class or imagery run')
        counts.append(dict(subject=int(subject),trials=len(g),left=int(sum(g.label==1)),right=int(sum(g.label==2))))
    report = dict(status='passed_loading_only',dataset='PhysioNet EEGMMIDB 1.0.0',subjects=subjects,
        channels=CHANNELS,shape=list(x.shape),units='volts',rms_median_volts=float(np.median(np.sqrt(np.mean(x*x,axis=(1,2))))),
        participant_counts=counts,provenance=files,predictions_generated=False,decoding_metrics_computed=False,
        prior_decoding_exposure='unknown; owner confirmation required')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='provenance'},indent=2))
