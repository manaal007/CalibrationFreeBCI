"""Small, scientifically motivated candidate set, selected using source data only."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline


class TrialRMS(BaseEstimator, TransformerMixin):
    """Label-free per-trial global scale removal; no target cohort fitting."""
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 3 or not np.isfinite(x).all():
            raise ValueError("Expected finite (trial, channel, time) EEG")
        scale = np.sqrt(np.mean(x * x, axis=(1, 2), keepdims=True))
        if np.any(scale <= 0):
            raise ValueError("Zero-energy trial")
        return x / scale


class TraceNormalize(BaseEstimator, TransformerMixin):
    """Separate covariance shape from global power after shrinkage estimation."""
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        trace = np.trace(x, axis1=-2, axis2=-1)
        if not np.isfinite(x).all() or np.any(trace <= 0):
            raise ValueError("Invalid covariance trace")
        return x / trace[:, None, None]


def build(name):
    from mne.decoding import CSP
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    if name == "csp_lda":
        return make_pipeline(CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False),
                             LinearDiscriminantAnalysis())
    if name == "rms_csp_lda":
        return make_pipeline(TrialRMS(), CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False),
                             LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    if name == "oas_mdm":
        return make_pipeline(Covariances(estimator="oas"), MDM(metric="riemann"))
    if name == "trace_oas_mdm":
        return make_pipeline(Covariances(estimator="oas"), TraceNormalize(), MDM(metric="riemann"))
    if name == "trace_tangent_lda":
        return make_pipeline(Covariances(estimator="oas"), TraceNormalize(), TangentSpace(metric="riemann"),
                             LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    raise ValueError(f"Unknown candidate: {name}")


def right_score(model, x, groups=None):
    """MDM signed distance; LDA decision function. Neither is called a probability."""
    if hasattr(model,'decision_by_subject'):
        return model.decision_by_subject(x,groups)
    if "mdm" in model.named_steps:
        transformed = x
        for step in model.steps[:-1]:
            transformed = step[1].transform(transformed)
        mdm = model.steps[-1][1]
        distances = mdm.transform(transformed)
        classes = list(mdm.classes_)
        return distances[:, classes.index(1)] - distances[:, classes.index(2)]
    if list(model.classes_) != [1, 2]:
        raise ValueError("Unexpected class order")
    return model.decision_function(x)
