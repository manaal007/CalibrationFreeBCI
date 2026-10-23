"""Filter-bank CSP on continuously filtered, aligned bands; no target fitting."""
import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
from .models import TrialRMS

BANDS = ((8.,30.), (8.,12.), (12.,18.), (18.,24.), (24.,30.))


class BankCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, bank=True):
        self.n_components = n_components
        self.bank = bank

    def fit(self, x, y):
        self._validate(x)
        self.indices_ = list(range(1,len(BANDS))) if self.bank else [0]
        self.csps_ = []
        for index in self.indices_:
            values = TrialRMS().transform(x[:,index])
            values -= values.mean(axis=1,keepdims=True)
            csp = CSP(n_components=self.n_components, reg='ledoit_wolf', log=True, norm_trace=False,
                      rank={'eeg':values.shape[1]-1})
            csp.fit(values, y)
            self.csps_.append(csp)
        return self

    def transform(self, x):
        check_is_fitted(self, 'csps_')
        self._validate(x)
        return np.concatenate([csp.transform(self._normalized(x[:,index]))
                               for index,csp in zip(self.indices_, self.csps_)], axis=1)

    def _normalized(self,x):
        x=TrialRMS().transform(x)
        return x-x.mean(axis=1,keepdims=True)

    def _validate(self,x):
        if x.ndim != 4 or x.shape[1] != len(BANDS) or x.shape[-1] < 2 or not np.isfinite(x).all():
            raise ValueError('Expected finite (trial, five bands, channel, time) EEG')


def build_bank(name):
    options = {'rms_csp4': (4,False), 'fbcsp2':(2,True), 'fbcsp4':(4,True)}
    if name not in options:
        raise ValueError(f'Unknown filter-bank candidate {name}')
    components, bank = options[name]
    return make_pipeline(BankCSP(components,bank),
                         LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
