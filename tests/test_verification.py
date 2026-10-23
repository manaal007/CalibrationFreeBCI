import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import joblib
import mne
import numpy as np
import pandas as pd
from scipy.io import savemat
from calibrationfreebci.data import load_bnci,CHANNELS
from calibrationfreebci.experiment import metrics
from calibrationfreebci.filterbank import build_bank,BANDS
from calibrationfreebci.models import build,right_score
from calibrationfreebci.nested import participant_folds
from calibrationfreebci.verification import verify_run,load_comparisons
from calibrationfreebci.audit import sha256


class ComprehensiveTests(unittest.TestCase):
    def test_matlab_loading_artifact_policy_and_sample_identity(self):
        signal = np.random.default_rng(2).normal(0,10,(4000,25))
        run = dict(X=signal,fs=250,y=np.array([1,2]),trial=np.array([251,2001]),
                   artifacts=np.array([0,1]),classes=np.array(['left hand','right hand','feet','tongue'],dtype=object))
        with TemporaryDirectory() as tmp:
            for session in ('T','E'):
                savemat(Path(tmp)/f'A01{session}.mat',{'data':np.array([run],dtype=object)})
            x,frame,files = load_bnci([1],tmp)
            clean,clean_frame,_ = load_bnci([1],tmp,artifact_policy='reject_flagged')
            bank,bank_frame,_ = load_bnci([1],tmp,BANDS)
        self.assertEqual(x.shape,(4,22,501))
        self.assertEqual(clean.shape,(2,22,501))
        self.assertEqual(bank.shape,(4,5,22,501))
        np.testing.assert_array_equal(frame.label,[1,2,1,2])
        np.testing.assert_array_equal(frame.trial_start_sample,[250,2000,250,2000])
        self.assertTrue((clean_frame.artifact_flag==0).all())
        np.testing.assert_allclose(x,bank[:,0],atol=1e-11)
        np.testing.assert_allclose(x.mean(axis=1),0,atol=1e-18)
        self.assertLess(np.max(np.abs(x)),1e-3)
        self.assertEqual(len(files),2)

    def test_nested_folds_never_split_a_participant(self):
        groups = np.repeat(np.arange(1,10),4)
        seen = []
        for train,test in participant_folds(groups,3,123):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            seen.extend(groups[test].tolist())
            for inner_train,inner_test in participant_folds(groups[train],3,123):
                self.assertFalse(set(groups[train][inner_train]) & set(groups[train][inner_test]))
                self.assertFalse(set(groups[test]) & set(groups[train][inner_train]))
        self.assertEqual(sorted(seen),sorted(groups.tolist()))

    def test_filterbank_serialization_gain_and_class_order(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(32,5,6,100))
        y = np.tile([1,2],16)
        x[y==1,:,0] *= 4
        x[y==2,:,1] *= 4
        with TemporaryDirectory() as tmp:
            for name in ('rms_csp4','fbcsp2','fbcsp4'):
                with self.subTest(candidate=name):
                    model = build_bank(name).fit(x,y)
                    for csp in model.named_steps['bankcsp'].csps_:
                        self.assertEqual(csp.rank,{'eeg':5})
                    scores = right_score(model,x)
                    self.assertGreater(metrics(y,np.where(scores>0,2,1),scores)['balanced_accuracy'],.95)
                    np.testing.assert_allclose(right_score(model,x*100),scores,atol=1e-8)
                    path = Path(tmp)/f'{name}.joblib'
                    joblib.dump(model,path)
                    np.testing.assert_array_equal(right_score(joblib.load(path),x),scores)

    def test_metric_validation_rejects_invalid_labels_shapes_and_infinite_scores(self):
        for pred,score in [([0,2],[0,1]),([1],[0,1]),([1,2],[0,np.inf])]:
            with self.assertRaises(ValueError):
                metrics(np.array([1,2]),np.array(pred),np.array(score))

    def test_saved_metrics_detect_corruption_and_missing_rows(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame(dict(model=['m']*4,subject=[1]*4,run=[0]*4,
                trial=[0,1,2,3],label=[1,2,1,2],prediction=[1,2,1,2],score_right=[-2,2,-1,1]))
            frame.to_csv(root/'predictions.csv',index=False)
            row = dict(model='m',subject=1,**metrics(frame.label,frame.prediction,frame.score_right))
            pd.DataFrame([row]).to_csv(root/'participant_metrics.csv',index=False)
            self.assertEqual(verify_run(root)['status'],'passed')
            row['auc'] = .2
            pd.DataFrame([row]).to_csv(root/'participant_metrics.csv',index=False)
            with self.assertRaises(AssertionError):
                verify_run(root)

    def test_confirmatory_exposure_gate(self):
        from calibrationfreebci.external import evaluate
        with TemporaryDirectory() as tmp:
            record = Path(tmp)/'exposure.json'
            record.write_text(json.dumps(dict(dataset='PhysioNet EEGMMIDB 1.0.0',subjects=[1],owner_confirmed=False,previously_used=False)))
            with self.assertRaisesRegex(ValueError,'Exposure record'):
                evaluate(tmp,Path(tmp)/'out',[1],exposure_record=record)
            self.assertFalse((Path(tmp)/'out').exists())

    def test_provisional_approval_does_not_establish_confirmatory_status(self):
        from calibrationfreebci.external import validate_exposure
        record=dict(dataset='PhysioNet EEGMMIDB 1.0.0',subjects=[1],
                    owner_confirmed=False,previously_used=None,evaluation_approved=True)
        validate_exposure(record,[1],'provisional')
        with self.assertRaisesRegex(ValueError,'Exposure record'):
            validate_exposure(record,[1],'confirmatory')
        with self.assertRaisesRegex(ValueError,'cohort mismatch'):
            validate_exposure(record,[2],'provisional')

    def test_provisional_evaluation_requires_recorded_approval(self):
        from calibrationfreebci.external import validate_exposure
        record=dict(dataset='PhysioNet EEGMMIDB 1.0.0',subjects=[1],
                    owner_confirmed=False,previously_used=None)
        with self.assertRaisesRegex(ValueError,'recorded approval'):
            validate_exposure(record,[1],'provisional')

    def test_cross_dataset_integration_on_synthetic_edf_reader(self):
        from calibrationfreebci.external import evaluate
        rng = np.random.default_rng(33)
        x = rng.normal(size=(24,22,100))*1e-6
        y = np.tile([1,2],12)
        x[y==1,0] *= 3
        x[y==2,1] *= 3
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = build('rms_csp_lda').fit(x,y)
            joblib.dump(model,root/'frozen_model.joblib')
            digest = sha256(root/'frozen_model.joblib')
            (root/'selection.json').write_text(json.dumps(dict(model='rms_csp_lda',model_sha256=digest,threshold=0)))
            (root/'comparison_manifest.json').write_text(json.dumps({'rms_csp_lda':dict(file='frozen_model.joblib',sha256=digest,threshold=0)}))
            exposure = root/'synthetic_exposure.json'
            exposure.write_text(json.dumps(dict(dataset='PhysioNet EEGMMIDB 1.0.0',subjects=[1],owner_confirmed=True,previously_used=False)))
            folder = root/'S001'
            folder.mkdir()
            for run in (4,8,12):
                (folder/f'S001R{run:02d}.edf').write_bytes(b'synthetic')
            raw = mne.io.RawArray(rng.normal(0,1e-6,(22,160*20)),mne.create_info(CHANNELS,160,'eeg'),verbose=False)
            raw.set_annotations(mne.Annotations([1,9],[4,4],['T1','T2']))
            with patch('mne.io.read_raw_edf',side_effect=lambda *a,**k: raw.copy()),patch('mne.datasets.eegbci.standardize'):
                evaluate(root,root/'external',[1],root,exposure)
            self.assertEqual(verify_run(root/'external')['prediction_rows'],6)
            self.assertTrue((root/'external/REPORT.md').is_file())
            self.assertEqual(len(pd.read_csv(root/'external/summary.csv')),4)
            (root/'frozen_model.joblib').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_comparisons(root)

    def test_common_reference_rank_survives_float32_roundtrip(self):
        rng=np.random.default_rng(71)
        x=rng.normal(size=(24,5,6,100))
        x-=x.mean(axis=2,keepdims=True)
        y=np.tile([1,2],12)
        model=build_bank('rms_csp4').fit(x,y)
        expected=right_score(model,x)
        np.testing.assert_allclose(right_score(model,x.astype(np.float32)),expected,atol=1e-5)

    def test_participant_bootstrap_is_reproducible(self):
        from calibrationfreebci.audit import bootstrap_mean
        self.assertEqual(bootstrap_mean([.5,.6,.8]),bootstrap_mean([.5,.6,.8]))
        for values in ([],[np.nan]):
            with self.assertRaises(ValueError):
                bootstrap_mean(values)


if __name__ == '__main__':
    unittest.main()
