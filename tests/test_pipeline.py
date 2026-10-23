import unittest
import numpy as np
import mne
from unittest.mock import patch
from calibrationfreebci.data import CHANNELS, epoch_raw
from calibrationfreebci.models import build, right_score
from calibrationfreebci.experiment import CANDIDATES, metrics


class PipelineTests(unittest.TestCase):
    def test_external_loader_task_onsets_channels_and_units(self):
        from calibrationfreebci.external import load_physionet
        from tempfile import TemporaryDirectory
        from pathlib import Path
        rng = np.random.default_rng(22)
        raw = mne.io.RawArray(rng.normal(0, 1e-6, (22, 160*20)),
                             mne.create_info(CHANNELS[::-1], 160, 'eeg'), verbose=False)
        raw.set_annotations(mne.Annotations([1, 9], [4,4], ['T1','T2']))
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)/'S001'
            folder.mkdir()
            for run in (4,8,12):
                (folder/f'S001R{run:02d}.edf').write_bytes(b'synthetic fixture')
            with patch('mne.io.read_raw_edf', side_effect=lambda *a,**k: raw.copy()), \
                 patch('mne.datasets.eegbci.standardize'):
                x, frame, provenance = load_physionet([1], tmp)
        self.assertEqual(x.shape, (6,22,501))
        np.testing.assert_array_equal(frame.label, [1,2,1,2,1,2])
        self.assertLess(np.max(np.abs(x)), 1e-4)
        self.assertEqual(len(provenance), 3)

    def test_trial_start_is_not_imagery_cue(self):
        raw = mne.io.RawArray(np.random.default_rng(9).normal(0, 1e-6, (22, 4000)),
                             mne.create_info(CHANNELS, 250, 'eeg'), verbose=False)
        epochs = epoch_raw(raw, np.array([[250, 0, 1], [2000, 0, 2]]))
        self.assertEqual(epochs.tmin, 2.5)
        self.assertEqual(epochs.tmax, 4.5)
        self.assertEqual(epochs.get_data().shape, (2, 22, 501))
        np.testing.assert_array_equal(epochs.events[:, 0], [250, 2000])

    def test_models_scores_predictions_and_gain(self):
        rng = np.random.default_rng(11)
        y = np.tile([1, 2], 20)
        x = rng.normal(size=(40, 6, 100))
        x[y == 1, 0] *= 3
        x[y == 2, 1] *= 3
        for name in CANDIDATES:
            with self.subTest(model=name):
                model = build(name).fit(x, y)
                scores = right_score(model, x)
                pred = np.where(scores > 0, 2, 1)
                np.testing.assert_array_equal(pred, model.predict(x))
                self.assertGreater(metrics(y, pred, scores)['balanced_accuracy'], .9)
                if name in ('rms_csp_lda','trace_oas_mdm','trace_tangent_lda'):
                    np.testing.assert_allclose(right_score(model, x*10), scores, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
