import unittest
import numpy as np
import pandas as pd
from calibrationfreebci.audit import recompute_predictions
from calibrationfreebci.models import TrialRMS, TraceNormalize


class IntegrityTests(unittest.TestCase):
    def test_trial_normalization_is_scale_invariant(self):
        x = np.random.default_rng(1).normal(size=(4, 3, 40))
        transform = TrialRMS()
        np.testing.assert_allclose(transform.transform(x), transform.transform(32 * x), atol=1e-12)

    def test_zero_trial_rejected(self):
        with self.assertRaises(ValueError):
            TrialRMS().transform(np.zeros((1, 3, 10)))

    def test_covariance_trace_invariant(self):
        x = np.array([np.diag([1., 2., 3.])])
        np.testing.assert_allclose(TraceNormalize().transform(x), TraceNormalize().transform(x * 1024))

    def test_rank_can_survive_constant_predictions(self):
        trials = pd.DataFrame(dict(subject=[7]*4, model=["mdm"]*4,
                                   imagery_class=["left", "left", "right", "right"],
                                   matlab_trial=[1, 2, 1, 2], true_label=[1, 1, 2, 2],
                                   predicted_label=[2]*4, score_right=[1., 2., 3., 4.]))
        result = recompute_predictions(trials).iloc[0]
        self.assertEqual(result.auc, 1.)
        self.assertEqual(result.balanced_accuracy, .5)
        with self.assertRaises(ValueError):
            recompute_predictions(pd.concat([trials, trials.iloc[:1]]))


if __name__ == "__main__":
    unittest.main()
