import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import joblib
import numpy as np
from calibrationfreebci.alignment import AlignedCSP,align_participant
from calibrationfreebci.models import right_score


class AlignmentTests(unittest.TestCase):
    def test_alignment_is_label_free_and_gain_invariant(self):
        self.assertEqual(list(inspect.signature(align_participant).parameters),['x'])
        x=np.random.default_rng(3).normal(size=(12,6,100))
        np.testing.assert_allclose(align_participant(x),align_participant(x*30),atol=1e-12)
        order=np.arange(len(x))[::-1]
        np.testing.assert_allclose(align_participant(x)[order],align_participant(x[order]),atol=1e-12)

    def test_participant_batches_do_not_mix_and_models_reload(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(32,6,100))
        y=np.tile([1,2],16)
        groups=np.repeat([1,2],16)
        x[y==1,0]*=3
        x[y==2,1]*=3
        model=AlignedCSP().fit(x,y,groups)
        score=right_score(model,x,groups)
        changed=x.copy()
        changed[groups==2,2]*=10
        np.testing.assert_allclose(score[groups==1],right_score(model,changed,groups)[groups==1],atol=1e-12)
        np.testing.assert_allclose(score[groups==1],right_score(model,x[groups==1]),atol=1e-12)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'aligned.joblib'
            joblib.dump(model,path)
            np.testing.assert_array_equal(right_score(joblib.load(path),x,groups),score)
        with self.assertRaises(ValueError):
            AlignedCSP().fit(x,y)


if __name__=='__main__':
    unittest.main()
