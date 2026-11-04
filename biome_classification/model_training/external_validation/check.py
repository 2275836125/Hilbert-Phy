"""Numerical contract checks; no fitting or checkpoint loading."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from foundation import ExternalData, classification_metrics, label_indices


class Checks(unittest.TestCase):
    def test_label_policy(self):
        y = label_indices(['A', 'NA', ' none ', '', ' A', 'unknown', 'B'], {'0': 'A', '1': 'B'})
        np.testing.assert_array_equal(y, [0, -1, -1, -1, -2, -2, 1])

    def test_absent_class_metrics(self):
        result = classification_metrics(np.array([0, 1]), np.array([[.8, .1, .1], [.1, .8, .1]]),
                                        {'0': 'A', '1': 'B', '2': 'C'})
        self.assertEqual(result['macro_f1_all_training_classes'], 2 / 3)
        self.assertEqual(result['macro_f1_present_classes'], 1.)
        self.assertEqual(result['macro_auroc_valid_classes'], 1.)
        self.assertEqual(result['auroc_valid_classes'], 2)
        self.assertIsNone(result['per_class'][2]['auroc'])

    def test_single_class_auc_undefined(self):
        result = classification_metrics(np.array([0, 0]), np.array([[.8, .2], [.7, .3]]), {'0': 'A', '1': 'B'})
        self.assertIsNone(result['macro_auroc_valid_classes'])
        self.assertEqual(result['auroc_valid_classes'], 0)

    def test_invalid_probabilities_rejected(self):
        for p in (np.array([[.2, .2]]), np.array([[np.nan, .5]]), np.array([[-.1, 1.1]])):
            with self.assertRaises(ValueError):
                classification_metrics(np.array([0]), p, {'0': 'A', '1': 'B'})

    def test_normalization_full_denominator_no_clip_raw_presence(self):
        data = ExternalData.__new__(ExternalData)
        data.counts = sparse.csr_matrix([[2, 0, 3], [8, 2, 0]])
        data.ids, data.tax_ids = np.array(['a', 'b']), np.array([1, 2, 3])
        data.denominators = np.array([10., 20.])
        with tempfile.TemporaryDirectory() as tmp:
            norm = Path(tmp) / 'norm.npz'
            np.savez(norm, minimum=[.2, .05, 0], maximum=[.3, .1, 0], constant=[False, False, True], tax_ids=data.tax_ids)
            batches = list(data.batches([0, 1], norm, batch_size=1))
            np.testing.assert_allclose(batches[0][1], [[0, -1, 0]])
            np.testing.assert_allclose(batches[1][1], [[2, 1, 0]])
            np.testing.assert_array_equal(batches[0][2], [[True, False, True]])
            reordered = list(data.batches([0], norm, order=[2, 0, 1]))[0]
            np.testing.assert_allclose(reordered[1], [[0, 0, -1]])

    def test_popphy_sum_before_log(self):
        data = ExternalData.__new__(ExternalData)
        data.counts = sparse.csr_matrix([[2, 1]])
        data.ids, data.tax_ids, data.denominators = np.array(['a']), np.array([1, 2]), np.array([10.])
        incidence = sparse.csr_matrix([[1, 1, 0], [1, 0, 1]])
        with tempfile.TemporaryDirectory() as tmp:
            norm = Path(tmp) / 'norm.npz'
            np.savez(norm, minimum=np.zeros(3), maximum=np.ones(3), scale=np.ones(3), constant=np.zeros(3, dtype=bool))
            _, x, present = next(data.batches([0], norm, method='popphy_cnn', incidence=incidence))
            np.testing.assert_allclose(x, np.log1p([[.3, .2, .1]]), rtol=1e-6)
            np.testing.assert_array_equal(present, [[True, True]])


if __name__ == '__main__':
    unittest.main()
