"""Small CPU invariance checks; no formal training or modified shared caches."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import numpy as np
import torch
from scipy import sparse
from networks import DeepPhylo
from networks import PopPhyCNN
from data import FoldDataset
from tree import path_distance

class Checks(unittest.TestCase):
    def test_original_edge_distances(self):
        # Unequal depths and retained unary ancestors; IDs have no metric role.
        a = [900,80,70,2,1]; b = [3,70,2,1]; c = [70000,1]
        self.assertEqual(path_distance(a,b),3)
        self.assertEqual(path_distance(a,a),0)
        self.assertEqual(path_distance(a,c),5)
        self.assertEqual(path_distance(a,b),path_distance(b,a))

    def test_presence_is_raw_and_denominator_not_reclosed(self):
        counts = sparse.csr_matrix([[2.,0.,1.],[0.,3.,0.]])
        ds = FoldDataset(counts,np.array([10.,20.]),np.array([0,1]),np.array([0,1]),
                         np.array([.2,0.,0.]),np.ones(3),np.zeros(3,dtype=bool),None,'deepphylo_taxonomy')
        x,p,y = ds[0]
        self.assertEqual(x[0].item(),0.)
        self.assertTrue(p[0]); self.assertFalse(p[1])
        self.assertAlmostEqual(x[2].item(),.1)

    def test_popphy_selected_descendant_sums_before_log(self):
        counts = sparse.csr_matrix([[2.,1.]])
        incidence = sparse.csr_matrix([[1,1,0],[1,0,1]])
        ds = FoldDataset(counts,np.array([10.]),np.array([0]),np.array([0]),np.zeros(3),np.ones(3),np.zeros(3,dtype=bool),incidence,'popphy_cnn')
        x,_,_ = ds[0]
        np.testing.assert_allclose(x.numpy(),np.log1p([.3,.2,.1]),rtol=1e-6)

    def test_deepphylo_padding_batch_invariance(self):
        torch.manual_seed(0)
        model = DeepPhylo(np.random.default_rng(0).normal(size=(9,4)),3).eval()
        abundance = torch.randn(3,9)
        present = torch.tensor([[1,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0]],dtype=torch.bool)
        with torch.no_grad():
            batch = model(abundance,present)
            for i in range(3):
                torch.testing.assert_close(batch[i],model(abundance[i:i+1],present[i:i+1])[0],atol=1e-6,rtol=1e-5)
        torch.nn.functional.cross_entropy(model(abundance,present),torch.tensor([0,1,2])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_popphy_multiclass_gradient(self):
        model = PopPhyCNN((10,30),4)
        logits = model(torch.zeros(2,1,10,30))
        self.assertEqual(logits.shape,(2,4))
        torch.nn.functional.cross_entropy(logits,torch.tensor([0,3])).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
