
import unittest

from testinfrastructure.InDirTest import InDirTest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CompartmentalSystems.BlockRhs import BlockRhs
class TestBlockRhs(InDirTest):
    def test_flat_rhs(self):
        x1 = np.ones((5, 5))
        x2 = np.ones((2,))
        brhs = BlockRhs(
            time_str="t",
            func_tups=[
                ((lambda x1: -x1), ["x1"]),
                ((lambda t, x2: -2 * t * x2), ["t", "x2"]),
            ],
        )    
        start_blocks = [("x1", x1), ("x2", x2)]
        flat_start_vec = np.concatenate([a.flatten() for (_,a) in start_blocks])
        fun = brhs.flat_rhs(
            block_shapes=[("x1", x1.shape), ("x2", x2.shape)]
        )
        #from IPython import embed; embed()
        res_flat = fun(0,flat_start_vec)
        ref_flat = np.concatenate([-1 * x1.flatten(), np.zeros_like(x2).flatten()])

