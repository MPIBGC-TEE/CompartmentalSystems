import unittest
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from testinfrastructure.InDirTest import InDirTest
from CompartmentalSystems.BlockIvp import BlockIvp
from CompartmentalSystems.BlockRhs import BlockRhs


class TestBlockIvp(InDirTest):
    def test_solution(self):
        x1_shape = (5, 5)
        x2_shape = (2,)
        start_blocks =[("x1", np.ones(x1_shape)), ("x2", np.ones(x2_shape))]
        block_shapes = [(n, a.shape) for (n, a) in start_blocks] 
        func_tups=[
            ((lambda x1: -x1), ["x1"]),
            ((lambda t, x2: -2 * t * x2), ["t", "x2"]),
        ]
        time_str="t"
        brsh = BlockRhs(
            time_str,
            func_tups,
        )
        bivp = BlockIvp(
            start_blocks=start_blocks,
            block_rhs=brsh
        )
        X_0 = bivp.start_vec
        rhs = bivp.rhs
        Xdot_0 = rhs(0,X_0)
        # the reference solution
        t_max = 2
        ref = {
            "x1": np.exp(-t_max) * np.ones(x1_shape),
            "x2": np.exp(-(t_max ** 2)) * np.ones(x2_shape),
        }
        res = bivp.block_solve(t_span=(0, t_max))
        self.assertTrue(np.allclose(res["x1"][-1], ref["x1"], rtol=1e-2))
        self.assertTrue(np.allclose(res["x2"][-1], ref["x2"], rtol=1e-2))
        
        # try the function output
        f_dict = bivp.block_solve_functions(t_span=(0, t_max))
        times=np.linspace(0,t_max,100)
        
        ref_x1_vals = np.stack(
            [
                val * np.ones(x1_shape) 
                for val in np.exp(-times)
            ]
        )
        self.assertTrue(
            np.allclose(
                f_dict['x1'](times),
                ref_x1_vals,
                atol=1e-2
            )    
        )    


        ref_x2_vals = np.stack(
           [
               val * np.ones(x2_shape) 
               for val in np.exp(-(times ** 2))
           ]
        )
        self.assertTrue(
            np.allclose(
                f_dict['x2'](times),
                ref_x2_vals,
                atol=1e-2
            )    
        )    


    def test_solution_no_explicit_time(self):
        # here we describe time by a variable with constant derivative
        # amd use it in the derivative of the second variable
        # to simulate a coupled system without feedback (skew product)
        x1_shape = (1,)
        x2_shape = (2,)
        start_blocks = [("x1", np.zeros(x1_shape)), ("x2", np.ones(x2_shape))]
        block_shapes = [(n, a.shape) for (n, a) in start_blocks] 
        func_tups=[
            ((lambda x1: np.ones(x1.shape)), ["x1"]),
            ((lambda x1, x2: -2 * x1 * x2), ["x1", "x2"]),
        ]
        time_str="t"
        brsh = BlockRhs(
            time_str,
            func_tups,
        )
        bivp = BlockIvp(
            start_blocks=start_blocks,
            block_rhs=brsh
        )
        # the reference solution
        t_max = 2
        ref = {
            "x1": t_max * np.ones(x1_shape),
            "x2": np.exp(-(t_max ** 2)) * np.ones(x2_shape),
        }
        res = bivp.block_solve(t_span=(0, t_max))
        self.assertTrue(np.allclose(res["x1"][-1], ref["x1"], rtol=1e-2))
        self.assertTrue(np.allclose(res["x2"][-1], ref["x2"], rtol=1e-2))


################################################################################


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(".", pattern=__file__)

    #    # Run same tests across 16 processes
    #    concurrent_suite = ConcurrentTestSuite(suite, fork_for_tests(1))
    #    runner = unittest.TextTestRunner()
    #    res=runner.run(concurrent_suite)
    #    # to let the buildbot fail we set the exit value !=0 if either a failure or error occurs
    #    if (len(res.errors)+len(res.failures))>0:
    #        sys.exit(1)

    unittest.main()
