import unittest

from testinfrastructure.InDirTest import InDirTest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CompartmentalSystems.PiecewiseBlockIvp import PiecewiseBlockIvp
from CompartmentalSystems.BlockRhs import BlockRhs


class TestPiecewiseBlockIvps(InDirTest):

    def test_solution_one_piece(self):
        x1_shape = (5, 5)
        x2_shape = (2,)
        start_blocks =[("x1", np.ones(x1_shape)), ("x2", np.ones(x2_shape))]
        block_shapes = [(n, a.shape) for (n, a) in start_blocks] 
        time_str="t"
        pwbivps = PiecewiseBlockIvp(
            start_blocks=start_blocks,
            block_rhss=[
                BlockRhs(
                    time_str,
                    func_tups=[
                        ((lambda x1: -x1), ["x1"]),
                        ((lambda t, x2: -2 * t * x2), ["t", "x2"]),
                    ]
                )
            ]
        )
        # the reference solution
        t_max = 2
        ref = {
            "x1": np.exp(-t_max) * np.ones(x1_shape),
            "x2": np.exp(-(t_max ** 2)) * np.ones(x2_shape),
        }
        res = pwbivps.block_solve(t_span=(0, t_max))
        self.assertTrue(np.allclose(res["x1"][-1], ref["x1"], rtol=1e-2))
        self.assertTrue(np.allclose(res["x2"][-1], ref["x2"], rtol=1e-2))
        

        # try the function output
        f_dict = pwbivps.block_solve_functions(t_span=(0, t_max))
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


    def test_solution_two_pieces(self):
        # we build a system where with 2 discontinous righthandsides
        # one for the first interval from 0 to t_1
        # one for the second interval from t_1 to t_max

        x1_shape = (5, 5)
        x2_shape = (2,)
        t_1 = 1
        t_max = t_1*2
        start_blocks =[("x1", np.ones(x1_shape)), ("x2", np.ones(x2_shape))]
        block_shapes = [(n, a.shape) for (n, a) in start_blocks] 
        time_str="t"
        brhs = BlockRhs(
            time_str,
            func_tups=[
                ((lambda x1: -x1), ["x1"]),
                ((lambda t, x2: -2 * t * x2), ["t", "x2"]),
            ]
        )
        pwbivps = PiecewiseBlockIvp(
            start_blocks=start_blocks,
            block_rhss=[
                brhs,
                brhs
            ],
            disc_times=[t_1]
        )
        # the reference solution
        t_max = 2
        ref = {
            "x1": np.exp(-t_max) * np.ones(x1_shape),
            "x2": np.exp(-(t_max ** 2)) * np.ones(x2_shape),
        }
        res = pwbivps.block_solve(t_span=(0, t_max))
        # we check at the end
        self.assertTrue(np.allclose(res["x1"][-1], ref["x1"], rtol=1e-2))
        self.assertTrue(np.allclose(res["x2"][-1], ref["x2"], rtol=1e-2))

        # try the function output
        f_dict = pwbivps.block_solve_functions(t_span=(0, t_max))
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
