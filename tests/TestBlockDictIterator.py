import unittest
import numpy as np
import matplotlib
from collections import  OrderedDict

from testinfrastructure.InDirTest import InDirTest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CompartmentalSystems.BlockDictIterator import BlockDictIterator


class TestBlockDictIterator(InDirTest):
    def test_solution(self):
        B_0 = np.array(
            [
                [ -1, 0.5],
                [0.5,  -1]
            ]
        )
        u_0 = np.array([1, 1])
        x_0 = np.array([0, 0])
        t_0 = 0
        delta_t = 2
        sd = {"x": x_0}
        bit = BlockDictIterator(
            iteration_str = "it",
            start_seed_dict=sd,
            present_step_funcs=OrderedDict({
                # these are functions that are  applied in order
                # on the start_seed_dict
                # they might compute variables that are purely  
                # diagnostic or those that are necessary for the
                # next step values
                "t": lambda it: t_0 + delta_t*it, #time
                "B": lambda it, x: B_0, # constant linear nonautonmous
                "u": lambda it, x: u_0, # constant
            }),
            next_step_funcs=OrderedDict({
                # these functions have to compute the seed for the next timestep
                "x": lambda x,B,u : B*x +u, 
            })
        )
        first=bit[0]
        self.assertTrue(isinstance(first,type(sd)))
        self.assertEqual(set(first.keys()),{"it","x","t","B","u"})
        self.assertTrue(np.all(first['t'] == t_0))
        self.assertTrue(np.all(first['x'] == x_0))
       
        # for a slice we get a tuple back
        first_ten = bit[0:10]
        self.assertTrue(isinstance(first_ten,tuple))
        self.assertTrue(np.all(first_ten[0]['x'] == x_0))
        self.assertTrue(np.all(first_ten[0]['t'] == t_0))


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
