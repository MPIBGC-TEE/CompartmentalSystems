import unittest
import numpy as np
import matplotlib
from collections import  OrderedDict
from CompartmentalSystems.ArrayDict import ArrayDict
import CompartmentalSystems.helpers_reservoir as hr
from testinfrastructure.InDirTest import InDirTest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CompartmentalSystems.BlockArrayIterator import BlockArrayIterator


class TestBlockArrayIterator(InDirTest):
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
        bit = BlockArrayIterator(
            iteration_str = "it",
            start_seed_dict=ArrayDict({"x": x_0}),
            present_step_funcs=OrderedDict({
                # these are functions that are  applied in order
                # on the start_seed_dict
                # they might compute variables that are purely  
                # diagnostic or those that are necessary for the
                # next step
                "t": lambda it: t_0 + delta_t*it, #time
                "B": lambda it, x: B_0, # constant linear nonautonmous
                "u": lambda it, x: u_0, # constant
            }),
            next_step_funcs=OrderedDict({
                # these functions have to compute the seed for the next timestep
                "x": lambda x,B,u : B@x +u, 
            })
        )
        first=bit[0]
        self.assertTrue(isinstance(first,ArrayDict))
        self.assertEqual(set(first.keys()),{"it","x","t","B","u"})
        self.assertTrue(np.all(first['t'] == t_0))
        self.assertTrue(np.all(first['x'] == x_0))
       
        # for a slice we now get an ArrayDict (Dictionary of Arrays back)
        ft = bit[0:10]
        self.assertTrue(isinstance(ft, ArrayDict))
        self.assertTrue(np.all(ft['x'][0] == x_0))
        self.assertTrue(np.all(ft['t'][0] == t_0))
        # we can also access the keys as attributes (for compatibility) 
        self.assertTrue(np.all(ft.x[0] == x_0))
        self.assertTrue(np.all(ft.t[0] == t_0))


    def test_averaged_values(self):
        u_0 = np.array([1, 1])
        x_0 = np.array([0, 0])
        t_0 = 0
        delta_t = 2
        bit = BlockArrayIterator(
            iteration_str = "it",
            start_seed_dict=ArrayDict({"x": x_0}),
            present_step_funcs=OrderedDict({
                # these are functions that are  applied in order
                # on the start_seed_dict
                # they might compute variables that are purely  
                # diagnostic or those that are necessary for the
                # next step
                "t": lambda it: t_0 + delta_t*it, #time
                "u": lambda it, x: u_0, # constant
            }),
            next_step_funcs=OrderedDict({
                # these functions have to compute the seed for the next timestep
                "x": lambda x,u : x +u, 
            })
        )
        n = 10
        step = 2
        # example with average every 2 values and n a multiple of step
        parts = hr.partitions(0, n, step)
        self.assertEqual(
            parts,
            [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
        )
        vals= bit[0: n]
        # just for readability we show vals.x 
        ref_vals_x= np.array(
           [
               [0, 0],
               [1, 1],
               [2, 2],
               [3, 3],
               [4, 4],
               [5, 5],
               [6, 6],
               [7, 7],
               [8, 8],
               [9, 9]
           ]
        )
        self.assertTrue((vals.x == ref_vals_x).all())

        av1 = vals.averages(parts) # using the averaging method of the results
        av2 = bit.averaged_values(parts) # using the iterator without storing
        # the values but only the averages 
        ref_x = np.array(
            [
                [0.5, 0.5],
                [2.5, 2.5],
                [4.5, 4.5],
                [6.5, 6.5],
                [8.5, 8.5]
            ]
        )

        self.assertTrue((av1.x==ref_x).all())
        self.assertTrue((av2.x==ref_x).all())

        ref_t = np.array([ 1.,  5.,  9., 13., 17.])
        self.assertTrue((av1.t==ref_t).all())
        self.assertTrue((av2.t==ref_t).all())

        # example with average every 3 values and n NOT a multiple of step
        # the first two averages are over 4 the only over 2 elements
        n = 10
        step = 4
        parts = hr.partitions(0, n, step)
        self.assertEqual(
            parts,
            [(0, 4), (4, 8), (8, 10)] 
        )
        av1 = vals.averages(parts) # using the averaging method of the results
        av2 = bit.averaged_values(parts) # using the iterator without storing
        # the values but only the averages 
        #from IPython import embed; embed()
        ref_x = np.array(
            [
                [1.5, 1.5],
                [5.5, 5.5],
                [8.5, 8.5]
            ]
        )

        self.assertTrue((av1.x==ref_x).all())
        self.assertTrue((av2.x==ref_x).all())

        ref_t = np.array([3, 11, 17])
        first = next(bit)
        second = next(bit)
        self.assertTrue((av1.t==ref_t).all())
        self.assertTrue((av2.t==ref_t).all())
    
    #def test_with_average_iterator(self):
    #    u_0 = np.array([1, 1])
    #    x_0 = np.array([0, 0])
    #    t_0 = 0
    #    delta_t = 2
    #    bit = BlockArrayIterator(
    #        iteration_str = "it",
    #        start_seed_dict=ArrayDict({"x": x_0}),
    #        present_step_funcs=OrderedDict({
    #            # these are functions that are  applied in order
    #            # on the start_seed_dict
    #            # they might compute variables that are purely  
    #            # diagnostic or those that are necessary for the
    #            # next step
    #            "t": lambda it: t_0 + delta_t*it, #time
    #            "u": lambda it, x: u_0, # constant
    #        }),
    #        next_step_funcs=OrderedDict({
    #            # these functions have to compute the seed for the next timestep
    #            "x": lambda x,u : x +u, 
    #        })
    #    )
    #    n = 10
    #    step = 2
    #    # example with average every 2 values and n a multiple of step
    #    parts = hr.partitions(0, n, step)
    #    self.assertEqual(
    #        parts,
    #        [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    #    )
    #    vals= bit[0: n]
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
