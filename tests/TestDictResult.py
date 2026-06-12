from unittest import TestCase
import numpy as np
from collections import  OrderedDict
from CompartmentalSystems.DictResult import  DictResult
from CompartmentalSystems.BlockDictIterator import BlockDictIterator

class TestDictResult(TestCase):
        
    def test_solution_1(self):
        t_0 = 0
        delta_t = 1
        x_0 = 0
        sd = {"x": 0}
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
            }),
            next_step_funcs=OrderedDict({
                # these functions have to compute the seed for the next timestep
                "x": lambda x: x + 1, 
            })
        )
        
        dr = DictResult(bit)
        #from IPython import embed; embed()
        self.assertTrue(dr[0]['t'] == t_0)
        self.assertTrue(dr[0]['x'] == x_0)

        # the iterator should already have moved one step
        self.assertTrue(dr[0]['it'] == 1) 

        self.assertTrue(dr[1]['t'] == t_0 + delta_t)
        self.assertTrue(dr[1]['x'] == 1)
        # the iterator should have alreay moved two steps
        self.assertTrue(dr[1]['it'] == 2)
        
        self.assertTrue(dr[2]['t'] == t_0 + 2 * delta_t)
        self.assertTrue(dr[2]['x'] == 2)
        # the iterator should have alreay moved three steps
        self.assertTrue(dr[2]['it'] == 3)
        
        # for a slice we now get a dictionary of tuples back)
        ff = dr[0:5]
        print(ff)
        self.assertTrue(ff['it'] == (1, 2, 3, 4,5))
        self.assertTrue(ff['t'] == (0, 1, 2, 3, 4))
        self.assertTrue(ff['x'] == (0, 1, 2, 3, 4))
