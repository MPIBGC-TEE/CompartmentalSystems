from unittest import TestCase
import numpy as np
from collections import  OrderedDict
from CompartmentalSystems.InfiniteIterator import InfiniteIterator
from CompartmentalSystems.IteratorResult import IteratorResult

class TestIteratorResult(TestCase):
    def test__getitem__(self):
        v_0 = 0

        def f(i, n):
            return n+1

        itr = InfiniteIterator(start_value=v_0, func=f)
        

        ir = IteratorResult(itr)
        # it should behave like the [] on a tuple 
        # integer index
        self.assertTrue(ir[0] == v_0)
        # 
        self.assertTrue(ir[1] == v_0+1)

        # (0,1,2,3)[0:0] = ()
        self.assertTrue(ir[0:0] == ())
        
        # (0,1,2,3)[0:1] = (0,) ## tuple
        self.assertTrue(ir[0: 1] == (v_0,))

        # (0,1,2,3)[1:3] = (1,2)
        self.assertTrue(ir[1: 3] == (1, 2))

        # if the iterator does not have max_iter this 
        # we want ir to throw an exception for undefined upper limits like [:]
        with self.assertRaises(IndexError) as cm:
            ir[:]    
        exc = cm.exception
        print(exc)
        # maybe later
