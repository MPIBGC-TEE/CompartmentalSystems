import unittest
import numpy as np
import matplotlib
from collections import  OrderedDict

from testinfrastructure.InDirTest import InDirTest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from CompartmentalSystems.InfiniteIterator import InfiniteIterator



class TestInfiniteIterator(InDirTest):

    def test__getitem__(self):
        v_0 = 1

        def f(i, n):
            return n+1

        itr = InfiniteIterator(start_value=v_0, func=f)
        

        # it should behave like the [] on a tuple 
        # integer index
        self.assertTrue(itr[0] == v_0)
        # 
        # (1,2,3)[0:0] = ()
        self.assertTrue(itr[0:0] == ())
        
        # (1,2,3)[0:1] = (1,) ## tuple
        self.assertTrue(itr[0: 1] == (v_0,))

        # (1,2,3)[1:3] = (2,3)
        self.assertTrue(itr[1: 3] == (2, 3))


    def test_dict(self):
        # the iterated varialbe can be of any type 
        # for some applications we use dictionaries
        I = (np.array([1, 1]).reshape(2, 1),)
        k = 0.5
        v_0={"X":np.array([1, 1]).reshape(2, 1)}

        def f(i, d):
            X=d["X"]
            return {"X": X + (I - k * X)}

        itr = InfiniteIterator(start_value=v_0, func=f)


        # first we get a single value
        self.assertTrue(np.all(itr[0] == v_0))
        # of 2,1 arrays
        
        # make sure that [] has no side effects
        result_1 = itr[0]
        result_2 = itr[0]
        self.assertTrue(result_1 == result_2)

        # tuples of dicts
        results_1 = itr[0:10]
        self.assertTrue(isinstance(results_1, tuple))
        
        # make sure that [] has no side effects
        results_2 = itr[0:10]
        #from IPython import embed; embed()
        self.assertTrue(
            [np.all(results_1[i]['X'] == results_2[i]['X']) for i in range(len(results_1))]
        )

    def test_numeric(self):
        # the iterated varialbe can be of any type 
        # e.g. an array
        I = np.array([1, 1]).reshape(2, 1)
        k = 0.5

        X_0 = np.array([1, 1]).reshape(2, 1)

        def f(i, X):
            return X + (I - k * X)

        itr = InfiniteIterator(start_value=X_0, func=f)

        # make sure that [] has no side effects

        # first we get a single value
        self.assertTrue(np.all(itr[0] == X_0))
        # the results will be tuples of length 1
        # of 2,1 arrays
        result_1 = itr[0]
        result_2 = itr[0]
        self.assertTrue(np.all(result_1 == result_2))

        # we should also get the same result with the following slice 0:1
        # but as a tuple
        self.assertTrue(isinstance(itr[0:1],tuple))
        self.assertTrue(np.all(itr[0] == itr[0:1][0]))


        # tuples of arrays
        results_1 = itr[0:10]
        results_2 = itr[0:10]
        #from IPython import embed; embed()
        self.assertTrue(np.all(np.stack(results_1) == np.stack(results_2)))

    
