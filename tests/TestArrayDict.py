from unittest import TestCase
import numpy as np
import matplotlib
from collections import OrderedDict
from CompartmentalSystems.ArrayDict import ArrayDict
import CompartmentalSystems.helpers_reservoir as hr


class TestArrayDict(TestCase):

    def test___getattribute__(self):
        ref_x = np.array([0, 1, 2, 3])
        ref_t = np.array([0, 1, 2, 3])
        ad = ArrayDict(
            {
                "t": ref_t,
                "x": ref_x
            }
        )
        self.assertTrue(np.allclose(ad.x, ref_x))

        self.assertTrue(np.allclose(ad['x'], ref_x))

    def test__fields(self):
        ref_x = np.array([0, 1, 2, 3])
        ref_t = np.array([0, 1, 2, 3])
        ad = ArrayDict(
            {
                "t": ref_t,
                "x": ref_x
            }
        )
        self.assertTrue(ad._fields == ad.keys())

    def test___add__(self):
        ref_x = np.array([0, 1, 2, 3])
        ref_t = np.array([0, 1, 2, 3])
        ad = ArrayDict(
            {
                "t": ref_t,
                "x": ref_x
            }
        )
        res = ad + ad
        self.assertTrue(np.all(res.x == ref_x + ref_x))
        self.assertTrue(np.all(res.t == ref_t + ref_t))

    def test___truediv__(self):
        ref_x = np.array([0, 1, 2, 3])
        ref_t = np.array([0, 1, 2, 3])
        ad = ArrayDict(
            {
                "t": ref_t,
                "x": ref_x
            }
        )
        res = ad / 2
        self.assertTrue(np.all(res.x == ref_x/2))
        self.assertTrue(np.all(res.t == ref_t/2))
    
    def test_averaged_values(self):
        ref_x = np.array([0, 1, 2, 3])
        ref_t = np.array([0, 1, 2, 3])
        ad = ArrayDict(
            {
                "t": ref_t,
                "x": ref_x
            }
        )
        res = ad.averaged_values([(0,2),(2,4)])
        self.assertTrue(np.all(res.x == np.array([0.5,2.5])))
