# vim:set ff=unix expandtab ts=4 sw=4:
from concurrencytest import ConcurrentTestSuite, fork_for_tests
import sys
import unittest
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol,Matrix, symbols, sin, Piecewise, DiracDelta, Function
from CompartmentalSystems.helpers_reservoir import factor_out_from_matrix, parse_input_function, melt, MH_sampling, stride, is_compartmental, func_subs, numerical_function_from_expression
from CompartmentalSystems.start_distributions import start_age_moments_from_empty_spin_up,start_age_moments_from_steady_state,compute_fixedpoint_numerically
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel

class TestStartDistributions(unittest.TestCase):
    def test_numeric_staedy_state(self):
        # two-dimensional nonlinear 
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')
        input_fluxes = {0: 4, 1: 2}
        output_fluxes = {0: C_0**2, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        res_1=compute_fixedpoint_numerically(srm,t0=0,x0=np.array([1,1]),parameter_set={},func_set={},max_order=1)

    def test_compute_start_age_moments_from_steady_state(self):
        # two-dimensional linear autonomous
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        age_moment_vector=start_age_moments_from_steady_state(srm,t0=0,parameter_set={},func_set={},max_order=2)
        self.assertEqual(age_moment_vector.shape,(2,2))
        # we only check the #expectation values since B is the identity the number are the same as in the input fluxes 
        ref_ex=np.array([1,2]) 
        for pool in range(srm.nr_pools):
            self.assertTrue(np.allclose(age_moment_vector[:,pool], ref_ex))
        
        # two-dimensional linear non-autonomous
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')
        input_fluxes = {0: 1*(sin(t)+1), 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        age_moment_vector=start_age_moments_from_steady_state(srm,t0=0,parameter_set={},func_set={},max_order=2)
        self.assertEqual(age_moment_vector.shape,(2,2))
        # we only check the #expectation values since B is the identity the number are the same as in the input fluxes 
        ref_ex=np.array([1,2]) 
        for pool in range(srm.nr_pools):
            self.assertTrue(np.allclose(age_moment_vector[:,pool], ref_ex))
        
        # two-dimensional linear but state dependent input non-autonomous 
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')
        input_fluxes = {0: 0.5*C_0+sin(t)+1, 1: 2}
        output_fluxes = {0: 1.5*C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        age_moment_vector=start_age_moments_from_steady_state(srm,t0=0,parameter_set={},func_set={},max_order=2)
        self.assertEqual(age_moment_vector.shape,(2,2))
        # we only check the #expectation values since B is the identity the number are the same as in the input fluxes 
        ref_ex=np.array([1,2]) 
        for pool in range(srm.nr_pools):
            self.assertTrue(np.allclose(age_moment_vector[:,pool], ref_ex))




if __name__ == '__main__':
    suite=unittest.defaultTestLoader.discover(".",pattern=__file__)
    # Run same tests across 16 processes
    concurrent_suite = ConcurrentTestSuite(suite, fork_for_tests(16))
    runner = unittest.TextTestRunner()
    res=runner.run(concurrent_suite)
    # to let the buildbot fail we set the exit value !=0 if either a failure or 
    # error occurs
    if (len(res.errors)+len(res.failures))>0:
        sys.exit(1)

