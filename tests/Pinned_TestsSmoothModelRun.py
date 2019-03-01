
#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:

from concurrencytest import ConcurrentTestSuite, fork_for_tests
import unittest
import sys 
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d 
from scipy.misc import factorial
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function

import example_smooth_reservoir_models as ESRM
import example_smooth_model_runs as ESMR

from testinfrastructure.InDirTest import InDirTest
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.smooth_model_run import SmoothModelRun 


class Pinned_TestSmoothModelRun(InDirTest):
    def test_solve_semi_symbolic_piecewise(self):
        # test semi-symbolic semi-numerical SmoothReservoirModel
        C_0, C_1, C_2 = symbols('C_0 C_1 C_2')
        lambda_0 = Symbol('lambda_0')
        t = Symbol('t')
        u_0_expr = Function('u_0')(C_0, C_1, t)
        u_2_expr = Function('u_2')(t)

        X = Matrix([C_0, C_1, C_2])
        t_min,t_max = 0,10
        u_data_0 = np.array([[ t_min ,  0.1], [ t_max ,  0.2]])
        u_data_2 = np.array([[ t_min ,  0.4], [ t_max ,  0.5]])
        input_fluxes = {0: u_data_0, 2: u_data_2}

        symbolic_input_fluxes = {0: u_0_expr, 2: u_2_expr}
        
        u_0_interp = interp1d(u_data_0[:,0], u_data_0[:,1])
        def u0_func(C_0_val, C_1_val, t_val):
            return C_0_val + C_1_val + u_0_interp(t_val)
        
        u_1_interp = interp1d(u_data_2[:,0], u_data_2[:,1])
        def u2_func(t_val):
            return u_1_interp(t_val)

        func_set = {u_0_expr: u0_func, u_2_expr: u2_func}
        
        output_fluxes = {0:Piecewise((lambda_0*C_0,t<t_max/2),(10*lambda_0*C_0,True))}
        internal_fluxes = {(0,1): 5*C_0, (1,0): 4*C_1**2}
        srm = SmoothReservoirModel(
            X, 
            t, 
            symbolic_input_fluxes, 
            #{0: 1, 2: 1},
            output_fluxes, 
            internal_fluxes
        )

        start_values = np.array([1, 2, 3])
        times = np.linspace(t_min,t_max, 11)
        smr = SmoothModelRun(srm, parameter_dict={lambda_0:.2}, start_values=start_values, times=times,func_set=func_set)
        
        soln = smr.solve()

    def test_linearize_piecewise(self):
        # Atmosphere, Terrestrial Carbon and Surface layer
        C_A, C_T, C_S = symbols('C_A C_T C_S')
        
        # equilibrium contents
        A_e, T_e, S_e = symbols('A_e T_e S_e')
        
        # equilibrium fluxes
        F_0, F_1, F_2 = symbols('F_0 F_1 F_2')
        
        # nonlinear coefficients
        alpha, beta = symbols('alpha beta')
        
        # external flux from surface layer to deep ocean
        F_ex = F_0*C_S/S_e
        
        # fossil fuel inputs
        u_A = symbols('u_A')
        
        
        #########################################
        
        state_vector = Matrix([C_A, C_T, C_S])
        time_symbol = symbols('tau')
        
        input_fluxes = {0: u_A, 1: 0, 2: F_0}
        output_fluxes = {0: Piecewise((1, time_symbol < 0), (0, True)), 1: 0, 2: F_0*C_S/S_e}
        internal_fluxes = {(0,1): F_2*(C_A/A_e)**alpha, # A --> T
                           (0,2): F_1*C_A/A_e,          # A --> S
                           (1,0): F_2*C_T/T_e,          # T --> A
                           (2,0): F_1*(C_S/S_e)**beta}  # S --> A
        nonlinear_srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)
        
        A_eq, T_eq, S_eq = (700.0, 3000.0, 1000.0) 
        par_dict = {  A_e:  A_eq,  T_e:   T_eq, S_e: S_eq, # equilibrium contents in Pg
                      F_0:  45.0,  F_1:  100.0, F_2: 60.0, # equilibrium fluxes in PgC/yr
                    alpha:   0.2, beta:   10.0           } # nonlinear coefficients
        
        
        # fossil fuel inputs
        par_dict[u_A] = 0
        
        # initialize model run 
        times = np.linspace(0, 10, 101)
        start_values = np.array([A_eq, T_eq, S_eq])
        nonlinear_smr = SmoothModelRun(nonlinear_srm, par_dict, start_values, times)
       
        linearized_smr = nonlinear_smr.linearize()
        #print(linearized_srm.F)
        soln = linearized_smr.solve()
        # system is in steady state, so the linearized solution
        # should stay constant
        self.assertTrue(np.allclose(soln[-1], start_values))


###############################################################################

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSmoothModelRunFail)
#    # Run same tests across 16 processes
#    concurrent_suite = ConcurrentTestSuite(suite, fork_for_tests(16))
#    concurrent_suite = ConcurrentTestSuite(suite, fork_for_tests(1))
#    runner = unittest.TextTestRunner()
#    res=runner.run(concurrent_suite)
#    # to let the buildbot fail we set the exit value !=0 if either a failure or error occurs
#    if (len(res.errors)+len(res.failures))>0:
#        sys.exit(1)

    unittest.main()
