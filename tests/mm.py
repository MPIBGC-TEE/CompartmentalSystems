from concurrencytest import ConcurrentTestSuite, fork_for_tests
import inspect
import matplotlib
import sys 
import unittest
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.integrate import quad
from scipy.interpolate import interp1d 
from scipy.misc import factorial
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function, ones
    
import example_smooth_reservoir_models as ESRM
import example_smooth_model_runs as ESMR

from testinfrastructure.InDirTest import InDirTest
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.smooth_model_run import SmoothModelRun 
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
output_fluxes = {0: 0, 1: 0, 2: F_0*C_S/S_e}
internal_fluxes = {(0,1): F_2*(C_A/A_e)**alpha, # A --> T
                   (0,2): F_1*C_A/A_e,          # A --> S
                   (1,0): F_2*C_T/T_e,          # T --> A
                   (2,0): F_1*(C_S/S_e)**beta}  # S --> A

nonlinear_srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

A_eq, T_eq, S_eq = (700.0, 3000.0, 1000.0) 
par_dict = {  A_e: A_eq,  T_e:  T_eq, S_e: S_eq, # equilibrium contents in Pg
              F_0: 45.0,  F_1: 100.0, F_2: 60.0, # equilibrium fluxes in PgC/yr
            alpha:  0.2, beta:  10.0           } # nonlinear coefficients

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
#self.assertTrue(np.allclose(soln[-1], start_values))
