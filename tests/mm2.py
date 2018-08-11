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
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.smooth_model_run import SmoothModelRun 
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function, ones,lambdify
C_0, C_1, C_2 = symbols('C_0 C_1 C_2')
t = Symbol('t')

u_0_expr = Function('u_0')(C_0, C_1, t)
u_2_expr = Function('u_2')(t)

X = Matrix([C_0, C_1, C_2])
t_min, t_max = 0, 10
u_data_0 = np.array([[ t_min ,  0.1], [ t_max ,  0.2]])
u_data_2 = np.array([[ t_min ,  0.4], [ t_max ,  0.5]])
input_fluxes = {0: u_data_0, 2: u_data_2}
symbolic_input_fluxes = {0: u_0_expr, 2: u_2_expr}

u_0_interp = interp1d(u_data_0[:,0], u_data_0[:,1])
def u0_func(C_0_val, C_1_val, t_val):
    return C_0_val*0 + C_1_val*0 + u_0_interp(t_val)

u_1_interp = interp1d(u_data_2[:,0], u_data_2[:,1])
def u2_func(t_val):
    return u_1_interp(t_val)
parameter_set={}
func_set = {u_0_expr: u0_func, u_2_expr: u2_func}

output_fluxes = {}
internal_fluxes = {(0,1): 5*C_0, (1,0): 4*C_1**2}
srm = SmoothReservoirModel(
    X, 
    t, 
    symbolic_input_fluxes, 
    output_fluxes, 
    internal_fluxes
)

start_values = np.array([1, 2, 3])
times = np.linspace(t_min,t_max, 11)
smr = SmoothModelRun(srm, parameter_set=parameter_set, start_values=start_values, times=times,func_set=func_set)

soln = smr.solve()

# we want B and u as python functions 
u_sym=srm.external_inputs
str_func_set = {str(key): val for key, val in func_set.items()}
cut_func_set = {key[:key.index('(')]: val 
    for key, val in str_func_set.items()}

tup = tuple(X) + (t,)
u_par=u_sym.subs(parameter_set)
print(u_par)
u_func = lambdify(tup, u_par, modules=[cut_func_set, 'numpy'])
u_func(1,1,1,0)
