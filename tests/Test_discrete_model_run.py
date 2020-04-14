#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:

import inspect
import sys 
import unittest
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.integrate import quad
from scipy.interpolate import interp1d 
from scipy.special import factorial
from scipy.integrate import solve_ivp,fixed_quad
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function, ones,var
from typing import Callable,Iterable,Union,Optional,List,Tuple 
from copy import copy
    
import CompartmentalSystems.example_smooth_reservoir_models as ESRM
import CompartmentalSystems.example_pwc_model_runs as ESMR

from testinfrastructure.InDirTest import InDirTest
#from testinfrastructure.helpers import pe
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.pwc_model_run import PWCModelRun 
from CompartmentalSystems.discrete_model_run import DiscreteModelRun
from CompartmentalSystems.helpers_reservoir import numerical_function_from_expression,numerical_rhs


class TestDiscreteModelRun(InDirTest):
    def test_from_PWCModelRun(self):
        x_0,x_1,t,k,u = symbols("x_1,x_2,k,t,u")
        inputs={
             0:u
            ,1:u
        }
        outputs={
             0:-x_0*k
            ,1:-x_1**3*k
        }
        internal_fluxes={}
        srm=SmoothReservoirModel([x_0,x_1],t,inputs,outputs,internal_fluxes)
        t_max=.5
        times = np.linspace(0, t_max, 11)
        x0=np.float(10)
        start_values = np.array([x0,x0])
        parameter_dict = {
             k: -1
            ,u:1}
        delta_t=np.float(1)
        
        pwc_mr = PWCModelRun(srm, parameter_dict, start_values, times)

        
        dmr = DiscreteModelRun.from_PWCModelRun(pwc_mr)
        pwc_mrs,_=pwc_mr.solve()
        dmrs=dmr.solve()
        self.assertTrue(np.allclose(dmrs,pwc_mrs))
        
        fig=plt.figure(figsize=(7,7))
        for i in [0,1]:
            ax=fig.add_subplot(2,1,1+i)
            plt.title("solutions")
            ax.plot(times,pwc_mrs[:,i],'*',color='red',label="pwc_mr"+str(i),markersize=12)
            ax.plot(times,dmrs[:,i],'*',color='blue',label="dmr"+str(i),markersize=8)
            ax.legend()
        fig.savefig("pool_contents.pdf")
        self.assertTrue(True)


    def test_start_value_format(self):

        ## create ReservoirModel
        C_1, C_2, C_3 = symbols('C_1 C_2 C_3')
        state_vector = Matrix(3, 1, [C_1, C_2, C_3]) 
        t = symbols('t')
        B = Matrix([[-2, 0, 1], [2, -2, 0], [0, 2, -2]])
        u = Matrix(3, 1, [1, 0, 0])
    
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)
    
        ## create ModelRun
        ss = (-B**(-1)*u)
        #start_values = np.array(ss).astype(np.float64).reshape((3,))
        start_values = np.array(ss).astype(np.float64)
        times = np.linspace(1919, 2009, 901)
        parameter_dict = {}
        pwc_mr = PWCModelRun(srm, parameter_dict, start_values, times)
        pwc_mr.initialize_state_transition_operator_cache(
            lru_maxsize=None
        )

        dmr = DiscreteModelRun.from_PWCModelRun(pwc_mr)

        


