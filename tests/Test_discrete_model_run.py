#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:

from concurrencytest import ConcurrentTestSuite, fork_for_tests
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
from scipy.misc import factorial
from scipy.integrate import solve_ivp,fixed_quad
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function, ones,var
from typing import Callable,Iterable,Union,Optional,List,Tuple 
from copy import copy
    
import example_smooth_reservoir_models as ESRM
import example_smooth_model_runs as ESMR

from testinfrastructure.InDirTest import InDirTest
from testinfrastructure.helpers import pe
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.smooth_model_run import SmoothModelRun 
from CompartmentalSystems.discrete_model_run import DiscreteModelRun
from CompartmentalSystems.helpers_reservoir import numerical_function_from_expression,numerical_rhs2


class TestDiscreteModelRun(InDirTest):
    def test_from_SmoothModelRun(self):
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
        
        smr = SmoothModelRun(srm, parameter_dict, start_values, times)
        
        # export the ingredients for an different ode solver 
        srm = smr.model
        state_vector, rhs = srm.age_moment_system(max_order=0)
        num_rhs = numerical_rhs2(
            state_vector,
            srm.time_symbol,
            rhs, 
            parameter_dict,
            {}
        )
        sf=solve_ivp(fun=num_rhs,t_span=[0,t_max],y0=start_values,t_eval=times)
        
        dmr = DiscreteModelRun.from_SmoothModelRun(smr)
        smrs=smr.solve()
        dmrs=dmr.solve()
        self.assertTrue(np.allclose(dmrs[:,1],sf.y[1,:]))
        self.assertTrue(np.allclose(dmrs[:,0],sf.y[0,:]))
        self.assertTrue(np.allclose(dmrs,smrs,rtol=1e-2,atol=1e-3))
        fig=plt.figure(figsize=(7,7))
        for i in [0,1]:
            ax=fig.add_subplot(2,1,1+i)
            plt.title("solutions")
            ax.plot(times,smrs[:,i],'*',color='red',label="smr"+str(i),markersize=12)
            ax.plot(times,dmrs[:,i],'*',color='blue',label="dmr"+str(i),markersize=8)
            n=len(sf.t)
            ax.plot(sf.t,sf.y[i].reshape(n,),'*',color='green',label="solve_ivp"+str(i),markersize=4)
            ax.legend()
        fig.savefig("pool_contents.pdf")
        self.assertTrue(True)

