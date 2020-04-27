#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:

import inspect
import sys 
import unittest
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
from CompartmentalSystems.model_run import plot_attributes, plot_stocks_and_fluxes

class TestDiscreteModelRun(InDirTest):
    def test_from_PWCModelRun(self):
        x_0,x_1,t,k,u = symbols("x_0,x_1,k,t,u")
        inputs={
             0: u*(2-2*sin(2*t))
            ,1: u
        }
        outputs={
             0: x_0*k
            ,1: x_1**2*k
        }
        internal_fluxes = {
            (0,1): x_0,
            (1,0): 0.5*x_1
        }
        srm=SmoothReservoirModel([x_0,x_1],t,inputs,outputs,internal_fluxes)
        t_max=2*np.pi
        times = np.linspace(0, t_max, 21)
        times_fine = np.linspace(0, t_max, 81)
        x0=np.float(10)
        start_values = np.array([x0,x0])
        parameter_dict = {
             k:  0.012
            ,u: 10.7}
        delta_t=np.float(1)
        
        pwc_mr = PWCModelRun(srm, parameter_dict, start_values, times)
        pwc_mr_fine = PWCModelRun(srm, parameter_dict, start_values, times_fine)

        
        xs, net_Us, net_Fs, net_Rs = pwc_mr.fake_net_discretized_output(times)
        xs, gross_Us, gross_Fs, gross_Rs = pwc_mr.fake_gross_discretized_output(times)
        xs_fine, gross_Us_fine, gross_Fs_fine, gross_Rs_fine = pwc_mr_fine.fake_gross_discretized_output(times_fine)

        dmr_from_pwc = DiscreteModelRun.from_PWCModelRun(pwc_mr)
        dmr_from_fake_net_data = DiscreteModelRun.reconstruct_from_fluxes_and_solution(
            times,
            xs,
            net_Fs,
            net_Rs
        )
        dmr_from_fake_gross_data_ffas = DiscreteModelRun.reconstruct_from_fluxes_and_solution(
            times,
            xs,
            gross_Fs,
            gross_Rs
        )
        dmr_from_fake_gross_data_ff = DiscreteModelRun.from_fluxes(
            start_values,
            times,
            gross_Us,
            gross_Fs,
            gross_Rs
        )
        dmr_from_fake_gross_data_ff_fine = DiscreteModelRun.from_fluxes(
            start_values,
            times_fine,
            gross_Us_fine,
            gross_Fs_fine,
            gross_Rs_fine
        )

        self.assertTrue(
            np.allclose(
                pwc_mr.solve(),
                dmr_from_pwc.solve()
            )
        )

        self.assertTrue(
            np.allclose(
                pwc_mr.solve(),
                dmr_from_fake_net_data.solve()
            )
        )
       
        # very unexpectedly the solution
        # can be reconstructed from the right start_value
        # the WRONG inputs WRONG internal fluxes and
        # WRONG outputs
        self.assertTrue(
            np.allclose(
                pwc_mr.solve(),
                dmr_from_fake_gross_data_ff.solve(),
                rtol=1e-3
            )
        )


        # Here we also expect different results.
        # We again abuse the DiscreteModelRun class
        # but this time we give it the right solution 
        # which will be reproduced
        self.assertTrue(
            np.allclose(
                pwc_mr.solve(),
                dmr_from_fake_gross_data_ffas.solve()
            )
        )
        # but the net influxes will be wrong
        self.assertFalse(
            np.allclose(
                pwc_mr.acc_net_external_input_vector(),
                dmr_from_fake_gross_data_ffas.net_Us
            )
        )
        #plot_attributes(
        #    [
        #        pwc_mr,
        #        dmr_from_fake_net_data,
        #        dmr_from_fake_gross_data_ff,
        #        dmr_from_fake_gross_data_ffas
        #    ],
        #    'plot.pdf'
        #)       
        plot_stocks_and_fluxes(
            [
                pwc_mr
                #,dmr_from_fake_net_data
                #,dmr_from_pwc
                ,dmr_from_fake_gross_data_ff
                ,dmr_from_fake_gross_data_ff_fine
            ],
            'stocks_and_fluxes.pdf'
        )       
        #plot_stocks_and_gross_fluxes(
        #    [
        #        pwc_mr,
        #        dmr_from_fake_net_data,
        #        dmr_from_fake_gross_data_ff,
        #        dmr_from_fake_gross_data_ffas
        #    ],
        #    'stocks_and_gross_fluxes.pdf'
        #)       


    #@unittest.skip        
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



