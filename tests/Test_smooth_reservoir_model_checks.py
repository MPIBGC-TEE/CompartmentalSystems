from concurrencytest import ConcurrentTestSuite, fork_for_tests
import sys
import unittest

import numpy as np
from sympy import Symbol, Matrix, symbols, diag, zeros, simplify, Function
from sympy.printing import pprint
from copy import deepcopy

from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from testinfrastructure.InDirTest import InDirTest
######### TestClass #############
class TestSmoothReservoirModelChecks(InDirTest):
    def test_free_symbols(self):
        t,C_1, C_2, C_3, k_1, k_2, k_3, a_12, a_13, a_21, a_23, a_31, a_32, u_1, u_2, u_3, gamma, xi \
        = symbols('t,C_1 C_2 C_3 k_1 k_2 k_3 a_12 a_13 a_21 a_23 a_31 a_32 u_1 u_2 u_3 gamma xi')
        C = Matrix(3,1, [C_1, C_2, C_3])
        u = Matrix(3,1, [u_1, u_2, u_3])
        B = gamma*Matrix([
                [-k_1, a_12, a_13],
                [a_21, -k_2, a_23],
                [a_31, a_32, -k_3]
            ])
        rm = SmoothReservoirModel.from_B_u(C,t,B,u)
        print(rm.free_symbols)

        rm_p1=rm.subs(
            {
                k_1:4,a_21:1,a_31:2,
                k_2:6,a_12:2,a_32:3,
                k_3:9,a_13:4,a_23:4,
                u_1:1,u_2:1,u_3:1,
                gamma:1,
                xi:0.5
            }
        )
        print(rm_p1.free_symbols)
    
    @unittest.skip('it does not work yet for the nonlinear example')
    def test_is_compartmental(self):
        # at the time of implementation this functionality sympy did not support 
        # relations in predicates yet.
        # So while the following works:
        #
        # with assuming(Q.positive(x) & Q.positive(y)):
        #    print(ask(Q.positive(2*x+y)
        #
        # it is not possible yet to get a meaningful answer to:
        #
        # with assuming(Q.is_true(x>0) & Q.is_true(y>0)):
        #    print(ask(Q.positive(2*x+y)
        # 
        # We therefore cannot implement more elaborate assumptions like k_1-(a_12+a_32)>=0 in the following
        # example  but still can assume all the state_variables to be positive.
        # Therefore we can check the compartmental_property best after all paramater value have been substituted

        # f = u + xi*B*C
        t,L,C_1, C_2, C_3, k_1, k_2, k_3, a_12, a_13, a_21, a_23, a_31, a_32, u_1, u_2, u_3, gamma, xi \
        = symbols('t,L,C_1 C_2 C_3 k_1 k_2 k_3 a_12 a_13 a_21 a_23 a_31 a_32 u_1 u_2 u_3 gamma xi')
        C = Matrix(3,1, [C_1, C_2, C_3])
        u = Matrix(3,1, [u_1, u_2, u_3])
        B = gamma*Matrix([
                [-k_1, a_12, a_13],
                [a_21, -k_2, a_23],
                [a_31, a_32, -k_3]
            ])
        rm = SmoothReservoirModel.from_B_u(C,t,B,u)
        
        # check that the method refuses if there are still free symbols (except the state variables and time)
        with self.assertRaises(Exception):
            print(rm.is_compartmental)
        
        # we first choose a parameter set that leads to a compartmental system
        rm_p1=rm.subs(
            {
                k_1:4,a_21:1,a_31:2,
                k_2:6,a_12:2,a_32:3,
                k_3:9,a_13:4,a_23:4,
                u_1:1,u_2:1,u_3:1,
                gamma:1,
                xi:0.5
            }
        )
        self.assertTrue(rm_p1.is_compartmental) 

        # now we chose paramtersets that do not lead to a compartmental system
        rm_p1=rm.subs(
            {
                k_1:2,a_21:1,a_31:2, #k is to small
                k_2:6,a_12:2,a_32:3,
                k_3:9,a_13:4,a_23:4,
                u_1:1,u_2:1,u_3:1,
                gamma:1,
                xi:0.5
            }
        )
        self.assertFalse(rm_p1.is_compartmental) 
        
        # As an edge case  we now choose a parameter set that leads to a compartmental system with a zero
        # flux in one of the pools
        rm_p1=rm.subs(
            {
                k_1:4,a_21:2,a_31:2,# zero flux
                k_2:6,a_12:2,a_32:3,
                k_3:9,a_13:4,a_23:4,
                u_1:1,u_2:1,u_3:1,
                gamma:1,
                xi:0.5
            }
        )
        self.assertTrue(rm_p1.is_compartmental) 
       # print(rm_p1)
        
        # try a nonlinear model
        B = gamma*Matrix([
                [-k_1 *C_1/(C_1+L), a_12, a_13],
                [ a_21*C_1/(C_1+L), -k_2, a_23],
                [ a_31*C_1/(C_1+L), a_32, -k_3]
            ])
        rm = SmoothReservoirModel.from_B_u(C,t,B,u)
        rm_p1=rm.subs(
            {   
                L  :10,
                k_1:4,a_21:1,a_31:2,
                k_2:6,a_12:2,a_32:3,
                k_3:9,a_13:4,a_23:4,
                u_1:1,u_2:1,u_3:1,
                gamma:1,
                xi:0.5
            }
        )
        print(rm_p1)
        # unfortunately sympy cannot handle this Michaelis Menten case yet
        with self.assertRaises(Exception):
            rm_p1.is_compartmental 
        
        # but it works for other  nonlinear models 
        B = gamma*Matrix([
                [-k_1 *C_1, a_12, a_13],
                [ a_21*C_1, -k_2, a_23],
                [ a_31*C_1, a_32, -k_3]
            ])
        rm = SmoothReservoirModel.from_B_u(C,t,B,u)
        rm_p1=rm.subs(
            {   
                k_1:4,a_21:1,a_31:2,
                k_2:6,a_12:2,a_32:3,
                k_3:9,a_13:4,a_23:4,
                u_1:1,u_2:1,u_3:1,
                gamma:1,
                xi:0.5
            }
        )
        self.assertTrue(rm_p1.is_compartmental) 
        #self.assertEqual(rm.input_fluxes, {0: u_1, 1: u_2, 2: u_3})
        #self.assertEqual(rm.output_fluxes, {0: 2*gamma*k_1*(1-t_21-t_31*k_1)*C_1*C_2,
        #                           1: -2*gamma*k_2*(-1+t_12*C_2+t_32)*C_2/C_3,
        #                           2: 2*gamma*k_3*(1-t_13-t_23)*C_3})
        #self.assertEqual(rm.internal_fluxes, {
        #    (0,1): 2*gamma*t_21*k_1*C_1*C_2, (0,2): 2*gamma*t_31*k_1**2*C_1*C_2,
        #    (1,0): 2*gamma*t_12*k_2*C_2**2/C_3, (1,2): 2*gamma*t_32*k_2*C_2/C_3,
        #    (2,0): 2*gamma*t_13*k_3*C_3, (2,1): 2*gamma*t_23*k_3*C_3})

