#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:

from copy import deepcopy
import inspect
import sys ,os
import unittest
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.integrate import quad
from scipy.interpolate import interp1d 
from scipy.special import factorial
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function, ones
    
import CompartmentalSystems.example_smooth_reservoir_models as ESRM
import CompartmentalSystems.example_smooth_model_runs as ESMR

from testinfrastructure.InDirTest import InDirTest
#from testinfrastructure.helpers import pe
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.smooth_model_run import SmoothModelRun 
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel 

#from CompartmentalSystems.smooth_model_run_14C import  pfile_C14Atm_NH

class TestSmoothModelRun(InDirTest):
#class TestSmoothModelRun(unittest.TestCase):
        
    def test_init(self):
        #create a valid model run complete with start ages
        symbs = symbols("x,k,t")
        x, t, k = symbs 
        srm = ESRM.minimal(symbs) 
        times = np.linspace(0, 20, 1600)
        start_values = np.array([10])
        pardict = {k: 1}
        smr = SmoothModelRun(srm, pardict, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        self.assertEqual(smr.start_values, start_values)
        self.assertTrue(all(smr.times==times))
        
        #create a valid model run without start ages
        smr = SmoothModelRun(srm, pardict, start_values, times=times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        #check if we can retrieve values back 
        #(although this looks too simple there was an error here)
        self.assertEqual(smr.start_values, start_values)
        self.assertTrue(all(smr.times==times))
       
        #check for incomplete param set
        pardict = {}
        with self.assertRaises(Exception):
            smr = SmoothModelRun(srm, pardict, start_values, times=times)

        

    def test_linearize(self):
       # initialize model run 
        start_year = 1765
        end_year = 2500
        max_age = 250
        times = np.arange(start_year, end_year+1, 1)
        
        time_symbol = symbols('tau')
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
        #u_A = symbols('u_A')
        u_A = Function('u_A')(time_symbol)
        
        # land use change flux
        f_TA = Function('f_TA')(time_symbol)
        
        
        #########################################
        
        state_vector = Matrix([C_A, C_T, C_S])
        
        input_fluxes = {0: u_A, 1: 0, 2: F_0}
        output_fluxes = {0: 0, 1: 0, 2: F_0*C_S/S_e}
        internal_fluxes = {(0,1): F_2*(C_A/A_e)**alpha, # A --> T
                           (0,2): F_1*C_A/A_e,          # A --> S
                           (1,0): F_2*C_T/T_e+f_TA,          # T --> A
                           (2,0): F_1*(C_S/S_e)**beta}  # S --> A
        
        nonlinear_srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)
        
        A_eq, T_eq, S_eq = (700.0, 3000.0, 1000.0) 
        par_dict = {  A_e: A_eq,  T_e:  T_eq, S_e: S_eq, # equilibrium contents in Pg
                      F_0: 45.0,  F_1: 100.0, F_2: 60.0, # equilibrium fluxes in PgC/yr
                    alpha:  0.2, beta:  10.0           } # nonlinear coefficients
        
        # fossil fuel inputs
        #par_dict[u_A] = 0
        # fossil fuel and land use change data
        p=Path(deepcopy(__file__))
        file_path = p.parents[1].joinpath('notebooks','PNAS','emissions.csv')
        ff_and_lu_data = np.loadtxt(file_path, usecols = (0,1,2), skiprows = 38)
        
        # column 0: time, column 1: fossil fuels
        ff_data = ff_and_lu_data[:,[0,1]]
        
        # linear interpolation of the (nonnegative) data points
        u_A_interp = interp1d(ff_data[:,0], np.maximum(ff_data[:,1], 0),fill_value="extrapolate")
        
        def u_A_func(t_val):
            # here we could do whatever we want to compute the input function
            # we return only the linear interpolation from above
            return u_A_interp(t_val)
        
        # column 0: time, column 2: land use effects
        lu_data = ff_and_lu_data[:,[0,2]]
        f_TA_func = interp1d(lu_data[:,0], lu_data[:,1],fill_value="extrapolate")
        
        # define a dictionary to connect the symbols with the according functions
        func_set = {u_A: u_A_func, f_TA: f_TA_func}
        
        #times = np.linspace(0, 10, 101)
        #start_values = np.array([A_eq/2, T_eq*2, S_eq/3])
        start_values = np.array([A_eq, T_eq, S_eq])
        nonlinear_smr = SmoothModelRun(nonlinear_srm, par_dict, start_values, times,func_set)
        nonlinear_smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        linearized_smr = nonlinear_smr.linearize()
        linearized_smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        nonlin_soln = nonlinear_smr.solve()
        lin_soln = linearized_smr.solve()
        self.assertTrue(
            np.allclose(
                nonlin_soln,
                lin_soln,
                rtol=5e-3
            )
        )

        # plot the solution
        fig=plt.figure(figsize=(10,7))
        ax=fig.add_subplot(1,1,1)
        #plt.title('Total carbon'+title_suffs[version])
        ax.plot(times, nonlin_soln[:,0],   alpha=0.2,color='blue', label='Atmosphere')
        ax.plot(times, nonlin_soln[:,1],   alpha=0.2,color='green', label='Terrestrial Biosphere')
        ax.plot(times, nonlin_soln[:,2],   alpha=0.2,color='purple', label='Surface ocean')
        ax.plot(times, nonlin_soln.sum(1), alpha=0.2,color='red', label='Total')
        ax.plot(times, lin_soln[:,0],   ls='--', color='blue', label='Atmosphere')
        ax.plot(times, lin_soln[:,1],   ls='--', color='green', label='Terrestrial Biosphere')
        ax.plot(times, lin_soln[:,2],   ls='--', color='purple', label='Surface ocean')
        ax.plot(times, lin_soln.sum(1), ls='--', color='red', label='Total')
        ax.set_xlim([1765,2500])
        ax.set_ylim([0,9000])
        ax.legend(loc=2)
        ax.set_xlabel('Time (yr)')
        ax.set_ylabel('Mass (PgC)')
        fig.savefig('plot.pdf')


    def test_linearize_old(self):
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
        start_values = np.array([A_eq/2, T_eq*2, S_eq/3])
        nonlinear_smr = SmoothModelRun(nonlinear_srm, par_dict, start_values, times)
        nonlinear_smr.initialize_state_transition_operator_cache(lru_maxsize=None)
      

        linearized_smr = nonlinear_smr.linearize_old()
        linearized_smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        nonlin_soln = nonlinear_smr.solve_old()
        lin_soln = linearized_smr.solve_old()
        self.assertTrue(
            np.allclose(
                nonlin_soln,
                lin_soln,
                rtol=5e-3
            )
        )

    def test_moments_from_densities(self):
        # two_dimensional
        start_values = np.array([1,2])
        def start_age_densities(a):
            p1 = np.exp(-a) * start_values[0]
            p2 = 2*np.exp(-2*a) * start_values[1]
        
            return np.array([p1, p2])

        max_order = 5
        moments = SmoothModelRun.moments_from_densities(max_order, start_age_densities)

        ref1 = np.array([factorial(n)/1**n for n in range(1, max_order+1)])
        ref2 = np.array([factorial(n)/2**n for n in range(1, max_order+1)]) 
        ref = np.array([ref1, ref2]).transpose()

        self.assertTrue(np.allclose(moments, ref,rtol=1e-3))

        # test empty pool
        start_values = np.array([0,2])
        def start_age_densities(a):
            p1 = np.exp(-a) * start_values[0]
            p2 = 2*np.exp(-2*a) * start_values[1]
        
            return np.array([p1, p2])

        max_order = 1
        moments = SmoothModelRun.moments_from_densities(max_order, start_age_densities)
        self.assertTrue(np.isnan(moments[0,0]))


    ########## public methods and properties ########## 
             
    
    def test_solve_symbolic(self):
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [0,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)
        smr = SmoothModelRun(srm, parameter_dict={}, start_values=np.array([1,1]), times=np.linspace(0,1,10))
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        a_ref = np.array(
            [[1.        , 1.        ],  
             [0.89488146, 0.90060046],
             [0.80071979, 0.82045707],
             [0.71649099, 0.75646401],
             [0.64117104, 0.70551544],
             [0.57378348, 0.66471948],
             [0.51342622, 0.63183319],
             [0.4594041 , 0.60546037],
             [0.41109603, 0.58445828],
             [0.36788092, 0.56768423]
            ]
        )

        ref = np.ndarray((10,2), float, a_ref)
        soln = smr.solve()
        self.assertTrue(np.allclose(soln, ref))


    def test_solve_semi_symbolic(self):
        # test semi-symbolic semi-numerical SmoothReservoirModel
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
        smr = SmoothModelRun(srm, parameter_dict={}, start_values=start_values, times=times,func_set=func_set)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        
        soln = smr.solve()


    ##### fluxes as functions #####


    def test_flux_funcs(self):
        # one-dimensional case, check that constant values do not lead
        # to problems like 1.subs({...})
        C_0 = Symbol('C_0')
        C_1 = Symbol('C_1')
        state_vector = [C_0,C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: time_symbol,1:1}
        output_fluxes = {0: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5,5])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        
        u = smr.external_input_flux_funcs()
        o = smr.external_output_flux_funcs()
        # check the scalar versions
        self.assertTrue(np.allclose(u[0](0.5), 0.5))
        self.assertTrue(np.allclose(u[1](0.5), 1))
        # check the vectorized versions
        self.assertTrue(np.allclose(u[0](times), times))
        self.assertTrue(np.allclose(u[1](times), np.ones_like(times)))
        #print(u[0](np.linspace(0,1,11)))
        #print(o[0](np.linspace(0,1,11)))
        

    def test_output_vector_func(self):
        # two-dimensional case
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1,3])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        res=smr.output_vector_func(1)
#        pe('res',locals())
        self.assertTrue(np.allclose(res, np.array([0.36809009, 1.10427026]),rtol=1e-3))
        

    ##### fluxes as vector-valued functions #####
    

    def test_external_input_vector_func(self):
        C_1, C_2 = symbols('C_1 C_2')
        state_vector = [C_1, C_2]
        time_symbol = Symbol('t')
        input_fluxes = {0: time_symbol, 1: 0}
        output_fluxes = {0: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5,2])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        
        u = smr.external_input_vector_func()
        self.assertTrue(np.allclose(u(0.5), np.array([0.5, 0])))


    ##### fluxes as vector over self.times #####


    def test_external_input_vector(self):
        # two-dimensional case
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 0.5*C_1} # even pool-dependent input
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1,3])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ref_a = np.array(
            [[ 1.,          1.42684416],
             [ 1.,          1.35725616],
             [ 1.,          1.29106198],
             [ 1.,          1.22809615],
             [ 1.,          1.16820118],
             [ 1.,          1.11122731],
             [ 1.,          1.05703214],
             [ 1.,          1.00548009],
             [ 1.,          0.95644224],
             [ 1.,          0.90979601]]
        )
        ref = np.ndarray((10, 2), float, ref_a)
        self.assertTrue(np.allclose(
            smr.external_input_vector[1:],
            ref
        ))


    def test_external_output_vector(self):
        # one-dimensional case
        C_0 = symbols('C_0')
        state_vector = [C_0]
        time_symbol = Symbol('t')
        input_fluxes = {}
        output_fluxes = {0: C_0}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1])
        times = np.linspace(0, 1, 11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ref_a = np.array(
            [[1.        ], 
             [0.90488336],
             [0.81872473],
             [0.7407776 ],
             [0.67029549],
             [0.60653188],
             [0.54884061],
             [0.49658265],
             [0.44930523],
             [0.40655558],
             [0.36788092]]
        )
        ref = np.ndarray((11, 1), float, ref_a)
        self.assertTrue(np.allclose(smr.external_output_vector, ref))

        # two-dimensional case
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1,3])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ref_a = np.array(
            [[1.        , 3.        ], 
             [0.90488336, 2.71465008],
             [0.81872473, 2.45617418],
             [0.7407776 , 2.22233281],
             [0.67029549, 2.01088647],
             [0.60653188, 1.81959565],
             [0.54884061, 1.64652182],
             [0.49658265, 1.48974795],
             [0.44930523, 1.3479157 ],
             [0.40655558, 1.21966675],
             [0.36788092, 1.10364277]]
        )
        ref = np.ndarray((11, 2), float, ref_a)
        self.assertTrue(np.allclose(smr.external_output_vector, ref))


    ##### age density methods #####


    def test_pool_age_densities_single_value(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        start_age_densities = lambda a: np.exp(-a)*start_values
        p_sv = smr.pool_age_densities_single_value(start_age_densities)

        a1_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 5.        ,  3.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 1.83939721,  1.10363832],
                  [ 1.83939724,  1.10363834],
                  [ 1.83939725,  1.10363835],
                  [ 1.83939724,  1.10363835],
                  [ 1.83939729,  1.10363837],
                  [ 1.83939727,  1.10363836]]])

        a2_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],

                 [[ 0.        ,  0.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ]],
                
                 [[ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ]]])
#
#        a2_ref = np.array(
#                [[[ 0.        ,  0.        ],
#                  [ 0.        ,  0.        ],
#                  [ 0.        ,  0.        ],
#                  [ 0.        ,  0.        ],
#                  [ 0.        ,  0.        ],
#                  [ 0.        ,  0.        ]],
#
#                 [[ 1.        ,  2.        ],
#                  [ 1.        ,  2.        ],
#                  [ 1.        ,  2.        ],
#                  [ 1.        ,  2.        ],
#                  [ 1.        ,  2.        ],
#                  [ 1.        ,  2.        ]],
#                
#                 [[ 0         ,  0         ],
#                  [ 0         ,  0         ],
#                  [ 0         ,  0         ],
#                  [ 0         ,  0         ],
#                  [ 0         ,  0         ],
#                  [  0.36788825, 0.73576626000000010         ]]])

        a_ref = a1_ref + a2_ref
        ref = np.ndarray((3,6,2), float, a_ref)
        y = p_sv(0,0)
        res_l = [[p_sv(a, t) for t in times] for a in ages]
        res = np.array(res_l)
#        print(res)
        self.assertTrue(np.allclose(res, ref,rtol=1e-3))


    def test_age_densities(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = Matrix([C_0, C_1])
        time_symbol = Symbol('t')
        #fixme: both input anoutput should be 1, 2, C_0, C_1
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        start_age_densities = lambda a: np.exp(-a)*start_values

        p = smr.pool_age_densities_func(start_age_densities)
        pool_age_densities = p(ages)
        system_age_density = smr.system_age_density(pool_age_densities)
        
        age_densities = smr.age_densities(pool_age_densities, system_age_density)

        a_ref = np.array(
            [[[ 0.        ,  0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        ]],
            
             [[ 5.        ,  3.        ,  8.        ],
              [ 1.        ,  2.        ,  3.        ],
              [ 1.        ,  2.        ,  3.        ],
              [ 1.        ,  2.        ,  3.        ],
              [ 1.        ,  2.        ,  3.        ],
              [ 1.        ,  2.        ,  3.        ]],
            
             [[ 1.83939721,  1.10363832,  2.94303553],
              [ 1.83939724,  1.10363834,  2.94303558],
              [ 1.83939726,  1.10363835,  2.94303561],
              [ 1.83939725,  1.10363835,  2.9430356 ],
              [ 1.8393973 ,  1.10363838,  2.94303568],
              [ 1.83939729,  1.10363837,  2.94303566]]])

        ref = np.ndarray((3,6,3), float, a_ref)
        self.assertTrue(np.allclose(age_densities, ref,rtol=1e-3))
        
        ## two-dimensional nonlinear with a noninvertable matrix at C_0 =1 point
        #output_fluxes = {0: (C_0-1)**2, 1: C_1}
        #internal_fluxes = {}
        #srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)
        #smr = SmoothModelRun(srm, {}, start_values, times)
        #lmr=smr.linearize_old()
        #ages = np.linspace(0,2,21)
        #p = lmr.pool_age_densities_func(start_age_densities)
        #pool_age_densities = p(ages)

    def test_system_age_density_single_value(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        start_age_densities = lambda a: np.exp(-a)*start_values
        p_sv = smr.system_age_density_single_value(start_age_densities)

        a1_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 5.        ,  3.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 1.83939721,  1.10363832],
                  [ 1.83939724,  1.10363834],
                  [ 1.83939725,  1.10363835],
                  [ 1.83939724,  1.10363835],
                  [ 1.83939729,  1.10363837],
                  [ 1.83939727,  1.10363836]]])

        a2_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 0.        ,  0.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ]],
                
                 [[ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ]]])

        a_ref = a1_ref + a2_ref
        ref = np.sum(np.ndarray((3,6,2), float, a_ref), axis=2)
        res_l = [[p_sv(a, t) for t in times] for a in ages]
        res = np.array(res_l)
        self.assertTrue(np.allclose(res, ref,rtol=1e-3))


    def test_system_age_density(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        start_age_densities = lambda a: np.exp(-a)*start_values
        p = smr.pool_age_densities_func(start_age_densities)
        age_densities = p(ages)
        system_age_density = smr.system_age_density(age_densities)

        a1_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 5.        ,  3.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 1.83939721,  1.10363832],
                  [ 1.83939724,  1.10363834],
                  [ 1.83939725,  1.10363835],
                  [ 1.83939724,  1.10363835],
                  [ 1.83939729,  1.10363837],
                  [ 1.83939727,  1.10363836]]])

        a2_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 0.        ,  0.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ]],
                
                 [[ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ]]])

        a_ref = a1_ref + a2_ref
        ref = np.sum(np.ndarray((3,6,2), float, a_ref), axis=2)
#        pe('system_age_density',locals())
#        pe('ref',locals())
#        pe('ref-system_age_density',locals())
        self.assertTrue(np.allclose(ref, system_age_density,rtol=1e-3))


    ##### age moments methods #####


    def test_age_moment_vector_from_densities(self):
        # test mean age
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [0,1])
        start_values = np.array([1,2])
        times = np.linspace(0,1,3)

        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)
        smr = SmoothModelRun(srm, parameter_dict={}, start_values=start_values, times=times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: 2*np.exp(-2*a)*start_values

        # the solution to be tested
        order = 1
        ma_from_dens = smr.age_moment_vector_from_densities(order, start_age_densities)

        # test against solution from mean age system
        start_mean_ages = [0.5,0.5]
        n = srm.nr_pools
        start_age_moments = np.ndarray((1,n), float, np.array(start_mean_ages))

        ref_ma = smr.age_moment_vector(1, start_age_moments)

        self.assertTrue(np.allclose(ma_from_dens, ref_ma,rtol=2e-3))


    def test_age_moment_vector_semi_explicit(self):
        x, y, t = symbols("x y t")
        X = Matrix([x,y])
        u = Matrix(2, 1, [1, 2])
        srm = SmoothReservoirModel.from_B_u(X, t, Matrix([[-1,0],[0,-1]]), u)
        
        start_values = np.array([1,1])
        times = np.linspace(0, 1, 10)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        n = smr.nr_pools

        def start_age_densities(a):
            p1 = np.exp(-a) * start_values[0]
            p2 = 2*np.exp(-2*a) * start_values[1]
        
            return np.array([p1, p2])

        start_age_moments = smr.moments_from_densities(1, start_age_densities)

        ma_ref = smr.age_moment_vector(1, start_age_moments)
        ma_semi_explicit = smr.age_moment_vector_semi_explicit(1, start_age_moments)
        self.assertTrue(np.allclose(ma_semi_explicit, ma_ref,rtol=1e-3))

        # test empty start_ages
        ma_ref = smr.age_moment_vector(1)
        ma_semi_explicit = smr.age_moment_vector_semi_explicit(1)
        self.assertTrue(np.allclose(ma_semi_explicit, ma_ref,rtol=1e-3))  

        # test that nothing fails for second moment
        start_age_moments = smr.moments_from_densities(2, start_age_densities)
        smr.age_moment_vector_semi_explicit(2, start_age_moments)
        smr.age_moment_vector_semi_explicit(2)

        # test empty second pool at beginning
        x, y, t = symbols("x y t")
        X = Matrix([x,y])
        u = Matrix(2, 1, [0, 1])
        srm = SmoothReservoirModel.from_B_u(X, t, Matrix([[-1,0],[0,-1]]), u)
        
        start_values = np.array([1,0])
        times = np.linspace(0, 1, 11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        n = smr.nr_pools

        def start_age_densities(a):
            p1 = np.exp(-a) * start_values[0]
            p2 = np.exp(-a) * start_values[1]
        
            return np.array([p1, p2])

        start_age_moments = smr.moments_from_densities(2, start_age_densities)

        ma_ref = smr.age_moment_vector(2, start_age_moments)
        ma_semi_explicit = smr.age_moment_vector_semi_explicit(2, start_age_moments)
        self.assertTrue(np.allclose(ma_semi_explicit, ma_ref, equal_nan=True,rtol=1e-3))


    def test_age_moment_vector(self):
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([1,1])
        smr = SmoothModelRun(srm, {}, start_values, times=np.linspace(0,1,10))
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a) * start_values
        start_age_moments = smr.moments_from_densities(1, start_age_densities)
    
        # set manually mean age in first pool to zero
        start_age_moments[0,0] = 0

        ma_vec = smr.age_moment_vector(1, start_age_moments)

        a_ref = np.array(
            [[0.        , 1.        ], 
             [0.08202136, 0.99408002],
             [0.14256488, 0.97750095],
             [0.19599202, 0.95232419],
             [0.24535025, 0.92082419],
             [0.29165268, 0.88523234],
             [0.33540979, 0.8477071 ],
             [0.37684872, 0.80984795],
             [0.41613677, 0.77309118],
             [0.45338235, 0.7384063 ]]
        )        
        ref = np.ndarray((10,2), float, a_ref)
        self.assertTrue(np.allclose(ma_vec, ref)) 

        # test empty initial pool, pool remains empty
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [9,0])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([1,0])
        smr = SmoothModelRun(srm, {}, start_values, times=np.linspace(0,1,10))
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a) * start_values
        start_age_moments = smr.moments_from_densities(1, start_age_densities)
    
        # set manually mean age in first pool to zero
        start_age_moments[0,0] = 0

        ma_vec = smr.age_moment_vector(1, start_age_moments)
        a_ref = np.array([[ 0.        , np.nan], 
                          [ 0.08202608, np.nan],
                          [ 0.14256586, np.nan],
                          [ 0.19599667, np.nan],
                          [ 0.24534558, np.nan],
                          [ 0.29165805, np.nan],
                          [ 0.33540452, np.nan],
                          [ 0.37683967, np.nan],
                          [ 0.41612361, np.nan],
                          [ 0.45337059, np.nan]])

        ref = np.ndarray((10,2), float, a_ref)
        self.assertTrue(np.allclose(ma_vec, ref, equal_nan=True,rtol=1e-3))

        # test empty initial pool, pool receives input
        # test second moment for technical problems
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([1,0])
        smr = SmoothModelRun(srm, {}, start_values, times=np.linspace(0,1,10))
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a) * start_values
        start_age_moments = smr.moments_from_densities(2, start_age_densities)
    
        # set manually mean age in first pool to zero
        start_age_moments[0,0] = 0

        ma_vec = smr.age_moment_vector(2, start_age_moments)
        a_ref = np.array(
            [[2.        ,     np.nan], 
             [0.98004619, 0.00388956],
             [0.64335829, 0.01466637],
             [0.48945203, 0.03104624],
             [0.41296535, 0.05182671],
             [0.37774886, 0.07590298],
             [0.36762348, 0.10227368],
             [0.37444289, 0.13006407],
             [0.39316946, 0.15850235],
             [0.42081205, 0.18696148]]
        )
        ref = np.ndarray((10,2), float, a_ref)
        self.assertTrue(np.allclose(ma_vec, ref, equal_nan=True))


    def test_system_age_moment(self):
        # create a parallel model with identical initial conditions and check that in this
        # case the pool age moments are both equal to the system age moments
        x, y, t = symbols("x y t")
        X = Matrix([x,y])
        u = Matrix([0,0])
        srm = SmoothReservoirModel.from_B_u(X, t, Matrix([[-1,0],[0,-1]]), u)

        start_values = np.array([1,1])
        smr = SmoothModelRun(srm, {}, start_values, np.linspace(0,1,10))
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        n = smr.nr_pools
        
        order = 2
        start_age_densities = lambda a: np.exp(-a) * start_values
        start_age_moments = smr.moments_from_densities(order, start_age_densities)
#        pe('start_age_moments',locals())
#        pe('start_age_moments.shape',locals())

        system_age_moment = smr.system_age_moment(order, start_age_moments)
        age_moment_vector = smr.age_moment_vector(order, start_age_moments)

        for pool in range(n):
            self.assertTrue(np.allclose(age_moment_vector[:,pool], system_age_moment))

        # test empty system and empty pools
        x, y, t = symbols("x y t")
        X = Matrix([x,y])
        u = Matrix(2, 1, [0,1])
        srm = SmoothReservoirModel.from_B_u(X, t, Matrix([[-1,0],[0,-1]]), u)

        start_values = np.array([0,0])
        smr = SmoothModelRun(srm, {}, start_values, np.linspace(0,1,10))
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        n = smr.nr_pools
        
        order = 1
        start_age_densities = lambda a: np.exp(-a) * start_values
        start_age_moments = smr.moments_from_densities(order, start_age_densities)

        age_moment_vector = smr.age_moment_vector(order, start_age_moments)
        system_age_moment = smr.system_age_moment(order, start_age_moments)

        self.assertTrue(np.all(np.isnan(age_moment_vector[:,0])))
        self.assertTrue(np.isnan(age_moment_vector[0,1]))
        self.assertTrue(np.allclose(age_moment_vector[:,1], system_age_moment, equal_nan=True))


    ##### transit time density methods #####


    def test_backward_transit_time_density_single_value_func(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        start_age_densities = lambda a: np.exp(-a)*start_values

        p_sv = smr.backward_transit_time_density_single_value_func(start_age_densities)
        self.assertTrue(np.allclose(p_sv(1, 1),(5+3)*np.exp(-1),rtol=1e-3 ))


    def test_backward_transit_time_density(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        start_age_densities = lambda a: np.exp(-a)*start_values

        a1_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 5.        ,  3.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 1.83939721,  1.10363832],
                  [ 1.83939724,  1.10363834],
                  [ 1.83939725,  1.10363835],
                  [ 1.83939724,  1.10363835],
                  [ 1.83939729,  1.10363837],
                  [ 1.83939727,  1.10363836]]])

        a2_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 0.        ,  0.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ]],
                
                 [[ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ]]])

        a_ref = a1_ref + a2_ref
        
        tt_a_ref = np.array(
                 [[ 0.        ,  0.        ,  0.       ,   0.        ,  0.        ,  0.        ],
                  [ 8.        ,  3.        ,  3.       ,   3.        ,  3.        ,  3.        ],
                  [ 2.94303553,  2.94303558,  2.9430356,   2.94303559,  2.94303566,  2.94303564]])
        tt_ref = np.ndarray((3,6), float, tt_a_ref)

        p = smr.pool_age_densities_func(start_age_densities)
        age_densities = p(ages)
        btt_dens = smr.backward_transit_time_density(age_densities)
        self.assertTrue(np.allclose(btt_dens, tt_ref,rtol=1e-3))


    def test_forward_transit_time_density_single_value_func(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        start_age_densities = lambda a: np.exp(-a)*start_values

        p_btt_sv = smr.backward_transit_time_density_single_value_func(
            start_age_densities
        )
        p_ftt_sv = smr.forward_transit_time_density_single_value_func()
    
        # we changed the behavior of u(t0) such as not to return 0 
        # by default to make the quantile computation consistent
        ## no input at time t0 --> no forward transit time
        #self.assertTrue(np.isnan(p_ftt_sv(1, 0)))

        self.assertEqual(round(p_btt_sv(0.5, 1), 5), round(p_ftt_sv(0.5, 0.5), 5))

        # test behaviour if t+a is out of bounds
        self.assertTrue(np.isnan(p_ftt_sv(1, 1)))


    def test_forward_transit_time_density(self):
        # two-dimensional, test FTT=BTT in steady state
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1, 2])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(0,1,3)
        # negative ages will be cut off automatically
        start_age_densities = lambda a: np.exp(-a)*start_values
        p = smr.pool_age_densities_func(start_age_densities)
        age_densities = p(ages)
        btt_arr = smr.backward_transit_time_density(age_densities)
        p_ftt = smr.forward_transit_time_density_func()
        ftt_arr = p_ftt(ages)

        for age in range(ftt_arr.shape[0]):
            for time in range(ftt_arr.shape[1]):
                if not np.isnan(ftt_arr[age, time]):
                    self.assertEqual(
                        round(ftt_arr[age, time], 3),
                        round(btt_arr[age,time], 3)
                    )
        

    def test_cumulative_transit_time_distributions_single_value_func(self):
        # two-dimensional, test FTT=BTT in steady state
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1, 2])
        times = np.linspace(0, 1, 6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(0, 1, 3)
        # negative ages will be cut off automatically
        start_age_densities = lambda a: np.exp(-a) * start_values
#        p = smr.pool_age_densities_func(start_age_densities)
#        age_densities = p(ages)
#        btt_arr = smr.backward_transit_time_density(age_densities)
#        p_ftt = smr.forward_transit_time_density_func()
#        ftt_arr = p_ftt(ages)

        F_btt_sv = smr.cumulative_backward_transit_time_distribution_single_value_func(start_age_densities)
        F_ftt_sv = smr.cumulative_forward_transit_time_distribution_single_value_func()

        for age in ages:
            for time in times:
                if not np.isnan(F_ftt_sv(age, time)):
                    self.assertEqual(
                        round(F_ftt_sv(age, time), 3),
                        round(F_btt_sv(age, time), 3)
                    )
        

    ##### transit time moment methods #####


    def test_backward_transit_time_moment_from_density(self):
        # test mean BTT 
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [0,1])
        start_values = np.array([1,2])
        times = np.linspace(0,1,3)

        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)
        smr = SmoothModelRun(srm, parameter_dict={}, start_values=start_values, times=times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: 2*np.exp(-2*a)*start_values

        # the solution to be tested
        order = 1
        mbtt_from_dens = smr.backward_transit_time_moment_from_density(order, start_age_densities)

        # test against solution from mean age system

        start_age_moments = smr.moments_from_densities(1, start_age_densities)
        ref_mbtt = smr.backward_transit_time_moment(1, start_age_moments)

        self.assertTrue(np.allclose(mbtt_from_dens, ref_mbtt,rtol=1e-3))
        

    def test_backward_transit_time_moment(self):
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([1,1])
        smr = SmoothModelRun(srm, {}, start_values, times=np.linspace(0,1,10))
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a) * start_values
        start_age_moments = smr.moments_from_densities(1, start_age_densities)
    
        # set manually mean age in first pool to zero
        start_age_moments[0,0] = 0 # first moment, first pool

        mbtt = smr.backward_transit_time_moment(1, start_age_moments)
        #mbtt_ref = smr.mean_backward_transit_time(tuple(start_age_moments[0]))
        mbtt_ref = np.array([
            0.66666667, 0.53307397, 0.46606145, 0.43533026, 0.42580755,
            0.42915127, 0.44056669, 0.45707406, 0.47678225, 0.49837578
        ])
        self.assertTrue(np.allclose(mbtt, mbtt_ref)) 

    @unittest.skip('FTT moments cannot be computed properly')
    def test_forward_transit_time_moment(self):
#        import warnings
#        from scipy.integrate import IntegrationWarning
#        warnings.filterwarnings(
#            'ignore',
#            category=IntegrationWarning
#        )

        # if we start in steady state
        # and keep the inputs constant
        # forward and backward transit times should coincide.

        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        k = 10
        B = Matrix([[-k,  0],
                    [ 0, -k]])
        u = Matrix(2, 1, [1,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array(-B**(-1)*u)
        smr = SmoothModelRun(srm, {}, start_values, times=np.linspace(0,1,11))
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a*k) / start_values*np.array(u)
        start_age_moments = smr.moments_from_densities(1, start_age_densities)
    
        mbtt = smr.backward_transit_time_moment(1, start_age_moments)
        mftt = smr.forward_transit_time_moment(1)

        self.assertTrue(np.allclose(mbtt[1:], mftt[1:],rtol=1e-2))
        self.assertTrue(np.isnan(mftt[0]))

        # test integration to infinity 
        x, t = symbols("x t")
        state_vector = Matrix([x])
        B = Matrix([-1])
        u = Matrix([1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)
        
        start_values = np.array([1])
        n = 101
        times = np.linspace(0, 100, n)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        # to keep the integration time in resonable bounds we lower the required accuracy 
        mftts = smr.forward_transit_time_moment(1,epsrel=1e-2)
        self.assertTrue(np.allclose(mftts[1:], np.ones((100,)),rtol=1e-2)) 

        # some code to show possible problems with the sto
#        Phi = smr._state_transition_operator
#        Phi(1, 0, [1])
#        B = smr._state_transition_operator_values[0,:,:,:].reshape((101,))
#
#        from scipy.integrate import odeint
#        def rhs(X, t):
#            return -X
#
#        def f(X, t):
#            return odeint(rhs, X, [53, t])[-1]
#
#
#        C = np.array([f(B[ti], t+1) for ti, t in enumerate(np.linspace(0, 99, 100))]).reshape((100,))
#
#        ft = np.linspace(7, 10, 1001)
#        print('B')
#        plt.plot(ft, [Phi(54+t, 54, [1]) for t in ft])
#        print('B')
#        plt.plot(ft, np.exp(-ft), color = 'red')
#        ##plt.plot(ft, [f(t) for t in ft], ls = ':', linewidth = 5)
#        plt.show()
#
#        def integrand(a):
#            return Phi(54+a, 54, [1]).sum()
#
#        from scipy.integrate import quad
#        print(quad(integrand, 0, np.infty)[0])
#
#        #print(B)
#        #print(np.exp(-np.linspace(0,100,101)))
#        #print(C)



    @unittest.skip('just for now')
    def test_apply_to_forward_transit_time_simulation(self):
        x, t = symbols("x t")
        state_vector = Matrix([x])
        B = Matrix([-1*(1.4+sin(2*pi/10*t))])
        #B = Matrix([-1])
        u = Matrix([1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)
        
        start_values = np.array([1])
        n = 101
        times = np.linspace(0, 100, n)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fine_times = np.linspace(times[1], times[-1], n*100)
        for M in [0, 3, 11]:
        #for M in []:
            N = 1000
            if M == 0: N = 100
            sim_dict = smr.apply_to_forward_transit_time_simulation(f_dict = {'mean': np.mean}, N = N, M = M)

            for f_name, sub_dict in sim_dict.items():
                points = ax.plot(times, sub_dict['values'], ls = '-', label = f_name + ', M=' + str(M))
                if M == 0:
                    ax.plot(fine_times, sub_dict['smoothing_spline'](fine_times), ls = '--', label = f_name + ', smoothing, M=' + str(M), color = points[-1].get_color())
                else:
                    ax.plot(fine_times, sub_dict['interpolation'](fine_times), ls = '--', label = f_name + ', interpolation, M=' + str(M), color = points[-1].get_color())
       
        sim_dict = smr.apply_to_forward_transit_time_simulation(f_dict = {'mean': np.mean}, N = 100, MH = True)
        for f_name, sub_dict in sim_dict.items():
            points = ax.plot(times, sub_dict['values'], ls = '-', label = f_name + ', MH')
            ax.plot(fine_times, sub_dict['interpolation'](fine_times), ls = '--', label = f_name + ', interpolation, MH', color = points[-1].get_color())

        # plot true value (if integration is possible)
        mftts = smr.forward_transit_time_moment(1)
        ax.plot(times, mftts, color = 'black', label = 'integrated')

        ax.legend() 
        fig.savefig('test.pdf')
        #plt.show()


    ##### comma separated values output methods #####


    def test_save_and_load_csv(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0:1+0.3*sin(1/4*time_symbol), 1: 3}
        output_fluxes = {0: 1/6*C_0}
        internal_fluxes = {(0,1): C_0, (1,0): 1/4*C_1}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1, 3])
        times = np.linspace(0,50,5)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(0,50,5)
        start_age_densities = lambda a: np.exp(-a)*start_values
        start_age_moments = smr.moments_from_densities(1, start_age_densities)

        # test if saving and loading yields no diferences
        p = smr.pool_age_densities_func(start_age_densities)
        pool_age_densities = p(ages)
        system_age_density = smr.system_age_density(pool_age_densities)
        filename = 'age_dens.csv'
        smr.save_pools_and_system_density_csv(filename, pool_age_densities, system_age_density, ages)
        loaded_age_densities = smr.load_pools_and_system_densities_csv(filename, ages)

        self.assertTrue(np.allclose(pool_age_densities, loaded_age_densities[:,:,:2]))

        pool_age_mean = smr.age_moment_vector(1, start_age_moments)
        system_age_mean = smr.system_age_moment(1, start_age_moments)
        smr.save_pools_and_system_value_csv('age_mean.csv', pool_age_mean, system_age_mean)

        loaded_pool_age_mean, loaded_system_age_mean = smr.load_pools_and_system_value_csv('age_mean.csv')
        self.assertTrue(np.allclose(pool_age_mean,loaded_pool_age_mean))
        self.assertTrue(np.allclose(system_age_mean,loaded_system_age_mean))

        filename = 'btt_dens.csv'
        btt_density = smr.backward_transit_time_density(pool_age_densities)
        smr.save_density_csv(filename, btt_density, ages)

        btt_mean = smr.backward_transit_time_moment(1, start_age_moments)
        smr.save_value_csv(filename, btt_mean)


    ##### plotting methods #####

    
    ## solutions ##


    def test_plot_solutions(self):
        fig = plt.figure()
        mr = ESMR.nonlinear_two_pool()
        mr.plot_solutions(fig)
        fig.savefig("plot.pdf")
        plt.close(fig.number)


    def test_plot_phase_plane(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        mr = ESMR.nonlinear_two_pool()
        mr.plot_phase_plane(ax, 0, 1)
        fig.savefig("plot.pdf")
        plt.close(fig.number)
    

    def test_plot_phase_planes(self):
        fig = plt.figure()
        mr = ESMR.emanuel_1()
        mr.plot_phase_planes(fig)
        fig.savefig("plot.pdf")
        plt.close(fig.number)
    

    ## fluxes ##


    def test_plot_internal_fluxes(self):
        fig = plt.figure()
        smr = ESMR.nonlinear_two_pool()
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        smr.plot_internal_fluxes(fig)
        fig.savefig("plot.pdf")
        plt.close(fig.number)


    def test_plot_external_output_fluxes(self):
        smr = ESMR.nonlinear_two_pool()
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        fig = plt.figure()
        smr.plot_external_output_fluxes(fig)
        fig.savefig("plot.pdf")
        plt.close(fig.number)


    def test_plot_external_input_fluxes(self):
        smr = ESMR.nonlinear_two_pool()
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        fig = plt.figure()
        smr.plot_external_input_fluxes(fig)
        fig.savefig("plot.pdf")
        plt.close(fig.number)


    ## means ##


    def test_plot_mean_ages(self):
        smr = ESMR.critics()
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        fig = plt.figure()
        smr.plot_mean_ages(fig, np.array([0,0]))
        fig.savefig("plot.pdf")
        plt.close(fig.number)


    def test_plot_mean_backward_transit_time(self):
        smr = ESMR.critics()
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        smr.plot_mean_backward_transit_time(ax, np.array([0,0]))
        fig.savefig("plot.pdf")
        plt.close(fig.number)



    ## densities ##


    # age #

    #fixme: make it work
#    def test_plotly(self):
#        # two-dimensional
#        C_0, C_1 = symbols('C_0 C_1')
#        state_vector = [C_0, C_1]
#        time_symbol = Symbol('t')
#        input_fluxes = {0:1+1/4*sin(time_symbol), 1: 3}
#        output_fluxes = {0: C_0}
#        internal_fluxes = {(1,0): 1/6*C_1}
#        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)
#
#        start_values = np.array([1, 3])
#        times = np.linspace(0,10,11)
#        smr = SmoothModelRun(srm, {}, start_values, times)
#
#        ages = np.linspace(0,10,11)
#        start_age_densities = lambda a: np.exp(-a)*start_values
#        p = smr.pool_age_densities_func(start_age_densities)
#        age_densities = p(ages)
#        start_mean_ages = [1,1]
#        pool = 0        
#
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        smr.plot_age_density_pool(ax, pool, age_densities, start_age_densities, ages, start_mean_ages) 
#
#        fig.savefig('testfig.pdf') 


    ##### cumulative distribution methods #####


    def test_cumulative_pool_age_distributions_single_value(self):
        # two-dimensional, no inputs
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 0, 1: 0}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        F_sv = smr.cumulative_pool_age_distributions_single_value(start_age_densities)

        ref = np.array([5-5*np.exp(-1), 3-3*np.exp(-1)])
        self.assertTrue(np.allclose(F_sv(1, 0), ref,rtol=1e-3))

        ref = np.array([(5-5*np.exp(-1))*np.exp(-1), (3-3*np.exp(-1))*np.exp(-1)])
        self.assertTrue(np.allclose(F_sv(2, 1), ref,rtol=1e-3))

        # two-dimensional, empty start system
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 0}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([0, 0])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        F_sv = smr.cumulative_pool_age_distributions_single_value(start_age_densities)
        ref = np.array([(1.0-np.exp(-1))-np.exp(-1.0/2)*(1-np.exp(-1.0/2)),0])
        self.assertTrue(np.allclose(F_sv(0.5, 1), ref))

        # two-dimensional, nonempty start system, input to first pool
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 0}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        F_sv = smr.cumulative_pool_age_distributions_single_value(start_age_densities)
        ref = np.array([(5-5*np.exp(-0.5))*np.exp(-0.5) + (1.0-np.exp(-0.5)),
                        (3-3*np.exp(-0.5))*np.exp(-0.5)])
        self.assertTrue(np.allclose(F_sv(1, 0.5), ref))


    def test_cumulative_system_age_distribution_single_value(self):
        # two-dimensional, nonempty start system, input to first pool
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 0}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        F_sv = smr.cumulative_system_age_distribution_single_value(start_age_densities)
        ref = np.array([(5-5*np.exp(-0.5))*np.exp(-0.5) + (1.0-np.exp(-0.5)),
                        (3-3*np.exp(-0.5))*np.exp(-0.5)])
        self.assertTrue(np.allclose(F_sv(1, 0.5), ref.sum()))


    def test_pool_age_distributions_quantiles(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 0, 1: 1}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(
            state_vector,
            time_symbol,
            input_fluxes,
            output_fluxes,
            internal_fluxes
        )

        start_values = np.array([1, 0])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        def start_age_densities(a):
            return np.exp(-a)*start_values
        
        # compute the median with different numerical methods
        start_age_moments = smr.moments_from_densities(1, start_age_densities)
        start_values_q = smr.age_moment_vector(1, start_age_moments)
        a_star_newton = smr.pool_age_distributions_quantiles(
            0.5,
            start_values=start_values_q,
            start_age_densities=start_age_densities,
            method='newton'
        )
        a_star_brentq = smr.pool_age_distributions_quantiles(
            0.5,
            start_age_densities=start_age_densities,
            method='brentq'
        )
        self.assertTrue(
            np.allclose(
                a_star_newton[:,0],
                np.log(2)+times,
                rtol=1e-3
            )
        )
        self.assertTrue(
            np.allclose(
                a_star_brentq[:,0],
                np.log(2)+times,
                rtol=1e-3
            )
        )

        a, t = symbols('a t')
        ref_sym = solve(Eq(1/2*(1-exp(-t)), 1 - exp(-a)), a)[0]
        ref = np.array(
            [ref_sym.subs({t: time}) for time in times],
            dtype=float
        )
        ref[0] = np.nan
        
        self.assertTrue(
            np.allclose(a_star_newton[:,1],
                ref,
                equal_nan=True,
                rtol=1e-03
            )
        )
        self.assertTrue(
            np.allclose(
                a_star_brentq[:,1],
                ref,
                equal_nan=True,
                rtol=1e-03
            )
        )

    
    def test_distribution_quantile(self):
        F = lambda a: 1-np.exp(-a)
        q = SmoothModelRun.distribution_quantile(
            quantile = 0.5,
            F = F,
            norm_const = None,
            method = 'brentq'
        )
        self.assertEqual(round(q, 5), round(np.log(2), 5))   

        F = lambda a: (1-np.exp(-a)) * 17
        q = SmoothModelRun.distribution_quantile(
            quantile = 0.5,
            F = F,
            norm_const = 17,
            start_value = 1.0,
            method = 'brentq'
        )
        self.assertEqual(round(q, 5), round(np.log(2), 5))   
 
    def test_pool_age_distributions_quantiles_by_ode(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 0, 1: 1}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1, 0])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        
        # compute the median with different numerical methods
        start_age_moments = smr.moments_from_densities(1, start_age_densities)
        start_values_q = smr.age_moment_vector(1, start_age_moments)
        a_star = smr.pool_age_distributions_quantiles_by_ode(0.5, start_age_densities=start_age_densities)
        self.assertTrue(np.allclose(a_star[:,0], np.log(2)+times,rtol=1e-3))

        a, t = symbols('a t')
        ref_sym = solve(Eq(1/2*(1-exp(-t)), 1 - exp(-a)), a)[0]
        ref = np.array([ref_sym.subs({t: time}) for time in times], dtype=float)
        ref[0] = np.nan
       
        self.assertTrue(
            np.allclose(
                a_star[:,1],
                ref,
                equal_nan=True,
                rtol=1e-03
            )
        )

    
    def test_system_age_distribution_quantiles(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 0, 1: 1}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1, 0])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        
        a_star_newton = smr.system_age_distribution_quantiles(0.5, start_age_densities=start_age_densities, method='newton')
        a_star_brentq = smr.system_age_distribution_quantiles(0.5, start_age_densities=start_age_densities, method='brentq')
        
        self.assertTrue(np.allclose(a_star_newton, np.log(2),rtol=1e-3))
        self.assertTrue(np.allclose(a_star_brentq, np.log(2),rtol=1e-3))

        # test empty start_system
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 0, 1: 1}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([0, 0])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        
        a_star = smr.system_age_distribution_quantiles(0.5, start_age_densities=start_age_densities)
        self.assertTrue(np.isnan(a_star[0]))


    def test_system_age_distribution_quantiles_by_ode(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 0, 1: 1}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1, 0])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        #F0 = lambda s: (1-np.exp(-s))*start_values
        a_star = smr.system_age_distribution_quantiles_by_ode(
            0.5, 
            start_age_densities,
            max_step=0.1
        )
        self.assertTrue(np.allclose(a_star, np.log(2),rtol=1e-3))

        # test empty start_system
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 0, 1: 1}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([0, 0])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        
        a_star = smr.system_age_distribution_quantiles_by_ode(
            0.5,
            start_age_densities,
            max_step=0.01
        )
        self.assertTrue(np.isnan(a_star[0]))

        # test steady state
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 1}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1, 1])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        a_star = smr.system_age_distribution_quantiles_by_ode(
            0.5,
            start_age_densities,
            max_step=0.1
        )
        self.assertTrue(np.allclose(a_star, np.log(2)))


    ########## private methods ########## 
             
    
    def test_solve_age_moment_system_single_value_old(self):
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)
        t_end = 1
        t_mid=(t_end-0)/2
        start_values = np.array([1,1])
        smr = SmoothModelRun(srm, {}, start_values, times=np.linspace(0,t_end ,11))
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a) * start_values
        start_age_moments = smr.moments_from_densities(1, start_age_densities)
    
        # set manually mean age in first pool to zero
        start_age_moments[0,0] = 0

        ams_func = smr._solve_age_moment_system_single_value_old(1, start_age_moments)
        ams = ams_func(t_mid)
        soln = ams[:2]
        self.assertTrue(np.allclose(soln, smr.solve_single_value_old()(t_mid)))
        ma = ams[2:]

        ref = np.array([ 0.26884456,  0.90341213])
        self.assertTrue(np.allclose(ma, ref)) 

        ## test missing start_age_moments
        ams_func = smr._solve_age_moment_system_single_value_old(1)
        ams=ams_func(t_mid)
        soln = ams[:2]
        self.assertTrue(np.allclose(soln, smr.solve_single_value_old()(t_mid)))
        ma = ams[2:]
        ref_ams_func = smr._solve_age_moment_system_single_value_old(1, np.zeros((1,2))) # 1 moment, 2 pools
        ref_ams=ref_ams_func(t_mid)
        ref_ma = ref_ams[2:]
        self.assertTrue(np.allclose(ma, ref_ma))

        # test second order moments!
        start_age_moments = smr.moments_from_densities(2, start_age_densities)
        ams_func = smr._solve_age_moment_system_single_value_old(2, start_age_moments)

        # test missing start_age_moments
        ams_func = smr._solve_age_moment_system_single_value_old(2)

    
    def test_solve_age_moment_system_func(self):
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)
        t_end = 1
        t_mid=(t_end-0)/2
        start_values = np.array([1,1])
        smr = SmoothModelRun(
            srm,
            {},
            start_values,
            times=np.linspace(0, t_end, 11)
        )
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a) * start_values
        start_age_moments = smr.moments_from_densities(1, start_age_densities)
    
        # set manually mean age in first pool to zero
        start_age_moments[0,0] = 0

        ams_func = smr._solve_age_moment_system_func(1, start_age_moments)
        ams = ams_func(np.array([t_mid,t_end]))
        #check that the time is the first dimension
        self.assertEqual(ams.shape,(2,4))

        ams = ams_func(t_mid)
        soln = ams[:2]
        self.assertTrue(
            np.allclose(
                soln,
                smr.solve_func()(t_mid),
                rtol=1e-03
            )
        )
        ma = ams[2:]
        ref = np.array([0.26884337, 0.90340101])
        self.assertTrue(np.allclose(ma, ref)) 

        ## test missing start_age_moments
        ams_func = smr._solve_age_moment_system_func(1)
        ams = ams_func(t_mid)
        soln = ams[:2]
        self.assertTrue(
            np.allclose(
                soln,
                smr.solve_func()(t_mid),
                rtol=1e-03
            )
        )
        ma = ams[2:]
        # 1 moment, 2 pools
        ref_ams_func = smr._solve_age_moment_system_func(1, np.zeros((1,2)))
        ref_ams=ref_ams_func(t_mid)
        ref_ma = ref_ams[2:]
        self.assertTrue(np.allclose(ma, ref_ma))

        # test second order moments!
        start_age_moments = smr.moments_from_densities(2, start_age_densities)
        ams_func = smr._solve_age_moment_system_func(2, start_age_moments)

        # test missing start_age_moments
        ams_func = smr._solve_age_moment_system_func(2)

    def test_solve_age_moment_system_old(self):
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([1,1])
        smr = SmoothModelRun(srm, {}, start_values, times=np.linspace(0,1,10))
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a) * start_values
        start_age_moments = smr.moments_from_densities(1, start_age_densities)
    
        # set manually mean age in first pool to zero
        start_age_moments[0,0] = 0

        ams = smr._solve_age_moment_system_old(1, start_age_moments)
        soln = ams[:,:2]
        self.assertTrue(
            np.allclose(
                soln,
                smr.solve(),
                rtol=1e-03
            )
        )
        ma = ams[:,2:]

        a_ref = np.array([[ 0.        ,  1.        ], 
                          [ 0.08202606,  0.99407994],
                          [ 0.14256583,  0.97750078],
                          [ 0.19599665,  0.95232484],
                          [ 0.24534554,  0.92082325],
                          [ 0.29165801,  0.8852548 ],
                          [ 0.33540449,  0.84768088],
                          [ 0.37683963,  0.80984058],
                          [ 0.41612357,  0.77309132],
                          [ 0.45337056,  0.73840585]])
        ref = np.ndarray((10,2), float, a_ref)
        self.assertTrue(np.allclose(ma, ref)) 

        # test missing start_age_moments
        ams = smr._solve_age_moment_system_old(1)
        soln = ams[:,:2]
        self.assertTrue(
            np.allclose(
                soln,
                smr.solve(),
                rtol=1e-03
            )
        )
        ma = ams[:,2:]
        ref_ams = smr._solve_age_moment_system_old(1, np.zeros((1,2))) # 1 moment, 2 pools
        ref_ma = ref_ams[:,2:]
        self.assertTrue(np.allclose(ma, ref_ma))

        # test second order moments!
        start_age_moments = smr.moments_from_densities(2, start_age_densities)
        ams = smr._solve_age_moment_system_old(2, start_age_moments)

        # test missing start_age_moments
        ams = smr._solve_age_moment_system_old(2)


    def test_solve_age_moment_system(self):
        x, y, t = symbols("x y t")
        nr_pools = 2
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(nr_pools, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([1,1])
        smr = SmoothModelRun(srm, {}, start_values, times=np.linspace(0,1,10))
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a) * start_values
        start_age_moments = smr.moments_from_densities(1, start_age_densities)
    
        # set manually mean age in first pool to zero
        start_age_moments[0,0] = 0

        ams,_ = smr._solve_age_moment_system(1, start_age_moments)
        soln = ams[:,:nr_pools]
        self.assertTrue(
                np.allclose(soln, smr.solve(), rtol=1e-03)
        )
        ma = ams[:,nr_pools:]


        #a_ref = np.array([[ 0.        ,  1.        ], 
        #                  [ 0.08202606,  0.99407994],
        #                  [ 0.14256583,  0.97750078],
        #                  [ 0.19599665,  0.95232484],
        #                  [ 0.24534554,  0.92082325],
        #                  [ 0.29165801,  0.8852548 ],
        #                  [ 0.33540449,  0.84768088],
        #                  [ 0.37683963,  0.80984058],
        #                  [ 0.41612357,  0.77309132],
        #                  [ 0.45337056,  0.73840585]])

        a_ref = np.array([[0.        , 1.        ], 
                          [0.08202136, 0.99408002],
                          [0.14256488, 0.97750095],
                          [0.19599202, 0.95232419],
                          [0.24535025, 0.92082419],
                          [0.29165268, 0.88523234],
                          [0.33540979, 0.8477071 ],
                          [0.37684872, 0.80984795],
                          [0.41613677, 0.77309118],
                          [0.45338235, 0.7384063 ]])
        ref = np.ndarray((10,nr_pools), float, a_ref)
        self.assertTrue(np.allclose(ma, ref)) 


        # test missing start_age_moments
        ams,_ = smr._solve_age_moment_system(1)
        soln = ams[:,:nr_pools]
        self.assertTrue(np.allclose(soln, smr.solve(), rtol=1e-03))
        ma = ams[:,nr_pools:]
        ref_ams,_ = smr._solve_age_moment_system(1, np.zeros((1,nr_pools))) # 1 moment, nr_pools pools
        ref_ma = ref_ams[:,nr_pools:]
        self.assertTrue(np.allclose(ma, ref_ma))

        # test second order moments!
        start_age_moments = smr.moments_from_densities(nr_pools, start_age_densities)
        ams,_ = smr._solve_age_moment_system(nr_pools, start_age_moments)

        # test missing start_age_moments
        ams,_ = smr._solve_age_moment_system(nr_pools)


    def test_output_rate_vector_at_t(self):
        # two-dimensional case
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1,3])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        self.assertTrue(np.allclose(smr.output_rate_vector_at_t(1), np.array([1, 1])))
        
    
    def test_output_rate_vector(self):
        # one-dimensional case
        C_0 = symbols('C_0')
        state_vector = [C_0]
        time_symbol = Symbol('t')
        input_fluxes = {}
        output_fluxes = {0: C_0}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1])
        times = np.linspace(0, 1, 11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ref = np.ones((11, 1))
        self.assertTrue(np.allclose(smr.output_rate_vector, ref))

        # two-dimensional case
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {}
        output_fluxes = {0: time_symbol*C_0, 1: 2*C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1,3])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ref_a = np.array([[ 1.        ,  3.        ], 
                          [ 0.90483744,  2.71451231],
                          [ 0.81873077,  2.4561923 ],
                          [ 0.74081821,  2.22245462],
                          [ 0.67032006,  2.01096019],
                          [ 0.60653067,  1.81959201],
                          [ 0.54881165,  1.64643494],
                          [ 0.49658532,  1.48975595],
                          [ 0.44932898,  1.34798693],
                          [ 0.40656968,  1.21970905],
                          [ 0.36787945,  1.10363835]])
        ref = np.ones((11, 2))
        ref[:,0] = np.linspace(0, 1, 11)
        ref[:,1] = 2
        self.assertTrue(np.allclose(smr.output_rate_vector, ref))


    ##### age density methods #####
    
    
    def test_age_densities_1_single_value(self):
        # one-dimensional
        C = Symbol('C')
        state_vector = [C]
        time_symbol = Symbol('t')
        input_fluxes = {}
        output_fluxes = {0: C}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        p1_sv = smr._age_densities_1_single_value(start_age_densities)

        # negative ages will be cut off automatically
        ages = np.linspace(-1,1,3)
        a_ref = np.array([[[ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ]],
                         
                          [[ 5.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ]],
                         
                          [[ 1.83939721],
                           [ 1.83939724],
                           [ 1.83939725],
                           [ 1.83939724],
                           [ 1.83939729],
                           [ 1.83939727]]])
                 
        ref = np.ndarray((3,6,1), float, a_ref)
        res_l = [[p1_sv(a, t) for t in times] for a in ages]
        res = np.array(res_l)
#        pe('(res-ref)/res',locals())
        self.assertTrue(np.allclose(res, ref,rtol=1e-3))

        # test missing start_age_densities
        a_ref = np.array([[[ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ]],
                         
                          [[ 5.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ]],
                         
                          [[ 0         ],
                           [ 0         ],
                           [ 0         ],
                           [ 0         ],
                           [ 0         ],
                           [ 1.83939727]]])
        
        ref = np.ndarray((3,6,1), float, a_ref)
        p1_sv = smr._age_densities_1_single_value()
        res_l = [[p1_sv(a,t) for t in times] for a in ages]
        res = np.array(res_l)
        self.assertTrue(np.allclose(res, ref,rtol=1e-3))

        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        start_age_densities = lambda a: np.exp(-a)*start_values
        p1_sv = smr._age_densities_1_single_value(start_age_densities)

        a_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 5.        ,  3.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 1.83939721,  1.10363832],
                  [ 1.83939724,  1.10363834],
                  [ 1.83939725,  1.10363835],
                  [ 1.83939724,  1.10363835],
                  [ 1.83939729,  1.10363837],
                  [ 1.83939727,  1.10363836]]])

        ref = np.ndarray((3,6,2), float, a_ref)
        res_l = [[p1_sv(a,t) for t in times] for a in ages]
        res = np.array(res_l)
        self.assertTrue(np.allclose(res, ref,rtol=1e-3))

        # test missing start_age_densities
        a_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 5.        ,  3.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 1.83939727,  1.10363836]]])

        ref = np.ndarray((3,6,2), float, a_ref)
        p1_sv = smr._age_densities_1_single_value()
        res_l = [[p1_sv(a,t) for t in times] for a in ages]
        res = np.array(res_l)
        self.assertTrue(np.allclose(res, ref,rtol=1e-3))


    def test_age_densities_1(self):
        # one-dimensional
        C = Symbol('C')
        state_vector = [C]
        time_symbol = Symbol('t')
        input_fluxes = {}
        output_fluxes = {0: C}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        p1 = smr._age_densities_1(start_age_densities)

        # negative ages will be cut off automatically
        ages = np.linspace(-1,1,3)
        a_ref = np.array([[[ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ]],
                         
                          [[ 5.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ]],
                         
                          [[ 1.83939721],
                           [ 1.83939724],
                           [ 1.83939725],
                           [ 1.83939724],
                           [ 1.83939729],
                           [ 1.83939727]]])
                 
        ref = np.ndarray((3,6,1), float, a_ref)
        self.assertTrue(np.allclose(p1(ages), ref,rtol=1e-3))

        # test missing start_age_densities
        a_ref = np.array([[[ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ]],
                         
                          [[ 5.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ]],
                         
                          [[ 0         ],
                           [ 0         ],
                           [ 0         ],
                           [ 0         ],
                           [ 0         ],
                           [ 1.83939727]]])
        
        ref = np.ndarray((3,6,1), float, a_ref)
        p1 = smr._age_densities_1()
        res = np.array(p1(ages))
        self.assertTrue(np.allclose(res, ref,rtol=1e-3))

        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        start_age_densities = lambda a: np.exp(-a)*start_values
        p1 = smr._age_densities_1(start_age_densities)

        a_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 5.        ,  3.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 1.83939721,  1.10363832],
                  [ 1.83939724,  1.10363834],
                  [ 1.83939725,  1.10363835],
                  [ 1.83939724,  1.10363835],
                  [ 1.83939729,  1.10363837],
                  [ 1.83939727,  1.10363836]]])

        ref = np.ndarray((3,6,2), float, a_ref)
        self.assertTrue(np.allclose(p1(ages), ref,rtol=1e-3))

        # test missing start_age_densities
        a_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 5.        ,  3.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 1.83939727,  1.10363836]]])

        ref = np.ndarray((3,6,2), float, a_ref)
        p1 = smr._age_densities_1()
        res = np.array(p1(ages))
        self.assertTrue(np.allclose(res, ref,rtol=1e-3))


    def test_age_densities_2_single_value(self):
        # one-dimensional
        C = Symbol('C')
        state_vector = [C]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1}
        output_fluxes = {0: C}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        p2_sv = smr._age_densities_2_single_value()

        # negative ages will be cut off automatically
        ages = np.linspace(-1,1,3)
        a_ref = np.array([[[ 0. ],
                           [ 0. ],
                           [ 0. ],
                           [ 0. ],
                           [ 0. ],
                           [ 0. ]],
                         
                          [[ 0. ],
                           [ 1. ],
                           [ 1. ],
                           [ 1. ],
                           [ 1. ],
                           [ 1. ]],
                         
                          [[ 0. ],
                           [ 0. ],
                           [ 0. ],
                           [ 0. ],
                           [ 0. ],
                           [ 0. ]]])
                 
        ref = np.ndarray((3,6,1), float, a_ref)
        res_l = [[p2_sv(a, t) for t in times] for a in ages]
        res = np.array(res_l)
        self.assertTrue(np.allclose(res, ref))

        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        p2_sv = smr._age_densities_2_single_value()

        a_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 0.        ,  0.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ]],
                
                 [[ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ]]])

        ref = np.ndarray((3,6,2), float, a_ref)
        res_l = [[p2_sv(a, t) for t in times] for a in ages]
        res = np.array(res_l)
        self.assertTrue(np.allclose(res, ref))


    def test_age_densities_2(self):
        # one-dimensional
        C = Symbol('C')
        state_vector = [C]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1}
        output_fluxes = {0: C}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        p2 = smr._age_densities_2()

        # negative ages will be cut off automatically
        ages = np.linspace(-1,1,3)
        a_ref = np.array([[[ 0. ],
                           [ 0. ],
                           [ 0. ],
                           [ 0. ],
                           [ 0. ],
                           [ 0. ]],
                         
                          [[ 0. ],
                           [ 1. ],
                           [ 1. ],
                           [ 1. ],
                           [ 1. ],
                           [ 1. ]],
                         
                          [[ 0. ],
                           [ 0. ],
                           [ 0. ],
                           [ 0. ],
                           [ 0. ],
                           [ 0. ]]])
                 
        ref = np.ndarray((3,6,1), float, a_ref)
        self.assertTrue(np.allclose(p2(ages), ref))

        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        p2 = smr._age_densities_2()

        a_ref = np.array(
                [[[ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ],
                  [ 0.        ,  0.        ]],
                
                 [[ 0.        ,  0.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ],
                  [ 1.        ,  2.        ]],
                
                 [[ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ],
                  [ 0         ,  0         ]]])

        ref = np.ndarray((3,6,2), float, a_ref)
        self.assertTrue(np.allclose(p2(ages), ref))

    
    ##### plot methods #####


    def test_density_plot(self):
        # actually tested by
        #   test_plot_age_density_pool
        #   test_plot_age_densities
        #   test_plot_system_age_densities
        pass


    ##### 14C methods #####


#    def test_to_14C_explicit(self):
#        # we test only that the construction works and can be solved,
#        # not the actual solution values
#        lamda_1, lamda_2, C_1, C_2 = symbols('lamda_1 lamda_2 C_1 C_2')
#        B = Matrix([[-lamda_1,        0],
#                    [       0, -lamda_2]])
#        u = Matrix(2, 1, [1, 1])
#        state_vector = Matrix(2, 1, [C_1, C_2])
#        time_symbol = Symbol('t')
#
#        srm = SmoothReservoirModel.from_B_u(
#            state_vector,
#            time_symbol,
#            B,
#            u
#        )
#
#        par_set = {lamda_1: 0.5, lamda_2: 0.2}
#        start_values = np.array([7,4])
#        start, end, ts = 1950, 2000, 0.5
#        times = np.linspace(start, end, int((end+ts-start)/ts))
#        smr = SmoothModelRun(srm, par_set, start_values, times)
#        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
#        soln = smr.solve()
#
#        atm_delta_14C = np.loadtxt(pfile_C14Atm_NH(), skiprows=1, delimiter=',')
#        F_atm_delta_14C = interp1d(
#            atm_delta_14C[:,0],
#            atm_delta_14C[:,1],
#            fill_value = 'extrapolate'
#        )
#
#        alpha = 1.18e-12
#        start_values_14C = smr.start_values * alpha
#        Fa_func = lambda t: alpha * (F_atm_delta_14C(t)/1000+1)
#        smr_14C = smr.to_14C_explicit(
#            start_values_14C,
#            Fa_func,
#            0.0001
#        )
#
#        soln_14C = smr_14C.solve()


    ##### temporary #####


    def test_FTTT_lambda_bar(self):
        # symmetric case
        lamda_1, lamda_2, C_1, C_2 = symbols('lamda_1 lamda_2 C_1 C_2')
        B = Matrix([[-lamda_1,        0],
                    [       0, -lamda_2]])
        u = Matrix(2, 1, [1/2, 1/2])
        state_vector = Matrix(2, 1, [C_1, C_2])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(state_vector,
                                            time_symbol,
                                            B,
                                            u)

        par_set = {lamda_1: 1, lamda_2: 1}
        start_values = np.array([u[0]/par_set[lamda_1],
                                 u[1]/par_set[lamda_2]]) # steady-state
        start, end = 0, 10
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        result = smr._FTTT_lambda_bar(end, 5, np.array(u).astype(np.float64))
        self.assertTrue(np.allclose(result,1,rtol=1e-3))

        # asymmetric case
        par_set = {lamda_1: 3, lamda_2: 2}
        start_values = np.array([u[0]/par_set[lamda_1],
                                 u[1]/par_set[lamda_2]]) # steady-state
        start, end = 0, 0.000005
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        s = (start+end)/2
        result = smr._FTTT_lambda_bar(end, s, np.array(u).astype(np.float64))
        self.assertEqual(round(result, 5),
                         round(-np.log((np.exp(-par_set[lamda_1]*(end-s))
                                       +np.exp(-par_set[lamda_2]*(end-s)))/2)/
                                        (end-s),
                                5))
        # at the same time we prove that for t1 --> t0
        # lambda_bar --> 1/2(lambda1+lambda2)
        self.assertEqual(round(result, 5),
                         1/2*(par_set[lamda_1]+par_set[lamda_2]))


    @unittest.skip('function for useless approach of Martin')
    def test_FTTT_lambda_bar_R(self):
        # one-dimensional in steady state
        # the result should be lambda
        lamda, I, C = symbols('lamda I C')
        B = Matrix([-lamda])
        u = Matrix([I])
        state_vector = Matrix([C])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(
            state_vector,
            time_symbol,
            B,
            u
        )

        par_set = {lamda: 2/5, I: 7}
        start_values = np.array(par_set[I]/par_set[lamda]) # steady state
        start, end = 0, 10
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        #print(start_values)
        #print('soln', smr.solve())

        result = smr._FTTT_lambda_bar_R(start, end)
        self.assertTrue(np.allclose(result, par_set[lamda],rtol=1e-3))
        

    def test_FTTT_lambda_bar_S(self):
        # one-dimensional in steady state
        # the result should be lambda
        lamda, I, C = symbols('lamda I C')
        B = Matrix([-lamda])
        u = Matrix([I])
        state_vector = Matrix([C])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(state_vector,
                                            time_symbol,
                                            B,
                                            u)

        par_set = {lamda: 2/5, I: 7}
        start_values = np.array(par_set[I]/par_set[lamda]) # steady state
        start, end = 0, 10
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        result = smr._FTTT_lambda_bar_S(start, end)
        self.assertEqual(round(result, 5), par_set[lamda])

        
    def test_FTTT_lambda_bar_R_left_limit(self):
        # symmetric case
        lamda_1, lamda_2, C_1, C_2 = symbols('lamda_1 lamda_2 C_1 C_2')
        B = Matrix([[-lamda_1,        0],
                    [       0, -lamda_2]])
        u = Matrix(2, 1, [1/2, 1/2])
        state_vector = Matrix(2, 1, [C_1, C_2])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(state_vector,
                                            time_symbol,
                                            B,
                                            u)

        par_set = {lamda_1: 1, lamda_2: 1}
        start_values = np.array([u[0]/par_set[lamda_1],
                                 u[1]/par_set[lamda_2]]) # steady-state
        start, end = 0, 10
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        result = smr._FTTT_lambda_bar_R_left_limit(start)
        self.assertEqual(result, 1*1/2+1*1/2)

        # asymmetric case
        par_set = {lamda_1: 3, lamda_2: 2}
        start_values = np.array([u[0]/par_set[lamda_1],
                                 u[1]/par_set[lamda_2]]) # steady-state
        start, end = 0, 10
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        result = smr._FTTT_lambda_bar_R_left_limit(start)
        self.assertEqual(result, (3*1/6+2*1/4)/(1/6+1/4)),


    def test_alpha_s_i_and_alpha_s(self):
        # symmetric case
        lamda_1, lamda_2, C_1, C_2 = symbols('lamda_1 lamda_2 C_1 C_2')
        B = Matrix([[-lamda_1,        0],
                    [       0, -lamda_2]])
        u = Matrix(2, 1, [1/2, 1/2])
        state_vector = Matrix(2, 1, [C_1, C_2])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(state_vector,
                                            time_symbol,
                                            B,
                                            u)

        par_set = {lamda_1: 1, lamda_2: 1}
        start_values = np.array([u[0]/par_set[lamda_1],
                                 u[1]/par_set[lamda_2]]) # steady-state
        start, end = 0, 10
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        s = 5
        t1 = 6
        result1 = 0
        u_val = smr.external_input_vector_func()(s)
        for i in range(smr.nr_pools):
            result1 += u_val[i] * smr._alpha_s_i(s, i, t1)
        result1 /= u_val.sum()

        result2 = smr._alpha_s(s, t1, u_val)

        self.assertTrue(np.allclose(result1, 1-np.exp(-(t1-s)),rtol=1e-3))
        self.assertTrue(np.allclose(result1, result2,rtol=1e-3))

        # asymmetric case
        par_set = {lamda_1: 3, lamda_2: 2}
        start_values = np.array([u[0]/par_set[lamda_1],
                                 u[1]/par_set[lamda_2]]) # steady-state
        start, end = 0, 10
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        s = 5
        t1 = 6
        result1 = 0
        u_val = smr.external_input_vector_func()(s)
        for i in range(smr.nr_pools):
            result1 += u_val[i] * smr._alpha_s_i(s, i, t1)
        result1 /= u_val.sum()

        result2 = smr._alpha_s(s, t1, u_val)

        self.assertEqual(round(result1,3), 
            round(1-(np.exp(-par_set[lamda_1]*(t1-s))*u_val[0]+
                     np.exp(-par_set[lamda_2]*(t1-s))*u_val[1])/
                    u_val.sum(), 3))
        self.assertEqual(round(result1,3), round(result2,3))


    def test_EFFTT_s_i(self):
        # asymmetric case
        lamda_1, lamda_2, C_1, C_2 = symbols('lamda_1 lamda_2 C_1 C_2')
        B = Matrix([[-lamda_1,        0],
                    [       0, -lamda_2]])
        u = Matrix(2, 1, [1/2, 1/2])
        state_vector = Matrix(2, 1, [C_1, C_2])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(state_vector,
                                            time_symbol,
                                            B,
                                            u)

        par_set = {lamda_1: 3, lamda_2: 2}
        start_values = np.array([u[0]/par_set[lamda_1],
                                 u[1]/par_set[lamda_2]]) # steady-state
        start, end = 0, 10
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        s = 5
        t1 = 6
        
        result0 = smr._EFFTT_s_i(s, 0, t1)
        ref=(np.exp(3)-4)/(3*np.exp(3)-3)
#        pe('(result0,ref)',locals())
        self.assertEqual(round(result0,3), 
                         round(ref,3))
        result1 = smr._EFFTT_s_i(s, 1, t1)
        self.assertEqual(round(result1,5), 
                         round((np.exp(2)-3)/(2*np.exp(2)-2),5))
    
        # check that splitting over the pools is necessary
        u_val = smr.external_input_vector_func()(s)
        Phi = smr._state_transition_operator

        def F_FTT(a):
            return 1 - Phi(s+a,s, u_val).sum()/u_val.sum()

        alpha_s = smr._alpha_s(s, t1, u_val)
        def integrand(a):
            return 1 - F_FTT(a)/alpha_s
      
        comb_res = quad(integrand, 0, t1-s)[0]
      
        self.assertFalse(comb_res == 1/2*result0+1/2*result1)


    @unittest.skip("result of the test unclear, function under test unused, probably finite time transit time stuff....")
    def test_TR(self):
        # asymmetric case
        lamda_1, lamda_2, C_1, C_2 = symbols('lamda_1 lamda_2 C_1 C_2')
        B = Matrix([[-lamda_1,        0],
                    [       0, -lamda_2]])
        u = Matrix(2, 1, [1/2, 1/2])
        state_vector = Matrix(2, 1, [C_1, C_2])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(state_vector,
                                            time_symbol,
                                            B,
                                            u)

        par_set = {lamda_1: 3, lamda_2: 2}
        start_values = np.array([u[0]/par_set[lamda_1],
                                 u[1]/par_set[lamda_2]]) # steady-state
        start, end = 0, 10
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        s = 5
        t1 = 6
        
        o = ones(smr.nr_pools, 1)
        Phi = smr._state_transition_operator
        v = Phi(t1, s, np.array(u).astype(np.float64))
        v_normed = v/sum(v)
        res=smr._TR(s, t1, v)-(t1-s)
        ref=(-o.T * (B.subs(par_set)**-1) * v_normed)[0]
        #print("#################################################")
        #print(res)
        #print(ref)
        #print(type(ref))
        #print("#################################################")
        self.assertEqual(round(res, 5), np.around(ref, 5)) 


    def test_FTTT_finite_plus_remaining(self):
        # asymmetric case
        lamda_1, lamda_2, C_1, C_2 = symbols('lamda_1 lamda_2 C_1 C_2')
        B = Matrix([[-lamda_1,        0],
                    [       0, -lamda_2]])
        u = Matrix(2, 1, [1/2, 1/2])
        state_vector = Matrix(2, 1, [C_1, C_2])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(state_vector,
                                            time_symbol,
                                            B,
                                            u)

        par_set = {lamda_1: 3, lamda_2: 2}
        start_values = np.array([u[0]/par_set[lamda_1],
                                 u[1]/par_set[lamda_2]], dtype=np.float64) # steady-state
        start, end = 0, 10
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        # dealing with s in the middle of the interval
        s = 5
        t1 = 6
        t0 = 0
        u_val = smr.external_input_vector_func()(s)

        n = smr.nr_pools
        alpha_s_is = [smr._alpha_s_i(s, i, t1) for i in range(n)]
        EFFT_s_is = [smr._EFFTT_s_i(s, i, t1) for i in range(n)]
        finite = sum([u_val[i] * alpha_s_is[i] * EFFT_s_is[i] 
                            for i in range(n)])

        Phi = smr._state_transition_operator
        v = Phi(t1,s,u_val)
        alpha_s = sum([u_val[i] * alpha_s_is[i] 
                            for i in range(n)])/u_val.sum()
        remaining = (1-alpha_s) * u_val.sum() * smr._TR(s, t1, v)


        self.assertEqual(round(smr._FTTT_finite_plus_remaining(s,t1,t0),3), 
                         round(finite+remaining,3))

        # dealing with s at the beginning of the interval
        s = 0
        t1 = 6
        t0 = 0
        u_val = smr.start_values

        n = smr.nr_pools
        alpha_s_is = [smr._alpha_s_i(s, i, t1) for i in range(n)]
        EFFT_s_is = [smr._EFFTT_s_i(s, i, t1) for i in range(n)]
        finite = sum([u_val[i] * alpha_s_is[i] * EFFT_s_is[i] 
                            for i in range(n)])

        Phi = smr._state_transition_operator
        v = Phi(t1,s,u_val)
        alpha_s = sum([u_val[i] * alpha_s_is[i] 
                            for i in range(n)])/u_val.sum()
        remaining = (1-alpha_s) * u_val.sum() * smr._TR(s, t1, v)


        self.assertEqual(round(smr._FTTT_finite_plus_remaining(s,t1,t0),4), 
                         round(finite+remaining,4))

    @unittest.skip("A function for Martin's idea, not used now")
    def test_FTTT_conditional(self):
        # one-dimensional autonomous case
        lamda, C = symbols('lamda C')
        B = Matrix([[-lamda]])
        u = Matrix(1, 1, [1])
        state_vector = Matrix(1, 1, [C])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(state_vector,
                                            time_symbol,
                                            B,
                                            u)

        par_set = {lamda: 1}
        start_values = np.array([u[0]/par_set[lamda]], dtype=np.float64) # steady-state
        start, end = 0, 10
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        t0 = 0
        t1 = 1
        result = np.array([smr._FTTT_conditional(t1,t0) 
                                for t1 in range(1, 11)])
        
        self.assertTrue(np.allclose(result, np.array([1]*10),rtol=1e-3))

        # symmetric case
        lamda_1, lamda_2, C_1, C_2 = symbols('lamda_1 lamda_2 C_1 C_2')
        B = Matrix([[-lamda_1,        0],
                    [       0, -lamda_2]])
        u = Matrix(2, 1, [1, 2])
        state_vector = Matrix(2, 1, [C_1, C_2])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(state_vector,
                                            time_symbol,
                                            B,
                                            u)

        par_set = {lamda_1: 1, lamda_2: 1}
        start_values = np.array([u[0]/par_set[lamda_1],
                                 u[1]/par_set[lamda_2]], dtype=np.float64) # steady-state
        start, end = 0, 10
        times = np.linspace(start, end, 101)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        t0 = 0
        result = np.array([smr._FTTT_conditional(t1,t0) 
                                for t1 in range(1, 11)])
        
        self.assertTrue(np.allclose(result, np.array([1]*10),rtol=1e-3))


        # asymmetric case
        lamda_1, lamda_2, C_1, C_2 = symbols('lamda_1 lamda_2 C_1 C_2')
        B = Matrix([[-lamda_1,        0],
                    [       0, -lamda_2]])
        u = Matrix(2, 1, [1, 1])
        state_vector = Matrix(2, 1, [C_1, C_2])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(state_vector,
                                            time_symbol,
                                            B,
                                            u)

        par_set = {lamda_1: 2, lamda_2: 1}
        start_values = np.array([u[0]/par_set[lamda_1],
                                 u[1]/par_set[lamda_2]], dtype=np.float64) # steady-state
        start, end = 0, 22
        times = np.linspace(start, end, 11)
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        t0 = 0
        result = np.array([smr._FTTT_conditional(t1,t0) 
                                for t1 in times[1:]])
        
        # tested is here only that no error eccurs
        #print(result)
        #self.assertTrue(np.allclose(result, np.array([1]*10)))

if __name__ == "__main__":
    unittest.main()
