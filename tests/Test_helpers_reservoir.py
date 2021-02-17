#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:
from concurrencytest import ConcurrentTestSuite, fork_for_tests
import sys
import unittest
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from sympy import (
    Symbol,
    Matrix,
    symbols,
    sin,
    Piecewise,
    DiracDelta,
    Function,
    simplify,
    zeros
)
from CompartmentalSystems.helpers_reservoir import (
    factor_out_from_matrix,
    parse_input_function,
    melt,
    MH_sampling,
    stride,
    is_compartmental,
    func_subs,
    numerical_function_from_expression,
    in_fluxes_by_index,
    internal_fluxes_by_index,
    out_fluxes_by_index,
    in_fluxes_by_symbol,
    internal_fluxes_by_symbol,
    out_fluxes_by_symbol,
)
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel

class TestHelpers_reservoir(unittest.TestCase):
    def test_in_fluxes_by_index_and_symbol(self):
        C_1, C_2, C_3, u_1, u_2, u_3 = symbols('C_1 C_2 C_3 u_1 u_2 u_3')
        C = Matrix(3, 1, [C_1, C_2, C_3])
        u = Matrix(3, 1, [u_1, u_2, u_3])
        self.assertEqual(
            in_fluxes_by_index(C, u),
            {0: u_1, 1: u_2, 2: u_3}
        )
        self.assertEqual(
            in_fluxes_by_symbol(C, u),
            {C_1: u_1, C_2: u_2, C_3: u_3}
        )

    def test_out_fluxes_by_index_and_symbol(self):
        t,C_1, C_2, C_3, k_1, k_2, k_3, a_12, a_13, a_21, a_23, a_31, a_32, gamma\
        = symbols('t,C_1 C_2 C_3 k_1 k_2 k_3 a_12 a_13 a_21 a_23 a_31 a_32 gamma')
        C = Matrix(3, 1, [C_1, C_2, C_3])
        B = gamma*Matrix([
                [-k_1, a_12, a_13],
                [a_21, -k_2, a_23],
                [a_31, a_32, -k_3]
            ])

        ofbi = out_fluxes_by_index(C, B)
        for key, val in ofbi.items():
            with self.subTest():
                ref_val = {
                    0: C_1*(-a_21*gamma-a_31*gamma+k_1*gamma),
                    1: C_2*(-a_12*gamma-a_32*gamma+k_2*gamma),
                    2: C_3*(-a_13*gamma-a_23*gamma+k_3*gamma)
                }[key]
                self.assertEqual(simplify(val-ref_val), 0)

        ofbi = out_fluxes_by_symbol(C, B)
        for key, val in ofbi.items():
            with self.subTest():
                ref_val = {
                    C_1: C_1*(-a_21*gamma-a_31*gamma+k_1*gamma),
                    C_2: C_2*(-a_12*gamma-a_32*gamma+k_2*gamma),
                    C_3: C_3*(-a_13*gamma-a_23*gamma+k_3*gamma)
                }[key]
                self.assertEqual(simplify(val-ref_val), 0)

    def test_internal_fluxes_by_index_and_symbol(self):
        t,C_1, C_2, C_3, k_1, k_2, k_3, a_12, a_13, a_21, a_23, a_31, a_32, gamma\
        = symbols('t,C_1 C_2 C_3 k_1 k_2 k_3 a_12 a_13 a_21 a_23 a_31 a_32 gamma')
        C = Matrix(3,1, [C_1, C_2, C_3])
        B = gamma*Matrix([
            [-k_1, a_12, a_13],
            [a_21, -k_2, a_23],
            [a_31, a_32, -k_3]
        ])

        self.assertEqual(
            internal_fluxes_by_index(C, B),
            {
                (0, 1): gamma*a_21*C_1, (0, 2): gamma*a_31*C_1,
                (1, 0): gamma*a_12*C_2, (1, 2): gamma*a_32*C_2,
                (2, 0): gamma*a_13*C_3, (2, 1): gamma*a_23*C_3
            }
        )
        
        self.assertEqual(
            internal_fluxes_by_symbol(C, B),
            {
                (C_1, C_2): gamma*a_21*C_1, (C_1, C_3): gamma*a_31*C_1,
                (C_2, C_1): gamma*a_12*C_2, (C_2, C_3): gamma*a_32*C_2,
                (C_3, C_1): gamma*a_13*C_3, (C_3, C_2): gamma*a_23*C_3
            }
        )
             
        # ## test backward conversion to compartmental matrix 
        # B2 = rm.compartmental_matrix
        # u2 = rm.external_inputs
        # self.assertEqual(simplify(u-u2), zeros(*u.shape))
        # self.assertEqual(simplify(B-B2), zeros(*B.shape))


    def test_numerical_function_from_expression(self):
        C_0, C_1, C_2 = symbols('C_0 C_1 C_2')
        t = Symbol('t')
        u_0_sym = Function('u_0')
        u_2_sym = Function('u_2')
        
        u_0_expr = u_0_sym(C_0, C_1, t)
        u_2_expr = u_2_sym(t)
        
        X = Matrix([C_0, C_1, C_2])
        t_min, t_max = 0, 10
        symbolic_input_fluxes = {0: u_0_expr, 2: u_2_expr}
        
        def u0_func(C_0_val, C_1_val, t_val):
            return C_0_val+ C_1_val + t_val
        
        def u2_func(t_val):
            return t_val
        parameter_dict={}
        func_set = {u_0_expr: u0_func, u_2_expr: u2_func}
        
        tup = (C_0,C_1) + (t,)
        u_0_func = numerical_function_from_expression(u_0_expr,tup,parameter_dict,func_set)
        self.assertEqual(u_0_func(1,2,3),1+2+3)
        
        tup = (t,)
        u_2_func=numerical_function_from_expression(u_2_expr,tup,parameter_dict,func_set)
        self.assertEqual(u_2_func(2),2)
        
        # wrong tup: C_1 is not necessary but it does not hurt
        # this behavior is convinient to make everything a variable of 
        # ALL statevariables and time 
        tup = (C_1,t,)
        u_2_func=numerical_function_from_expression(u_2_expr,tup,parameter_dict,func_set)
        # the superflous first argument just does not have any influence
        self.assertEqual(u_2_func(1002103413131,2),2)
        
        # wrong tup: C_0 is missing but necessary
        # this is a real error
        tup = (C_1,t)
        with self.assertRaises(Exception) as e:
            u_0_func=numerical_function_from_expression(u_0_expr,tup,parameter_dict,func_set)

    def test_func_subs(self):
        # t is in the third position
        C_0, C_1  = symbols('C_0 C_1')
        t= Symbol('t')  
        u_0_expr = Function('u_0')(C_0, C_1, t)
        def u_0_func(C_0,C_1,t):
            return (C_0*C_1)**t

        ref=u_0_func(1,2,3)
        u_0_part=func_subs(t,u_0_expr,u_0_func,3)
        res=u_0_part(1,2)
        self.assertEqual(ref,res)
       
        # this time t is in the first position
        u_0_expr = Function('u_0')(t,C_0, C_1)
        def u_0_func(C_0,C_1,t):
            return (C_0*C_1)**t

        ref=u_0_func(1,2,3)
        u_0_part=func_subs(t,u_0_expr,u_0_func,1)
        res=u_0_part(2,3)
        self.assertEqual(ref,res)
    

    def test_parse_input_function(self):
        t = symbols('t')
        u = (1+sin(t) + DiracDelta(2-t) 
            +5*DiracDelta(3-t) + Piecewise((1,t<=1), (2,True)))
        impulses, jump_times = parse_input_function(u, t)
        self.assertEqual(impulses, [{'time': 2, 'intensity': 1}, 
                                    {'time': 3,'intensity': 5}])
        self.assertEqual(jump_times, [1,2,3])

        u = 1
        impulses, jump_times = parse_input_function(u, t)
        self.assertEqual(impulses, [])
        self.assertEqual(jump_times, [])


    def test_factor_out_from_matrix(self):
        gamma, k_1 = symbols('gamma k_1')
        M = Matrix([[12*gamma*k_1, 0], [3*gamma**2, 15*gamma]])
        cf = factor_out_from_matrix(M)

        self.assertEqual(cf, 3*gamma)


    def test_melt(self):
        ndarr = np.arange(24).reshape(3,4,2)
        
        a_ref = [[0,  0,  0,  0], 
                 [0,  0,  1,  1],
                 [0,  1,  0,  2],
                 [0,  1,  1,  3],
                 [0,  2,  0,  4],
                 [0,  2,  1,  5],
                 [0,  3,  0,  6],
                 [0,  3,  1,  7],
                 [1,  0,  0,  8],
                 [1,  0,  1,  9],
                 [1,  1,  0, 10],
                 [1,  1,  1, 11],
                 [1,  2,  0, 12],
                 [1,  2,  1, 13],
                 [1,  3,  0, 14],
                 [1,  3,  1, 15],
                 [2,  0,  0, 16],
                 [2,  0,  1, 17],
                 [2,  1,  0, 18],
                 [2,  1,  1, 19],
                 [2,  2,  0, 20],
                 [2,  2,  1, 21],
                 [2,  3,  0, 22],
                 [2,  3,  1, 23]]
        ref = np.array(a_ref).reshape((24,4))
        melted = melt(ndarr)
        self.assertTrue(np.all(melted==ref))

        ages = np.linspace(0,4,3)
        times = np.linspace(0,0.75,4)
        pools = [0,1]

        a_ref = [[0.,     0.  ,   0.,     0.  ], 
                 [0.,     0.  ,   1.,     1.  ],
                 [0.,     0.25,   0.,     2.  ],
                 [0.,     0.25,   1.,     3.  ],
                 [0.,     0.5 ,   0.,     4.  ],
                 [0.,     0.5 ,   1.,     5.  ],
                 [0.,     0.75,   0.,     6.  ],
                 [0.,     0.75,   1.,     7.  ],
                 [2.,     0.  ,   0.,     8.  ],
                 [2.,     0.  ,   1.,     9.  ],
                 [2.,     0.25,   0.,    10.  ],
                 [2.,     0.25,   1.,    11.  ],
                 [2.,     0.5 ,   0.,    12.  ],
                 [2.,     0.5 ,   1.,    13.  ],
                 [2.,     0.75,   0.,    14.  ],
                 [2.,     0.75,   1.,    15.  ],
                 [4.,     0.  ,   0.,    16.  ],
                 [4.,     0.  ,   1.,    17.  ],
                 [4.,     0.25,   0.,    18.  ],
                 [4.,     0.25,   1.,    19.  ],
                 [4.,     0.5 ,   0.,    20.  ],
                 [4.,     0.5 ,   1.,    21.  ],
                 [4.,     0.75,   0.,    22.  ],
                 [4.,     0.75,   1.,    23.  ]]
        melted = melt(ndarr, [ages, times, pools])
        ref = np.array(a_ref).reshape((24,4))
        self.assertTrue(np.all(melted==ref))

    def test_MH_sampling(self):
        # fixme:
        # fails sometimes because of bad random values
        # perhaps a fixed random seed would help


        # uniform distribution on [0,1]
        PDF = lambda x: 1 if x>=0 and x<=1 else 0
        rvs = MH_sampling(200000, PDF)
        self.assertTrue(abs(np.mean(rvs)-0.5) < 0.05)
        self.assertTrue(abs(np.std(rvs, ddof=1)-np.sqrt(1/12)) < 0.05)

        # exponential distribution
        l = 2
        PDF = lambda x: l*np.exp(-l*x)
        rvs = MH_sampling(100000, PDF, start = 1/l)
        #print(rvs)
        #print(np.mean(rvs))
        #count, bins, ignored = plt.hist(rvs, 100, normed=True)
        #ts = np.linspace(0, 5, 101)
        #plt.plot(ts, [PDF(t) for t in ts], color='red')
        #plt.show()
        self.assertTrue(abs(np.mean(rvs)-1/l) < 0.05)
        self.assertTrue(abs(np.std(rvs, ddof=1)-1/l) < 0.05)


    def test_stride(self):
        data = np.array([i*10+np.linspace(0,9,10) for i in range(20)])
        strided_data = stride(data, (2,4))
        ref = np.array([[   0.,    4.,    8.,    9.], 
                        [  20.,   24.,   28.,   29.],
                        [  40.,   44.,   48.,   49.],
                        [  60.,   64.,   68.,   69.],
                        [  80.,   84.,   88.,   89.],
                        [ 100.,  104.,  108.,  109.],
                        [ 120.,  124.,  128.,  129.],
                        [ 140.,  144.,  148.,  149.],
                        [ 160.,  164.,  168.,  169.],
                        [ 180.,  184.,  188.,  189.],
                        [ 190.,  194.,  198.,  199.]])
        self.assertTrue(np.all(strided_data==ref))

        times = np.linspace(0,100,101)
        strided_times = stride(times, 25)
        self.assertTrue(np.all(strided_times==np.array([0, 25, 50, 75, 100])))
        
        strided_times = stride(times, 1)
        self.assertTrue(np.all(strided_times==times))

    def test_is_compartmental(self):
        k=Symbol('k',positive=True, real=True)
        l=Symbol('l',positive=True, real=True)
        M=Matrix([[-k,0],[0,-l]])
        self.assertTrue(is_compartmental(M))



################################################################################


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

