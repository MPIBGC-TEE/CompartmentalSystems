import unittest

from testinfrastructure.InDirTest import InDirTest
#from testinfrastructure.helpers import pe
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must come before any import of matplotlib.pyplot or pylab to get rid of the stupid warning!
import matplotlib.pyplot as plt
from sympy import Symbol, Piecewise
from scipy.interpolate import interp1d

from CompartmentalSystems.BlockIvp import BlockIvp
from CompartmentalSystems.myOdeResult import solve_ivp_pwc


class TestBlockIvp(InDirTest):

    @unittest.skip
    def test_solution(self):
        x1_shape=(5,5)
        x2_shape=(2,)
        
        bivp=BlockIvp(
             time_str='t'
            ,start_blocks=[('x1',np.ones(x1_shape)),('x2',np.ones(x2_shape))]
            ,functions=[
                 ((lambda x1   : - x1     ),    [     'x1'     ])
                ,((lambda t,x2 : - 2*t*x2 ),    ['t' ,     'x2'])
             ])   
        # the reference solution 
        t_max=2
        ref={'x1':np.exp(-t_max)*np.ones(x1_shape),
             'x2':np.exp(-t_max**2)*np.ones(x2_shape)
        }
        res = bivp.block_solve(t_span=(0,t_max))
        self.assertTrue(np.allclose(res['x1'][-1],ref['x1'],rtol=1e-2))
        self.assertTrue(np.allclose(res['x2'][-1],ref['x2'],rtol=1e-2))

        # here we describe time by a variable with constant derivative
        # amd use it in the derivative of the second variable
        # to simulate a coupled system without feedback (skew product)
        x1_shape=(1,)
        x2_shape=(2,)
        bivp=BlockIvp(
             time_str='t'
            ,start_blocks=[('x1',np.zeros(x1_shape)),('x2',np.ones(x2_shape))]
            ,functions=[
                 ((lambda x1   :   np.ones(x1.shape)       ),    ['x1'])
                ,((lambda x1,x2 : - 2*x1*x2 ),    ['x1' ,     'x2'])
             ])   
        # the reference solution 
        t_max=2
        ref={'x1':t_max*np.ones(x1_shape),
             'x2':np.exp(-t_max**2)*np.ones(x2_shape)
        }
        res = bivp.block_solve(t_span=(0,t_max))
        self.assertTrue(np.allclose(res['x1'][-1],ref['x1'],rtol=1e-2))
        self.assertTrue(np.allclose(res['x2'][-1],ref['x2'],rtol=1e-2))


    def test_solve_ivp_pwc(self):
        t = Symbol('t')
        ms = [1, -1, 1]
        disc_times = [410, 820]
        t_start = 0
        t_end = 1000
        ref_times = np.arange(t_start, t_end, 10)
        times = np.array([0, 300, 600, 900])
        t_span = (t_start, t_end)
        #first build a single rhs
        m = Piecewise(
            (ms[0], t<disc_times[0]),
            (ms[1], t<disc_times[1]),
            (ms[2], True)
        )
        rhs = lambda t, x: m.subs({'t': t})
        x0 = np.asarray([0])
        # To see the differencte between a many piece and
        # a one piece solotion 
        # we deliberately chose a combination
        # of method and first step where the 
        # solver will get lost if it does not restart
        # at the disc times.
        sol_obj = solve_ivp_pwc(
            rhs,
            t_span,
            x0,
            t_eval = times,
            method = 'RK45',
            first_step = None
        )
        def funcmaker(m):
            return lambda t, x: m
        
        rhss = [funcmaker(m) for m in ms]            

        ref_func = interp1d(
            [t_start] + list(disc_times) + [t_end],
            [0.,410.,0.,180.]
        )


        self.assertFalse(
            np.allclose(
                ref_func(times),
                sol_obj.y[0,:],
                atol = 400
            )
        )

        sol_obj_pw = solve_ivp_pwc(
            rhss,
            t_span,
            x0,
            #t_eval = times,
            method = 'RK45',
            first_step = None,
            disc_times = disc_times
        )


        self.assertTrue(
            np.allclose(
                ref_func(sol_obj_pw.t),
                sol_obj_pw.y[0,:]
            )
        )

        self.assertTrue(
            np.allclose(
                ref_func(ref_times),
                sol_obj_pw.sol(ref_times)[0,:]
            )
        )

        sol_obj_pw_t_eval = solve_ivp_pwc(
            rhss,
            t_span,
            x0,
            t_eval = times,
            method = 'RK45',
            first_step = None,
            disc_times = disc_times
        )
        
        self.assertTrue(
            np.allclose(
                times,
                sol_obj_pw_t_eval.t
            )
        )
        
        self.assertTrue(
            np.allclose(
                ref_func(times),
                sol_obj_pw_t_eval.y[0,:]
            )
        )


        fig, ax = plt.subplots(
                nrows = 1,
                ncols = 1
        )
        ax.plot(
            ref_times
            ,ref_func(ref_times)
            ,color = 'blue'
            ,label='ref'
        #    ,ls = '-'
        )
        ax.plot(
            times
            ,ref_func(times)
            ,'*'
            ,label='ref points'
        )
        ax.plot(
            sol_obj.t
            ,sol_obj.y[0,:]
            ,'o'
            ,label='pure solve_ivp'
            #ls = '--'
        )

        ax.plot(
            sol_obj_pw.t
            ,sol_obj_pw.y[0,:]
            ,'+'
            ,label='solve_ivp_pwc'
            #ls = '--'
        )
        ax.legend()
        fig.savefig('inaccuracies.pdf', tight_layout = True)


################################################################################


if __name__ == '__main__':
    suite=unittest.defaultTestLoader.discover(".",pattern=__file__)

#    # Run same tests across 16 processes
#    concurrent_suite = ConcurrentTestSuite(suite, fork_for_tests(1))
#    runner = unittest.TextTestRunner()
#    res=runner.run(concurrent_suite)
#    # to let the buildbot fail we set the exit value !=0 if either a failure or error occurs
#    if (len(res.errors)+len(res.failures))>0:
#        sys.exit(1)

    unittest.main()
