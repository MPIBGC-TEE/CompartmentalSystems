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
from CompartmentalSystems.myOdeResult import custom_solve_ivp


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


    def test_custom_solve_ivp(self):
        t=Symbol('t')
        #ms = [1, 1/2, 2, -1/2, -1, -2]
        #disc_times=[1,2,3,4]
        #m = Piecewise(
        #    (ms[0], t<disc_times[0]),
        #    (ms[1], t<disc_times[1]),
        #    (ms[2], t<disc_times[2]),
        #    (ms[3], t<disc_times[3]),
        #    (-2, True)
        #)
        #t_start = disc_times[0]-1
        #t_end = disc_times[-1]+1
        #  
        #times = np.linspace(t_start, t_end, t_end*100+1)
################ new
        ms = [1, -1, 1]
        disc_times = [410, 820]
        t_start = 0
        t_end = 1000
        ref_times = np.arange(t_start, t_end, 10)
        times = np.array([0, 300, 600, 900])
        t_span = (t_start, t_end)

        m_l = Piecewise(
            (ms[0], t <= disc_times[0]),
            (ms[1], t <=  disc_times[1]),
            (ms[2], True)
        )
        m_r = Piecewise(
            (ms[0], t < disc_times[0]),
            (ms[1], t <  disc_times[1]),
            (ms[2], True)
        )
        def func_l(t, x):
            return np.ones_like(x)*m_l.subs({'t':t})

        def func_r(t, x):
            return np.ones_like(x)*m_r.subs({'t':t})

        x0 = np.asarray([0])

        ref_func = interp1d(
            [t_start] + list(disc_times) + [t_end],
            [0.,410.,0.,180.]
        )
        #ref_func = interp1d(ref_times, ref[0,:], fill_value='extrapolate')

        # To see the differencte between a many piece and
        # a one piece solotion 
        # we deliberately chose a combination
        # of method and first step where the 
        # solver will get lost if it does not restart
        # at the disc times.
        sol_obj = custom_solve_ivp(
            func_l,
            t_span,
            x0,
            t_eval = times,
            dense_output = True,
            method = 'RK45',
            first_step = None
        )

        self.assertFalse(
            np.allclose(
                ref_func(times),
                sol_obj.y[0,:],
                atol = 400
            )
        )

        sol_obj_pw = custom_solve_ivp(
            func_l,
            t_span,
            x0,
            #t_eval = times,
            dense_output = True,
            method = 'RK45',
            first_step = None,
            disc_times = disc_times,
            deriv_r = func_r
        )

        self.assertTrue(
            np.allclose(
                ref_func(sol_obj_pw.t),
                sol_obj_pw.y[0,:]
            )
        )

        self.assertTrue(
            np.allclose(
                ref_func(sol_obj_pw.t),
                sol_obj_pw.sol(sol_obj_pw.t)[0,:]
            )
        )
        #labels=['obj.y','obj_2.y','obj.sol(times)','obj_2.sol(times)']
        ress = [
                sol_obj.y, 
                #sol_obj_2.y, 
                #sol_obj.sol(times), 
                #sol_obj_2.sol(times)
        ]
        labels=['obj.sol(times)','obj_2.sol(times)']
        fig,axs = plt.subplots(
                nrows = 2,
                ncols = 2
        )
        ax = axs[0,0]
        ax.plot(
            ref_times
            ,ref_func(ref_times),
            color = 'blue'
        #    ,ls = '-'
        )
        ax.plot(
            times
            ,ref_func(times)
            ,'*'
        )
        ax.plot(
            sol_obj.t
            ,sol_obj.y[0,:]
            ,'o'
            #ls = '--'
        )

        ax.plot(
            sol_obj_pw.t
            ,sol_obj_pw.y[0,:]
            ,'+'
            #ls = '--'
        )
       # ax = axs[1,0]
       # for nr, res in enumerate(ress):
       #     ax.plot(
       #         times
       #         ,(ref-res)[0,:]
       #         ,label = labels[nr]
       #     )
       # ax.legend()
        fig.savefig('inaccuracies.pdf',tight_layout = True)






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
