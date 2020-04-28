import unittest

from testinfrastructure.InDirTest import InDirTest
#from testinfrastructure.helpers import pe
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must come before any import of matplotlib.pyplot or pylab to get rid of the stupid warning!
import matplotlib.pyplot as plt

from CompartmentalSystems.BlockIvp import BlockIvp, custom_solve_ivp

class TestBlockIvp(InDirTest):
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
        from sympy import Symbol, Piecewise
        t=Symbol('t')
        ms = [1, 1/2, 2, -1/2, -1, -2]
        disc_times=[1,2,3,4]
        m = Piecewise(
            (ms[0], t<disc_times[0]),
            (ms[1], t<disc_times[1]),
            (ms[2], t<disc_times[2]),
            (ms[3], t<disc_times[3]),
            (-2, True)
        )
        t_start = disc_times[0]-1
        t_end = disc_times[-1]+1
          
        def func(t, x):
            return np.ones_like(x)*m.subs({'t':t})
            #return np.ones_like(x)*(t_end/2-np.ceil(t))

        times = np.linspace(t_start, t_end, t_end*100+1)

        #for dim in [1,2]:
        #    with self.subTest():   
        dim = 1
        x0 = np.asarray([1])
        ref = np.zeros((dim,len(times)))
        dt = np.diff(times)
        ref[0,0] = x0
        for k, t in enumerate(times[:-1]):
            ref[0,k+1] = ref[0,k] + func(t,646549)*dt[k]

        t_span = (times[0], times[-1])
        
        sol_obj = custom_solve_ivp(
            func,
            t_span,
            x0,
            t_eval = times,
            dense_output = True,
            #method = 'LSODA'
        )
        print('#########################3 sol_obj.y')
        #print(sol_obj.y)
        print(sol_obj.y.shape)
        
        sol_obj_2 = custom_solve_ivp(
            func,
            t_span,
            x0,
            t_eval = times,
            disc_times   = disc_times,
            dense_output = True,
            #method = 'LSODA'
        )
       # # compare values 
       # self.assertTrue(
       #     np.allclose(
       #         ref,
       #         sol_obj.y,
       #         atol = 1e-03
       #     )
       # )
       # self.assertTrue(
       #     np.allclose(
       #         ref,
       #         sol_obj_2.y,
       #         atol = 1e-03
       #     )
       # )

       # # compare functions 
       # self.assertTrue(
       #     np.allclose(
       #         ref,
       #         sol_obj.sol(times),
       #         atol = 1e-03 
       #     )
       # )
       # self.assertTrue(
       #     np.allclose(
       #         ref,
       #         sol_obj_2.sol(times),
       #         atol = 1e-03 
       #     )
       # )
        #labels=['obj.y','obj_2.y','obj.sol(times)','obj_2.sol(times)']
        ress = [
                sol_obj.y, 
                #sol_obj_2.y, 
                #sol_obj.sol(times), 
                sol_obj_2.sol(times)
        ]
        labels=['obj.sol(times)','obj_2.sol(times)']
        fig,axs = plt.subplots(
                nrows = 2,
                ncols = 2
        )
        ax = axs[0,0]
        ax.plot(
            times
            ,ref[0,:]
        )
        ax.plot(
            times
            ,ress[0][0,:]
        )
        ax.plot(
            times
            ,ress[1][0,:]
        )
        #ax.legend()
        #ax = axs[1,0]
        ##ress = [sol_obj.y ,sol_obj_2.y ,sol_obj.sol(times) ,sol_obj_2.sol(times)]
        #ress = [sol_obj.sol(times) ,sol_obj_2.sol(times)]
        #for nr, res in enumerate(ress):
        #    ax.plot(
        #        times
        #        ,(ref-res)[0,:]
        #        ,label = labels[nr]
        #    )
        #ax.legend()
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
