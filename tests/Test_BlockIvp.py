import unittest

from testinfrastructure.InDirTest import InDirTest
#from testinfrastructure.helpers import pe
import numpy as np

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
        def func(t, x):
            return -0.2*t*x
        
        disc_times = np.linspace(0, 10, 11)
        for dim in [1,2]:
            with self.subTest():   
                x0 = 1 + np.arange(dim).reshape(dim,)
                t_span = (disc_times[0], disc_times[-1])
        
                sol_obj = custom_solve_ivp(
                    func,
                    t_span,
                    x0,
                    disc_times,
                    dense_output=True
                )
            
                self.assertTrue(
                    np.allclose(
                        sol_obj.y,
                        sol_obj.sol(disc_times)
                    )
                )


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
