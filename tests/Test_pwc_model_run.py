import unittest

import numpy as np
from scipy.linalg import inv

from sympy import Function, Matrix, sin, symbols

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.pwc_model_run import PWCModelRun 

class TestPWCModelRun(unittest.TestCase):

    def setUp(self):
        x, y, t, k = symbols("x y t k")
        u_1 = Function('u_1')(x,t)
        state_vector = Matrix([x,y])
        B = Matrix([[-1,  1.5],
                    [ k, -2  ]])
        u = Matrix(2, 1, [u_1, 1])
        self.srm = SmoothReservoirModel.from_B_u(
            state_vector,
            t,
            B,
            u
        )

        start_values = np.array([10, 40])
        t_0 = 0
        t_max = 10
        times = np.linspace(t_0, t_max, 11)
        disc_times = [5]
        
        parameter_dicts = [{k:1}, {k:.5}]
        func_dicts = [{u_1: lambda x, t: 9}, {u_1: lambda x, t: 3}]

        self.pwc_mr = PWCModelRun(
            self.srm,
            parameter_dicts,
            start_values,
            times,
            disc_times,
            func_dicts
        )

        timess = [
            np.linspace(t_0, disc_times[0], 6),
            np.linspace(disc_times[0], t_max, 6)
        ]

        smrs = []
        tmp_start_values = start_values
        for i in range(len(disc_times)+1):
            smrs.append(
                SmoothModelRun(
                    self.srm,
                    parameter_dict = parameter_dicts[i],
                    start_values = tmp_start_values,
                    times = timess[i],
                    func_set = func_dicts[i]
                ) 
            )
            tmp_start_values = smrs[i].solve()[-1]
        self.smrs = smrs


    def test_nr_pools(self):
        self.assertEqual(self.srm.nr_pools, self.pwc_mr.nr_pools)


    def test_dts(self):
        dts_smrs = [smr.dts for smr in self.smrs]
        dts_ref = np.concatenate(dts_smrs)
        self.assertTrue(np.all(dts_ref == self.pwc_mr.dts))


    def test_init(self):
        # automatically tested by setUp
        pass


    def test_solve(self):
        soln_smrs = [smr.solve() for smr in self.smrs]
        l = [soln[:-1] for soln in soln_smrs[:-1]] + [soln_smrs[-1]]
        soln_ref = np.concatenate(l, axis=0)
        self.assertTrue(np.allclose(soln_ref, self.pwc_mr.solve()))
    

    @unittest.skip
    def test_acc_gross_external_input_vector(self):
        raise(Exception('To be implemented'))

    @unittest.skip
    def test_acc_gross_internal_flux_matrix(self):
        raise(Exception('To be implemented'))
    
    @unittest.skip
    def test_acc_gross_external_output_vector(self) :
        raise(Exception('To be implemented'))
    
    @unittest.skip
    def test_acc_net_external_input_vector(self):
        raise(Exception('To be implemented'))

    @unittest.skip
    def test_acc_net_internal_flux_matrix(self):
        raise(Exception('To be implemented'))
    
    @unittest.skip
    def test_acc_net_external_output_vector(self) :
        raise(Exception('To be implemented'))


################################################################################


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(".",pattern=__file__)
    unittest.main()




