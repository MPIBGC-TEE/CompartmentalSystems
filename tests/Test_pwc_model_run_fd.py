import unittest

import numpy as np
from scipy.linalg import inv

from sympy import Function, Matrix, sin, symbols

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.pwc_model_run_fd import PWCModelRunFD 

class TestPWCModelRunFD(unittest.TestCase):

    def test_init(self):
        x, y, t, k = symbols("x y t k")
        u_1 = Function('u_1')(x,t)
        state_vector = Matrix([x,y])
        B = Matrix([[-1,  1.5],
                    [ k, -2  ]])
        u = Matrix(2, 1, [u_1, 1])
        srm = SmoothReservoirModel.from_B_u(
            state_vector,
            t,
            B,
            u
        )

        start_values = np.array([10, 40])
        t_0 = 0
        t_max = 10
        disc_times = [5]
        
        ps = [{k:1}, {k:.5}]
        fs = [{u_1: lambda x, t: 9}, {u_1: lambda x, t: 3}]
        timess = [
            np.linspace(t_0, disc_times[0], 10),
            np.linspace(disc_times[0], t_max, 10)
        ]

        smrs = []
        tmp_start_values = start_values
        for i in range(len(disc_times)+1):
            smrs.append(
                SmoothModelRun(
                    srm,
                    parameter_dict = ps[i],
                    start_values = tmp_start_values,
                    times = timess[i],
                    func_set = fs[i]
                ) 
            )
            tmp_start_values = smrs[i].solve()[-1]

        gross_Us = np.concatenate(
            [
                smr.acc_gross_external_input_vector(
                    np.array([smr.times[0], smr.times[-1]])
                ) 
                for smr in smrs
            ],
            axis = 0
        )

        gross_Fs = np.concatenate(
            [
                smr.acc_gross_internal_flux_matrix(
                    np.array([smr.times[0], smr.times[-1]])
                ) 
                for smr in smrs
            ],
            axis = 0
        )

        gross_Rs = np.concatenate(
            [
                smr.acc_gross_external_output_vector(
                    np.array([smr.times[0], smr.times[-1]])
                ) 
                for smr in smrs
            ],
            axis = 0
        )

        pwc_mr_fd = PWCModelRunFD(
            t,
            np.array([t_0]+disc_times+[t_max]),
            start_values,
            gross_Us,
            gross_Fs,
            gross_Rs
        )


################################################################################


if __name__ == '__main__':
    unittest.main()



