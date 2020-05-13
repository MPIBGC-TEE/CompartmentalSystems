import unittest

import numpy as np
from sympy import Function, Matrix, symbols

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.pwc_model_run import PWCModelRun
from CompartmentalSystems.smooth_model_run_14C import SmoothModelRun_14C
from CompartmentalSystems.pwc_model_run_14C import PWCModelRun_14C


class TestPWCModelRun(unittest.TestCase):

    def setUp(self):
        x, y, t, k = symbols("x y t k")
        u_1 = Function('u_1')(x, t)
        state_vector = Matrix([x, y])
        B = Matrix([[-1,  1.5],
                    [k, -2]])
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

        parameter_dicts = [{k: 1}, {k: 0.5}]
        func_dicts = [{u_1: lambda x_14C, t: 9}, {u_1: lambda x_14C, t: 3}]

        pwc_mr = PWCModelRun(
            self.srm,
            parameter_dicts,
            start_values,
            times,
            disc_times,
            func_dicts
        )

        alpha = 1e-12
        start_values_14C = start_values * alpha

        def Fa_func(t): return alpha
        decay_rate = 0.5

        self.pwc_mr_14C = PWCModelRun_14C(
            pwc_mr,
            start_values_14C,
            Fa_func,
            decay_rate
        )

        timess = [
            np.linspace(t_0, disc_times[0], 6),
            np.linspace(disc_times[0], t_max, 6)
        ]

        smrs_14C = []
        tmp_start_values_14C = start_values_14C
        for i in range(len(disc_times)+1):
            smr = SmoothModelRun(
                self.srm,
                parameter_dict=parameter_dicts[i],
                start_values=tmp_start_values_14C,
                times=timess[i],
                func_set=func_dicts[i]
            )

            smrs_14C.append(
                SmoothModelRun_14C(
                    smr,
                    tmp_start_values_14C,
                    Fa_func,
                    decay_rate
                )
            )
            tmp_start_values_14C = smrs_14C[i].solve()[-1]

        self.smrs_14C = smrs_14C

    def test_acc_gross_external_output_vector(self):
        ageov_smrs_14C = [smr_14C.acc_gross_external_output_vector()
                          for smr_14C in self.smrs_14C]
        ageov_14C_ref = np.concatenate(ageov_smrs_14C, axis=0)
        self.assertTrue(
            np.allclose(
                ageov_14C_ref,
                self.pwc_mr_14C.acc_gross_external_output_vector()
            )
        )

    def test_acc_net_external_output_vector(self):
        aneov_smrs_14C = [smr_14C.acc_net_external_output_vector()
                          for smr_14C in self.smrs_14C]
        aneov_14C_ref = np.concatenate(aneov_smrs_14C, axis=0)
        self.assertTrue(
            np.allclose(
                aneov_14C_ref,
                self.pwc_mr_14C.acc_net_external_output_vector()
            )
        )


###############################################################################


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(".", pattern=__file__)
    unittest.main()
