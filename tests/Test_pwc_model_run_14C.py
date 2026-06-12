import unittest

import numpy as np
from sympy import Function, Matrix, symbols

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.pwc_model_run import PWCModelRun
from CompartmentalSystems.smooth_model_run_14C import SmoothModelRun_14C
from CompartmentalSystems.pwc_model_run_14C import PWCModelRun_14C


class TestPWCModelRun_14C(unittest.TestCase):

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

        self.alpha = 0.5
        start_values_14C = start_values * self.alpha

        def Fa_func(t): return self.alpha
        decay_rate = 1.0

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
        tmp_start_values = start_values
        tmp_start_values_14C = start_values_14C
        for i in range(len(disc_times)+1):
            smr = SmoothModelRun(
                self.srm,
                parameter_dict=parameter_dicts[i],
                start_values=tmp_start_values,
                times=timess[i],
                func_set=func_dicts[i]
            )
            tmp_start_values = smr.solve()[-1]

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

    def test_solve(self):
        soln_smrs_14C = [smr_14C.solve() for smr_14C in self.smrs_14C]
        L = [soln[:-1] for soln in soln_smrs_14C[:-1]]
        L += [soln_smrs_14C[-1]]
        soln_14C_ref = np.concatenate(L, axis=0)

        self.assertTrue(
            np.allclose(
                soln_14C_ref,
                self.pwc_mr_14C.solve()
            )
        )

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

    # Delta 14C methods

    def test_solve_Delta_14C(self):
        soln_smrs_Delta_14C = [
            smr_14C.solve_Delta_14C(alpha=self.alpha)
            for smr_14C in self.smrs_14C
        ]
        L = [soln[:-1] for soln in soln_smrs_Delta_14C[:-1]]
        L += [soln_smrs_Delta_14C[-1]]
        Delta_14C_ref = np.concatenate(L, axis=0)

        self.assertTrue(
            np.allclose(
                Delta_14C_ref,
                self.pwc_mr_14C.solve_Delta_14C(alpha=self.alpha),
                equal_nan=True
            )
        )

    def test_Delta_14C(self):
        methods = [
            "acc_gross_external_input_vector_Delta_14C",
            "acc_net_external_input_vector",
            "acc_gross_external_output_vector",
            "acc_net_external_output_vector",
            "acc_gross_internal_flux_matrix",
            "acc_net_internal_flux_matrix"
        ]

        for method in methods:
            with self.subTest():
                Delta_14C = [getattr(smr_14C, method)()
                             for smr_14C in self.smrs_14C]
                Delta_14C_ref = np.concatenate(Delta_14C, axis=0)
                self.assertTrue(
                    np.allclose(
                        Delta_14C_ref,
                        getattr(self.pwc_mr_14C, method)(),
                        equal_nan=True
                    )
                )


###############################################################################


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(".", pattern=__file__)
    unittest.main()
