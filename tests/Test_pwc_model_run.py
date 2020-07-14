import unittest

import numpy as np
from sympy import Function, Matrix, symbols

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.pwc_model_run import PWCModelRun


class TestPWCModelRun(unittest.TestCase):

    def setUp(self):
        x, y, t, k = symbols("x y t k")
        u_1 = Function('u_1')(x, t)
        state_vector = Matrix([x, y])
        B = Matrix([[-1,  1.5],
                    [k/(t+1), -2]])
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

        parameter_dicts = [{k: 1}, {k: 0.5*t}]
        func_dicts = [{u_1: lambda x, t: 9}, {u_1: lambda x, t: 3*t}]

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
                    parameter_dict=parameter_dicts[i],
                    start_values=tmp_start_values,
                    times=timess[i],
                    func_set=func_dicts[i]
                )
            )
            tmp_start_values = smrs[i].solve()[-1]
        self.smrs = smrs

    def test_nr_intervals(self):
        self.assertEqual(len(self.smrs), self.pwc_mr.nr_intervals)

    def test_boundaries(self):
        smrs = self.smrs
        self.assertTrue(
            np.all(
                [smrs[0].times[0]]
                + self.pwc_mr.disc_times
                + [smrs[-1].times[-1]]
                == self.pwc_mr.boundaries
            )
        )

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
        L = [soln[:-1] for soln in soln_smrs[:-1]] + [soln_smrs[-1]]
        soln_ref = np.concatenate(L, axis=0)
        self.assertTrue(np.allclose(soln_ref, self.pwc_mr.solve()))

    def test_B_func(self):
        B_func_smrs = [smr.B_func() for smr in self.smrs]
        for k, B_func in enumerate(B_func_smrs):
            with self.subTest():
                t = self.smrs[k].times[0]
                self.assertTrue(
                    np.all(B_func(t) == self.pwc_mr.B_func()(t))
                )

    def test_external_input_vector_func(self):
        u_func_smrs = [smr.external_input_vector_func() for smr in self.smrs]

        u_func_pwc = self.pwc_mr.external_input_vector_func()
        for k, u_func_k in enumerate(u_func_smrs):
            with self.subTest():
                t = self.smrs[k].times[0]
                self.assertTrue(np.all(u_func_k(t) == u_func_pwc(t)))

    def test_acc_gross_external_input_vector(self):
        ageiv_smrs = [smr.acc_gross_external_input_vector()
                      for smr in self.smrs]
        ageiv_ref = np.concatenate(ageiv_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                ageiv_ref,
                self.pwc_mr.acc_gross_external_input_vector()
            )
        )

    def test_acc_gross_internal_flux_matrix(self):
        agifm_smrs = [smr.acc_gross_internal_flux_matrix()
                      for smr in self.smrs]
        agifm_ref = np.concatenate(agifm_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                agifm_ref,
                self.pwc_mr.acc_gross_internal_flux_matrix()
            )
        )

    def test_acc_gross_external_output_vector(self):
        ageov_smrs = [smr.acc_gross_external_output_vector()
                      for smr in self.smrs]
        ageov_ref = np.concatenate(ageov_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                ageov_ref,
                self.pwc_mr.acc_gross_external_output_vector()
            )
        )

    def test_acc_net_external_input_vector(self):
        aneiv_smrs = [smr.acc_net_external_input_vector()
                      for smr in self.smrs]
        aneiv_ref = np.concatenate(aneiv_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                aneiv_ref,
                self.pwc_mr.acc_net_external_input_vector()
            )
        )

    def test_acc_net_internal_flux_matrix(self):
        anifm_smrs = [smr.acc_net_internal_flux_matrix()
                      for smr in self.smrs]
        anifm_ref = np.concatenate(anifm_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                anifm_ref,
                self.pwc_mr.acc_net_internal_flux_matrix()
            )
        )

    def test_acc_net_external_output_vector(self):
        aneov_smrs = [smr.acc_net_external_output_vector()
                      for smr in self.smrs]
        aneov_ref = np.concatenate(aneov_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                aneov_ref,
                self.pwc_mr.acc_net_external_output_vector()
            )
        )

    ###########################################################################

    def test_fake_discretized_Bs(self):
        Bs_smrs = [smr.fake_discretized_Bs()
                   for smr in self.smrs]
        Bs_ref = np.concatenate(Bs_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                Bs_ref,
                self.pwc_mr.fake_discretized_Bs()
            )
        )

        # test with cache
        self.pwc_mr.initialize_state_transition_operator_cache(
            100,
            True,
            4
        )
        self.assertTrue(
            np.allclose(
                Bs_ref,
                self.pwc_mr.fake_discretized_Bs(),
                rtol=1e-03
            )
        )

###############################################################################


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(".", pattern=__file__)
    unittest.main()
