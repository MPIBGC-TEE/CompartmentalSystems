from sympy import symbols, Matrix, Symbol
import numpy as np
import unittest
from scipy.linalg import expm

from CompartmentalSystems.smooth_model_run_14C import (
    SmoothModelRun_14C,
)
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.helpers_reservoir import (
    ALPHA_14C,
    DECAY_RATE_14C_DAILY
)


class TestSmoothModelRun_14C(unittest.TestCase):

    def setUp(self):
        C_1, C_2, k = symbols('C_1 C_2 k')
        B = Matrix([[-2, 0],
                    [k, -2]])
        u = Matrix(2, 1, [1, 1])
        state_vector = Matrix(2, 1, [C_1, C_2])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(
            state_vector,
            time_symbol,
            B,
            u
        )

        parameter_dict = {k: 1}
        start_values = np.array([1.0/3.0, 4.0/9.0])
        times = np.linspace(0, 5, 6)
        smr = SmoothModelRun(srm, parameter_dict, start_values, times)
#        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        self.alpha = 0.5
#        self.alpha = ALPHA_14C
        start_values_14C = smr.start_values * self.alpha

        def Fa_func_14C(t): return self.alpha
        self.smr_14C = SmoothModelRun_14C(
            smr,
            start_values_14C,
            Fa_func_14C,
            1.0 # or use DECAY_RATE_14C_DAILY
        )

    def test_init(self):
        self.smr_14C.solve()

    def test_acc_gross_external_output_vector(self):
        # relies on 14C model in steady statei, i.e. decay_rate = 1
        self.assertTrue(
            np.allclose(
                np.array([1/3, 8/9]) * np.ones((5, 2)) * self.alpha,
                self.smr_14C.acc_gross_external_output_vector()
            )
        )

    def test_acc_net_external_output_vector(self):
        mr = self.smr_14C
        Phi_14C = expm(mr.B_func()(0))
        times = mr.times

        for k in range(len(times)-1):
            with self.subTest():
                Phi = Phi_14C * np.exp(mr.decay_rate*(times[k+1]-times[k]))

                here_with_decay = np.array(
                    [
                        Phi_14C.sum(0)[i] * mr.solve()[k][i]
                        for i in range(2)
                    ]
                )
                here_without_decay = np.array(
                    [
                        Phi.sum(0)[i] * mr.solve()[k][i] for i in range(2)
                    ]
                )
                gone_by_decay = here_without_decay - here_with_decay
                gone_with_decay = np.array(
                    [
                        (1 - Phi_14C.sum(0)[i]) * mr.solve()[k][i]
                        for i in range(2)
                    ]
                )
#                gone_without_decay = np.array(
#                    [
#                        (1 - Phi.sum(0)[i]) * mr.solve()[k][i]
#                        for i in range(2)
#                    ]
#                )
                gone_without_decay = gone_with_decay - gone_by_decay
                ref = np.array([gone_without_decay])

                self.assertTrue(
                    np.allclose(
                        ref,
                        self.smr_14C.acc_net_external_output_vector()[k]
                    )
                )

    # Delta 14C methods

    def test_Delta_14C(self):
        def F_Delta_14C(C12, C14, alpha):
            with np.errstate(invalid='ignore'):
                res = (C14/C12/alpha-1) * 1000

            return res

        smr_14C = self.smr_14C

        methods = [
            'solve',
            "acc_gross_external_input_vector",
            "acc_net_external_input_vector",
            "acc_gross_external_output_vector",
            "acc_net_external_output_vector",
            "acc_gross_internal_flux_matrix",
            "acc_net_internal_flux_matrix"
        ]

        for method in methods:
            with self.subTest():
                self.assertTrue(
                    np.allclose(
                        F_Delta_14C(
                            getattr(smr_14C.smr, method)(),
                            getattr(smr_14C, method)(),
                            alpha=self.alpha
                        ),
                        getattr(smr_14C, method+'_Delta_14C')(
                            alpha=self.alpha
                        ),
                        equal_nan=True
                    )
                )

    @unittest.skip
    def test_Delta_14C_gross_vs_net(self):
        # maybe get a feeling for gross vs net Delta 14C
        # in dependence of step size
        def F_Delta_14C(C12, C14, alpha):
            with np.errstate(invalid='ignore'):
                res = (C14/C12/alpha-1) * 1000

            return res

        smr_14C = self.smr_14C

        methods = [
            "external_input_vector_Delta_14C",
            "external_output_vector_Delta_14C",
            "internal_flux_matrix_Delta_14C",
        ]

        nrs = [6, 11, 21, 51, 101, 501, 5001]
        data_timess = np.array(
            [np.linspace(1, 5, nr) for nr in nrs]
        )
        for method in methods:
            print(method)
            for nr, data_times in zip(nrs, data_timess):
                with self.subTest():
                    gross = getattr(smr_14C, 'acc_gross_' + method)(
                        data_times=data_times,
                        alpha=self.alpha
                    )

                    net = getattr(smr_14C, 'acc_net_' + method)(
                        data_times=data_times,
                        alpha=self.alpha
                    )

                    max_abs_err = np.nanmax(np.abs(net-gross))
#                    print(nr, max_abs_err)
                    gross[gross == 0] = np.nan
                    max_rel_err = np.nanmax(np.abs(net-gross)/gross) * 100
#                    print(nr, max_rel_err)
