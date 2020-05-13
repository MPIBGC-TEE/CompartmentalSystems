from sympy import symbols, Matrix, Symbol
import numpy as np
import unittest
from scipy.linalg import expm

from CompartmentalSystems.smooth_model_run_14C import (
    SmoothModelRun_14C,
)
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel


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

        alpha = 0.5
        start_values_14C = smr.start_values * alpha

        def Fa_func(t): return alpha
        self.smr_14C = SmoothModelRun_14C(
            smr,
            start_values_14C,
            Fa_func,
            1.0
        )

    def test_init(self):
        self.smr_14C.solve()

    def test_acc_gross_external_output_vector(self):
        self.assertTrue(
            np.allclose(
                np.array([1/3, 8/9]) * np.ones((5, 2)) * 0.5,
                self.smr_14C.acc_gross_external_output_vector()
            )
        )

    def test_accc_net_external_output_vector(self):
        mr = self.smr_14C
        Phi_14C = expm(mr.B_func()(0))
        Phi = Phi_14C * np.exp(mr.decay_rate)

        here_with_decay = np.array(
            [
                Phi_14C.sum(0)[i] * mr.start_values[i]
                for i in range(2)
            ]
        )
        here_without_decay = np.array(
            [
                Phi.sum(0)[i] * mr.start_values[i] for i in range(2)
            ]
        )
        gone_by_decay = here_without_decay - here_with_decay
        gone_with_decay = np.array(
            [
                (1 - Phi_14C.sum(0)[i]) * mr.start_values[i]
                for i in range(2)
            ]
        )
#        gone_without_decay = np.array(
#            [
#                (1 - Phi.sum(0)[i]) * mr.start_values[i]
#                for i in range(2)
#            ]
#        )
        gone_without_decay = gone_with_decay - gone_by_decay
        ref = np.array([gone_without_decay] * 5)

        self.assertTrue(
            np.allclose(
                ref,
                self.smr_14C.acc_net_external_output_vector()
            )
        )
