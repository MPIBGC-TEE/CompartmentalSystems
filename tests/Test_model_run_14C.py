# import unittest
from testinfrastructure.InDirTest import InDirTest
import numpy as np
from sympy import symbols, Matrix

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.smooth_model_run_14C import SmoothModelRun_14C
from CompartmentalSystems.pwc_model_run_fd import PWCModelRunFD
from CompartmentalSystems.pwc_model_run_14C import PWCModelRun_14C
from CompartmentalSystems.discrete_model_run import DiscreteModelRun
from CompartmentalSystems.discrete_model_run_14C import DiscreteModelRun_14C
from CompartmentalSystems.model_run import (
    plot_stocks_and_fluxes
)


class TestModelRun_14C(InDirTest):
    def setUp(self):
        x, y, t = symbols("x y t")
        state_vector = Matrix([x, y])
        B = Matrix([[-1, 1.5],
                    [0.5, -2]])
        u = Matrix(2, 1, [9, 1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([10, 40])
        self.start_values = start_values
        self.t_0 = 0
        self.t_max = 10
        self.ntmo = 10
        self.fac = 2
        self.times = np.linspace(self.t_0, self.t_max, self.ntmo+1)

        self.smr = SmoothModelRun(srm, {}, start_values, self.times)

        alpha = 0.5
        self.decay_rate = 1.0
        self.start_values_14C = alpha * self.start_values

        def Fa_func(t): return alpha
        self.Fa_func = Fa_func

        self.smr_14C = SmoothModelRun_14C(
            self.smr,
            self.start_values_14C,
            self.Fa_func,
            self.decay_rate
        )

    def test_DiscreteModelRun_14CFromFakeData(self):
        dmr_from_smr_14C = DiscreteModelRun.from_SmoothModelRun(self.smr_14C)
        dmr_14C = DiscreteModelRun_14C(
            DiscreteModelRun.from_SmoothModelRun(self.smr),
            self.start_values_14C,
            dmr_from_smr_14C.net_Us,
            self.decay_rate
        )
 
        meths = [
            "solve",
            "acc_net_external_input_vector",
            "acc_net_external_output_vector",
            "acc_net_internal_flux_matrix"
        ]
        for meth in meths:
            with self.subTest():
                self.assertTrue(
                    np.allclose(
                        getattr(self.smr_14C, meth)(),
                        getattr(dmr_14C, meth)()
                    )
                )

    def test_PWCModelRunFD_14C(self):
        times = self.smr.times

        xs, gross_Us, gross_Fs, gross_Rs \
            = self.smr.fake_gross_discretized_output(times)

        pwc_mr_fd = PWCModelRunFD.from_gross_fluxes(
            self.smr.model.time_symbol,
            times,
            self.smr.start_values,
            gross_Us,
            gross_Fs,
            gross_Rs
        )
        pwc_mr_fd_14C = PWCModelRun_14C(
            pwc_mr_fd.pwc_mr,
            self.start_values_14C,
            self.Fa_func,
            self.decay_rate
        )

        meths = [
            "solve",
            "acc_gross_external_input_vector",
            "acc_net_external_input_vector",
            "acc_gross_external_output_vector",
            "acc_net_external_output_vector",
            "acc_gross_internal_flux_matrix",
            "acc_net_internal_flux_matrix"
        ]
        for meth in meths:
            with self.subTest():
                ref = getattr(self.smr_14C, meth)()
                res = getattr(pwc_mr_fd_14C, meth)()
                self.assertTrue(
                    np.allclose(
                        ref,
                        res,
                        rtol=3e-02
                    )
                    # For this linear constant model
                    # the error should actually be zero
                    # and is only due to numerical inaccuracy.
                )
        plot_stocks_and_fluxes(
            [
                self.smr_14C,
                pwc_mr_fd_14C
            ],
            'stocks_and_fluxes.pdf'
        )
