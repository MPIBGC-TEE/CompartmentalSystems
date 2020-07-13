#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:

# import unittest
import numpy as np
from sympy import symbols, Matrix

from testinfrastructure.InDirTest import InDirTest
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.smooth_model_run_14C import SmoothModelRun_14C
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
from CompartmentalSystems.discrete_model_run_14C import DiscreteModelRun_14C


class TestDiscreteModelRun_14C(InDirTest):
    def setUp(self):
        x, t = symbols('x t')
        B = Matrix(1, 1, [-1])
        u = Matrix(1, 1, [1])
        state_vector = Matrix(1, 1, [x])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)
        parameter_dict = {}
        start_values = np.array([1])
        times = np.arange(0, 6, 1)

        smr = SmoothModelRun(srm, parameter_dict, start_values, times)

        self.alpha = 1
        start_values_14C = start_values * self.alpha
        decay_rate = 1.0

        def Fa_func(t): return self.alpha
        self.smr_14C = SmoothModelRun_14C(
            smr,
            start_values_14C,
            Fa_func,
            decay_rate
        )

        dmr_from_pwc = DMR.from_SmoothModelRun(smr)
        fake_net_Us = DMR.from_SmoothModelRun(self.smr_14C).net_Us

        # we cannot infer net_Us_14C coming from data, hence we use the
        # net_Us from the 14C model coming from the Smooth 14C model
        # this is of no use in pracitical situations tough since once
        # we have smr, we can use smr_14C immediately instead of going
        # through DMRs
        self.dmr_from_pwc_14C = DiscreteModelRun_14C(
            dmr_from_pwc,
            start_values_14C,
            fake_net_Us,
            decay_rate
        )

    def test_from_SmoothModelRun(self):
        methods = [
            'solve',
            "acc_net_external_input_vector",
            "acc_net_external_output_vector",
            "acc_net_internal_flux_matrix"
#            'solve_Delta_14C',
#            "acc_net_external_input_vector_Delta_14C",
#            "acc_net_external_output_vector_Delta_14C",
#            "acc_net_internal_flux_matrix_Delta_14C"
        ]

        for method in methods:
            with self.subTest():
                self.assertTrue(
                    np.allclose(
                        getattr(self.smr_14C, method)(),
                        getattr(self.dmr_from_pwc_14C, method)(),
                        rtol=1e-04
                    )
                )

    def test_Delta_14C(self):
        methods = [
            'solve_Delta_14C',
            "acc_net_external_input_vector_Delta_14C",
            "acc_net_external_output_vector_Delta_14C",
            "acc_net_internal_flux_matrix_Delta_14C"
        ]

        for method in methods:
            with self.subTest():
                self.assertTrue(
                    np.allclose(
                        getattr(self.smr_14C, method)(),
                        getattr(self.dmr_from_pwc_14C, method)(),
                        rtol=1e-04,
                        equal_nan=True
                    )
                )
