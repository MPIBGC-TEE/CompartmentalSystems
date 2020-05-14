#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:

# import unittest
import numpy as np
from sympy import sin, symbols, Matrix

from testinfrastructure.InDirTest import InDirTest
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.smooth_model_run_14C import SmoothModelRun_14C
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
from CompartmentalSystems.discrete_model_run_14C import DiscreteModelRun_14C
from CompartmentalSystems.model_run import plot_stocks_and_fluxes


class TestDiscreteModelRun_14C(InDirTest):
    def test_from_SmoothModelRun(self):
 #       x_0, x_1, t, k, u = symbols("x_0 x_1 k t u")
 #       inputs = {
 #           0: u,#*(2-2*sin(2*t)),
 #           1: u
 #       }
 #       outputs = {
 #           0: x_0*k,
 #           1: x_1#**2*k
 #       }
 #       internal_fluxes = {
 #           (0, 1): x_0,
 #           (1, 0): 0.5*x_1
 #       }
 #       srm = SmoothReservoirModel(
 #           Matrix(2, 1, [x_0, x_1]),
 #           t,
 #           inputs,
 #           outputs,
 #           internal_fluxes
 #       )

 #       t_max = 2*np.pi
 #       times = np.linspace(0, t_max, 21)
 #       times_fine = np.linspace(0, t_max, 81)
 #       x0 = np.float(10)
 #       start_values = np.array([x0, x0])
 #       parameter_dict = {
 #           k: 0.012,
 #           u: 10.7}

        x, t = symbols('x t')
        B = Matrix(1, 1, [-1])
        u = Matrix(1, 1, [1])
        state_vector = Matrix(1, 1, [x])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)
        parameter_dict = {}
        start_values = np.array([1])
        times = np.arange(0, 6, 1)
 
        smr = SmoothModelRun(srm, parameter_dict, start_values, times)

        alpha = 0.1
        start_values_14C = start_values * alpha
        decay_rate = 1.0

        def Fa_func(t): return alpha
        smr_14C = SmoothModelRun_14C(
            smr,
            start_values_14C,
            Fa_func,
            decay_rate
        )
#        smr_fine = SmoothModelRun(
#            srm,
#            parameter_dict,
#            start_values,
#            times_fine
#        )

        xs, net_Us, net_Fs, net_Rs = smr.fake_net_discretized_output(times)
        xs, gross_Us, gross_Fs, gross_Rs \
            = smr.fake_gross_discretized_output(times)
#        xs_fine, gross_Us_fine, gross_Fs_fine, gross_Rs_fine \
#            = smr_fine.fake_gross_discretized_output(times_fine)
#
        dmr_from_pwc = DMR.from_SmoothModelRun(smr)
        print('X')
        dmr_ref = DMR.from_SmoothModelRun(smr_14C)
        print('Y')
        dmr_from_pwc_14C = DiscreteModelRun_14C(
            dmr_from_pwc,
            start_values_14C,
            #Fa_func,
            dmr_ref.net_Us,
            decay_rate
        )
        print('B')
        print(dmr_ref.Bs)
        print(dmr_from_pwc_14C.Bs)
        print('u')
        #print(net_Us)
        #print(gross_Us)
        print(dmr_ref.net_Us)
        print(dmr_from_pwc_14C.net_Us)

#        dmr_from_fake_net_data = DMR.reconstruct_from_fluxes_and_solution(
#            times,
#            xs,
#            net_Fs,
#            net_Rs
#        )
#        dmr_from_fake_gross_data_ffas \
#            = DMR.reconstruct_from_fluxes_and_solution(
#                times,
#                xs,
#                gross_Fs,
#                gross_Rs
#            )
#        dmr_from_fake_gross_data_ff = DMR.from_fluxes(
#            start_values,
#            times,
#            gross_Us,
#            gross_Fs,
#            gross_Rs
#        )
#        dmr_from_fake_gross_data_ff_fine = DMR.from_fluxes(
#            start_values,
#            times_fine,
#            gross_Us_fine,
#            gross_Fs_fine,
#            gross_Rs_fine
#        )
#
        #print(smr_14C.solve())
        #print(dmr_from_pwc_14C.solve())
        self.assertTrue(
            np.allclose(
                smr_14C.solve(),
                dmr_from_pwc_14C.solve()
            )
        )

#        self.assertTrue(
#            np.allclose(
#                smr.solve(),
#                dmr_from_fake_net_data.solve()
#            )
#        )
#
#        # very unexpectedly the solution
#        # can be reconstructed from the right start_value
#        # the WRONG inputs WRONG internal fluxes and
#        # WRONG outputs
#        self.assertTrue(
#            np.allclose(
#                smr.solve(),
#                dmr_from_fake_gross_data_ff.solve(),
#                rtol=1e-3
#            )
#        )
#
#        # Here we also expect different results.
#        # We again abuse the DiscreteModelRun class
#        # but this time we give it the right solution
#        # which will be reproduced
#        self.assertTrue(
#            np.allclose(
#                smr.solve(),
#                dmr_from_fake_gross_data_ffas.solve()
#            )
#        )
#        # but the net influxes will be wrong
#        self.assertFalse(
#            np.allclose(
#                smr.acc_net_external_input_vector(),
#                dmr_from_fake_gross_data_ffas.net_Us
#            )
#        )
#        plot_stocks_and_fluxes(
#            [
#                smr,
#                # dmr_from_fake_net_data,
#                # dmr_from_pwc,
#                dmr_from_fake_gross_data_ff,
#                dmr_from_fake_gross_data_ff_fine
#            ],
#            'stocks_and_fluxes.pdf'
#        )
