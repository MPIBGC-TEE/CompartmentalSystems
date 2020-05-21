import unittest

import numpy as np

from sympy import Function, Matrix, symbols

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.pwc_model_run_fd import PWCModelRunFD


class TestPWCModelRunFD(unittest.TestCase):

#    def test_init(self):
#        x, y, t, k = symbols("x y t k")
#        u_1 = Function('u_1')(x, t)
#        state_vector = Matrix([x, y])
#        B = Matrix([[-1,  1.5],
#                    [ k, -2  ]])  # noqa: E201, E202
#        u = Matrix(2, 1, [u_1, 1])
#        srm = SmoothReservoirModel.from_B_u(
#            state_vector,
#            t,
#            B,
#            u
#        )
#
#        start_values = np.array([10, 40])
#        t_0 = 0
#        t_max = 10
#        disc_times = [5]
#
#        ps = [{k: 1}, {k: 0.5}]
#        fs = [{u_1: lambda x, t: 9}, {u_1: lambda x, t: 3}]
#        timess = [
#            np.linspace(t_0, disc_times[0], 10),
#            np.linspace(disc_times[0], t_max, 10)
#        ]
#
#        smrs = []
#        tmp_start_values = start_values
#        for i in range(len(disc_times)+1):
#            smrs.append(
#                SmoothModelRun(
#                    srm,
#                    parameter_dict=ps[i],
#                    start_values=tmp_start_values,
#                    times=timess[i],
#                    func_set=fs[i]
#                )
#            )
#            tmp_start_values = smrs[i].solve()[-1]
#
#        gross_Us = np.concatenate(
#            [
#                smr.acc_gross_external_input_vector(
#                    np.array([smr.times[0], smr.times[-1]])
#                )
#                for smr in smrs
#            ],
#            axis=0
#        )
#
#        gross_Fs = np.concatenate(
#            [
#                smr.acc_gross_internal_flux_matrix(
#                    np.array([smr.times[0], smr.times[-1]])
#                )
#                for smr in smrs
#            ],
#            axis=0
#        )
#
#        gross_Rs = np.concatenate(
#            [
#                smr.acc_gross_external_output_vector(
#                    np.array([smr.times[0], smr.times[-1]])
#                )
#                for smr in smrs
#            ],
#            axis=0
#        )
#
#        pwc_mr_fd = PWCModelRunFD(  # noqa: F841
#            t,
#            np.array([t_0]+disc_times+[t_max]),
#            start_values,
#            gross_Us,
#            gross_Fs,
#            gross_Rs
#        )


    def test_pwc_with_logm(self):
        x, y, t, k = symbols("x y t k")
        u_1 = Function('u_1')(x, t)
        state_vector = Matrix([x, y])
        B = Matrix([[-0.1,  0.15],
                    [   k, -0.2  ]])  # noqa: E201, E202
        u = Matrix(2, 1, [u_1, 1])
        srm = SmoothReservoirModel.from_B_u(
            state_vector,
            t,
            B,
            u
        )

        start_values = np.array([100, 100])
        t_0 = 0
        t_max = 10
        disc_times = [5]

        ps = [{k: 0.1}, {k: 0.05}]
        fs = [{u_1: lambda x, t: 9}, {u_1: lambda x, t: 3}]
        timess = [
            np.linspace(t_0, disc_times[0], 6),
            np.linspace(disc_times[0], t_max, 6)
        ]

        smrs = []
        tmp_start_values = start_values
        for i in range(len(disc_times)+1):
            smrs.append(
                SmoothModelRun(
                    srm,
                    parameter_dict=ps[i],
                    start_values=tmp_start_values,
                    times=timess[i],
                    func_set=fs[i]
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
            axis=0
        )

        gross_Fs = np.concatenate(
            [
                smr.acc_gross_internal_flux_matrix(
                    np.array([smr.times[0], smr.times[-1]])
                )
                for smr in smrs
            ],
            axis=0
        )

        gross_Rs = np.concatenate(
            [
                smr.acc_gross_external_output_vector(
                    np.array([smr.times[0], smr.times[-1]])
                )
                for smr in smrs
            ],
            axis=0
        )

        pwc_mr_fd = PWCModelRunFD(  # noqa: F841
            t,
            np.array([t_0]+disc_times+[t_max]),
            start_values,
            gross_Us,
            gross_Fs,
            gross_Rs
        )

        # test solve
        soln_smrs = [smr.solve() for smr in smrs]
        L = [start_values.reshape(1, 2)] + [soln[-1,:].reshape(1, 2) for soln in soln_smrs]
        soln_ref = np.concatenate(L, axis=0)
        self.assertTrue(np.allclose(soln_ref, pwc_mr_fd.solve()))
        

        from scipy.linalg import expm
        from CompartmentalSystems.discrete_model_run import DiscreteModelRun

        B0 = pwc_mr_fd.Bs[0]
        dt0 = pwc_mr_fd.dts[0]
        print(B0)
        print(expm(B0*dt0))
        B0_dmr_gross = DiscreteModelRun.reconstruct_B(
            start_values,
            gross_Fs[0],
            gross_Rs[0]
        )
        print('B0_dmr_gross')
        print(B0_dmr_gross)

        data_times = np.array([0, 5, 10])
        B0_dmr_net = DiscreteModelRun.reconstruct_B(
            start_values,
            pwc_mr_fd.acc_net_internal_flux_matrix()[0],
            pwc_mr_fd.acc_net_external_output_vector()[0]
        )
        print('B0_dmr_net')
        print(B0_dmr_net)

        print('fake_Bs')
        print(pwc_mr_fd.fake_discretized_Bs(data_times)[0])
 

###############################################################################


if __name__ == '__main__':
    unittest.main()
