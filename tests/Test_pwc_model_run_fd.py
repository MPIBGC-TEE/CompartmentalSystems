import unittest

import numpy as np
from sympy import Function, Matrix, symbols

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.pwc_model_run_fd import PWCModelRunFD

import os.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestPWCModelRunFD(unittest.TestCase):

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

        parameter_dict = {k: 1}
        func_dict = {u_1: lambda x, t: 9}

        self.smr = SmoothModelRun(
            self.srm,
            parameter_dict,
            start_values,
            times,
            func_dict
        )
            

    def test_reconstruction_accuracy(self):
        smr = self.smr
        xs, gross_Us, gross_Fs, gross_Rs =\
            smr.fake_gross_discretized_output(smr.times)

        # integration_method = 'solve_ivp'
        pwc_mr_fd = PWCModelRunFD.from_gross_fluxes(
            smr.model.time_symbol,
            smr.times,
            xs[0, :],
            gross_Us,
            gross_Fs,
            gross_Rs
        )
    
        with self.subTest():
            self.assertTrue(
                np.allclose(
                    smr.solve(),
                    pwc_mr_fd.solve(),
                    rtol=1e-03
                )
            )
    
        with self.subTest():
            self.assertTrue(
                np.allclose(
                    smr.acc_gross_external_input_vector(),
                    pwc_mr_fd.acc_gross_external_input_vector(),
                    rtol=1e-4
                )
            )
    
        with self.subTest():
            self.assertTrue(
                np.allclose(
                    smr.acc_gross_external_output_vector(),
                    pwc_mr_fd.acc_gross_external_output_vector(),
                    rtol=1e-4
                )
            )
    
        with self.subTest():
            self.assertTrue(
                np.allclose(
                    smr.acc_gross_internal_flux_matrix(),
                    pwc_mr_fd.acc_gross_internal_flux_matrix(),
                    rtol=1e-4
                )
            )
    
        # integration_method = 'trapezoidal'
        # nr_nodes = 3 result in insufficient accuracy
        pwc_mr_fd = PWCModelRunFD.from_gross_fluxes(
            smr.model.time_symbol,
            smr.times,
            xs[0, :],
            gross_Us,
            gross_Fs,
            gross_Rs,
            integration_method='trapezoidal',
            nr_nodes=3
            )
    
        with self.subTest():
            self.assertFalse(
                np.allclose(
                    smr.solve(),
                    pwc_mr_fd.solve(),
                    rtol=1e-03
                )
            )
    
        with self.subTest():
            self.assertTrue(
                np.allclose(
                    smr.acc_gross_external_input_vector(),
                    pwc_mr_fd.acc_gross_external_input_vector(),
                    rtol=1e-04
                )
            )
    
        with self.subTest():
            self.assertFalse(
                np.allclose(
                    smr.acc_gross_external_output_vector(),
                    pwc_mr_fd.acc_gross_external_output_vector(),
                    rtol=1e-04
                )
            )
    
        with self.subTest():
            self.assertFalse(
                np.allclose(
                    smr.acc_gross_internal_flux_matrix(),
                    pwc_mr_fd.acc_gross_internal_flux_matrix(),
                    rtol=1e-04
                )
            )

                
        # integration_method = 'trapezoidal'
        # nr_nodes = 3 result in insufficient accuracy
        pwc_mr_fd = PWCModelRunFD.from_gross_fluxes(
            smr.model.time_symbol,
            smr.times,
            xs[0, :],
            gross_Us,
            gross_Fs,
            gross_Rs,
            integration_method='trapezoidal',
            nr_nodes=151
            )
    
        with self.subTest():
            self.assertTrue(
                np.allclose(
                    smr.solve(),
                    pwc_mr_fd.solve(),
                    rtol=1e-03
                )
            )
    
        with self.subTest():
            self.assertTrue(
                np.allclose(
                    smr.acc_gross_external_input_vector(),
                    pwc_mr_fd.acc_gross_external_input_vector(),
                    rtol=1e-04
                )
            )
    
        with self.subTest():
            self.assertTrue(
                np.allclose(
                    smr.acc_gross_external_output_vector(),
                    pwc_mr_fd.acc_gross_external_output_vector(),
                    rtol=1e-04
                )
            )
    
        with self.subTest():
            self.assertTrue(
                np.allclose(
                    smr.acc_gross_internal_flux_matrix(),
                    pwc_mr_fd.acc_gross_internal_flux_matrix(),
                    rtol=1e-04
                )
            )


###############################################################################


if __name__ == '__main__':
    unittest.main()
