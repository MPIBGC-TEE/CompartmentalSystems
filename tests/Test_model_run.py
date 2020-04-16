import unittest
import numpy as np
from sympy import symbols, Matrix

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  

from CompartmentalSystems.pwc_model_run import PWCModelRun
from CompartmentalSystems.pwc_model_run_fd import PWCModelRunFD
from CompartmentalSystems.discrete_model_run import DiscreteModelRun


class TestModelRun(unittest.TestCase):
    def setUp(self):
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([1,1])
        self.start_values = start_values
        times=np.linspace(0,1,10)
        self.times=times

        self.pwc_mr = PWCModelRun(srm, {}, start_values, times)
        xs, Fs, rs, us = self.pwc_mr._fake_discretized_output(times)
        self.pwc_mr_fd = PWCModelRunFD(t, times, start_values, xs , Fs, rs, us)
        self.dmr_from_pwc = DiscreteModelRun.from_PWCModelRun(self.pwc_mr)

    def test_roundtrip(self):

        xs, Fs, rs, us = self.pwc_mr._fake_discretized_output(self.times)
        dmr_2 = DiscreteModelRun.reconstruct_from_data(self.times, self.start_values,xs, Fs, rs, us)
        self.assertTrue(np.all(self.pwc_mr.solve()==self.dmr_from_pwc.solve()))
        
        self.assertTrue(np.allclose(self.pwc_mr.solve(),self.pwc_mr_fd.solve(),rtol=1e-3))

    def test_external_input_vector(self):
        self.assertTrue(
            np.allclose(
                self.pwc_mr.external_input_vector[:-1],
                self.dmr_from_pwc.external_input_vector,
#                rtol=1e-03
            )
        )

    def test_internal_flux_matrix(self):
        self.assertTrue(
            np.allclose(
                self.pwc_mr.internal_flux_matrix[:-1],
                self.dmr_from_pwc.internal_flux_matrix,
#                rtol=1e-03
            )
        )

    def test_external_output_vector(self):
        self.assertTrue(
            np.allclose(
                self.pwc_mr.external_output_vector[:-1],
                self.dmr_from_pwc.external_output_vector,
#                rtol=1e-03
            )
        )
