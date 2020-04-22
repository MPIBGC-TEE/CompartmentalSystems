import unittest
import numpy as np
from sympy import symbols, Matrix

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  

from CompartmentalSystems.pwc_model_run import PWCModelRun
from CompartmentalSystems.pwc_model_run_fd import PWCModelRunFD
from CompartmentalSystems.discrete_model_run_with_gross_fluxes import DiscreteModelRunWithGrossFluxes
from CompartmentalSystems.model_run import (
    plot_attributes,
    plot_stocks_and_net_fluxes,
    plot_stocks_and_gross_fluxes
)

class TestModelRun(unittest.TestCase):
    def setUp(self):
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1,    2],
                    [ 0.5, -2]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([1,1])
        self.start_values = start_values
        times=np.linspace(0,1,11)
        self.times=times

        self.pwc_mr = PWCModelRun(srm, {}, start_values, times)
        xs, net_Us, net_Fs, net_Rs = self.pwc_mr.fake_net_discretized_output(times)
        xs, gross_Us, gross_Fs, gross_Rs = self.pwc_mr.fake_gross_discretized_output(times)
        
        self.pwc_mr_fd = PWCModelRunFD(
            t
		    ,times
		    ,start_values
		    ,xs
		    ,gross_Us
		    ,gross_Fs
		    ,gross_Rs
        )
        
        self.dmr_from_pwc = DiscreteModelRunWithGrossFluxes.from_PWCModelRun(
            self.pwc_mr)

        self.dmr_from_fake_data = DiscreteModelRunWithGrossFluxes.reconstruct_from_fluxes_and_solution(
           times,
           xs,
           net_Us,
           net_Fs,
           net_Rs,
           gross_Us,
           gross_Fs,
           gross_Rs
        )

    #@unittest.skip
    def test_solve(self):
        self.assertTrue(np.allclose(
            self.pwc_mr.solve(),
            self.dmr_from_pwc.solve()
        ))
        self.assertTrue(np.allclose(
            self.pwc_mr.solve(),
            self.dmr_from_fake_data.solve()
        ))
        
        self.assertTrue(np.allclose(
            self.pwc_mr.solve(),
            self.pwc_mr_fd.solve(),
            rtol = 1e-03
        ))

    #@unittest.skip
    def test_acc_gross_external_input_vector(self):
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_gross_external_input_vector(),
                self.dmr_from_pwc.acc_gross_external_input_vector()
            )
        )

        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_gross_external_input_vector(),
                self.dmr_from_fake_data.acc_gross_external_input_vector()
            )
        )
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_gross_external_input_vector(),
                self.pwc_mr_fd.acc_gross_external_input_vector()
            )
        )

    def test_acc_net_external_input_vector(self):
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_net_external_input_vector(),
                self.dmr_from_pwc.acc_net_external_input_vector()
            )
        )
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_net_external_input_vector(),
                self.dmr_from_fake_data.acc_net_external_input_vector()
            )
        )
        # the rtol is due to inaccuracies in the numeric
        # computation of the xs and would not occure if
        # the piecewise solution was exactly identical to 
        # the continous, although they should be 
        # identical.
        # It is NOT due to the difference between net and gross 
        # fluxes.
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_net_external_input_vector(),
                self.pwc_mr_fd.acc_net_external_input_vector(),
                rtol = 3.75e-03
            )
        )

    #@unittest.skip
    def test_acc_gross_internal_flux_matrix(self):
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_gross_internal_flux_matrix(),
                self.dmr_from_pwc.acc_gross_internal_flux_matrix()
            )
        )

        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_gross_internal_flux_matrix(),
                self.dmr_from_fake_data.acc_gross_internal_flux_matrix()
            )
        )
        # the rtol is due to inaccuracies in the numeric
        # computation of the xs and would not occure if
        # the piecewise solution was exactly identical to 
        # the continous, although they should be 
        # identical for constant B.
        # It is NOT due to the difference between net and gross 
        # fluxes.
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_gross_internal_flux_matrix(),
                self.pwc_mr_fd.acc_gross_internal_flux_matrix(),
                rtol = 3.75e-03
            )
        )

    def test_acc_net_internal_flux_matrix(self):
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_net_internal_flux_matrix(),
                self.dmr_from_pwc.acc_net_internal_flux_matrix()
            )
        )
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_net_internal_flux_matrix(),
                self.dmr_from_fake_data.acc_net_internal_flux_matrix()
            )
        )
        # the rtol is due to inaccuracies in the numeric
        # computation of the xs and would not occure if
        # the piecewise solution was exactly identical to 
        # the continous, although they should be 
        # identical.
        # It is NOT due the difference between net and gross 
        # fluxes
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_net_internal_flux_matrix(),
                self.pwc_mr_fd.acc_net_internal_flux_matrix(),
                rtol = 3.75e-03
            )
        )

    #@unittest.skip
    def test_acc_gross_external_output_vector(self):
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_gross_external_output_vector(),
                self.dmr_from_pwc.acc_gross_external_output_vector()
            )
        )

        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_gross_external_output_vector(),
                self.dmr_from_fake_data.acc_gross_external_output_vector()
            )
        )
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_gross_external_output_vector(),
                self.pwc_mr_fd.acc_gross_external_output_vector(),
                rtol = 3.75e-03
            )
        )

    def test_acc_net_external_output_vector(self):
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_net_external_output_vector(),
                self.dmr_from_pwc.acc_net_external_output_vector()
            )
        )
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_net_external_output_vector(),
                self.dmr_from_fake_data.acc_net_external_output_vector()
            )
        )
        # the rtol is due to inaccuracies in the numeric
        # computation of the xs and would not occure if
        # the piecewise solution was exactly identical to 
        # the continous, although they should be 
        # identical.
        # It is NOT due to the difference between net and gross 
        # fluxes.
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_net_external_output_vector(),
                self.pwc_mr_fd.acc_net_external_output_vector(),
                rtol = 3.75e-03
            )
        )



