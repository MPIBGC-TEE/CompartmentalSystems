import unittest
import numpy as np
from sympy import symbols, Matrix

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  

from CompartmentalSystems.pwc_model_run import PWCModelRun
from CompartmentalSystems.pwc_model_run_fd import PWCModelRunFD
from CompartmentalSystems.discrete_model_run_with_gross_inputs import DiscreteModelRunWithGrossInputs


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
        times=np.linspace(0,1,11)
        self.times=times

        self.pwc_mr = PWCModelRun(srm, {}, start_values, times)
        xs, Fs, rs, Us = self.pwc_mr._fake_discretized_output(times)
        
        self.pwc_mr_fd = PWCModelRunFD(
            t, times, start_values, xs, Fs, rs, Us)
        
        
        self.dmr_from_pwc = DiscreteModelRunWithGrossInputs.from_PWCModelRun(
            self.pwc_mr)


        self.dmr = DiscreteModelRunWithGrossInputs.reconstruct_from_fluxes_and_solution(
           times, xs, Fs, rs, Us)

    #@unittest.skip
    def test_solve(self):

        self.assertTrue(np.allclose(
            self.pwc_mr.solve(),
            self.dmr_from_pwc.solve()
        ))
        self.assertTrue(np.allclose(
            self.pwc_mr.solve(),
            self.dmr.solve()
        ))
        
        self.assertTrue(np.allclose(
            self.pwc_mr.solve(),
            self.pwc_mr_fd.solve(),
            rtol = 1e-03
        ))

    def test_external_input_vector(self):
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_external_input_vector(),
                self.dmr_from_pwc.acc_external_input_vector()
            )
        )

        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_external_input_vector(),
                self.dmr.acc_external_input_vector()
            )
        )

        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_external_input_vector(),
                self.pwc_mr_fd.acc_external_input_vector()
            )
        )

    @unittest.skip
    def test_acc_internal_flux_matrix(self):
        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_internal_flux_matrix(),
                self.dmr_from_pwc.acc_internal_flux_matrix()
#                rtol=1e-03
            )
        )

        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_internal_flux_matrix(),
                self.dmr.acc_internal_flux_matrix()
#                rtol=1e-03
            )
        )

        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_internal_flux_matrix(),
                self.pwc_mr_fd.acc_internal_flux_matrix()
#                rtol=1e-03
            )
        )

    #@unittest.skip
    def test_acc_external_output_vector(self):
        print('Bs')
        print(self.dmr.Bs)
        print(self.dmr_from_pwc.Bs)

        print('acc_out')
        print(self.dmr.acc_external_output_vector())
        print(self.dmr_from_pwc.acc_external_output_vector())
        self.assertTrue(
            np.allclose(
                self.dmr.acc_external_output_vector(),
                self.dmr_from_pwc.acc_external_output_vector(),
#                rtol=1e-03
            )
        )
        #self.assertTrue(
        #    np.allclose(
        #        self.pwc_mr.acc_external_output_vector(),
        #        self.dmr_from_pwc.acc_external_output_vector(),
#       #         rtol=1e-03
        #    )
        #)

        #self.assertTrue(
        #    np.allclose(
        #        self.pwc_mr.acc_external_output_vector(),
        #        self.dmr.acc_external_output_vector(),
#       #         rtol=1e-03
        #    )
        #)

        #print(self.pwc_mr.acc_external_output_vector())
        #print(self.pwc_mr_fd.acc_external_output_vector())

        self.assertTrue(
            np.allclose(
                self.pwc_mr.acc_external_output_vector(),
                self.pwc_mr_fd.acc_external_output_vector(),
                rtol=1e-03
            )
        )


