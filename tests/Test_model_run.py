import unittest
import numpy as np
from sympy import symbols, Matrix

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  

from CompartmentalSystems.pwc_model_run import PWCModelRun
from CompartmentalSystems.pwc_model_run_fd import PWCModelRunFD
from CompartmentalSystems.discrete_model_run import DiscreteModelRun


class TestModelRun(unittest.TestCase):
    def test_roundtrip(self):
        # copied from test_age_moment_vector
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([1,1])
        times=np.linspace(0,1,10)
        pwc_mr = PWCModelRun(srm, {}, start_values, times)
        #pwc_mr.initialize_state_transition_operator_cache(lru_maxsize=None)
        dmr_1 = DiscreteModelRun.from_PWCModelRun(pwc_mr)

        xs, Fs, rs, us = pwc_mr._fake_discretized_output(times)
        dmr_2 = DiscreteModelRun.reconstruct_from_data(times, start_values,xs, Fs, rs, us)
        self.assertTrue(np.all(pwc_mr.solve()==dmr_1.solve()))
        
        pwc_mr_fd = PWCModelRunFD(t, times, start_values, xs , Fs, rs, us)
        self.assertTrue(np.allclose(pwc_mr.solve(),pwc_mr_fd.solve(),rtol=1e03))

