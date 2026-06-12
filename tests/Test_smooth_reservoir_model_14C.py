import unittest

from sympy import Function, Matrix, symbols, simplify

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_reservoir_model_14C import (
    SmoothReservoirModel_14C
)


class TestSmoothReservoirModel_14C(unittest.TestCase):

    def setUp(self):
        x, y, t, k = symbols("x y t k")
        u_1 = Function('u_1')(x, t)
        state_vector = Matrix([x, y])
        B = Matrix([[-1,  1.5],
                    [k, -2]])
        u = Matrix(2, 1, [u_1, 1])
        srm = SmoothReservoirModel.from_B_u(
            state_vector,
            t,
            B,
            u
        )

        decay_symbol = symbols('lamda')
        Fa = Function('Fa')(t)
        self.srm_14C = SmoothReservoirModel_14C(
            srm,
            decay_symbol,
            Fa
        )

    def test_init(self):
        self.assertTrue(isinstance(self.srm_14C, SmoothReservoirModel_14C))
        self.assertTrue(isinstance(self.srm_14C, SmoothReservoirModel))
        self.assertTrue(hasattr(self.srm_14C, 'decay_symbol'))

    def test_output_fluxess_corrected_for_decay(self):
        x, y, k = symbols("x y k")
        ref = {0: x*(1-k), 1: 0.5*y}
        expr_dict = {k: simplify(e) for k, e
                     in self.srm_14C.output_fluxes_corrected_for_decay.items()}
        self.assertEqual(ref, expr_dict)


###############################################################################


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(".", pattern=__file__)
    unittest.main()
