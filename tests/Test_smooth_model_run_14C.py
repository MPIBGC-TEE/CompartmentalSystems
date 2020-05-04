
from sympy import  symbols, Matrix, Symbol
import numpy as np
import unittest
from scipy.interpolate import interp1d 

from CompartmentalSystems.smooth_model_run_14C import (
    SmoothModelRun_14C, 
    pfile_C14Atm_NH
) 
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  

class TestSmoothModelRun_14C(unittest.TestCase):
    def test_from_SmoothModelRun(self):
        # we test only that the construction works and can be solved,
        # not the actual solution values
        lamda_1, lamda_2, C_1, C_2 = symbols('lamda_1 lamda_2 C_1 C_2')
        B = Matrix([[-lamda_1,        0],
                    [       0, -lamda_2]])
        u = Matrix(2, 1, [1, 1])
        state_vector = Matrix(2, 1, [C_1, C_2])
        time_symbol = Symbol('t')

        srm = SmoothReservoirModel.from_B_u(state_vector,
                                            time_symbol,
                                            B,
                                            u)

        par_set = {lamda_1: 0.5, lamda_2: 0.2}
        start_values = np.array([7,4])
        start, end, ts = 1950, 2000, 0.5
        times = np.linspace(start, end, int((end+ts-start)/ts))
        smr = SmoothModelRun(srm, par_set, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        soln = smr.solve()
        
        atm_delta_14C = np.loadtxt(pfile_C14Atm_NH(), skiprows=1, delimiter=',')
        F_atm_delta_14C = interp1d(
            atm_delta_14C[:,0],
            atm_delta_14C[:,1],
            fill_value = 'extrapolate'
        )

        alpha = 1.18e-12
        start_values_14C = smr.start_values * alpha
        Fa_func = lambda t: alpha * (F_atm_delta_14C(t)/1000+1)
        smr_14C = SmoothModelRun_14C.from_SmoothModelRun(
            smr,
            start_values_14C,
            Fa_func,
            0.0001
        )
        soln_14C = smr_14C.solve()
