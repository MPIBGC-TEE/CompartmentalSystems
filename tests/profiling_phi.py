import timeit
import cProfile
import pstats
import numpy as np
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function, ones
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel

def test_age_densities_1_single_value():
    # one-dimensional
    C = Symbol('C')
    state_vector = [C]
    time_symbol = Symbol('t')
    input_fluxes = {}
    output_fluxes = {0: C}
    internal_fluxes = {}
    srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

    start_values = np.array([5])
    times = np.linspace(0,1,6)
    smr = SmoothModelRun(srm, {}, start_values, times)

    start_age_densities = lambda a: np.exp(-a)*start_values
    p1_sv = smr._age_densities_1_single_value(start_age_densities)

    # negative ages will be cut off automatically
    ages = np.linspace(-1,1,3)
    a_ref = np.array([[[ 0.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [ 0.        ]],
                     
                      [[ 5.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [ 0.        ]],
                     
                      [[ 1.83939721],
                       [ 1.83939724],
                       [ 1.83939725],
                       [ 1.83939724],
                       [ 1.83939729],
                       [ 1.83939727]]])
             
    ref = np.ndarray((3,6,1), np.float, a_ref)
    res_l = [[p1_sv(a, t) for t in times] for a in ages]
    res = np.array(res_l)

    # test missing start_age_densities
    a_ref = np.array([[[ 0.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [ 0.        ]],
                     
                      [[ 5.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [ 0.        ]],
                     
                      [[ 0         ],
                       [ 0         ],
                       [ 0         ],
                       [ 0         ],
                       [ 0         ],
                       [ 1.83939727]]])
    
    ref = np.ndarray((3,6,1), np.float, a_ref)
    p1_sv = smr._age_densities_1_single_value()
    res_l = [[p1_sv(a,t) for t in times] for a in ages]
    res = np.array(res_l)

    # two-dimensional
    C_0, C_1 = symbols('C_0 C_1')
    state_vector = [C_0, C_1]
    time_symbol = Symbol('t')
    input_fluxes = {}
    output_fluxes = {0: C_0, 1: C_1}
    internal_fluxes = {}
    srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

    start_values = np.array([5, 3])
    times = np.linspace(0,1,6)
    smr = SmoothModelRun(srm, {}, start_values, times)

    ages = np.linspace(-1,1,3)
    # negative ages will be cut off automatically
    start_age_densities = lambda a: np.exp(-a)*start_values
    p1_sv = smr._age_densities_1_single_value(start_age_densities)

    a_ref = np.array(
            [[[ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ]],
            
             [[ 5.        ,  3.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ]],
            
             [[ 1.83939721,  1.10363832],
              [ 1.83939724,  1.10363834],
              [ 1.83939725,  1.10363835],
              [ 1.83939724,  1.10363835],
              [ 1.83939729,  1.10363837],
              [ 1.83939727,  1.10363836]]])

    ref = np.ndarray((3,6,2), np.float, a_ref)
    res_l = [[p1_sv(a,t) for t in times] for a in ages]
    res = np.array(res_l)

    # test missing start_age_densities
    a_ref = np.array(
            [[[ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ]],
            
             [[ 5.        ,  3.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ],
              [ 0.        ,  0.        ]],
            
             [[ 0         ,  0         ],
              [ 0         ,  0         ],
              [ 0         ,  0         ],
              [ 0         ,  0         ],
              [ 0         ,  0         ],
              [ 1.83939727,  1.10363836]]])

    ref = np.ndarray((3,6,2), np.float, a_ref)
    p1_sv = smr._age_densities_1_single_value()
    res_l = [[p1_sv(a,t) for t in times] for a in ages]
    res = np.array(res_l)
# main
#res=timeit.timeit(
#    test_age_densities_1_single_value
#    ,number=10
#)
#print('res',res)

with cProfile.Profile() as pr:
    test_age_densities_1_single_value()

st=pstats.Stats(pr)
st.sort_stats('time')
st.print_stats()    


