import timeit
from copy import deepcopy
import time
import cProfile
import pstats
import numpy as np
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function, ones
from CompartmentalSystems.moothmodel_run import SmoothModelRun
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel

def smr_1d(nc):
    #one-dimensional
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
    smr.build_state_transition_operator_cache(nc)
    return deepcopy(smr)


def smr_2d(nc):
    # two-dimensional
    C_0, C_1 = symbols('C_0 C_1')
    state_vector = [C_0, C_1]
    time_symbol = Symbol('t')
    input_fluxes = {}
    output_fluxes = {0: C_0, 1: C_1}
    internal_fluxes = {}
    srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

    start_values = np.array([5, 3])
    times = np.linspace(0,1,100)
    smr = SmoothModelRun(srm, {}, start_values, times)
    smr.build_state_transition_operator_cache(nc)
    return deepcopy(smr)

def age_densities(smr):#_1D(smr):
    start_age_densities = lambda a: np.exp(-a)*smr.start_values
    p=smr.pool_age_densities_func(start_age_densities)
    p1_sv = smr._age_densities_1_single_value(start_age_densities)

    # negative ages will be cut off automatically
    ages = np.linspace(-1,1,3)
    res=p(ages)        
# main
reps=10
def funcmaker(f,*args):
    def f_wihtout_args():
        return f(*args)

    return f_wihtout_args

for smr_func in [smr_1d,smr_2d]:
    print('#####################################')
    for nc in [10,100,1000]:#,10000]:
        smr=smr_func(nc)
        res=timeit.timeit(
            #funcmaker(age_densities_1_single_value_2D,smr)
            funcmaker(age_densities,smr)
            ,number=10
        )
        print('res',res)

#with cProfile.Profile() as pr:
#    test_age_densities_1_single_value()
#
#st=pstats.Stats(pr)
#st.sort_stats('time')
#st.print_stats()    


