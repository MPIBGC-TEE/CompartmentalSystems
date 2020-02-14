
from concurrencytest import ConcurrentTestSuite, fork_for_tests
import sys
import unittest
import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol,Matrix, symbols, sin, Piecewise, DiracDelta, Function
from CompartmentalSystems.helpers_reservoir import factor_out_from_matrix, parse_input_function, melt, MH_sampling, stride, is_compartmental, func_subs, numerical_function_from_expression,pe
from CompartmentalSystems.start_distributions import \
    start_age_moments_from_empty_spinup, \
    start_age_moments_from_steady_state, \
    start_age_moments_from_zero_initial_content, \
    compute_fixedpoint_numerically, \
    start_age_distributions_from_steady_state, \
    start_age_distributions_from_empty_spinup, \
    start_age_distributions_from_zero_initial_content
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from testinfrastructure.InDirTest import InDirTest
C_0, C_1 = symbols('C_0 C_1')
state_vector = [C_0, C_1]
t = Symbol('t')

f_expr = Function('f')(t)


input_fluxes = {0: C_0*f_expr+2, 1: 2}
output_fluxes = {0: C_0, 1: C_1}
internal_fluxes = {(0,1):0.5*C_0**3}
srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)

parameter_set={}
def f_func( t_val):
    return np.sin(t_val)+1.0

func_set = {f_expr: f_func}

t_min = 0
t_max = 2*np.pi
n_steps=11
times = np.linspace(t_min,t_max,n_steps) 
# create a model run that starts with all pools empty
smr = SmoothModelRun(srm, parameter_set=parameter_set, start_values=np.zeros(srm.nr_pools), times=times,func_set=func_set)
# choose a t_0 somewhere in the times
t0_index = int(n_steps/2)
t0       = times[t0_index]
a_dens_func_t0,pool_contents=start_age_distributions_from_empty_spinup(srm,t_max=t0,parameter_set=parameter_set,func_set=func_set)
pe('pool_contents',locals())

# construct a function p that takes an age array "ages" as argument
# and gives back a three-dimensional ndarray (ages x times x pools)
# from the a array-valued function representing the start age density
p=smr.pool_age_densities_func(start_age_distributions_from_zero_initial_content(srm))

# for this particular example we are only interrested in ages that are smaller than t_max
# the particular choice ages=times means that t_0_ind is the same in both arrays
ages=times 
t0_age_index=t0_index

pool_dens_data=p(ages)
n =0

fig=smr.plot_3d_density_plotly("pool {0}".format(n),pool_dens_data[:,:,n],ages)
# plot the computed start age density for t0 on top
trace_on_surface = go.Scatter3d(
    x=np.array([-t0 for a in ages]),
    y=np.array([a for a in ages]),
    z=np.array([a_dens_func_t0(a)[n] for a in ages]),
    mode = 'lines',
    line=dict(
        color='#FF0000',
        width=15
        )
    #,
    #showlegend = legend_on_surface
)
#smr.add_equilibrium_surface_plotly(fig)
fig.add_scatter3d(
    x=np.array([-t0 for a in ages]),
    y=np.array([a for a in ages]),
    z=np.array([a_dens_func_t0(a)[n] for a in ages]),
    mode = 'lines',
    line=dict(
        color='#FF0000',
        width=15
        )
)
#plot(fig,filename="test_{0}.html".format(n),auto_open=False)
plot(fig,filename="test_{0}.html".format(n))

# make sure that the values for the model run at t0 conince with the values computed by the             # function returned by the function under test
res_data=np.array([a_dens_func_t0(a)[n] for a in ages])
ref_data=pool_dens_data[:,t0_index,n]
self.assertTrue(np.allclose(res_data,ref_data,rtol=1e-3))

# make sure that the density is zero for all values of age bigger than t0
self.assertTrue(np.all(res_data[t0_age_index:]==0))
