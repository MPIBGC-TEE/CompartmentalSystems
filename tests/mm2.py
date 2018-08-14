from concurrencytest import ConcurrentTestSuite, fork_for_tests
import inspect
import matplotlib
import sys 
import unittest
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
from plotly.offline import plot
from pathlib import Path
from scipy.integrate import quad
from scipy.interpolate import interp1d 
from scipy.linalg import inv
from scipy.misc import factorial
from bgc_md.Model import Model
    
import example_smooth_reservoir_models as ESRM
import example_smooth_model_runs as ESMR
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.helpers_reservoir import numerical_function_from_expression
from CompartmentalSystems.smooth_model_run import SmoothModelRun 
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function, ones,lambdify
C_0, C_1 = symbols('C_0 C_1')
t = Symbol('t')

#u_0_expr = Function('u_0')(C_0, C_1, t)
u_0_expr = Function('u_0')(t)

X = Matrix([C_0, C_1 ])
t_min, t_max = 0,2 
symbolic_input_fluxes = {0: u_0_expr, 1: u_0_expr}

#def u0_func(C_0_val, C_1_val, t_val):
#    return C_0_val*0 + C_1_val*0 + u_0_interp(t_val)
def u0_func(t):
    return (np.cos(2*t)+1)*5

parameter_set={}
parameter_set = {'T_s0': 15,
               'sigma': 4.5,
               'xi_b': 2,
               's_0': 100,
               'f_i': 0.33,
               'alpha': 0.5, # not in the code --> duble F_NPP0 to 100
               'rho': '0.65', # value of x in the paper
               'b_11': 0.67,
               'b_22': 0.2,
               'b_33': 0.04,
               'b_41': 0.5092, 'b_42': 0.0260, 'b_44': 2.5,
               'b_51': 0.1608, 'b_52': 0.1740, 'b_55': 0.4,
               'b_63': 0.04, 'b_66': 0.25,
               'b_74': 1.1250, 'b_75': 0.1530, 'b_76': 0.06, 'b_77': 0.7,
                   'b_78': 0.0103, 'b_79': 0.00016875,
               'b_85': 0.042, 'b_86': 0.07, 'b_87': 0.3525, 'b_88': 0.023,
                   'b_89': 0, # following the code
               'b_97': 0.0045, 'b_98': 5.8995e-05, 'b_99': 0.000375}
func_set = {u_0_expr: u0_func}

output_fluxes =  {0: 2*C_0, 1: 2*C_1}
internal_fluxes = {}
#{(0,1): C_0, (1,0): C_1}
srm = SmoothReservoirModel(
    X, 
    t, 
    symbolic_input_fluxes, 
    output_fluxes, 
    internal_fluxes
)

m=Model.from_file("/home/mm/bgc-md/bgc_md/data/all_records/Rasmussen2016JMB.yaml")
srm=m.reservoir_model
u_sym=srm.external_inputs
print(u_sym)
B_sym=srm.compartmental_matrix
print(B_sym)
# we want B and u as python functions 
#str_func_set = {str(key): val for key, val in func_set.items()}
#cut_func_set = {key[:key.index('(')]: val 
#    for key, val in str_func_set.items()}
#
#tup = tuple(X) + (t,)
#u_par=u_sym.subs(parameter_set)
#print(u_par)
#u_func = lambdify(tup, u_par, modules=[cut_func_set, 'numpy'])
tup = tuple(X) + (t,)
tup = (t,) # for the linear case 
u_func=numerical_function_from_expression(u_sym,tup,parameter_set,func_set)
u_func(0)
B_func=numerical_function_from_expression(B_sym,tup,parameter_set,func_set)
B0=B_func(0)
u0=u_func(0)
x0=-inv(B0)@u0
start_values = x0.reshape(srm.nr_pools) # np.array([1, 2, 3])
print(x0)
lapm=LinearAutonomousPoolModel(Matrix(u0),Matrix(B0))
a_dens_function =lambda y:start_values*np.array(lapm.a_density(y)).astype(np.float).reshape(srm.nr_pools)
mean_age_vec=lapm.a_expected_value



#times = np.linspace(t_min,t_max, 31)
times = np.linspace(0, 650, 211)
#times = np.linspace(0, 65, 21)
smr = SmoothModelRun(srm, parameter_set=parameter_set, start_values=start_values, times=times,func_set=func_set)
fn='sto.cache'
try:
    smr.load_state_transition_operator_cache(fn)
#except FileNotFoundError:
except Exception as e:
    print(e)
    smr.build_state_transition_operator_cache(1001)
    smr.save_state_transition_operator_cache(fn)


p=smr.pool_age_densities_func(a_dens_function)
ages=np.linspace(0,50,51)
dfn="age_dens.csv"
try:
    dens_data=smr.load_pools_and_system_density_csv(dfn,ages)
    pool_age_densities = dens_data[:,:,:smr.nr_pools]
    system_age_density = dens_data[:,:,smr.nr_pools]

except Exception:
    pool_dens_data=p(ages)
    system_dens_data=smr.system_age_density(pool_dens_data)
    smr.save_pools_and_system_density_csv(dfn,pool_dens_data,system_dens_data,ages)

fig=smr.plot_3d_density_plotly('pool 1',pool_dens_data[:,:,0],ages)
trace_on_surface = go.Scatter3d(
    #name=name,
    #x=-strided_times, y=strided_data, z=strided_z,
    #x=[-times[5:10]],
    #y=ages,
    x=np.array([-times[0] for a in ages]),
    y=np.array([a for a in ages]),
    z=np.array([a_dens_function(a)[0] for a in ages]),
    #z=np.array([2 for a in ages]),
    mode = 'lines',
    line=dict(
        color='#FF0000',
        width=15
        )
    #,
    #showlegend = legend_on_surface
)
smr.add_equilibrium_surface_plotly(fig)
fig['data'] += [trace_on_surface]
#fig['data'] = [trace_on_surface]
plot(fig,filename='test.html')

mean_age_arr=np.array(lapm.a_expected_value).astype(np.float).reshape(srm.nr_pools)
mpfig=plt.figure()
smr.plot_mean_ages(mpfig,mean_age_arr)
#mpfig2=plt.figure()
#smr.plot_solutions(mpfig2)
plt.show()
