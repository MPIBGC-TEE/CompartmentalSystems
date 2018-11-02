#!/usr/bin/env python
# all array-like data structures are numpy.array
import numpy as np
from copy import copy
from string import Template
from sympy import Matrix, symbols, Symbol, Function, latex, atan ,pi,sin,lambdify,Piecewise
import matplotlib.pyplot as plt
# load the compartmental model packages
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.helpers_reservoir import  numsol_symbolic_system ,numerical_function_from_expression
from testinfrastructure.helpers  import pe

# load other files in the same directory that support this script
from Classes import BastinModel,BastinModelRun
import plotFuncs #panel_one,all_in_one,plot_epsilon_family
import drivers #contains the fossil fuel and TB->At interpolating functions
from limiters import cubic,deceleration,atan_ymax,half_saturation #


########## symbol and symbolic function definitions ##########
assert_non_negative_sym=Function('assert_non_negative_sym')
def assert_non_negative_num(Net_SD_DS): # asserts that the netflux is non negative
    assert(Net_SD_DS>0) 
    return Net_SD_DS

def phi_eps(z_sym,eps_sym):
    return z_sym/(eps_sym+z_sym)

def FluxLimiter(expression,limit):
    sf=2/pi*limit
    res=sf*atan(1/sf*expression)
    return res
# time symbol
time_symbol = symbols('t')
# Atmosphere, Terrestrial Carbon and Surface ocean
C_A, C_T, C_S = symbols('C_A C_T C_S')
z= symbols('z') #virtual pool for controller
epsilon=Symbol('epsilon')# parameter for 
# fossil fuel inputs
u_A = Function('u_A')(time_symbol)
#u = 1+sin(time_symbol/50)

eps=Symbol("epsilon")
# land use change flux
f_TA = Function('f_TA')
# nonlinear effects
alpha, beta = symbols('alpha beta')

########## model structure: equilibrium values and fluxes ##########

# equilibrium values
A_eq, T_eq, S_eq = (700.0, 3000.0, 1000.0)

state_vector = Matrix([C_A, C_T, C_S])

# fluxes
    

F_AT = 60.0*(C_A/A_eq)**alpha
F_AS = 100.0*C_A/A_eq
F_TA = 60.0*C_T/T_eq + f_TA(time_symbol)
F_SA = 100.0*(C_S/S_eq)**beta
F_DS = 45.0
F_SD=F_DS*C_S/S_eq
input_fluxes = {0: u_A, 1: 0, 2: F_DS} 
output_fluxes = {2: F_SD}
net_input_fluxes = {0: u_A } 
net_output_fluxes = {2: assert_non_negative_sym(F_SD-F_DS)}
internal_fluxes = {(0,1): F_AT, (0,2): F_AS,
                   (1,0): F_TA, (2,0): F_SA}
3
lim_inf_300  = {
        (0,1): FluxLimiter(F_AT,70),#120
        (0,2): FluxLimiter(F_AS,300), #90-110
        (1,0): FluxLimiter(F_TA,70), #120
        #(1,0): F_TA, 
        (2,0): FluxLimiter(F_SA,300) #90-110
        #(2,0): F_SA
      }

lim_inf_90  = {
        (0,1): FluxLimiter(F_AT,70),#120
        (0,2): FluxLimiter(F_AS,90), #90-110
        (1,0): FluxLimiter(F_TA,70), #120
        #(1,0): F_TA, 
        (2,0): FluxLimiter(F_SA,90) #90-110
        #(2,0): F_SA
      }

#phi_sym=Function('phi_sym')
u_z_exp=phi_eps(z,epsilon)
z_max=Symbol("z_max")
u_z_exp=cubic(z,z_max)
control_start=1900


u_t_z_exp=Piecewise((1,time_symbol<control_start),(u_z_exp,True))

start_values = 1.01*np.array([A_eq, T_eq, S_eq])
# define a dictionary to connect the symbolic function  with the according implementations 
func_dict = {
    u_A: drivers.u_A_func, 
    f_TA: drivers.f_TA_func, 
    assert_non_negative_sym:assert_non_negative_num
}
par_dict_v1 = {alpha: 0.2, beta: 10.0} # nonlinear
#par_dict_v2 = {alpha: 1.0, beta:  1.0} # linear
par_dict_v2 = copy(par_dict_v1)
par_dict_v1.update({epsilon:10})
# define the time windows of interest
#start_year = 100
start_year = 1765
end_year = 2500
times = np.arange(start_year, end_year+1,1)# (end_year-start_year)/1000)

fig=plt.figure(figsize=(24,16))
subplotArr=fig.subplots(4,3)
# the following call has a (desired) side effect on the subplots
par_dict_v2.update({z_max:100})
plotFuncs.model_run(
        state_vector, 
        time_symbol, 
        net_input_fluxes, 
        lim_inf_300, 
        net_output_fluxes, 
        internal_fluxes,
        z,
        epsilon,
        u_t_z_exp,
        u_A,
        f_TA,
        func_dict,
        #par_dict_v1,
        par_dict_v2,
        start_values, 
        z0=par_dict_v2[z_max],
        #z0=0,
        times=times,
        subplots=subplotArr[:,0]
)
par_dict_v2.update({z_max:500})
plotFuncs.model_run(
        state_vector, 
        time_symbol, 
        net_input_fluxes, 
        lim_inf_300, 
        net_output_fluxes, 
        internal_fluxes,
        z,
        epsilon,
        u_t_z_exp,
        u_A,
        f_TA,
        func_dict,
        #par_dict_v1,
        par_dict_v2,
        start_values, 
        #z0=200,
        z0=par_dict_v2[z_max],
        times=times,
        subplots=subplotArr[:,1]
)
#par_dict_v2[epsilon]=100
par_dict_v2.update({z_max:2500})
plotFuncs.model_run(
        state_vector, 
        time_symbol, 
        net_input_fluxes, 
        lim_inf_300, 
        net_output_fluxes, 
        internal_fluxes,
        z,
        epsilon,
        u_t_z_exp,
        u_A,
        f_TA,
        func_dict,
        #par_dict_v1,
        par_dict_v2,
        start_values, 
        #z0=2000,
        z0=par_dict_v2[z_max],
        times=times,
        subplots=subplotArr[:,2]
)
plt.subplots_adjust(hspace=0.4)
#file_name=my_func_name()+'_'+file_name_str+'.pdf'
file_name='compare.pdf'
fig.savefig(file_name)
