#!/usr/bin/env python
# all array-like data structures are numpy.array
import numpy as np
from copy import copy
from sympy import Matrix, symbols, Symbol, Function, latex, atan ,pi,sin,lambdify,Piecewise

# load the compartmental model packages
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.helpers_reservoir import  numsol_symbolic_system ,numerical_function_from_expression
from testinfrastructure.helpers  import pe

# load other files in the same directory that support this script
from Classes import BastinModel,BastinModelRun
from plotFuncs import panel_one,all_in_one,plot_epsilon_family
import drivers #contains the fossil fuel interpolating functions

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
limited_internal_fluxes  = {
        (0,1): FluxLimiter(F_AT,70),#120
        (0,2): FluxLimiter(F_AS,300), #90-110
        (1,0): FluxLimiter(F_TA,70), #120
        #(1,0): F_TA, 
        (2,0): FluxLimiter(F_SA,300) #90-110
        #(2,0): F_SA
      }

# create the Models
nonlinear_srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

limited_srm = SmoothReservoirModel(state_vector, time_symbol, net_input_fluxes, net_output_fluxes, limited_internal_fluxes)

phi_sym=Function('phi_sym')
u_z_eps_exp=phi_eps(z,epsilon)
control_start=1900


u_t_z_exp=Piecewise((1,time_symbol<control_start),(u_z_eps_exp,True))

bm_phi_num=BastinModel(limited_srm,phi_sym(time_symbol,z),z)

bm_u_exp=BastinModel(limited_srm,u_t_z_exp,z)



# define the time windows of interest
start_year = 1765
end_year = 2500
times = np.arange(start_year, end_year+1,1)# (end_year-start_year)/1000)

# define a dictionary to connect the symbolic function  with the according implementations 
func_dict = {
    u_A: drivers.u_A_func, 
    f_TA: drivers.f_TA_func, 
    assert_non_negative_sym:assert_non_negative_num
}
# the systems start a little higher than the equilibrium
# of the system with unconstrained fluxes
start_values = 1.05*np.array([A_eq, T_eq, S_eq])


# possibly nonlinear effects as a parameter dictionary
par_dict_v1 = {alpha: 0.2, beta: 10.0} # nonlinear
#par_dict_v2 = {alpha: 1.0, beta:  1.0} # linear
par_dict_v1.update({epsilon:10})
u_t_z_exp_par=u_t_z_exp.subs(par_dict_v1)
u_t_z_num=lambdify((time_symbol,z),u_t_z_exp_par,modules=[func_dict,'numpy'])
func_dict_phi_num=copy(func_dict)
func_dict_phi_num.update({phi_sym:u_t_z_num})

    
#f.savefig("limited_fluxes_ast.pdf")
z0=200000
control_start_values = np.array(list(start_values)+[z0])



# we implement the kick in of the control by defining phi_eps as a piecewise function
# with respect to time 
        
# call the plot functions you want and comment the remaining ones
# each of them creates a pdf with the name of the function
all_in_one(nonlinear_srm,limited_srm,bm_u_exp, par_dict_v1, control_start_values, times, func_dict,u_A)
#all_in_one(nonlinear_srm,limited_srm,bm_phi_num, par_dict_v1, control_start_values, times, func_dict_phi_num,u_A)
#epsilons=[1,100,1000]
#plot_epsilon_family( limited_srm, par_dict_v1, control_start_values, times, func_dict, epsilons)    
#panel_one(limited_srm,bm, par_dict_v1, control_start_values, times, func_dict)
