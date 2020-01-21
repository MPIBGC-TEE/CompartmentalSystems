#!/usr/bin/env python
# all array-like data structures are numpy.array
import numpy as np
from copy import copy
from string import Template
from sympy import Matrix, symbols, Symbol, Function, latex, atan, pi, sin, lambdify, Piecewise 
import matplotlib.pyplot as plt
# load the compartmental model packages
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.helpers_reservoir import  numsol_symbolic_system ,numerical_function_from_expression
from testinfrastructure.helpers  import pe

# load other files in the same directory that support this script
from Classes import BastinModel, BastinModelRun
import plotFuncs #panel_one,all_in_one,plot_epsilon_family
import drivers #contains the fossil fuel and TB->At interpolating functions
import  limiters # module of different phi expression


########## symbol and symbolic function definitions ##########
assert_non_negative_sym=Function('assert_non_negative_sym')
def assert_non_negative_num(Net_SD_DS): # asserts that the netflux is non negative
    
    #assert(np.array(Net_SD_DS>0).all()) 
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
# fossil fuel inputs
u_A = Function('u_A')(time_symbol)
#u = 1+sin(time_symbol/50)

# land use change flux
#f_TA = Function('f_TA')
# nonlinear effects
alpha, beta = symbols('alpha beta')

########## model structure: equilibrium values and fluxes ##########

# equilibrium values
A_eq, T_eq, S_eq = (700.0, 3000.0, 1000.0)

state_vector = Matrix([C_A, C_T, C_S])

# fluxes
F_AT = 60.0*(C_A/A_eq)**alpha
F_AS = 100.0*C_A/A_eq
F_TA = 60.0*C_T/T_eq #+ f_TA(time_symbol)
F_SA = 100.0*(C_S/S_eq)**beta
F_DS = 45.0
F_SD=F_DS*C_S/S_eq
input_fluxes = {0: u_A, 1: 0, 2: F_DS} 
output_fluxes = {2: F_SD}
net_input_fluxes = {0: u_A } 
net_output_fluxes = {2: assert_non_negative_sym(F_SD-F_DS)}
internal_fluxes = {(0,1): F_AT, (0,2): F_AS,
                   (1,0): F_TA, (2,0): F_SA}

lim_inf  = {
        (0,1): FluxLimiter(F_AT,90),#120
        (0,2): FluxLimiter(F_AS,200), #90-110
        (1,0): FluxLimiter(F_TA,90), #120
        #(1,0): F_TA, 
        (2,0): FluxLimiter(F_SA,200) #90-110
        #(2,0): F_SA
      }

#lim_inf_90  = {
#        (0,1): FluxLimiter(F_AT,70),#120
#        (0,2): FluxLimiter(F_AS,90), #90-110
#        (1,0): FluxLimiter(F_TA,70), #120
#        #(1,0): F_TA, 
#        (2,0): FluxLimiter(F_SA,90) #90-110
#        #(2,0): F_SA
#      }

#phi_sym=Function('phi_sym')
epsilon=Symbol('epsilon')# parameter for 
z_max=Symbol("z_max")
alph=Symbol('alph')
control_start=1900


half_saturation_utz_exp=Piecewise((1,time_symbol<control_start),(limiters.half_saturation(z,epsilon),True))
cubic_utz_exp=Piecewise((1,time_symbol<control_start),(limiters.cubic(z,z_max),True))
deceleration_utz_exp=Piecewise((1,time_symbol<control_start),(limiters.deceleration(z,z_max,alph),True))

start_values = 1.01*np.array([A_eq, T_eq, S_eq])
# define a dictionary to connect the symbolic function  with the according implementations 
func_dict = {
    u_A: drivers.u_A_func, 
#    f_TA: drivers.f_TA_func, 
    assert_non_negative_sym:assert_non_negative_num
}
par_dict= {alpha: 0.2, beta: 10.0} # nonlinear
#par_dict_v2 = {alpha: 1.0, beta:  1.0} # linear

par_dict_half_saturation_10 = copy(par_dict)
par_dict_half_saturation_10.update({epsilon:10})

par_dict_cubic_fast = copy(par_dict)
par_dict_cubic_fast.update({z_max:50})
par_dict_cubic_mid = copy(par_dict)
par_dict_cubic_mid.update({z_max:500})
par_dict_cubic_slow = copy(par_dict)
par_dict_cubic_slow.update({z_max:1000})

par_dict_deceleration_10_10 = copy(par_dict)
par_dict_deceleration_10_10.update({z_max:10,alph:10})

# define the time windows of interest
#start_year = 100
start_year = 1765
end_year = 2500
times = np.arange(start_year, end_year+1,1)# (end_year-start_year)/1000)


# create some models 
unlimited_srm = SmoothReservoirModel(
        state_vector, time_symbol,
        net_input_fluxes, net_output_fluxes, internal_fluxes
)
limited_srm = SmoothReservoirModel(
        state_vector, time_symbol, net_input_fluxes, net_output_fluxes, 
        lim_inf
)
#limited_srm_90 = SmoothReservoirModel(
#        state_vector, time_symbol, net_input_fluxes, net_output_fluxes, 
#        lim_inf_90
#)
#half_saturation_bm_300=BastinModel(
#        limited_srm_300,half_saturation_utz_exp,z
#)
cubic_bm=BastinModel(
        limited_srm,cubic_utz_exp,z
)
#deceleration_bm_300=BastinModel(
#        limited_srm_300,deceleration_utz_exp,z
#)

# create a dictionary of model runs
all_mrs={ 
        "unlimited_smr" : 
            SmoothModelRun(
                unlimited_srm, par_dict, start_values, times, func_dict), 
        "limited_smr":
            SmoothModelRun(
                limited_srm, par_dict, start_values , times, func_dict),
#        "limited_90_smr":
#            SmoothModelRun(
#                limited_srm_90, par_dict, start_values , times, func_dict),
#        "limited_300_controlled_half_saturation_10_20":
#            BastinModelRun( 
#                half_saturation_bm_300, par_dict_half_saturation_10, 
#                start_values=np.array(list(start_values)+[20]), 
#                times=times, func_dict=func_dict),
        "limited_controlled_cubic_fast":
            BastinModelRun(
                cubic_bm, par_dict_cubic_fast, 
                start_values=np.array(list(start_values)
                    +[par_dict_cubic_fast[z_max]]), 
                times=times, func_dict=func_dict),
        "limited_controlled_cubic_mid":
            BastinModelRun(
                cubic_bm, par_dict_cubic_mid, 
                start_values=np.array(list(start_values)
                    +[par_dict_cubic_mid[z_max]]), 
                times=times, func_dict=func_dict),
        "limited_controlled_cubic_slow":
            BastinModelRun(
                cubic_bm, par_dict_cubic_slow, 
                start_values=np.array(list(start_values)
                    +[par_dict_cubic_slow[z_max]]), 
                times=times, func_dict=func_dict)
#        "limited_300_controlled_deceleration_10_10":
#            BastinModelRun(
#                deceleration_bm_300, par_dict_deceleration_10_10, 
#                start_values=np.array(list(start_values)
#                    +[par_dict_deceleration_10_10[z_max]]), 
#                times=times, func_dict=func_dict)
        }

# make plots
# comment the ones you do not need and add new ones using the components
# there is actually not much need to delete
plotFuncs.limiter_comparison()
plotFuncs.panel_one(
        limited_srm,
        cubic_bm, 
        par_dict_cubic_fast, 
        control_start_values= np.array(list(start_values)+[par_dict_cubic_fast[z_max]]), 
        times=times, 
        func_dict=func_dict)

plotFuncs.panel_two(
        cubic_bm,
        par_dict_cubic_fast,
        control_start_values= np.array(list(start_values)+[par_dict_cubic_fast[z_max]]),
        times=times,
        func_dict=func_dict)

#plotFuncs.epsilon_family(
#        limited_srm_300, par_dict_half_saturation_10, 
#        control_start_values=np.array(list(start_values)+[2000]), 
#        times=times, func_dict=func_dict,epsilons=[1,100,1000] )    

plotFuncs.cubic_family(
        limited_srm, par_dict, start_values, 
        times=times, func_dict=func_dict,zs=[50,500,1000] )    

#plotFuncs.deceleration_family(
#        limited_srm_300, par_dict, start_values, 
#        times=times, func_dict=func_dict,zs=[100,1000,10000],alphas=[1.5,2.5,3.5] )    
#plotFuncs.epsilon_family_2(
#        limited_srm_300, par_dict, start_values, 
#        times=times, func_dict=func_dict,zs=[100,1000,10000],epsilons=[1.5,2.5,3.5] )    

#pf=plotFuncs.compare_model_runs(
#    {
##        "limited_uncontrolled_90":all_mrs["limited_90_smr"]
##        ,
#        "limited_uncontrolled":all_mrs["limited_smr"]
#        ,
#        "unlimited_uncontrolled":all_mrs["unlimited_smr"]
#    },
#    drivers.u_A_func
#)
pf=plotFuncs.compare_model_runs(
    { 
        key:all_mrs[key] for key in [
            "limited_smr"
            ,"limited_controlled_cubic_fast"  
#            "limited_controlled_cubic_fast" , 
#            "limited_controlled_cubic_mid",
#            "limited_controlled_cubic_slow"
##            "limited_300_controlled_deceleration_10_10",
##            "limited_300_controlled_half_saturation_10_20"
        ]
    },
    drivers.u_A_func
)


pf=plotFuncs.compare_controlers(
    { 
        key:all_mrs[key] for key in [
            "limited_controlled_cubic_fast" , 
            "limited_controlled_cubic_mid",
            "limited_controlled_cubic_slow"
        ]
    },
    drivers.u_A_func
)
