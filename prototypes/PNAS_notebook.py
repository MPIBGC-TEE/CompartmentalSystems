#!/usr/bin/env python
# all array-like data structures are numpy.array
import  os
import numpy as np

# for 2d plots we use Matplotlib
import matplotlib.pyplot as plt

from sympy import Matrix, symbols, Symbol, Function, latex, atan ,pi,sin
from scipy.interpolate import interp1d
# load the compartmental model packages
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.helpers_reservoir import  numsol_symbolic_system 

########## symbol definitions ##########

# time symbol
time_symbol = symbols('t')
# Atmosphere, Terrestrial Carbon and Surface ocean
C_A, C_T, C_S = symbols('C_A C_T C_S')
#virtual pool for controller
z= symbols('z')
# fossil fuel inputs
u_A = Function('u_A')(time_symbol)
u = 1+sin(time_symbol/50)
# land use change flux
f_TA = Symbol('f_TA')(time_symbol)
# nonlinear effects
alpha, beta = symbols('alpha beta')


########## model structure: equilibrium values and fluxes ##########

# equilibrium values
A_eq, T_eq, S_eq = (700.0, 3000.0, 1000.0)

state_vector = Matrix([C_A, C_T, C_S])

# fluxes
def FluxLimiter(expression,limit):
    sf=2/pi*limit
    res=sf*atan(1/sf*expression)
    return res
    #return expression
    

F_AT = 60*(C_A/A_eq)**alpha
F_AS = 100*C_A/A_eq
F_TA = 60*C_T/T_eq + f_TA
F_SA = 100*(C_S/S_eq)**beta

input_fluxes = {0: u*u_A, 1: 0, 2: 45} 
#input_fluxes = {0: 0, 1: 0, 2: 45} # without fossil fuel
o=45*C_S/S_eq
output_fluxes = {2: o}
internal_fluxes = {(0,1): F_AT, (0,2): F_AS,
                   (1,0): F_TA, (2,0): F_SA}
                   

limited_internal_fluxes  = {(0,1): FluxLimiter(F_AT,10), (0,2): F_AS,
                   (1,0): F_TA, (2,0): F_SA}
#limited_internal_fluxes  = {(0,1): FluxLimiter(F_AT,10), (0,2): FluxLimiter(F_AS,10),
#                   (1,0): FluxLimiter(F_TA,10), (2,0): FluxLimiter(F_SA,10)}
# create the SmoothReservoirModel
nonlinear_srm = SmoothReservoirModel(state_vector, 
                                     time_symbol, 
                                     input_fluxes, 
                                     output_fluxes, 
                                     internal_fluxes)

limited_srm = SmoothReservoirModel(state_vector, 
                                     time_symbol, 
                                     input_fluxes, 
                                     output_fluxes, 
                                     limited_internal_fluxes)

# define the time and age windows of interest
start_year = 1765
end_year = 2500
#end_year = 2015
max_age = 250

times = np.arange(start_year, end_year+1, 1)
ages = np.arange(0, max_age+1, 1)

# fossil fuel and land use change data
ff_and_lu_data = np.loadtxt('emissions.csv', usecols = (0,1,2), skiprows = 38)
# column 0: time, column 1: fossil fuels
ff_data = ff_and_lu_data[:,[0,1]]
# linear interpolation of the (nonnegative) data points
u_A_interp = interp1d(ff_data[:,0], np.maximum(ff_data[:,1], 0))

def u_A_func(t_val):
    # here we could do whatever we want to compute the input function
    # we return only the linear interpolation from above
    return u_A_interp(t_val)

# column 0: time, column 2: land use effects
lu_data = ff_and_lu_data[:,[0,2]]
f_TA_func = interp1d(lu_data[:,0], lu_data[:,1])

# define a dictionary to connect the symbols with the according functions
func_set = {u_A: u_A_func, f_TA: f_TA_func}
# the system starts in equilibrium
start_values = np.array([A_eq, T_eq, S_eq])

# possibly nonlinear effects as a parameter dictionary
par_dict_v1 = {alpha: 0.2, beta: 10.0} # nonlinear
#par_dict_v2 = {alpha: 1.0, beta:  1.0} # linear
#
# create the nonlinear model run
rhs=nonlinear_srm.F
soln = numsol_symbolic_system(
    state_vector,
    time_symbol,
    rhs,
    par_dict_v1,
    func_set,
    start_values, 
    times
)
nonlinear_smr = SmoothModelRun(nonlinear_srm, par_dict_v1, start_values, times, func_set)
limited_smr = SmoothModelRun(limited_srm, par_dict_v1, start_values, times, func_set)
### solve the model
soln = nonlinear_smr.solve()
limited_soln = limited_smr.solve()

fig1=plt.figure(figsize=(10,7))
#fig1.title('Total carbon'+title_suffs[version])
#ax=fig1.add_subplot(1,1,1)
plt.plot(times, soln[:,0], color='blue', label='Atmosphere')
plt.plot(times, soln[:,1], color='green', label='Terrestrial Biosphere')
plt.plot(times, soln[:,2], color='purple', label='Surface ocean')
plt.plot(times, soln.sum(1), color='red', label='Total')
plt.plot(times, limited_soln[:,0],  label='lim Atmosphere')
plt.plot(times, limited_soln[:,1], label='lim Terrestrial Biosphere')
plt.plot(times, limited_soln[:,2], label='lim Surface ocean')
plt.plot(times, limited_soln.sum(1), label='lim Total')
plt.xlim([1765,2500])
plt.ylim([0,9000])
plt.legend(loc=2)
plt.xlabel('Time (yr)')
plt.ylabel('Mass (PgC)')
#plt.show()
fig1.savefig('poolcontents.pdf')

##
fig2 = plt.figure()
nonlinear_smr.plot_internal_fluxes(fig2,fontsize=20)
fig2.savefig('fluxes.pdf')
#
