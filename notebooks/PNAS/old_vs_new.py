
import os
import numpy as np
import matplotlib.pyplot as plt
#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=False)
from sympy import Matrix, symbols, Symbol, Function, latex
from scipy.interpolate import interp1d
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun

folder='output_data_new'
if not os.path.exists(folder):
    os.makedirs(folder)
    print("Created output data folder named '%s'" % folder)


# time symbol
time_symbol = symbols('t')

# Atmosphere, Terrestrial Carbon and Surface ocean
C_A, C_T, C_S = symbols('C_A C_T C_S')

# fossil fuel inputs
u_A = Function('u_A')(time_symbol)

# land use change flux
f_TA = Function('f_TA')(time_symbol)

# nonlinear effects
alpha, beta = symbols('alpha beta')


# Now, we define the model.

# In[4]:


########## model structure: equilibrium values and fluxes ##########

# equilibrium values
A_eq, T_eq, S_eq = (700.0, 3000.0, 1000.0)

state_vector = Matrix([C_A, C_T, C_S])

# fluxes
F_AT = 60*(C_A/A_eq)**alpha
F_AS = 100*C_A/A_eq
F_TA = 60*C_T/T_eq + f_TA
F_SA = 100*(C_S/S_eq)**beta

input_fluxes = {0: u_A, 1: 0, 2: 45} 
output_fluxes = {2: 45*C_S/S_eq}
internal_fluxes = {(0,1): F_AT, (0,2): F_AS,
                   (1,0): F_TA, (2,0): F_SA}

# create the SmoothReservoirModel
nonlinear_srm = SmoothReservoirModel(state_vector, 
                                     time_symbol, 
                                     input_fluxes, 
                                     output_fluxes, 
                                     internal_fluxes)

# define the time and age windows of interest
start_year = 1765
end_year = 2500
max_age = 250

times = np.arange(start_year, end_year+1, 1)
ages = np.arange(0, max_age+1, 1)


# We read in the fossil fuel inputs and the land use change effects from a csv file and create linear interpolation functions from the data points. Then we connect these function with the symbols from the model.

# fossil fuel and land use change data
ff_and_lu_data = np.loadtxt('emissions.csv', usecols = (0,1,2), skiprows = 38)

# column 0: time, column 1: fossil fuels
ff_data = ff_and_lu_data[:,[0,1]]

# linear interpolation of the (nonnegative) data points
u_A_interp = interp1d(ff_data[:,0], np.maximum(ff_data[:,1], 0),fill_value='extrapolate')

def u_A_func(t_val):
    # here we could do whatever we want to compute the input function
    # we return only the linear interpolation from above
    return u_A_interp(t_val)

# column 0: time, column 2: land use effects
lu_data = ff_and_lu_data[:,[0,2]]
f_TA_func = interp1d(lu_data[:,0], lu_data[:,1],fill_value='extrapolate')

# define a dictionary to connect the symbols with the according functions
func_set = {u_A: u_A_func, f_TA: f_TA_func}

# the system starts in equilibrium
start_values = np.array([A_eq, T_eq, S_eq])

# possibly nonlinear effects as a parameter dictionary
par_dict_v1 = {alpha: 0.2, beta: 10.0} # nonlinear


# create the nonlinear model run
nonlinear_smr = SmoothModelRun(nonlinear_srm, par_dict_v1, start_values, times, func_set)
# create a linear model with the same solution trajectory
linear_smr = nonlinear_smr.linearize()
linear_smr_2 = nonlinear_smr.linearize_2()

print(nonlinear_smr.model)
print(linear_smr.model)
#sol_vals_2,sol_func_2=nonlinear_smr.solve_2()
#sol_vals=nonlinear_smr.solve()

#lin_sol_vals_2,lin_sol_func_2=linear_smr_2.solve_2()
#lin_sol_vals=linear_smr.solve()
#
#fig=plt.figure()
#ax=fig.add_subplot(1,1,1)
#for pool_nr in [0,1,2]:
#    ax.plot(
#        times
#        ,[sol_func_2(t)[pool_nr] for t in times]
#        ,color='blue'
#        ,label='sol_2'
#        ,linestyle='--'
#    )
#    ax.plot(
#        times
#        ,sol_vals[:,pool_nr]
#        ,color='red'
#        ,label='sol'
#        ,linestyle='-.'
#    )
#    ax.plot(
#        times
#        ,[lin_sol_func_2(t)[pool_nr] for t in times]
#        ,color='green'
#        ,label='lin_sol_2'
#    )
#    ax.plot(
#        times
#        ,lin_sol_vals[:,pool_nr]
#        ,color='yellow'
#        ,label='lin_sol'
#        ,linestyle=':'
#    )
#
#fig.savefig('old_vs_new.pdf')

xi, T, N, C, u = linear_smr.model.xi_T_N_u_representation() # version does not matter
B = xi*T*N

# consider fossil fuel input only, no deep ocean input
u[2] = 0 

# set up fossil fuel only system
start_values_ff_only = np.zeros((3,)) # no fossil fuels in the system in 1765

srms_ff_only = []
smrs_ff_only = []

linear_srm_ff_only = SmoothReservoirModel.from_B_u(state_vector, time_symbol, B, u)
linear_smr_ff_only = SmoothModelRun(
     linear_srm_ff_only
    ,linear_smr.parameter_dict
    ,start_values_ff_only
    ,linear_smr.times
    ,linear_smr.func_set
)

    
# the trick is to use the same state transition operator as before
# --> the fossil fuel carbon moves through the system as if all other carbon were there, too

linear_soln_ff_only = linear_smr_ff_only.solve()

## plot the solutions
#fig=plt.figure(figsize=(10,7))
#ax=fig.add_subplot(1,1,1)
#ax.set_title('Fossil fuel derived carbon')
#ax.plot(times, linear_soln_ff_only[:,0], color='blue', label='Atmosphere')
#ax.plot(times, linear_soln_ff_only[:,1], color='green', label='Terrestrial Biosphere')
#ax.plot(times, linear_soln_ff_only[:,2], color='purple', label='Surface ocean')
#ax.plot(times, linear_soln_ff_only.sum(1), color='red', label='Total')
#ax.set_xlim([1765,2500])
#ax.set_ylim([0, 2700])
##ax.set_legend(loc=2)
#ax.set_xlabel('Time (yr)')
#ax.set_ylabel('Mass (PgC)')
#fig.savefig('old_vs_new_ff.pdf')


# Now, we compute the state transition operator cache and save it to a file. If this file already exists, we simply load it instead of having to recompute it. Depending on the size of the state stransition operator cache, this might take several hours. But all time we invest at this point will be saved later on during density and quantile computations.
# 
# Furthermore, we solve the linearized model and plot the solution trajectories for the different compartments.

# In[8]:


##### (build and save or) load state transition operator cache #####

# the cache size indicates at how many intermediate time points the
# state transition operator is going to be cached to increase
# the speed of upcoming computations massively
cache_size =1001


#print('Building state transition operator cache')
ca_2b=linear_smr_2._compute_state_transition_operator_cache_2b(size = cache_size)
#ca_2=linear_smr_2._compute_state_transition_operator_cache_2(size = cache_size)
#ca  =linear_smr._compute_state_transition_operator_cache(size = cache_size)
#
#print(np.nanmax((ca_2-ca)/ca*100))
print(ca_2b)
