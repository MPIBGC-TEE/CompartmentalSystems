#!/usr/bin/env python
# all array-like data structures are numpy.array
import  os
import inspect
import numpy as np

# for 2d plots we use Matplotlib
import matplotlib.pyplot as plt

from copy import copy,deepcopy
from sympy import Matrix, symbols, Symbol, Function, latex, atan ,pi,sin,lambdify,Piecewise
from scipy.interpolate import interp1d
# load the compartmental model packages
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.helpers_reservoir import  numsol_symbolic_system ,numerical_function_from_expression
from testinfrastructure.helpers  import pe

########## symbol definitions ##########
Net_sym=Function('Net_sym')
def Net_num(Net_SD_DS):
    if Net_SD_DS>0 :
        return Net_SD_DS
    else:
        raise Exception("negative Netflux")

def phi_maker(epsilon):
    def phi(expression):
        res=expression/(epsilon+expression)
        return res
    return phi
class BastinModel():
    # Bastonification of a reservoir model
    def __init__(self,srm,u_expr,z_sym):
        self.z_sym=z_sym
        self.u_expr=u_expr
        crm=deepcopy(srm)
        cof= crm.output_fluxes
        cif= crm.input_fluxes
        # up to now we can only built 
        # single input single output models
        assert(len(cof)==1)
        assert(len(cif)==1)
        F_SD=list(cof.values())[0]
        cif= crm.input_fluxes
        #index of the single input receiving pool
        ii=list(cif.items())[0][0]
        d=cif[ii]
        cif[ii] = u_expr*d
        crm.input_fluxes=cif
    
        self.state_vector=Matrix(list(srm.state_vector)+[z_sym])
        z_dot=F_SD-u_expr*d
        #rhs
        self.F=Matrix(list(crm.F)+[z_dot])
        self.time_symbol=srm.time_symbol



class BastinModelRun():
    def __init__(self,bm,par_dict, start_values, times, func_set):
        self.bm=bm
        self.par_dict=par_dict
        self.start_values=start_values
        self.times=times
        self.func_set=func_set

    def solve(self):
        bm=self.bm
        soln = numsol_symbolic_system(
            bm.state_vector,
            bm.time_symbol,
            bm.F,
            self.par_dict,
            self.func_set,
            self.start_values, 
            self.times
        )
        return soln
    def phi_num(self,tup):
        bm=self.bm
        u_num= numerical_function_from_expression(bm.u_expr,tup,self.par_dict,self.func_set)
        return u_num

# time symbol
time_symbol = symbols('t')
# Atmosphere, Terrestrial Carbon and Surface ocean
C_A, C_T, C_S = symbols('C_A C_T C_S')
#virtual pool for controller
z= symbols('z')
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
def FluxLimiter(expression,limit):
    sf=2/pi*limit
    res=sf*atan(1/sf*expression)
    return res
    

F_AT = 60.0*(C_A/A_eq)**alpha
F_AS = 100.0*C_A/A_eq
F_TA = 60.0*C_T/T_eq + f_TA(time_symbol)
F_SA = 100.0*(C_S/S_eq)**beta
F_DS = 45.0
F_SD=F_DS*C_S/S_eq
input_fluxes = {0: u_A, 1: 0, 2: F_DS} 
output_fluxes = {2: F_SD}
net_input_fluxes = {0: u_A } 
net_output_fluxes = {2: Net_sym(F_SD-F_DS)}
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
# create the SmoothReservoirModels
nonlinear_srm = SmoothReservoirModel(state_vector, 
                                     time_symbol, 
                                     input_fluxes, 
                                     output_fluxes, 
                                     internal_fluxes)

limited_srm = SmoothReservoirModel(state_vector, 
                                     time_symbol, 
                                     net_input_fluxes, 
                                     net_output_fluxes, 
                                     limited_internal_fluxes)

# define the time windows of interest
start_year = 1765
#start_year = 1900
#end_year =1800 
end_year = 2500
#end_year = 2015
max_age = 250

times = np.arange(start_year, end_year+1,1)# (end_year-start_year)/1000)
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

def u_A_step(t_val):
    t_step=2100
    lower,higher=10,30
    res = lower if t_val<t_step else higher
    return res

# column 0: time, column 2: land use effects
lu_data = ff_and_lu_data[:,[0,2]]
f_TA_func = interp1d(lu_data[:,0], lu_data[:,1])

# define a dictionary to connect the symbols with the according functions
func_set = {
    u_A: u_A_func, 
    f_TA: f_TA_func, 
    Net_sym:Net_num
}

# the systems start a little higher than the equilibrium
# of the system with unconstrained fluxes
start_values = 1.05*np.array([A_eq, T_eq, S_eq])


# possibly nonlinear effects as a parameter dictionary
par_dict_v1 = {alpha: 0.2, beta: 10.0} # nonlinear
#par_dict_v2 = {alpha: 1.0, beta:  1.0} # linear
#
# create the nonlinear model run

#Net_SD_DS=F_SD-F_DS

### solve the model
#soln = nonlinear_smr.solve()
#limited_soln = limited_smr.solve()
#

def poolsizes(ax,soln):
    ax.plot(times, soln[:,0],label='Atmosphere')
    ax.plot(times, soln[:,1],label='Terrestrial Biosphere')
    ax.plot(times, soln[:,2],label='Surface ocean')
    ax.plot(times, soln[:,0:3].sum(1), label='lim Total')
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('Mass (PgC)')
    if soln.shape[1]>3:
        ax.plot(times, soln[:,3], color='black', label='z')

    ax.legend(loc=2)
    return(ax)

def my_func_name():
    cf=inspect.currentframe()
    callerName=cf.f_back.f_code.co_name
    return callerName

def panel_one(limited_smr,bmr):
    soln_uncontrolled = limited_smr.solve()

    soln_controlled = bmr.solve()
    fig=plt.figure(figsize=(10,10))
    #fig1.title('Total carbon'+title_suffs[version])
    ax1=fig.add_subplot(2,1,1)
    ax1.plot(times, soln_uncontrolled[:,0],color='blue' ,label='Atmosphere')
    ax1.plot(times, soln_uncontrolled[:,1],color='green',label='Terrestrial Biosphere')
    ax1.plot(times, soln_uncontrolled[:,2],color='red'  ,label='Surface ocean')
    ax1.set_ylabel('Mass (PgC)')
    ax1.legend(loc=2)
    ax1.set_title("Uncontrolled")

    ax2=fig.add_subplot(2,1,2)
    ax2.plot(times, soln_controlled[:,0],color='blue' ,label='Atmosphere')
    ax2.plot(times, soln_controlled[:,1],color='green',label='Terrestrial Biosphere')
    ax2.plot(times, soln_controlled[:,2],color='red'  ,label='Surface ocean')
    ax2.set_ylabel('Carbon stocks (PgC)')
    ax2.set_xlabel('Time (yr)')
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_title("Controlled")
    
    #limited_soln_uncontrolled 
    fig.savefig(my_func_name()+'.pdf')


def all_in_one(nonlinear_smr,limited_smr,bmr):
    rhs=nonlinear_srm.F 
    soln = nonlinear_smr.solve()
    limited_soln_uncontrolled = limited_smr.solve()

    limited_soln = bmr.solve()
    fig=plt.figure(figsize=(18,30))
    #fig.title('Total carbon'+title_suffs[version])
    ax1=fig.add_subplot(5,1,1)
    ax2=fig.add_subplot(5,1,2)
    ax3=fig.add_subplot(5,1,3)
    ax4=fig.add_subplot(5,1,4)
    ax5=fig.add_subplot(5,1,5)
    
    ax1=poolsizes(ax1,soln)
    ax1.set_title("unlimited uncontrolled")

    ax2=poolsizes(ax2,limited_soln_uncontrolled)
    ax2.set_title("limited uncontrolled")

    ax3=poolsizes(ax3,limited_soln)
    ax3.set_title("limited controlled")
    
    
    # since we do not know the actual phi of the bastin model run 
    # we assume the most general case that after all paramteters
    # and functions have been substituted t,z remain as arguments
    # The actual expression might not even contain t but that is no
    # problem
    bm=bmr.bm
    tup=(bm.time_symbol,bm.z_sym)
    times=bmr.times
    phi_num=bmr.phi_num(tup)
    ax4.set_title("control")
    zval=limited_soln[:,3]
    u_vals=phi_num(times,zval)
    pe('times.shape',locals())
    pe('zval.shape',locals())
    ax4.plot(times, u_vals, label='u')
    ax4.legend(loc=3)
    
    ax5.plot(times, u_A_func(times),label='u_A')
    ax5.legend(loc=2)
    ax5.set_xlabel('Time (yr)')
    ax5.set_ylabel('Mass (PgC)')
    fig.savefig(my_func_name()+'.pdf')
    

#def as_sa_fluxes(mr):
#    fig=plt.figure()
#    fl_at=mr.internal_flux_funcs()[(0,1)]
#    fl_ta=mr.internal_flux_funcs()[(1,0)]
#    fl_as=mr.internal_flux_funcs()[(0,2)]
#    fl_sa=mr.internal_flux_funcs()[(2,0)]
#    ax1=fig.add_subplot(4,1,1)
#    ax2=fig.add_subplot(4,1,2)
#    ax3=fig.add_subplot(4,1,3)
#    ax4=fig.add_subplot(4,1,4)
#    ax1.plot(times,fl_as(times))
#    ax1.plot(times,fl_sa(times))
#    ax2.plot(times,fl_as(times)-fl_sa(times))
#
#    ax3.plot(times,fl_at(times))
#    ax3.plot(times,fl_ta(times))
#    ax4.plot(times,fl_as(times)-fl_sa(times))
#    return fig
def plot_epsilon_family(
        limited_srm,
        par_dict,
        control_start_values, 
        times,
        func_set,
        epsilons
    ):    
    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1)
    ax1.set_title("control u for different values of epsilon")
    for eps in epsilons:
        phi_eps=phi_maker(eps)
        z=Symbol('z')
        u_z_exp=phi_eps(z)
        bm=BastinModel(limited_srm,u_z_exp,z)
        bmr=BastinModelRun(
            bm, 
            par_dict,
            control_start_values, 
            times,
            func_set
        )
        phi_num=bmr.phi_num((z,))
        soln=bmr.solve() 
        z=soln[:,3]
        pe('bm.u_expr',locals())
        u=phi_num(z)
        ax1.plot(times,u)
        ax1.legend(loc=3)
     
    pe('__name__',locals())
    fig.savefig("epsilon_famamly.pdf")
    
    

    
epsilons=[1,100,1000]
###
#fig2 = plt.figure()
#nonlinear_smr.plot_internal_fluxes(fig2,fontsize=20)
#fig2.savefig('fluxes.pdf')
#
#f=as_sa_fluxes(nonlinear_smr)
#f.savefig("fluxes_ast.pdf")
#f=as_sa_fluxes(limited_smr)
#f.savefig("limited_fluxes_ast.pdf")
z0=200
control_start_values = np.array(list(start_values)+[z0])
plot_epsilon_family(
        limited_srm,
        par_dict_v1,
        control_start_values, 
        times,
        func_set,
        epsilons)    

nonlinear_smr = SmoothModelRun(nonlinear_srm, par_dict_v1, start_values, times, func_set)
limited_smr = SmoothModelRun(limited_srm, par_dict_v1, start_values, times, func_set)
eps_val=10
phi_eps=phi_maker(10)
u_z_exp=phi_eps(z)
control_start=1900
phi_sym=Function('phi')
u_t_z_exp=Piecewise((1,time_symbol<control_start),(u_z_exp,True))
u_t_z_num=lambdify((time_symbol,z),u_t_z_exp,modules='numpy')
func_set[phi_sym]=u_t_z_num

# we implement the kick in of the control by defining phi_eps as a piecewise function
# with respect to time 
        

#bm=BastinModel(limited_srm,u_z_exp,z)
bm=BastinModel(limited_srm,phi_sym(time_symbol,z),z)
bmr=BastinModelRun(
    bm, 
    par_dict_v1,
    control_start_values, 
    times,
    func_set
)
all_in_one(nonlinear_smr,limited_smr,bmr)
panel_one(limited_smr,bmr)
