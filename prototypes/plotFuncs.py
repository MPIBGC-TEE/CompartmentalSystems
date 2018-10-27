
import inspect
from sympy import Symbol
# for 2d plots we use Matplotlib
import matplotlib.pyplot as plt
from testinfrastructure.helpers  import pe
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from Classes import BastinModel,BastinModelRun

def poolsizes(ax,times,soln):
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

def panel_one(limited_srm,bm, par_dict_v1, control_start_values, times, func_dict):
    start_values=control_start_values[:-1]
    limited_smr = SmoothModelRun(limited_srm, par_dict_v1, start_values, times, func_dict)
    bmr=BastinModelRun( bm, par_dict_v1, control_start_values, times, func_dict)

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


def all_in_one(unlimited_srm,limited_srm,bm,par_dict_v1, control_start_values, times, func_dict,u_A):
    start_values=control_start_values[:-1]
    unlimited_smr = SmoothModelRun(unlimited_srm, par_dict_v1, start_values, times, func_dict)
    limited_smr = SmoothModelRun(limited_srm, par_dict_v1, start_values, times, func_dict)
    bmr=BastinModelRun( bm, par_dict_v1, control_start_values, times, func_dict)

    soln = unlimited_smr.solve()
    limited_soln_uncontrolled = limited_smr.solve()

    limited_soln = bmr.solve()
    fig=plt.figure(figsize=(18,30))
    #fig.title('Total carbon'+title_suffs[version])
    ax1=fig.add_subplot(5,1,1)
    ax2=fig.add_subplot(5,1,2)
    ax3=fig.add_subplot(5,1,3)
    ax4=fig.add_subplot(5,1,4)
    ax5=fig.add_subplot(5,1,5)
    
    ax1=poolsizes(ax1,times,soln)
    ax1.set_title("unlimited uncontrolled")

    ax2=poolsizes(ax2,times,limited_soln_uncontrolled)
    ax2.set_title("limited uncontrolled")

    ax3=poolsizes(ax3,times,limited_soln)
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
   
    
    ax5.plot(times, func_dict[u_A](times),label='u_A')
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
        func_dict,
        epsilons
    ):    
    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1)
    ax1.set_title("control u for different values of epsilon")
    for eps_val in epsilons:
        z=Symbol('z')
        eps=Symbol('eps')
        u_z_exp=z/(eps+z)
        par_dict[eps]=eps_val
        bm=BastinModel(limited_srm,u_z_exp,z)
        bmr=BastinModelRun(
            bm, 
            par_dict,
            control_start_values, 
            times,
            func_dict
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
    
    

