
import inspect
from sympy import Symbol,lambdify
from copy import copy
import numpy as np
# for 2d plots we use Matplotlib
import matplotlib.pyplot as plt
from testinfrastructure.helpers  import pe
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.helpers_reservoir import  numsol_symbolic_system ,numerical_function_from_expression
from testinfrastructure.helpers  import pe
from Classes import BastinModel,BastinModelRun
import drivers #contains the fossil fuel interpolating functions
from string import Template
from limiters import cubic,deceleration,half_saturation,atan_ymax

def poolsizes(ax,times,soln):
    ax.plot(times, soln[:,0],label='Atmosphere')
    ax.plot(times, soln[:,1],label='Terrestrial Biosphere')
    ax.plot(times, soln[:,2],label='Surface ocean')
    #ax.plot(times, soln[:,0:3].sum(1), label='lim Total')
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('Mass (PgC)')
    #if soln.shape[1]>3:
    #    ax.plot(times, soln[:,3], color='black', label='z')

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


    

def deceleration_family(
        limited_srm,
        par_dict,
        start_values, 
        times,
        func_dict,
        zs,
        alphas
    ):    
    z=Symbol('z')
    z_max=Symbol('z_max')
    alph=Symbol('alph')
    u_z_exp=deceleration(z,z_max,alph)
    bm=BastinModel(limited_srm,u_z_exp,z)
    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1)
    ax1.set_title("control u with deceleration limiter for different values of alph")
    for z_max_val in zs:
        for alpha_val in alphas:
            control_start_values=np.array(list(start_values)+[z_max_val])
            par_dict[z_max]=z_max_val
            par_dict[alph]=alpha_val
            bmr=BastinModelRun(
                bm, 
                par_dict,
                control_start_values, 
                times,
                func_dict
            )
            phi_num=bmr.phi_num((z,))
            soln=bmr.solve() 
            z_sol=soln[:,3]
            pe('bm.u_expr',locals())
            u=phi_num(z_sol)
            ax1.plot(times,u,label="alph:"+str(alpha_val)+",z_max="+str(z_max_val))
    ax1.legend(loc=3)
     
    fig.savefig(my_func_name()+'.pdf')
def cubic_family(
        limited_srm,
        par_dict,
        start_values, 
        times,
        func_dict,
        zs
    ):    
    z=Symbol('z')
    z_max=Symbol('z_max')
    u_z_exp=cubic(z,z_max)
    bm=BastinModel(limited_srm,u_z_exp,z)
    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1)
    ax1.set_title("control u for different values of z_max")
    for z_max_val in zs:
        control_start_values=np.array(list(start_values)+[z_max_val])
        par_dict[z_max]=z_max_val
        bmr=BastinModelRun(
            bm, 
            par_dict,
            control_start_values, 
            times,
            func_dict
        )
        phi_num=bmr.phi_num((z,))
        soln=bmr.solve() 
        z_sol=soln[:,3]
        pe('bm.u_expr',locals())
        u=phi_num(z_sol)
        ax1.plot(times,u,label="z_max="+str(z_max_val))
        ax1.legend(loc=3)
     
    fig.savefig(my_func_name()+'.pdf')
def epsilon_family(
        limited_srm,
        par_dict,
        control_start_values, 
        times,
        func_dict,
        epsilons
    ):    
    z=Symbol('z')
    eps=Symbol('eps')
    u_z_exp=half_saturation(z,eps)
    bm=BastinModel(limited_srm,u_z_exp,z)
    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1)
    ax1.set_title("control u for different values of epsilon")
    for eps_val in epsilons:
        par_dict[eps]=eps_val
        bmr=BastinModelRun(
            bm, 
            par_dict,
            control_start_values, 
            times,
            func_dict
        )
        phi_num=bmr.phi_num((z,))
        soln=bmr.solve() 
        z_sol=soln[:,3]
        pe('bm.u_expr',locals())
        u=phi_num(z_sol)
        ax1.plot(times,u)
        ax1.legend(loc=3)
     
    fig.savefig(my_func_name()+'.pdf')

def epsilon_family_2(
        limited_srm,
        par_dict,
        start_values, 
        times,
        func_dict,
        zs,
        epsilons
    ):    
    z=Symbol('z')
    eps=Symbol('eps')
    z0=Symbol('z0')
    u_z_exp=half_saturation(z,eps)
    bm=BastinModel(limited_srm,u_z_exp,z)
    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1)
    ax1.set_title("control u for different values of epsilon")
    for z0_val in zs:
        for eps_val in epsilons:
            control_start_values=np.array(list(start_values)+[z0_val])
            par_dict[eps]=eps_val
            par_dict[z0]=z0_val,
            bmr=BastinModelRun(
                bm, 
                par_dict,
                control_start_values, 
                times,
                func_dict
            )
            phi_num=bmr.phi_num((z,))
            soln=bmr.solve() 
            z_sol=soln[:,3]
            pe('bm.u_expr',locals())
            u=phi_num(z_sol)
            ax1.plot(times,u,label="eps:"+str(eps_val)+",z0="+str(z0_val))
    ax1.legend(loc=3)
     
    fig.savefig(my_func_name()+'.pdf')


def compare_model_runs(mr_dict,u_A_func):
    nc=len(mr_dict) 
    if nc==1:
        # the subplot array would become one dimensional 
        nc=nc+1 
    fig=plt.figure(figsize=(nc*3,20))
    subplotArr=fig.subplots(7,nc)
    for ind,key in enumerate(mr_dict):
        ax0=subplotArr[0,ind]
        mr=mr_dict[key]
        soln = mr.solve()
        times=mr.times
        
        ax0=poolsizes(ax0,times,soln)
        ax0.set_title(key)

        ax1=subplotArr[1,ind]
        ax1.plot(times, u_A_func(times),label='u_A')
        ax1.legend(loc=2)
        ax1.set_xlabel('Time (yr)')
        ax1.set_ylabel('Mass (PgC)')
        
        ax2=subplotArr[2,ind]
        ax2.set_title("solutions")
        eifl=mr.sol_funcs()
        for i,f in enumerate(eifl):
            values=f(times)
            ax2.plot(times,values,label=str(i))
        ax2.legend(loc=0)

        ax3=subplotArr[3,ind]
        ax3.set_title("external inputs")
        eifl=mr.external_input_flux_funcs()
        for key in eifl.keys():
            f=eifl[key]
            values=f(times)
            ax3.plot(times,values,label=key)
        ax3.legend(loc=0)


        #ax3.legend(loc=0)
        ax4=subplotArr[4,ind]
        ax4.set_title("internal fluxes")
        ifl=mr.internal_flux_funcs()
        for key,fluxFunc in ifl.items():
            ax4.plot(times,fluxFunc(times),label=key)
        ax4.legend(loc=0)
        
        ax5=subplotArr[5,ind]
        ax5.set_title("external outputs")
        ofl=mr.output_flux_funcs()
        for key in ofl.keys():
            f=ofl[key]
            values=f(times)
            ax5.plot(times,values,label=key)
        ax5.legend(loc=0)

        if type(mr)==BastinModelRun:
            ax6=subplotArr[6,ind]
            ax6.set_title("control ")
            bm=mr.bm
            tup=(bm.time_symbol,bm.z_sym)
            phi_num=mr.phi_num(tup)
            z_vals=soln[:,3]
            u_vals=phi_num(times,z_vals)
            #sum_vals=limited_soln_controlled[:,0:3].sum(1)
            ax6.plot(times, u_vals, label='u')
            ax6.legend(loc=0)
        #ax6.set_ylim(ax_1_1.get_ylim())
        plt.subplots_adjust(hspace=0.6)

    
    fig.savefig(my_func_name()+'.pdf')

