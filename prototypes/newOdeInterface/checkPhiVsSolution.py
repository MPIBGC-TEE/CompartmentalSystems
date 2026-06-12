import numpy as np
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
from sympy import var,Matrix
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.helpers_reservoir import numerical_function_from_expression,array_integration_by_ode,array_quad_result,array_integration_by_values

def test_stateTransitionOperator_by_different_methods():
    # The state transition operator Phi can be used to reproduce the solution
    k_0_val=1
    k_1_val=2
    x0_0=np.float(0.5)
    x0_1=np.float(1.5)
    delta_t=np.float(1./4.)
    # 
    var(["x_0","x_1","k_0","k_1","t","u"])
    #
    inputs={
         0:u
        ,1:u*t
    }
    outputs={
         0:k_0*x_0**2
        ,1:k_1*x_1
    }
    internal_fluxes={}
    svec=Matrix([x_0,x_1])
    srm=SmoothReservoirModel(
             state_vector       =svec
            ,time_symbol        =t
            ,input_fluxes       =inputs
            ,output_fluxes      =outputs
            ,internal_fluxes    =internal_fluxes
    )
    t_0     = 0
    t_max   = 4
    nt=5
    times = np.linspace(t_0, t_max, nt)
    double_times = np.linspace(t_0, t_max, 2*(nt-1)+1)
    quad_times = np.linspace(t_0, t_max, 4*(nt-1)+1)
    parameter_dict = {
         k_0: k_0_val
        ,k_1: k_1_val
        ,u:1}
    func_dict={}
    start_x= np.array([x0_0,x0_1]) #make it a column vector for later use
    #create the model run
    smr=SmoothModelRun(
         model=srm
        ,parameter_dict= parameter_dict
        ,start_values=start_x
        ,times=times
        ,func_set=func_dict
    )
    smr.build_state_transition_operator_cache(size=4)
    nr_pools=smr.nr_pools
    # to be able to compare the results we have to compute them for a 
    # set of n linear independent vectors
    def baseVector(i):
        e_i = np.zeros((nr_pools,1))
        e_i[i] = 1
        return e_i
    bvs = [ baseVector(i) for i in range(nr_pools)]
    #pe('Phi_skew(2,1,bvs[0])',locals())
    #raise
    
    test_times=np.linspace(t_0, t_max, 11)


# We now rebuild the solution by means of phi and plot it along with the original solution 
    original_sol,sol_func   =smr.solve()
    
    u_sym=srm.external_inputs
    u_num=numerical_function_from_expression(u_sym,(t,),parameter_dict,{})
    
    
    def vectorlist2array(l):
        return np.stack( [vec.flatten() for vec in l],1)
#    def lists_dict2array_dict(d):
#        return {key:vectorlist2array(val) for key,val in d.items()}
#    
    def continiuous_integral_values(integrator,times):
        start=time.time()
        res=vectorlist2array([integrator( lambda tau : smr._state_transition_operator(t,tau,u_num(tau)) ,t_0 ,t) for t in times])
        stop=time.time()
        exec_time=stop-start
        #pe('exec_time',locals())
        return (times,res,exec_time)

    def discrete_integral_values(integrator,times):
        start=time.time()
        res=vectorlist2array([integrator(lambda tau:smr._state_transition_operator(t,tau,u_num(tau)),taus=+times[0:i+1]) for i,t in enumerate(times)]) 
        stop=time.time()
        exec_time=stop-start
        #pe('exec_time',locals())
        return (times,res,exec_time)
    

    ## reconstruct the solution with Phi and the integrand
    # x_t=Phi(t,t0)*x_0+int_t0^t Phi(tau,t0)*u(tau) dtau
    # x_t=a(t)+b(t)
    et=bvs[0]+bvs[1]    
    phi_arrays= {
        'skew'  :(times,vectorlist2array([  smr._state_transition_operator(t,t_0,et).reshape(srm.nr_pools,1) for t in times]))
    }
    
    a_arrays={
        'skew'   :(       times,vectorlist2array([smr._state_transition_operator(t,t_0,start_x).reshape(srm.nr_pools,1) for t in times]))
        ,'trapez1':(       times,vectorlist2array([smr._state_transition_operator(t,t_0,start_x).reshape(srm.nr_pools,1) for t in times]))
        ,'trapez2':(double_times,vectorlist2array([smr._state_transition_operator(t,t_0,start_x).reshape(srm.nr_pools,1) for t in double_times]))
        ,'trapez4':(quad_times  ,vectorlist2array([smr._state_transition_operator(t,t_0,start_x).reshape(srm.nr_pools,1) for t in quad_times]))
    }
    nested_boundary_tuples=[(0,t) for t in reversed(times)]
    
    b_arrays_trapez={
    }
    b_arrays={
        'skew'      :continiuous_integral_values( array_integration_by_ode    ,times)
        ,'trapez1'  :discrete_integral_values(  array_integration_by_values ,times)
        ,'trapez2'  :discrete_integral_values(  array_integration_by_values,double_times)
        ,'trapez4'  :discrete_integral_values(  array_integration_by_values,quad_times)
    }
     

    b_arrays_quad={
        'skew'     :continiuous_integral_values(  array_quad_result,times)
    }

    
    x_arrays={key:(a_arrays[key][0],a_arrays[key][1]+b_arrays[key][1]) for key in a_arrays.keys()}
    #x_arrays['trapez']=(times,a_arrays['skew'][1]+b_arrays['trapez'][1])
        
    styleDict=OrderedDict({
         'skew'     :('green',6)
        ,'trapez1'  :('black',4)
        ,'trapez2'  :('blue',4)
        ,'trapez4'  :('brown',2)
    })
    def plot_comparison(axl,axr,d):
        for key in styleDict.keys():
            if key in d.keys(): 
                val=d[key]
                if len(val)==3:
                    time="{:7.1e}".format(val[2])
                else:
                    time=""
                axl.plot(val[0],val[1][0,:],'+',color=styleDict[key][0],markersize=styleDict[key][1] ,label=key+"[0]"+time)
                axr.plot(val[0],val[1][1,:],'x',color=styleDict[key][0],markersize=styleDict[key][1] ,label=key+"[1]"+time)


    fig=plt.figure(figsize=(17,27))
    rpn=5
    cpn=2
    r=1
    axl=fig.add_subplot(rpn,cpn,r)
    plt.title("""phi components, nonlinear part of the system (x[0]) """)
    axr=fig.add_subplot(rpn,cpn,r+1)
    plt.title("""phi components, linear part of the system (x[1]) """)
    plot_comparison(axl,axr,phi_arrays) 
    axl.legend()
    

    r+=cpn
    axl=fig.add_subplot(rpn,cpn,r)
    plt.title('''
    original solution and reconstruction via phi, 
    imprecise for trapez_rule and wrong for the old method
    '''
    )
    axr=fig.add_subplot(rpn,cpn,r+1)
    axl.plot(times,original_sol[:,0],'o',color='blue' ,label="original_sol[:,0]")
    axr.plot(times,original_sol[:,1],'o',color='blue' ,label="original_sol[:,1]")
    
    plot_comparison(axl,axr,x_arrays) 
    axl.legend()
    axr.legend()
     

    r+=cpn
    axl=fig.add_subplot(rpn,cpn,r)
    plt.title('phi(t,ti-0) x0 ')
    axr=fig.add_subplot(rpn,cpn,r+1)
    ax=fig.add_subplot(rpn,cpn,r)
    plot_comparison(axl,axr,a_arrays) 
    axl.legend()
    axr.legend()
    

    r+=cpn
    axl=fig.add_subplot(rpn,cpn,r)
    plt.title('\int_{t0}^t phi(tau,t) u(tau) d tau')
    axr=fig.add_subplot(rpn,cpn,r+1)
    plot_comparison(axl,axr,b_arrays) 
    axl.legend()
    axr.legend()

    #r+=cpn
    r+=cpn
    axl=fig.add_subplot(rpn,cpn,r)
    plt.title('\int_{t0}^t phi(tau,t) u(tau) d tau by quad')
    axr=fig.add_subplot(rpn,cpn,r+1)
    plot_comparison(axl,axr,b_arrays_quad) 
    axl.legend()
    axr.legend()



    fig.savefig("solutions.pdf")
    
test_stateTransitionOperator_by_different_methods()
