import numpy as np
import time
from collections import OrderedDict
from sympy import var,symbols,Symbol
from scipy.integrate import solve_ivp, quad
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from sympy import var,Matrix
from sympy.printing import pprint

from testinfrastructure.InDirTest import InDirTest
#from testinfrastructure.helpers import pe
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.smooth_model_run import SmoothModelRun 
from CompartmentalSystems.BlockIvp import BlockIvp
from CompartmentalSystems.helpers_reservoir import numerical_function_from_expression,numerical_rhs2,x_phi_ivp,integrate_array_func_for_nested_boundaries,array_quad_result,array_integration_by_ode,array_integration_by_values

class TestPhi(InDirTest):
    def test_phi_2d_linear(self):
        C_0,C_1 = symbols('C_0,C_1')
        state_vector = [C_0,C_1]
        time_symbol = Symbol('t')
        input_fluxes = {}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([1,2])
        t_0     = 0
        t_max   = 4
        nt=200
        times = np.linspace(t_0, t_max, nt)
        smr = SmoothModelRun(srm, {}, start_values, times)
        

        nr_pools=srm.nr_pools

        def baseVector(i):
            e_i = np.zeros((nr_pools,1))
            e_i[i] = 1
            return e_i
        bvs = [ baseVector(i) for i in range(nr_pools)]
        
        blivp= x_phi_ivp(
            smr.model
            ,smr.parameter_dict
            ,smr.func_set
            ,smr.start_values
            ,x_block_name='sol'
            ,phi_block_name='Phi_1d'
        )
        sol_dict=blivp.block_solve(t_span=(t_0,t_max),t_eval=times)
        
        sol_skew=sol_dict['sol'][...,-1]
        phi_mat=sol_dict['Phi_1d'][...,-1].reshape(nr_pools,nr_pools)
        phi_ref=np.eye(2)*np.exp(-(t_max-t_0))

        self.assertTrue( np.allclose( phi_mat,phi_ref,atol=1e-4))
        for x in bvs:
            with self.subTest():
                self.assertTrue( 
                    np.allclose(
                        smr._state_transition_operator_for_linear_systems(t_max,t_0,x),
                        np.matmul(phi_ref,x).reshape(nr_pools,)
                    )
                )

    def test_phi_2d_non_linear(self):
        k_0_val=1
        k_1_val=2
        x0_0=np.float(0.5)
        x0_1=np.float(1.5)
        delta_t=np.float(1./4.)
        # 
        x_0,x_1,k_0,k_1,t,u=symbols("x_0,x_1,k_0,k_1,t,u")
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
        nt=200
        times = np.linspace(t_0, t_max, nt)
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

        nr_pools=srm.nr_pools

        def baseVector(i):
            e_i = np.zeros((nr_pools,1))
            e_i[i] = 1
            return e_i
        bvs = [ baseVector(i) for i in range(nr_pools)]
        
        smrl=smr.linearize()
        
        blivp= x_phi_ivp(
            smr.model
            ,smr.parameter_dict
            ,smr.func_set
            ,smr.start_values
            ,x_block_name='sol'
            ,phi_block_name='Phi_2d'
        )
        sol_dict=blivp.block_solve(t_span=(t_0,t_max),t_eval=times)
        #print(sol_dict)
        phi_mat=sol_dict['Phi_2d'][-1,...]
        
        blivp_l= x_phi_ivp(
            smrl.model
            ,smrl.parameter_dict
            ,smrl.func_set
            ,smrl.start_values
            ,x_block_name='sol'
            ,phi_block_name='Phi_2d'
        )
        sol_dict_l=blivp.block_solve(t_span=(t_0,t_max),t_eval=times)
        
        phi_mat_l=sol_dict_l['Phi_2d'][-1,...]
        print(phi_mat)
        
        for mat in [phi_mat , phi_mat_l]:
            for x in bvs:
                with self.subTest():
                    phi_x_old=smrl._state_transition_operator_for_linear_systems(t_max,t_0,x),
                    phi_x_mat=np.matmul(mat,x).reshape(nr_pools,)
                    self.assertTrue( 
                        np.allclose(
                            phi_x_old, 
                            phi_x_mat,
                            atol=1e-5
                        )
                    )


    def test_phi_cache_vals(self):
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
        nt=2000 # the old way relies on the interpolation and goes wild for 
        # small nt...
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
        nr_pools=srm.nr_pools
        #smr.build_state_transition_operator_cache_2b(size=3)
        cache= smr._compute_state_transition_operator_cache_2b(size=2)

        phi_2_0=cache.values[0]
        phi_4_2=cache.values[1]


        def baseVector(i):
            e_i = np.zeros((nr_pools,1))
            e_i[i] = 1
            return e_i
        bvs = [ baseVector(i) for i in range(nr_pools)]
        smrl=smr.linearize()
        for ind,phi in enumerate(cache.values):
            tmin=cache.keys[ind]
            tmax=cache.keys[ind+1]
            for x in bvs:
                with self.subTest():
                    phi_x_old=smrl._state_transition_operator_for_linear_systems(tmax,tmin,x)
                    phi_x_mat=np.matmul(phi,x).reshape(nr_pools,)
                    self.assertTrue( 
                        np.allclose(
                            phi_x_old, 
                            phi_x_mat,
                            rtol=1e-2
                        )
                    )

    def test_x_phi_ivp_linear(self):
        # We compute Phi_t0(t) =Phi(t,t_0) with fixed t_0
        # This can be computed by a skew product system
        # and can be used to compute the statetransition operator as function of both time arguments
        k_0_val=1
        k_1_val=2
        x0_0=np.float(1)
        x0_1=np.float(1)
        delta_t=np.float(1)
        # 
        var(["x_0","x_1","k_0","k_1","t","u"])
        #
        inputs={
             0:u
            ,1:u
        }
        outputs={
             0:x_0*k_0
            ,1:x_1*k_1
        }
        internal_fluxes={}
        srm=SmoothReservoirModel(
                 state_vector       =[x_0,x_1]
                ,time_symbol        =t
                ,input_fluxes       =inputs
                ,output_fluxes      =outputs
                ,internal_fluxes    =internal_fluxes
        )
        # for this system we know the state transition operator (for fixed t0) to be a simple matrix exponential
        def Phi_ref(times):
            if isinstance(times,np.ndarray): 
                #3d array
                res=np.zeros((nr_pools,nr_pools,len(times)))
                res[0,0,:]=np.exp(-k_0_val*times)
                res[1,1,:]=np.exp(-k_1_val*times)
            else: 
                #2d array for scalar time
                res=np.zeros((nr_pools,nr_pools))
                res[0,0]=np.exp(-k_0_val*times)
                res[1,1]=np.exp(-k_1_val*times)
            return(res)

        t_max=4
        times = np.linspace(0, t_max, 11)
        parameter_dict = {
             k_0: k_0_val
            ,k_1: k_1_val
            ,u:1

        }
        func_dict={}
        nr_pools=srm.nr_pools
        nq=nr_pools*nr_pools
        sol_rhs=numerical_rhs2(
             srm.state_vector
            ,srm.time_symbol
            ,srm.F
            ,parameter_dict
            ,func_dict
        )
        #
        start_x= np.array([x0_0,x0_1])
        start_Phi_1d=np.identity(nr_pools).reshape(nr_pools**2)
        start_Int_Phi_u=np.zeros(nr_pools)
        t_span=(0,t_max)
        #        
        #)
        block_ivp=x_phi_ivp(srm,parameter_dict,func_dict,start_x,x_block_name='sol',phi_block_name='Phi')
        s_block_ivp=block_ivp.solve(t_span=t_span)
        t_block_rhs    = block_ivp.get_values("t"         ,t_span=t_span)
        sol_block_rhs  = block_ivp.get_values("sol"       ,t_span=t_span)
        Phi_block_rhs  = block_ivp.get_values("Phi"       ,t_span=t_span)
        Phi_block_rhs_mat   =Phi_block_rhs.reshape(nr_pools,nr_pools,len(t_block_rhs))
        # for comparison solve the original system without the phi rows
        sol=solve_ivp(fun=sol_rhs,t_span=(0,t_max),y0=start_x,max_step=delta_t,method='LSODA')
        # and make sure that is identical with the block rhs by using the interpolation function
        # for the block system and apply it to the grid that the solver chose for sol
        sol_func  = block_ivp.get_function("sol"       ,t_span=t_span,dense_output=True)
        self.assertTrue(np.allclose(sol.y,sol_func(sol.t),rtol=1e-2))
        # check Phi
        self.assertTrue(np.allclose(Phi_block_rhs_mat,Phi_ref(t_block_rhs),atol=1e-2))

        fig=plt.figure(figsize=(7,7))
        ax1=fig.add_subplot(2,1,1)
        ax1.plot(sol.t,               sol.y[0,:],'o',color='red' ,label="sol[0]")
        ax1.plot(sol.t,               sol.y[1,:],'x',color='red' ,label="sol[1]")
        #
        ax1.plot(sol.t,               sol_func(sol.t)[0,:],'*',color='blue' ,label="sol[0]")
        ax1.plot(sol.t,               sol_func(sol.t)[1,:],'+',color='blue' ,label="sol[1]")
        #
        ax1.plot(t_block_rhs, sol_block_rhs[0,:],'*',color='green',label="sol_block_rhs[0]")
        ax1.plot(t_block_rhs, sol_block_rhs[1,:],'+',color='green',label="sol_block_rhs[1]")
        #
        ax1.legend()

        ax2=fig.add_subplot(2,1,2)
        ax2.plot(t_block_rhs  ,Phi_ref(t_block_rhs)[0,0,:],'o',color='red'  ,label="Phi_ref[0,0]")
        ax2.plot(t_block_rhs  ,Phi_ref(t_block_rhs)[1,1,:],'x',color='red'  ,label="Phi_ref[1,1]")

        ax2.plot(t_block_rhs, Phi_block_rhs_mat[0,0,:],'*',color='green',label="Phi_block_rhs_mat[0,0]")
        ax2.plot(t_block_rhs, Phi_block_rhs_mat[1,1,:],'+',color='green',label="Phi_block_rhs_mat[1,1]")
        ax2.legend()
        fig.savefig("solutions.pdf")

    def test_stateTransitionOperator_by_different_methods(self):
        # The state transition operator Phi is defined for linear systems only
        # To compute it we have to create a linear system first by substituting
        # the solution into the righthandside
        # This could be done in different ways:
        # 1.)   By solving the ODE with the actual start vector first and then
        #       substituting the interpolation into the righthandside used to compute Phi(t,s)
        #       directly as solution of the ivp d/dt Phi = B(x(tau),tau ) 
        #       a) with startvalue I (identity matrix) integrated from s to t
        #       b) with the n column vectors of the Identity matrix seperately (this is the original approach)

        # 2.)   Creation of a skewproductsystem whose solution yields
        #       the solution for the initial value problem and the state transition operator for Phi(t,t_0) simultaniously.
        #       and computing Phi(t,s)=Phi(t,t_0)* Phi(s,t_0)^-1

        # This test makes sure that all approaches yield the same result       
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
        nr_pools=srm.nr_pools
        # now produce the state transition operator by different methods
        # if called without x the first two yield a matrix
        def Phi_skew(t,s,x=None):
            return smr._state_transition_operator_by_skew_product_system(t,s,x)

        def Phi_direct(t,s,x=None):
            return smr._state_transition_operator_by_direct_integration(t,s,x)

        # the original implementation requires always a vector phi is applied to 
        # to avoid the inversion and solve a single system instead
        def Phi_orig(t,s,x):
            return smr._state_transition_operator(t,s,x)
        
        # the old implementation assumed linear a system, which would be
        # disastrous if the model had not been linearized
        with self.assertRaises(Exception) as e:
            smr._state_transition_operator_for_linear_systems(t_max,t_0,start_x)
       
        # to be able to compare it we have to use the linearized_system
        smr_lin=smr.linearize()
        def Phi_old(t,s,x):
            return smr_lin._state_transition_operator_for_linear_systems(t,s,x)
        
        
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
        args=[(t,s,e) for t in test_times for s in test_times if s<=t for e in bvs]
        resultTupels=[
            (
                     np.matmul(   Phi_skew(t,s)    ,e_i).flatten() 
                    ,Phi_skew(t,s,e_i) 
                    ,np.matmul( Phi_direct(t,s)    ,e_i).flatten() 
                    ,Phi_direct(t,s,e_i)
                    ,Phi_orig(t,s ,e_i)
                    ,Phi_old(t,s ,e_i)
            )        
            for (t,s,e_i) in args
        ]
        

        rtol=2e-2 #2%
        def check_tuple(tup):
            skew_mat,skew_vec,direct_mat,direct_vec,orig_vec,old_vec=tup
            if not np.allclose(skew_mat,skew_vec,rtol=rtol):
                print('skew_mat,skew_vec')
                pprint((skew_mat,skew_vec))
                return False
            if not np.allclose(direct_mat,direct_vec,rtol=rtol):
                print('direct_mat,direct_vec')
                pprint((direct_mat,direct_vec))
                return False
            if not np.allclose(skew_mat,direct_mat,rtol=rtol):
                print('skew_mat,direct_mat')
                pprint((direct_mat,direct_vec))
                return False
            if not np.allclose(direct_vec,orig_vec,rtol=rtol):
                print('direct_vec,orig_vec')
                pprint((direct_vec,orig_vec))
                return False
            if not np.allclose(direct_vec,orig_vec,rtol=rtol):
                print('orig_vec','old_vec')
                pprint((orig_vec,old_vec))
                return False
            return True

        ok=[ check_tuple(tup) for tup in resultTupels ]

        self.assertTrue(all(ok))

#################### We now rebuild the solution by means of phi and plot it along with the original #################################
        original_sol   =smr.solve()
        
        u_sym=srm.external_inputs
        u_num=numerical_function_from_expression(u_sym,(t,),parameter_dict,{})
        
        
        def vectorlist2array(l):
            return np.stack( [vec.flatten() for vec in l],1)
        def lists_dict2array_dict(d):
            return {key:vectorlist2array(val) for key,val in d.items()}
        
        def continiuous_integral_values(Phi_func,integrator,times):
            start=time.time()
            res=vectorlist2array([integrator( lambda tau : Phi_func(t,tau,u_num(tau)) ,t_0 ,t) for t in times])
            stop=time.time()
            exec_time=stop-start
            #pe('exec_time',locals())
            return (times,res,exec_time)

        def discrete_integral_values(Phi_func,integrator,times):
            start=time.time()
            res=vectorlist2array([integrator(lambda tau:Phi_func(t,tau,u_num(tau)),taus=+times[0:i+1]) for i,t in enumerate(times)]) 
            stop=time.time()
            exec_time=stop-start
            #pe('exec_time',locals())
            return (times,res,exec_time)
        

        ## reconstruct the solution with Phi and the integrand
        # x_t=Phi(t,t0)*x_0+int_t0^t Phi(tau,t0)*u(tau) dtau
        # x_t=a(t)+b(t)
        et=bvs[0]+bvs[1]    
        phi_arrays= {
             'old'   :(times,vectorlist2array([   Phi_old(t,t_0,et).reshape(srm.nr_pools,1) for t in times]))
            ,'skew'  :(times,vectorlist2array([  Phi_skew(t,t_0,et).reshape(srm.nr_pools,1) for t in times]))
            ,'direct':(times,vectorlist2array([Phi_direct(t,t_0,et).reshape(srm.nr_pools,1) for t in times]))
        }
        
        a_arrays={
             'old'    :(       times,vectorlist2array([ Phi_old(t,t_0,start_x).reshape(srm.nr_pools,1) for t in times]))
            ,'skew'   :(       times,vectorlist2array([Phi_skew(t,t_0,start_x).reshape(srm.nr_pools,1) for t in times]))
            ,'trapez1':(       times,vectorlist2array([Phi_skew(t,t_0,start_x).reshape(srm.nr_pools,1) for t in times]))
            ,'trapez2':(double_times,vectorlist2array([Phi_skew(t,t_0,start_x).reshape(srm.nr_pools,1) for t in double_times]))
            ,'trapez4':(quad_times  ,vectorlist2array([Phi_skew(t,t_0,start_x).reshape(srm.nr_pools,1) for t in quad_times]))
        }
        nested_boundary_tuples=[(0,t) for t in reversed(times)]
        
        b_arrays_trapez={
        }
        b_arrays={
             'old'      :continiuous_integral_values(   Phi_old,array_integration_by_ode    ,times)
            ,'skew'     :continiuous_integral_values(  Phi_skew,array_integration_by_ode    ,times)
            ,'direct'   :continiuous_integral_values(Phi_direct,array_integration_by_ode    ,times)
            ,'trapez1'  :discrete_integral_values(     Phi_skew,array_integration_by_values ,times)
            ,'trapez2'  :discrete_integral_values(  Phi_skew,array_integration_by_values,double_times)
            ,'trapez4'  :discrete_integral_values(  Phi_skew,array_integration_by_values,quad_times)
        }
         

        b_arrays_quad={
             'old'      :continiuous_integral_values(   Phi_old,array_quad_result,times)
            ,'skew'     :continiuous_integral_values(  Phi_skew,array_quad_result,times)
            ,'direct'   :continiuous_integral_values(Phi_direct,array_quad_result,times)
        }
    
        
        x_arrays={key:(a_arrays[key][0],a_arrays[key][1]+b_arrays[key][1]) for key in a_arrays.keys()}
        #x_arrays['trapez']=(times,a_arrays['skew'][1]+b_arrays['trapez'][1])
            
        styleDict=OrderedDict({
             'old'      :('red',8)
            ,'skew'     :('green',6)
            ,'direct'   :('orange',4)
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
        axl.plot(times,original_sol[:,0],'o',color='blue' ,label="original_sol[0]")
        axr.plot(times,original_sol[:,1],'o',color='blue' ,label="original_sol[1]")
        
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
    
        
        #pe('b_arrays["skew"][1]-b_arrays["direct"][1]',locals())
    def test_save_and_load_state_transition_operator_cache(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = Matrix([C_0, C_1])
        time_symbol = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5, 3])
        times = np.linspace(0,1,6)

        smr = SmoothModelRun(srm, {}, start_values, times)
        
        ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        start_age_densities = lambda a: np.exp(-a)*start_values
        smr.build_state_transition_operator_cache()

        ca = smr._state_transition_operator_values
        size = smr._cache_size

        filename = 'sto.cache'
        smr.save_state_transition_operator_cache(filename)
        smr.load_state_transition_operator_cache(filename)
    
        self.assertEqual(size, smr._cache_size)
        self.assertTrue(np.all(ca==smr._state_transition_operator_values))


    def test_state_transition_operator_1d(self):
        # one-dimensional case
        C = Symbol('C')
        state_vector = [C]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1} # are inputs really ignored in the computation of Phi?
        output_fluxes = {0: C}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)

        x = np.array([1])

        Phix = smr._state_transition_operator(1,0,x)

        self.assertEqual(Phix.shape, (1,))
        self.assertTrue(abs(Phix-np.exp(-1))<1e-03)
        print(type(Phix))

    def test_state_transition_operator_2d(self):
        # two-dimensional case
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5,3])
        times = np.linspace(0,1,11)
        smr = SmoothModelRun(srm, {}, start_values, times)

        x = np.array([1,3])
        Phix = smr._state_transition_operator(1,0,x)
       
        self.assertEqual(Phix.shape, (2,))
        
        # test t<t_0
        with self.assertRaises(Exception):
            Phix = smr._state_transition_operator(0,1,x)

        # test if operator works correctly also late in time
        C = Symbol('C')
        state_vector = [C]
        time_symbol = Symbol('t')
        input_fluxes = {0: 1} # are inputs really ignored in the computation of Phi?
        output_fluxes = {0: C}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, time_symbol, input_fluxes, output_fluxes, internal_fluxes)

        start_values = np.array([5])
        times = np.linspace(0,100,101)
        smr = SmoothModelRun(srm, {}, start_values, times)

        x = np.array([1])

        Phix = smr._state_transition_operator(91,89,x)
        self.assertTrue(abs(Phix-np.exp(-2))<1e-03)
