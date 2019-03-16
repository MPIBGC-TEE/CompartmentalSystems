import numpy as np
from sympy import var
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from sympy import var,Matrix
from sympy.printing import pprint

from testinfrastructure.InDirTest import InDirTest
from testinfrastructure.helpers import pe
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.smooth_model_run import SmoothModelRun 
from CompartmentalSystems.BlockIvp import BlockIvp
from CompartmentalSystems.helpers_reservoir import numerical_function_from_expression,numerical_rhs2,x_phi_ivp

class TestPhi(InDirTest):
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
        x0_0=np.float(2)
        x0_1=np.float(1)
        delta_t=np.float(1./4.)
        # 
        var(["x_0","x_1","k_0","k_1","t","u"])
        #
        inputs={
             0:u
            ,1:u*t
        }
        outputs={
             0:k_0*x_0#**2
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
        times = np.linspace(t_0, t_max, 25)
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
                     np.matmul(   Phi_skew(t,s)    ,e_i) 
                    ,Phi_skew(t,s,e_i) 
                    ,np.matmul( Phi_direct(t,s)    ,e_i)
                    ,Phi_direct(t,s,e_i)
                    ,Phi_orig(t,s ,e_i)
            )        
            for (t,s,e_i) in args
        ]
        

        rtol=1e-2
        def check_tuple(tup):
            skew_mat,skew_vec,direct_mat,direct_vec,orig_vec=tup
            if not np.allclose(skew_mat,skew_vec,rtol=rtol):
                print('skew_mat,skew_vec')
                pprint((skew_mat,skew_vec))
                return False
            if not np.allclose(direct_mat,direct_vec,rtol=rtol):
                print('direct_mat,direct_vec')
                pprint((direct_mat,direct_vec))
                return False
            if not np.allclose(direct_vec,orig_vec,rtol=rtol):
                print('direct_vec,orig_vec')
                pprint((direct_vec,orig_vec))
                return False
            return True

        ok=[ check_tuple(tup) for tup in resultTupels ]
        #ok=[np.allclose(a,b,rtol=rtol) and np.allclose(a,c,rtol=rtol) for (a,b,c) in resultTupels])
        pe('ok',locals())
#        ts   =my_x_phi_ivp.get_values("t",t_span=t_span,max_step=.2)
#        original_sol   =my_x_phi_ivp.get_values("sol",t_span=t_span)
#        phis =my_x_phi_ivp.get_values("Phi_1d",t_span=t_span)
        original_sol   =smr.solve()
        pe('original_sol.shape',locals())
#        
#        
#        rs=(nr_pools,len(ts))
#        
        u_sym=srm.external_inputs
        u_num=numerical_function_from_expression(u_sym,(t,),parameter_dict,{})
#        #
        def trapez_integral(i):
            # We compute the integral 
            # NOT as the solution of an ivp but with an integration rule that 
            # works with arrays instead of functions
            # this allows us to predict where we need the state transition operator
            # but sacrifices precision
            t=times[i]
            if i==0:
                return np.zeros((nr_pools,1))
            # the integrals boundaries grow with time
            # so the vector for the trapezrule becomes longer and longer
            taus=times[0:i]
            sh=(nr_pools,len(taus)) 
            phi_vals=np.array([Phi_skew(t,tau) for tau in taus])
            integrand_vals=np.stack([np.matmul(Phi_skew(t,tau),u_num(tau)).flatten() for tau in taus],1)
            val=np.trapz(y=integrand_vals,x=taus).reshape(nr_pools,1)
            return val
        
        def continiuous_integral(t,Phi_func):
            # We compute the integral of the continious function
            # as an ivp
            def rhs(tau,X):
                # although we do not need X we have to provide a 
                # righthandside suitable for solve_ivp
                return np.matmul(Phi_func(t,tau),u_num(tau)).flatten()
        
            ys= solve_ivp(rhs,y0=np.zeros(nr_pools),t_span=(t_0,t)).y
            val=ys[:,-1].reshape(nr_pools,1)
            return val
        

        ## reconstruct the solution with Phi and the integrand
        # x_t=Phi(t,t0)*x_0+int_t0^t Phi(tau,t0)*u(tau) dtau
        # x_t=a(t)+b(t)
        a1_list=[np.matmul(  Phi_skew(t,t_0),start_x.reshape(srm.nr_pools,1)) for t in times]
        a2_list=[np.matmul(Phi_direct(t,t_0),start_x.reshape(srm.nr_pools,1)) for t in times]
        b_list=[ trapez_integral(i) for i,t in enumerate(times)]
        b_cont1_list=[continiuous_integral(t,Phi_skew) for t in times]
        b_cont2_list=[continiuous_integral(t,Phi_direct) for t in times]
        x2_list=[a1_list[i]+b_list[i] for i,t in enumerate(times)]
        x3_list=[a1_list[i]+b_cont1_list[i] for i,t in enumerate(times)]
        x4_list=[a2_list[i]+b_cont2_list[i] for i,t in enumerate(times)]
        def vectorlist2array(l):
            return np.stack( [vec.flatten() for vec in l],1)
        
        a1,a2,b,b_cont1,b_cont2,x2,x3,x4=map(
                vectorlist2array
                ,[a1_list,a2_list,b_list,b_cont1_list,b_cont2_list,x2_list,x3_list,x4_list])
        pe('b_cont1-b_cont2',locals())
        fig=plt.figure(figsize=(10,17))
        ax1=fig.add_subplot(2,1,1)
        ax1.plot(times,original_sol[:,0],'o',color='blue' ,label="original_sol[0]")
        ax1.plot(times,original_sol[:,1],'o',color='blue' ,label="original_sol[1]")
        
        ax1.plot(times,x2[0,:],'+',color='red' ,label="x2[0]")
        ax1.plot(times,x2[1,:],'+',color='red' ,label="x2[0]")
        
        ax1.plot(times,x3[0,:],'x',color='green' ,label="x3[0]")
        ax1.plot(times,x3[1,:],'x',color='green' ,label="x3[1]")
        
        ax1.plot(times,x3[0,:],'+',color='orange' ,label="x4[0]")
        ax1.plot(times,x3[1,:],'+',color='orange' ,label="x4[1]")
        ax1.legend()
        
        ax2=fig.add_subplot(2,1,2)
        ax2.plot(times,b_cont1[0,:],'x',color='green' ,label="b1[0]")
        ax2.plot(times,b_cont1[1,:],'x',color='green' ,label="b1[0]")
        
        ax2.plot(times,b_cont2[0,:],'+',color='orange' ,label="b2[0]")
        ax2.plot(times,b_cont2[1,:],'+',color='orange' ,label="b2[0]")
        
        ax2.plot(times,a1[0,:],'x',color='green' ,label="a1[0]")
        ax2.plot(times,a1[1,:],'x',color='green' ,label="a1[0]")
        
        ax2.plot(times,a2[0,:],'+',color='orange' ,label="a2[0]")
        ax2.plot(times,a2[1,:],'+',color='orange' ,label="a2[0]")
        
        ax2.legend()
        
        fig.savefig("solutions.pdf")
    
        
