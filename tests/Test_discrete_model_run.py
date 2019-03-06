#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:

from concurrencytest import ConcurrentTestSuite, fork_for_tests
import inspect
import sys 
import unittest
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.integrate import quad
from scipy.interpolate import interp1d 
from scipy.misc import factorial
from scipy.integrate import solve_ivp,OdeSolver,odeint
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function, ones,var
    
import example_smooth_reservoir_models as ESRM
import example_smooth_model_runs as ESMR

from testinfrastructure.InDirTest import InDirTest
from testinfrastructure.helpers import pe
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.smooth_model_run import SmoothModelRun 
from CompartmentalSystems.discrete_model_run import DiscreteModelRun
from CompartmentalSystems.helpers_reservoir import numerical_function_from_expression,numerical_rhs2



class TestDiscreteModelRun(InDirTest):
    def test_from_SmoothModelRun(self):
        x_0,x_1,t,k,u = symbols("x_1,x_2,k,t,u")
        inputs={
             0:u
            ,1:u
        }
        outputs={
             0:-x_0*k
            ,1:-x_1*k
        }
        internal_fluxes={}
        srm=SmoothReservoirModel([x_0,x_1],t,inputs,outputs,internal_fluxes)
        t_max=4
        times = np.linspace(0, t_max, 11)
        x0=np.float(10)
        start_values = np.array([x0,x0])
        parameter_dict = {
             k: -1
            ,u:1}
        delta_t=np.float(1)
        
        smr = SmoothModelRun(srm, parameter_dict, start_values, times)
        
        # export the ingredients for an different ode solver 
        srm = smr.model
        state_vector, rhs = srm.age_moment_system(max_order=0)
        num_rhs = numerical_rhs2(
            state_vector,
            srm.time_symbol,
            rhs, 
            parameter_dict,
            {}
        )
        sf=solve_ivp(fun=num_rhs,t_span=[0,t_max],y0=start_values,max_step=delta_t,vectorized=False,method='LSODA')
        
        dmr = DiscreteModelRun.from_SmoothModelRun(smr)
        smrs=smr.solve()
        dmrs=dmr.solve()
        fig=plt.figure(figsize=(7,7))
        ax1=fig.add_subplot(2,1,1)
        ax1.plot(times,smrs[:,0],'*',color='red',label="smr")
        ax1.plot(times,dmrs[:,0],'*',color='blue',label="dmr")
        n=len(sf.t)
        ax1.plot(sf.t,sf.y[0].reshape(n,),'*',color='green',label="solve_ivp")
        ax1.legend()
        ax2=fig.add_subplot(2,1,2)
        ax2.plot(times,smrs[:,1],'*',color='red',label="smr")
        ax2.plot(times,dmrs[:,1],'*',color='blue',label="dmr")
        n=len(sf.t)
        ax2.plot(sf.t,sf.y[1].reshape(n,),'*',color='green',label="solve_ivp")
        ax2.legend()
        fig.savefig("pool_contents.pdf")
        self.assertTrue(True)

#--------------------------------------------------------------------------
    def test_SkewProductSystem(self):
        nr_pools=2 
        nq=nr_pools*nr_pools
        k_0_val=-1
        k_2_val=-2
        # construct a simple (diagonal) matrix exponential
        # (with the 3 dimension representing time)
        def Phi_ref(times):
            if isinstance(times,np.ndarray): 
                #3d array
                res=np.zeros((nr_pools,nr_pools,len(times)))
                res[0,0,:]=np.exp(k_0_val*times)
                res[1,1,:]=np.exp(k_2_val*times)
            else: 
                #2d array for scalar time
                res=np.zeros((nr_pools,nr_pools))
                res[0,0]=np.exp(k_0_val*times)
                res[1,1]=np.exp(k_2_val*times)
            return(res)
        
        #build a system that has this solution and solve it
        B=np.array([[k_0_val,0],[0,k_2_val]])
        start_matrix_1d=np.identity(nr_pools).reshape(nq,)
        def f(t,X):
            return (np.matmul(B,X.reshape((nr_pools,nr_pools)))).reshape(nq,)
        
        t_max=1 
        nr_times=11
        times = np.linspace(0, t_max, nr_times)
        delta_t=np.float(1)
        
        sf=solve_ivp(fun=f,t_span=[0,t_max],y0=start_matrix_1d,max_step=delta_t,vectorized=False,method='LSODA')
        sf_mat=sf.y.reshape(nr_pools,nr_pools,len(sf.t)) 

        # now solve a new system where the righthandside depends on the solution
        # in this case an integral of the product with a (in general) timedependent vector
        u           =lambda t: np.array([1,1])
        integrand   =lambda t: np.matmul(Phi_ref(t),u(t))
         
        # for an integral the argument X is ignored
        rhs_subs =lambda t,X:integrand(t)
        start_u=np.zeros((nr_pools,))
        ss=solve_ivp(fun=rhs_subs,t_span=[0,t_max],y0=start_u,max_step=delta_t,vectorized=False,method='LSODA')
        

        # now we build the skewproduct system that yields the matrixexponential and the derived variable simultaniusly
        def rhs_skew(t,X):
            #split into the parts for Phi and f
            Phi=X[0:nq]
            Phi_mat=Phi.reshape(nr_pools,nr_pools)
            return np.append( f(t,Phi) ,np.matmul(Phi_mat,u(t)))
            
        start_skew=np.append(start_matrix_1d,start_u)
        s_skew=solve_ivp(fun=rhs_skew,t_span=[0,t_max],y0=start_skew,max_step=delta_t,vectorized=False,method='LSODA')
        Phi_skew=s_skew.y[0:nq,:]
        Phi_skew_mat=Phi_skew.reshape(nr_pools,nr_pools,len(s_skew.t))
        derivedVals=s_skew.y[nq:,:]
        print(derivedVals.shape)

        fig=plt.figure(figsize=(7,7))
        ax1=fig.add_subplot(2,1,1)
        ax1.plot(times  ,Phi_ref(times)[0,0,:],'*',color='red'  ,label="s[0,0]")
        ax1.plot(sf.t   ,        sf_mat[0,0,:],'*',color='blue' ,label="ivp[0,0]")
        ax1.plot(s_skew.t,  Phi_skew_mat[0,0,:],'*',color='green',label="skew_mat[0,0]")
        ax1.plot(times  ,Phi_ref(times)[1,1,:],'+',color='red' ,label="s[1,1]")
        ax1.plot(sf.t   ,        sf_mat[1,1,:],'+',color='blue',label="ivp[1,1]")
        ax1.plot(s_skew.t,  Phi_skew_mat[1,1,:],'*',color='green',label="skew_mat[1,1]")
        ax1.legend()
        ax2=fig.add_subplot(2,1,2)
        ax2.plot(ss.t   ,        ss.y[0,:],'*',color='blue'  ,label="subs[0]")
        ax2.plot(ss.t   ,        ss.y[1,:],'+',color='blue'  ,label="subs[1]")
        ax2.plot(s_skew.t, derivedVals[0,:],'*',color='green' ,label="skew[0]")
        ax2.plot(s_skew.t, derivedVals[1,:],'+',color='green' ,label="skew[1]")
        ax2.legend()
        fig.savefig("solutions.pdf")

        # now we use the smr to build the skewproduct system automatically
        # first we create an equivalent model 
        x_0,x_1,t,k_0,k_2,u = symbols("x_1,x_2,k_0,k_2,t,u")
        inputs={
             0:u
            ,1:u
        }
        outputs={
             0:-x_0*k_1
            ,1:-x_1*k_2
        }
        internal_fluxes={}
        srm=SmoothReservoirModel([x_0,x_1],t,inputs,outputs,internal_fluxes)
        m_no_inputs=srm.no_input_model
        t_max=4
        times = np.linspace(0, t_max, 11)
        parameter_dict = {
             k_1: k_0_val
            ,k_2: k_1_val
            ,u:1

        }
        #B=srm.compartmental_matrix
        nr_oools=srm.nr_pools
        #tup=(t,)+tuple(srm.state_vector)
        #B_num=numerical_function_from_expression(B,tup,parameter_dict,{})
        vec_num_rhs=numerical_rhs2(
             m_no_inputs.state_vector
            ,m_no_inputs.time_symbol
            ,m_no_inputs.F
            ,parameter_dict
            ,{}
        )
        # for the numerical rhs we have to create a vector valued function, but we want all columns of 
        # the state transition operator we have to rearange it as one long vector
        # since B in the nonlinear case also depends on x we also have to compute B for every portion seperately      
        def mat_num_rhs_1d(t,X):
            # cut X in chunks
            xs=[X[i*nr_pools:(i+1)*nr_pools] for i in range(nr_pools)]
            ys=[vec_num_rhs(t,x) for x in xs]
            res=np.empty_like(X)
            # combine to one array again
            for i in range(nr_pools):
                res[i*nr_pools:(i+1)*nr_pools]=ys[i]
            return res
        
        start_matrix_1d=np.identity(nr_pools).reshape(nq,)
        sf=solve_ivp(fun=mat_num_rhs_1d,t_span=[0,t_max],y0=start_matrix_1d,max_step=delta_t,vectorized=False,method='LSODA')

    def test_linearization_skew_linear(self):
        # The state transition operator is defined for linear systems only
        # to compute it we have to create a linear system first by substituting
        # the solution into the righthandside
        # This could be done in different waya:
        # 1.)   By solving the ODE with the actual start vector first and then
        #       substituting the interpolation into the righthandside used to compute the state transition operator
        # 2.)   Creation of a skewproductsystem whose solution yields
        #       the solution for the initial value problem and the state transition operator simultaniously.
        # We first check the result for a linear system, where the whole procedure is not neccesary
        k_0_val=1
        k_1_val=2
        x0_0=np.float(1)
        x0_1=np.float(1)
        start_x= np.array([x0_0,x0_1])
        delta_t=np.float(1)
        
        var(["x_0","x_1","k_0","k_1","t","u"])

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
        m_no_inputs=srm.no_input_model
        t_max=4
        times = np.linspace(0, t_max, 11)
        parameter_dict = {
             k_0: k_0_val
            ,k_1: k_1_val
            ,u:1

        }
        func_dict={}
        #B=srm.compartmental_matrix
        nr_pools=srm.nr_pools
        nq=nr_pools*nr_pools
        #tup=(t,)+tuple(srm.state_vector)
        #B_num=numerical_function_from_expression(B,tup,parameter_dict,{})
        sol_num_rhs=numerical_rhs2(
             srm.state_vector
            ,srm.time_symbol
            ,srm.F
            ,parameter_dict
            ,func_dict
        )
        # for the rhs of the complete system have to creat a vector valued function, 
        # The first part of the vector will be the solution x
        # Then the components of the state transition operator follow
        B_sym=srm.compartmental_matrix
        
        tup=(t,)+tuple(srm.state_vector)
        B_func=numerical_function_from_expression(B_sym,tup,parameter_dict,func_dict)
        x_i_start=0
        x_i_end=nr_pools
        Phi_1d_i_start=x_i_end
        Phi_1d_i_end=(nr_pools+1)*nr_pools
        int_i_start=Phi_1d_i_end
        int_i_end=int_i_start+nr_pools
        def mat_num_rhs_1d(t,X):
            x=X[x_i_start:x_i_end]
            # cut out the solution part of X
            res_sol=sol_num_rhs(t,x)
            B=B_func(t,*x)
            #the next chunks are the columns of PHI
            Phi_1d=X[Phi_1d_i_start:Phi_1d_i_end]
            # we could hust reshape it to a matrix 
            # Phi_mat =Phi_1d.reshape((nr_pools,nr_pools)
            # and write res=np.matmul(B,Phi_mat)
            # but we can also separate the columns (e.g to compute them in parallel)
            Phi_cols=[Phi_1d[i*nr_pools:(i+1)*nr_pools] for i in range(nr_pools)]
            Phi_ress=[np.matmul(B,pc) for pc in Phi_cols]

            # combine the results to one 1d array again
            res=np.empty_like(X)
            res[x_i_start:x_i_end]=res_sol
            # the columns of PHi
            for i in range(nr_pools):
                res[(i+1)*nr_pools:(i+2)*nr_pools]=Phi_ress[i]
            return res

        start_Phi_1d=np.identity(nr_pools).reshape(nq,)
        start_skew=np.append(start_x,start_Phi_1d)
        pe('start_skew',locals())
        s_skew=solve_ivp(fun=mat_num_rhs_1d,t_span=[0,t_max],y0=start_skew,max_step=delta_t,vectorized=False,method='LSODA')
        # extract the solution for x, and Phi
        sol_skew=s_skew.y[x_i_start:x_i_end,:]
        Phi_skew=s_skew.y[Phi_1d_i_start:Phi_1d_i_end,:]
        Phi_skew_mat=Phi_skew.reshape(nr_pools,nr_pools,len(s_skew.t))

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
        # for comparison solve the original system 
        sol=solve_ivp(fun=sol_num_rhs,t_span=[0,t_max],y0=start_x,max_step=delta_t,vectorized=False,method='LSODA')


        # we now express the solution as expression of the state transition operator
        # x_t=Phi(t,t0)*x_0+int_t0^t Phi(tau,t0)*u(tau) dtau
        # and check that we get the original solution back
        # We build a skewproduct system  for int_t0^t Phi(tau,t0)*u(tau) dtau 
        # ontop of the skew product system for Phi 
        # The the initial values for the additional Variables we set to 0 
        u_sym=srm.external_inputs
        tup=(t,)
        u_num=numerical_function_from_expression(u_sym,tup,parameter_dict,func_dict)
        def skew_skew_num_rhs_1d(t,X):
            #cut out the part of X for the solution and the state transition operator
            x_and_Phi=X[x_i_start:Phi_1d_i_end]
            res_x_and_Phi=mat_num_rhs_1d(t,x_and_Phi)
            Phi_1d=X[Phi_1d_i_start:Phi_1d_i_end]
            Phi_mat=Phi_1d.reshape(nr_pools,nr_pools)
            #int_vars=X[Phi_1d_i_end:] # in this case not necessary since this part of the rhs does not depend on t
            #the according state (only indirectly via Phi ..)
            res_int=np.matmul(Phi_mat,u_num(t))
            return np.append(res_x_and_Phi,res_int)

        start_skew_skew=np.append(start_skew,np.zeros(nr_pools))
        s_skew_skew=solve_ivp(fun=skew_skew_num_rhs_1d,t_span=[0,t_max],y0=start_skew_skew,max_step=delta_t,vectorized=False,method='LSODA')
        t_skew_skew=s_skew_skew.t
        sol_skew_skew=s_skew_skew.y[x_i_start:x_i_end,:]
        
        Phi_skew_skew       =s_skew_skew.y[Phi_1d_i_start:Phi_1d_i_end,:]
        Phi_skew_skew_mat   =Phi_skew_skew.reshape(nr_pools,nr_pools,len(t_skew_skew))
        int_skew_skew       =s_skew_skew.y[int_i_start:int_i_end,:]
        pe('int_skew_skew.shape',locals())
        sol2_skew_skew=np.stack(
            [np.matmul(Phi_skew_skew_mat[:,:,i],start_x)+int_skew_skew[:,i] for i in range(len(t_skew_skew))]
           ,1
        )
        pe('sol2_skew_skew.shape',locals())

        fig=plt.figure(figsize=(7,7))
        ax1=fig.add_subplot(2,1,1)
        ax1.plot(sol.t,               sol.y[0,:],'*',color='blue' ,label="sol[0]")
        ax1.plot(sol.t,               sol.y[1,:],'+',color='blue' ,label="sol[1]")

        ax1.plot(t_skew_skew, sol_skew_skew[0,:],'*',color='green',label="sol_skew_skew[0]")
        ax1.plot(t_skew_skew, sol_skew_skew[1,:],'+',color='green',label="sol_skew_skew[1]")

        ax1.plot(t_skew_skew,sol2_skew_skew[0,:],'*',color='orange',label="sol2_skew_skew[0]")
        ax1.plot(t_skew_skew,sol2_skew_skew[1,:],'+',color='orange',label="sol2_skew_skew[1]")

        #ax1.plot(s_skew_skew.t, s_skew_skew.y[int_i_start,:],'*',color='orange',label="sol_skew_skew[0]")
        #ax1.plot(s_skew_skew.t, s_skew_skew.y[  int_i_end,:],'+',color='orange',label="sol_skew_skew[1]")
        
        ax1.legend()

        ax2=fig.add_subplot(2,1,2)
        ax2.plot(times  ,Phi_ref(times)[0,0,:],'*',color='red'  ,label="Phi_ref[0,0]")
        ax2.plot(times  ,Phi_ref(times)[1,1,:],'+',color='red'  ,label="Phi_ref[1,1]")

        ax2.plot(t_skew_skew, Phi_skew_skew_mat[0,0,:],'*',color='green',label="Phi_skew_skew_mat[0,0]")
        ax2.plot(t_skew_skew, Phi_skew_skew_mat[1,1,:],'+',color='green',label="Phi_skew_skew_mat[1,1]")
        ax2.legend()
        fig.savefig("solutions.pdf")
        

        

