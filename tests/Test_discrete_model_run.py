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
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function, ones
    
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
        pardict = {
             k: -1
            ,u:1}
        delta_t=np.float(1)
        
        smr = SmoothModelRun(srm, pardict, start_values, times)
        
        # export the ingredients for an different ode solver 
        srm = smr.model
        state_vector, rhs = srm.age_moment_system(max_order=0)
        num_rhs = numerical_rhs2(
            state_vector,
            srm.time_symbol,
            rhs, 
            pardict,
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

    def test_SkewProductSystem(self):
        nr_pools=2 
        nq=nr_pools*nr_pools
        k_1_val=-1
        k_2_val=-2
        # construct a simple (diagonal) matrix exponential
        # (with the 3 dimension representing time)
        def Phi_ref(times):
            if isinstance(times,np.ndarray): 
                #3d array
                res=np.zeros((nr_pools,nr_pools,len(times)))
                res[0,0,:]=np.exp(k_1_val*times)
                res[1,1,:]=np.exp(k_2_val*times)
            else: 
                #2d array for scalar time
                res=np.zeros((nr_pools,nr_pools))
                res[0,0]=np.exp(k_1_val*times)
                res[1,1]=np.exp(k_2_val*times)

            return(res)
        
        #build a system that has this solution and solve it
        B=np.array([[k_1_val,0],[0,k_2_val]])
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
        sskew=solve_ivp(fun=rhs_skew,t_span=[0,t_max],y0=start_skew,max_step=delta_t,vectorized=False,method='LSODA')
        Phi_skew=sskew.y[0:nq,:]
        Phi_skew_mat=Phi_skew.reshape(nr_pools,nr_pools,len(sskew.t))
        derivedVals=sskew.y[nq:,:]
        print(derivedVals.shape)

        fig=plt.figure(figsize=(7,7))
        ax1=fig.add_subplot(2,1,1)
        ax1.plot(times  ,Phi_ref(times)[0,0,:],'*',color='red'  ,label="s[0,0]")
        ax1.plot(sf.t   ,        sf_mat[0,0,:],'*',color='blue' ,label="ivp[0,0]")
        ax1.plot(sskew.t,  Phi_skew_mat[0,0,:],'*',color='green',label="skew_mat[0,0]")
        ax1.plot(times  ,Phi_ref(times)[1,1,:],'+',color='red' ,label="s[1,1]")
        ax1.plot(sf.t   ,        sf_mat[1,1,:],'+',color='blue',label="ivp[1,1]")
        ax1.plot(sskew.t,  Phi_skew_mat[1,1,:],'*',color='green',label="skew_mat[1,1]")
        ax1.legend()
        ax2=fig.add_subplot(2,1,2)
        ax2.plot(ss.t   ,        ss.y[0,:],'*',color='blue'  ,label="subs[0]")
        ax2.plot(ss.t   ,        ss.y[1,:],'+',color='blue'  ,label="subs[1]")
        ax2.plot(sskew.t, derivedVals[0,:],'*',color='green' ,label="skew[0]")
        ax2.plot(sskew.t, derivedVals[1,:],'+',color='green' ,label="skew[1]")
        ax2.legend()
        fig.savefig("solutions.pdf")

        # now we use the smr to build the skewproduct system automatically
        # first we create an equivalent model 
        x_0,x_1,t,k_1,k_2,u = symbols("x_1,x_2,k_1,k_2,t,u")
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
        pardict = {
             k_1: k_1_val
            ,k_2: k_2_val
            ,u:1

        }
        #B=srm.compartmental_matrix
        nr_poools=srm.nr_pools
        #tup=(t,)+tuple(srm.state_vector)
        #B_num=numerical_function_from_expression(B,tup,pardict,{})
        vec_num_rhs=numerical_rhs2(
             m_no_inputs.state_vector
            ,m_no_inputs.time_symbol
            ,m_no_inputs.F
            ,pardict
            ,{}
        )
        # for the numerical rhs we have to create a vector valued function, but we want all columns of 
        # the state transition operator we have to rearange it as one long vector
        # since B in the nonlinear case also depends on x we also have to compute B for every portion seperately      
        def mat_num_rhs(t,X):
            # cut X in chunks
            xs=[X[i*nr_pools:(i+1)*nr_pools] for i in range(nr_pools)]
            ys=[vec_num_rhs(t,x) for x in xs]
            res=np.empty_like(X)
            for i in range(nr_pools):
                res[[i*nr_pools:(i+1)*nr_pools]=yx[i]
            return res




        

