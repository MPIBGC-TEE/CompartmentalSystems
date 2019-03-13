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
from scipy.integrate import solve_ivp,fixed_quad
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function, ones,var
from typing import Callable,Iterable,Union,Optional,List,Tuple 
from copy import copy
    
import example_smooth_reservoir_models as ESRM
import example_smooth_model_runs as ESMR

from testinfrastructure.InDirTest import InDirTest
from testinfrastructure.helpers import pe
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.smooth_model_run import SmoothModelRun 
from CompartmentalSystems.discrete_model_run import DiscreteModelRun
from CompartmentalSystems.helpers_reservoir import numerical_function_from_expression,numerical_rhs2
def X_Phi_IVP(srm,parameter_dict,func_dict,start_x):






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
        sf=solve_ivp(fun=num_rhs,t_span=[0,t_max],y0=start_values)#,max_step=delta_t,vectorized=False,method='LSODA')
        
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
#
    def test_block_rhs_versus_block_ivp(self):
        pass
        #s_block_rhs=solve_ivp(
        #    fun=block_rhs(
        #         time_str='t'
        #         ,X_blocks  = [('sol',nr_pools),('Phi',nr_pools*nr_pools),('Int_Phi_u',nr_pools)]
        #         ,functions = [
        #             (sol_rhs,['t','sol'])
        #             ,(Phi_rhs,['t','sol','Phi'])
        #             ,(Int_phi_u_rhs,['t','Phi'])
        #          ]
        #    )
        #    ,t_span=t_span
        #    ,y0=np.concatenate([ start_x,start_Phi_1d,start_Int_Phi_u])
        #)
        #t_block_rhs         =s_block_rhs.t
        #sol_block_rhs       =s_block_rhs.y[x_i_start:x_i_end,:]
        #Phi_block_rhs       =s_block_rhs.y[Phi_1d_i_start:Phi_1d_i_end,:]
        #Phi_block_rhs_mat   =Phi_block_rhs.reshape(nr_pools,nr_pools,len(t_block_rhs))
        #int_block_rhs       =s_block_rhs.y[int_i_start:int_i_end,:]
        #
        ## even more compactly the same system
        #block_ivp=BlockIvp(
        #    time_str='t'
        #    ,start_blocks  = [('sol',start_x),('Phi',start_Phi_1d),('Int_Phi_u',start_Int_Phi_u)]
        #    ,functions = [
        #         (sol_rhs,['t','sol'])
        #        ,(Phi_rhs,['t','sol','Phi'])
        #        ,(Int_phi_u_rhs,['t','Phi'])
        #     ]
        #)
        ## but we can also acces single blocks of the result
        #self.assertTrue(np.array_equal( t_block_rhs     ,x_phi_ivp.get("t"         ,t_span=t_span)))
        #self.assertTrue(np.array_equal( sol_block_rhs   ,x_phi_ivp.get("sol"       ,t_span=t_span)))
        #self.assertTrue(np.array_equal( Phi_block_rhs   ,x_phi_ivp.get("Phi"       ,t_span=t_span)))
        #self.assertTrue(np.array_equal( int_block_rhs   ,x_phi_ivp.get("Int_Phi_u" ,t_span=t_span)))
        ## we can get the same solution object we get from solve_ivp
        ##print(x_phi_ivp.get("sol",t_span=t_span))
        ##
        ##
        


