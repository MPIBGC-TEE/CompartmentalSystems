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
    #B=srm.compartmental_matrix
    nr_pools=srm.nr_pools
    nq=nr_pools*nr_pools
    #tup=(t,)+tuple(srm.state_vector)
    sol_rhs=numerical_rhs2(
         srm.state_vector
        ,srm.time_symbol
        ,srm.F
        ,parameter_dict
        ,func_dict
    )
    # for comparison solve the original system 
    
    B_sym=srm.compartmental_matrix
    ## now use the interpolation function to compute B in an alternative way.
    #symbolic_sol_funcs = {sv: Function(sv.name + '_sol')                        for sv in svec}
    #sol_func_exprs =     {sv: symbolic_sol_funcs[sv](srm.time_symbol)           for sv in svec}# To F add the (t) 
    #def func_maker(pool):
    #    def func(t):
    #        return sol.sol(t)[pool]
    #
    #    return(func)
    #
    #sol_dict = {symbolic_sol_funcs[svec[pool]]:func_maker(pool) for pool in range(srm.nr_pools)}
    #lin_func_dict=copy(func_dict)
    #lin_func_dict.update(sol_dict)
    #
    #linearized_B = B_sym.subs(sol_func_exprs)
    #
    tup=(t,)+tuple(srm.state_vector)
    B_func=numerical_function_from_expression(B_sym,tup,parameter_dict,func_dict)
    x_i_start=0
    x_i_end=nr_pools
    Phi_1d_i_start=x_i_end
    Phi_1d_i_end=(nr_pools+1)*nr_pools
    #
    def Phi_rhs(t,x,Phi_1d):
        B=B_func(t,*x)
        Phi_cols=[Phi_1d[i*nr_pools:(i+1)*nr_pools] for i in range(nr_pools)]
        Phi_ress=[np.matmul(B,pc) for pc in Phi_cols]
        return np.stack([np.matmul(B,pc) for pc in Phi_cols]).flatten()

    #create the additional startvecot for the components of Phi
    start_Phi_1d=np.identity(nr_pools).reshape(nr_pools**2)

    block_ivp=BlockIvp(
        time_str='t'
        ,start_blocks  = [('sol',start_x),('Phi_1d',start_Phi_1d)]
        ,functions = [
             (sol_rhs,['t','sol'])
            ,(Phi_rhs,['t','sol','Phi_1d'])
         ]
    )
    return block_ivp

def block_rhs(
         time_str  : str
        ,X_blocks  : List[ Tuple[str,int] ]
        ,functions : List[ Tuple[Callable,List[str]]]
    )->Callable[[np.double,np.ndarray],np.ndarray]:
    """
    The function returns a function dot_X=f(t,X) suitable as the righthandside 
    for the ode solver scipy.solve_ivp from a collection of vector valued
    functions that compute blocks of dot_X from time and blocks of X 
    rather than from single equations.

    A special application is the creation of block triangular systems, to 
    integrate variables whose time derivative depend on the solution
    of an original system instantaniously along with it.
    
    Assume that
    X_1(t) is the solution of the initial value problem (ivp)
    
    ivp_1:
    dot_X_1=f_1(t,X) ,X_1(t_0) 

    and X_2(t) the solution of another ivp 

    ivp_2:
    dot_X_2=f_2(t,X_1,X_2) ,X_2(t_0) whose righthand side depends on x_1
    
    Then we can obtain the solution of both ivps simultaniously by
    combining the them into one.
    
    (dot_X_1, dox_X_2)^t = (f_1(t,X_1),f_2(t,X_1,X_2))^t

    For n instead of 2 variables one has:  
    (dot_X_1, dox_X_2,...,dot_X_n)^t = (f_1(t,X_1),f_2(t,X_1,X_2),...,f_n(t,X_1,...X_n))^t
    
    For a full lower triangular system  the block derivative dot_X_i depend on t,
    and ALL the blocks X_1,...,X_i but often they will only depend on 
    SOME of the previous blocks so that f_m has a considerably smaller argument list.

    This function therefore allows to specify WHICH blocks the f_i depend on.
    Consider the following 7 dimensional block diagonal example:

    b_s=block_rhs(
         time_str='t'
        ,X_blocks=[('X1',5),('X2',2)]
        ,functions=[
             ((lambda x   : x*2 ),  ['X1']    )
            ,((lambda t,x : t*x ),  ['t' ,'X2'])
         ])   
    

    The first argument 'time_str' denotes the alias for the t argument to be used
    later in the signature of the blockfunctions.
    The second argument 'X_blocks' describes the decomposition of X into blocks
    by a list of tuples of the form ('Name',size)
    The third argument 'functions' is a list of tuples of the function itself
    and the list of the names of its block arguments as specified in the
    'X_blocks' argument. 
    Order is important for the 'X_blocks' and the 'functions'
    It is assumed that the i-th function computes the derivative of the i-th block.
    The names of the blocks itself are arbitrary and have no meaning apart from
    their correspondence in the X_blocks and functions argument.
    """
    block_names=[t[0] for t in X_blocks]
    dims=[t[1] for t in X_blocks]
    nb=len(dims)
    strArgLists=[f[1] for f in functions]
    # make sure that all argument lists are really lists
    assert(all([isinstance(l,list) for l in strArgLists])) 
    # make sure that the function argument lists do not contain block names
    # that are not mentioned in the Xblocks argument
    flatArgList=[arg for argList in strArgLists for arg in argList]
    assert(set(flatArgList).issubset(block_names+[time_str]))
    

    # first compute the indices of block boundaries in X by summing the dimensions 
    # of the blocks
    indices=[0]+[ sum(dims[:(i+1)]) for i in range(nb)]
    def rhs(t,X):
        blockDict={block_names[i]: X[indices[i]:indices[i+1]] for i in range(nb)}
        #pe('blockDict',locals())
        blockDict[time_str]=t
        arg_lists=[ [blockDict[k] for k in f[1]] for f in functions]
        blockResults=[ functions[i][0]( *arg_lists[i] )for i in range(nb)]
        #pe('blockResults',locals())
        return np.concatenate(blockResults).reshape(X.shape)
    
    return rhs




class BlockIvp:
    def __init__(
         self
        ,time_str       : str
        ,start_blocks   : List[ Tuple[str,np.ndarray] ]
        ,functions      : List[ Tuple[Callable,List[str]]]):
        
        self.time_str=time_str
        names           =[sb[0]   for sb in start_blocks]
        start_arrays    =[sb[1]   for sb in start_blocks]
        shapes          =[a.shape  for a in start_arrays]
        #assert that we have only vectors as startvalues
        assert(all([len(s)==1 for s in shapes]))
        #
        dims=[s[0] for s in shapes]
        nb=len(dims)
        r=range(nb)
        X_blocks=[(names[i],dims[i]) for i in r]
        indices=[0]+[ sum(dims[:(i+1)]) for i in r]
        self.index_dict={names[i]:(indices[i],indices[i+1]) for i in r}
        self.rhs=block_rhs(
             time_str  = time_str
            ,X_blocks  = X_blocks
            ,functions = functions
        )
        self.start_vec=np.concatenate(start_arrays)
        self._cache=dict()
        print(self.rhs(0,self.start_vec))
        
    def solve(self,t_span,dense_output=False,**kwargs):
        # this is just a caching proxy for scypy.solve_ivp
        # remember the times for the solution
        if not(isinstance(t_span,tuple)):
            raise Exception('''
            scipy.solve_ivp actually allows a list for t_span, but we insist 
            that it should be a tuple, since we want to cache the solution 
            and want to use t_span as a hash.''')
        cache_key=(t_span,dense_output)
        if cache_key in self._cache.keys():
            # caching can be made much more sophisticated by
            # starting at the end of previos solution with
            # the same start times and a smaller t_end
            return self._cache[cache_key]
        #
        if 'vectorized' in kwargs.keys():
            del(kwargs['vectorized'])
            print('''The vectorized flag is forbidden for $c 
            since we rely on decomposing the argument vector'''.format(s=self.__class__))
        sol=solve_ivp(
             fun=self.rhs
            ,y0=self.start_vec
            ,t_span=t_span
            ,dense_output=dense_output
            ,**kwargs
        )
        self._cache[cache_key]=sol
        return sol
    #
    def get_values(self,block_name,t_span,**kwargs):
        sol=self.solve(t_span=t_span,**kwargs)
        if block_name==self.time_str:
            return sol.t
        if block_name in self.index_dict.keys():
            lower,upper=self.index_dict[block_name] 
            return sol.y[lower:upper,:]
        else:
            raise Exception("There is no block with this name")
        
    def get_function(self,block_name,t_span,**kwargs):
        if block_name==self.time_str:
            print("""
            warning:
            $s interpolated with respect to s$ is probably an accident...
            """.format(s=self.time_str))
            # this is silly since it means somebody asked for the interpolation of t with respect to t
            # so we give back the identiy
            return lambda t:t

        elif block_name in self.index_dict.keys():
            lower,upper=self.index_dict[block_name] 
            complete_sol_func=self.solve(t_span=t_span,dense_output=True).sol
            def block(t):
                return complete_sol_func(t)[lower:upper]
            
            return block

class TestBlockRhs(InDirTest):
    def test_block_rhs(self):
        b_s=block_rhs(
             time_str='t'
            ,X_blocks=[('X1',5),('X2',2)]
            ,functions=[
                 ((lambda x   : x*2 ),  ['X1']    )
                ,((lambda t,x : t*x ),  ['t' ,'X2'])
             ])   
        # it should take time and 
        example_X=np.append(np.ones(5),np.ones(2))
        res=b_s(0,example_X)
        ref=np.array([2, 2, 2, 2, 2, 0, 0])
        self.assertTrue(np.array_equal(res,ref))
        # solve the resulting system

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
#
    def test_stateTransitionOperator_from_linear_System_by_skew_product_system(self):
        # We first check the result for a linear system, where the whole procedure is not neccesary
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
        # for this system we know the state transition operator to be a simple matrix exponential
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
        #B=srm.compartmental_matrix
        nr_pools=srm.nr_pools
        nq=nr_pools*nr_pools
        #tup=(t,)+tuple(srm.state_vector)
        #B_num=numerical_function_from_expression(B,tup,parameter_dict,{})
        sol_rhs=numerical_rhs2(
             srm.state_vector
            ,srm.time_symbol
            ,srm.F
            ,parameter_dict
            ,func_dict
        )
        B_sym=srm.compartmental_matrix
        #
        tup=(t,)+tuple(srm.state_vector)
        B_func=numerical_function_from_expression(B_sym,tup,parameter_dict,func_dict)
        x_i_start=0
        x_i_end=nr_pools
        Phi_1d_i_start=x_i_end
        Phi_1d_i_end=(nr_pools+1)*nr_pools
        int_i_start=Phi_1d_i_end
        int_i_end=int_i_start+nr_pools
        #
        def Phi_rhs(t,x,Phi_1d):
            B=B_func(t,*x)
            Phi_cols=[Phi_1d[i*nr_pools:(i+1)*nr_pools] for i in range(nr_pools)]
            Phi_ress=[np.matmul(B,pc) for pc in Phi_cols]
            return np.stack([np.matmul(B,pc) for pc in Phi_cols]).flatten()
        #
        #
        start_x= np.array([x0_0,x0_1])
        start_Phi_1d=np.identity(nr_pools).reshape(nr_pools**2)
        start_Int_Phi_u=np.zeros(nr_pools)
        t_span=(0,t_max)
        #        
        # even more compactly the same system
        block_ivp=BlockIvp(
            time_str='t'
            ,start_blocks  = [('sol',start_x),('Phi',start_Phi_1d)]
            ,functions = [
                 (sol_rhs,['t','sol'])
                ,(Phi_rhs,['t','sol','Phi'])
             ]
        )
        s_block_ivp=block_ivp.solve(t_span=t_span)
        t_block_rhs    = block_ivp.get("t"         ,t_span=t_span)
        sol_block_rhs  = block_ivp.get("sol"       ,t_span=t_span)
        Phi_block_rhs  = block_ivp.get("Phi"       ,t_span=t_span)
        Phi_block_rhs_mat   =Phi_block_rhs.reshape(nr_pools,nr_pools,len(t_block_rhs))
        #print(block_ivp.get("sol",t_span=t_span))
        # for comparison solve the original system 
        sol=solve_ivp(fun=sol_rhs,t_span=(0,t_max),y0=start_x,max_step=delta_t,method='LSODA')
        # and make sure that is identical with the block rhs by using the interpolation function
        # for the block system and apply it to the grid that the solver chose for sol
        sol_func  = block_ivp.get("sol"       ,t_span=t_span,dense_output=True)
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
        

    def test_linearization_by_skew_product(self):
        # The state transition operator Phi is defined for linear systems only
        # To compute it we have to create a linear system first by substituting
        # the solution into the righthandside
        # This could be done in different ways:
        # 1.)   By solving the ODE with the actual start vector first and then
        #       substituting the interpolation into the righthandside used to compute the state transition operator
        # 2.)   Creation of a skewproductsystem whose solution yields
        #       the solution for the initial value problem and the state transition operator for Phi(t,t_0) simultaniously.
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
             0:x_0*k_0
            ,1:x_1*k_1
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
        t_max=4
        times = np.linspace(0, t_max, 11)
        parameter_dict = {
             k_0: k_0_val
            ,k_1: k_1_val
            ,u:1
        
        }
        func_dict={}
        start_x= np.array([x0_0,x0_1])
        x_phi_ivp=X_Phi_IVP(srm,parameter_dict,func_dict,start_x)
        
        # we now express the solution as integral expression of the state transition operator
        # x_t=Phi(t,t0)*x_0+int_t0^t Phi(t,tau)*u(tau) dtau
        # and check that we get the original solution back
        t_0=0
        t_span=(t_0,t_max)
        Phi_t0 =x_phi_ivp.get_function("Phi_1d",t_span=t_span)
        times = np.linspace(t_0, t_max, 11)
        ts   =x_phi_ivp.get_values("t",t_span=t_span,max_step=1)
        xs   =x_phi_ivp.get_values("sol",t_span=t_span)
        phis =x_phi_ivp.get_values("Phi_1d",t_span=t_span)
        
        def Phi_t0_mat(t):
            return np.matrix(Phi_t0(t).reshape(srm.nr_pools,srm.nr_pools))
        
        def Phi(t,s):
            # this whole function would make sense if 
            # one could guarantee that the state transition operator will
            # be required only for a fixed grid 
            # (ideally to be determined by the functions unsing it before this function is called)
            # so that no interpolation takes place (as it is now)
            # but only cached values would be required
            # The the cost of computing the inverses for every time step would be
            # justified
            """
            For t_0 <s <t we have
            
            Phi(t,t_0)=Phi(t,s)*Phi(s,t_0)
            
            If we know $Phi(s,t_0) \forall s \in [t,t_0] $
            
            We can reconstruct Phi(t,s)=Phi(t,t_0)* Phi(s,t_0)^-1
            This is what this function does.
            """
            # first check 
            if t_0 > t:
                raise(Error("Evaluation before t0 is not possible"))
            if t_0 == t:
                return np.identity(srm.nr_pools)
            # fixme: mm 3/9/2019
            # the following inversion is very expensive and should be cached 
            # to save space in a linear succession state Transition operators from step to step
            # then everything can be reached by a 
            return np.matmul(Phi_t0_mat(t),np.linalg.inv(Phi_t0_mat(s)))
        
        # We compute the integral 
        # NOT as the solution of an ivp but with an integration rule that 
        # works with arrays instead of functions
        nr_pools=srm.nr_pools
        rs=(nr_pools,len(ts))
        
        u_sym=srm.external_inputs
        u_num=numerical_function_from_expression(u_sym,(t,),parameter_dict,{})
        #
        def integral(i,t):
            if i==0:
                return np.zeros(nr_pools)
            # the integrals boundaries grow with time
            # so the vector for the trapezrule becomes longer and longer
            taus=ts[0:i]
            sh=(nr_pools,len(taus)) 
            #t=taus[-1]
            phi_vals=np.array([Phi(t,tau) for tau in taus])
            print(phi_vals)
            integrand_vals=np.array([np.matmul(Phi(t,tau),u_num(tau)) for tau in taus]).reshape(sh)
            pe('integrand_vals',locals())
            return np.trapz(y=integrand_vals,x=taus).reshape(nr_pools)

        
        ## reconstruct the solution with Phi and the integrand
        # x_t=Phi(t,t0)*x_0+int_t0^t Phi(tau,t0)*u(tau) dtau
        a=np.array([np.matmul(Phi_t0_mat(t),start_x) for t in ts]).reshape(rs)
        b=np.array([ integral(i,t) for i,t in enumerate(ts)]).reshape(rs)
        xs2=a+b.reshape(rs) 
        fig=plt.figure(figsize=(10,17))
        ax1=fig.add_subplot(2,1,1)
        ax1.plot(ts,xs[0,:],'*',color='blue' ,label="sol[0]")
        ax1.plot(ts,xs[1,:],'+',color='blue' ,label="sol[1]")

        ax1.plot(ts,xs2[0,:],'o',color='red' ,label="sol2[0]")
        #ax1.plot(ts,xs2[1,:],'x',color='red' ,label="sol2[1]")

        #ax1.plot(ts,a[0,:],'o',color='orange' ,label="a[0]")
        #ax1.plot(ts,a[1,:],'x',color='orange' ,label="a[1]")

        #ax1.plot(ts,b[0,:],'o',color='green' ,label="b[0]")
        #ax1.plot(ts,b[1,:],'x',color='green' ,label="b[1]")
        ax1.legend()

        #ax2=fig.add_subplot(2,1,2)
        #ax2.plot(ts,integrand_vals[0,:],'x',color='green' ,label="integrand")
        #ax2.plot(ts,phi_int_vals1[0,:],'x',color='red' ,label="phi 1")
        #ax2.plot(ts,phi_int_vals2[0,:],'x',color='red' ,label="phi 2")
        #ax2.legend()
        
        fig.savefig("solutions.pdf")

