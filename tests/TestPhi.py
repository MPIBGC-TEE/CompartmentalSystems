import numpy as np
import time
from collections import OrderedDict
from sympy import var,symbols,Symbol,Function
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
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
from CompartmentalSystems.helpers_reservoir import numerical_function_from_expression,numerical_rhs,x_phi_ivp,integrate_array_func_for_nested_boundaries,array_quad_result,array_integration_by_ode,array_integration_by_values

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
        smr.build_state_transition_operator_cache(size=2)
        

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
            ,phi_block_name='Phi_2d'
        )
        for s in  np.linspace(t_0,t_max,5):
            for t in  np.linspace(s,t_max,5):
                sol_dict=blivp.block_solve(t_span=(s,t))
                
                sol_skew=sol_dict['sol'][-1,...]
                phi_mat=sol_dict['Phi_2d'][-1,...]
                
                phi_ref=np.eye(2)*np.exp(-(t-s))
                # test the matrix valued results
                with self.subTest():
                    self.assertTrue( np.allclose( phi_mat,phi_ref,rtol=1e-2))
                
                # test the vectored valued results
                for x in bvs:
                    for phi_x in [
                            smr._state_transition_operator(t,s,x)
                            ,
                            smr._state_transition_operator_for_linear_systems(t,s,x)
                        ]:
                        with self.subTest():
                            self.assertTrue( 
                                np.allclose(
                                    phi_x,
                                    np.matmul(phi_ref,x).reshape(nr_pools,),
                                    rtol=1e-2
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
        nt=20000
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
        smr.build_state_transition_operator_cache(size=2)

        nr_pools=srm.nr_pools

        def baseVector(i):
            e_i = np.zeros((nr_pools,1))
            e_i[i] = 1
            return e_i
        bvs = [ baseVector(i) for i in range(nr_pools)]
        
        smrl=smr.linearize_old()
        
        for s in  np.linspace(t_0,t_max,5):
            for t in  np.linspace(s,t_max,5):
                #sol_dict=blivp.block_solve(t_span=(s,t))
                #print(sol_dict)
                #phi_mat=sol_dict['Phi_2d'][-1,...]
                
                #sol_dict_l=blivp.block_solve(t_span=(s,t))
                
                #phi_mat_l=sol_dict_l['Phi_2d'][-1,...]
                
                #for mat in [ phi_mat ]:#, phi_mat_l ]:
                for x in bvs:
                    with self.subTest():
                        self.assertTrue( 
                            np.allclose(
                                smr._state_transition_operator(t,s,x),
                                smrl._state_transition_operator_for_linear_systems(t,s,x),
                                rtol=1.5*1e-2
                            )
                        )

        blivp= x_phi_ivp(
            smr.model
            ,smr.parameter_dict
            ,smr.func_set
            ,smr.start_values
            ,x_block_name='sol'
            ,phi_block_name='Phi_2d'
        )
        blivp_l= x_phi_ivp(
            smrl.model
            ,smrl.parameter_dict
            ,smrl.func_set
            ,smrl.start_values
            ,x_block_name='sol'
            ,phi_block_name='Phi_2d'
        )
        for t in  np.linspace(t_0,t_max,5):
            for phi_mat in [
                                blivp.block_solve(t_span=(t_0,t))['Phi_2d'][-1,...] ,
                                blivp_l.block_solve(t_span=(t_0,t))['Phi_2d'][-1,...]
                            ]:
                for x in bvs:
                    with self.subTest():
                        self.assertTrue( 
                            np.allclose(
                                np.matmul(phi_mat,x).reshape(nr_pools,),
                                smr._state_transition_operator(t,t_0,x),
                                rtol=1.5*1e-2
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
        cache= smr._compute_state_transition_operator_cache(size=2)

        phi_2_0=cache.values[0]
        #print(phi_2_0)
        phi_4_2=cache.values[1]
        #print(phi_4_2)


        def baseVector(i):
            e_i = np.zeros((nr_pools,1))
            e_i[i] = 1
            return e_i
        bvs = [ baseVector(i) for i in range(nr_pools)]
        smrl=smr.linearize_old()
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
            #if isinstance(times,np.ndarray): 
            #    #3d array
            res=np.zeros((len(times),nr_pools,nr_pools))
            res[:,0,0]=np.exp(-k_0_val*times)
            res[:,1,1]=np.exp(-k_1_val*times)
            #else: 
            #    #2d array for scalar time
            #    res=np.zeros((nr_pools,nr_pools))
            #    res[0,0]=np.exp(-k_0_val*times)
            #    res[1,1]=np.exp(-k_1_val*times)
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
        sol_rhs=numerical_rhs(
             srm.state_vector
            ,srm.time_symbol
            ,srm.F
            ,parameter_dict
            ,func_dict
        )
        #
        start_x= np.array([x0_0,x0_1])
        start_Int_Phi_u=np.zeros(nr_pools)
        t_span=(0,t_max)
        #        
        #)
        block_ivp=x_phi_ivp(srm,parameter_dict,func_dict,start_x,x_block_name='sol',phi_block_name='Phi')

        sol_dict =block_ivp.block_solve(t_span=t_span,t_eval=times) 
        sol_values  = sol_dict["sol"]
        Phi_values  = sol_dict["Phi"]
        
        # for comparison solve the original system without the phi rows
        sol=solve_ivp(fun=sol_rhs,t_span=(0,t_max),y0=start_x,t_eval=times,method='LSODA')
        # and make sure that is identical with the block rhs by using the interpolation function
        # for the block system and apply it to the grid that the solver chose for sol
        sol_func_dict  = block_ivp.block_solve_functions(t_span=t_span)
        sol_func  = sol_func_dict["sol"]
        Phi_func  = sol_func_dict["Phi"]

        self.assertTrue(np.allclose(np.moveaxis(sol.y,-1,0),sol_func(sol.t),rtol=1e-2))
        # check Phi
        self.assertTrue(np.allclose(Phi_values,Phi_ref(times),atol=1e-2))
        self.assertTrue(
            np.allclose(
                np.stack([Phi_func(t) for t in times],axis=0),
                Phi_ref(times),
                atol=1e-2
            )
        )

        fig=plt.figure(figsize=(7,7))
        ax1=fig.add_subplot(2,1,1)
        ax1.plot(times,               sol.y[0,:],'o',color='red' ,label="sol[0]")
        ax1.plot(times,               sol.y[1,:],'x',color='red' ,label="sol[1]")
        #
        ax1.plot(times,               sol_func(times)[:,0],'*',color='blue' ,label="sol[0]")
        ax1.plot(times,               sol_func(times)[:,1],'+',color='blue' ,label="sol[1]")
        #
        ax1.plot(times, sol_values[:,0],'*',color='green',label="sol_values[0]")
        ax1.plot(times, sol_values[:,1],'+',color='green',label="sol_values[1]")
        #
        ax1.legend()

        ax2=fig.add_subplot(2,1,2)
        ax2.plot(times  ,Phi_ref(times)[:,0,0],'o',color='red'  ,label="Phi_ref[0,0]")
        ax2.plot(times  ,Phi_ref(times)[:,1,1],'x',color='red'  ,label="Phi_ref[1,1]")

        ax2.plot(times, Phi_values[:,0,0],'*',color='green',label="Phi_values[0,0]")
        ax2.plot(times, Phi_values[:,1,1],'+',color='green',label="Phi_values[1,1]")
        ax2.legend()
        fig.savefig("solutions.pdf")

        
    def test_cache_hash(self): 
        # test semi-symbolic semi-numerical SmoothReservoirModel
        C_0, C_1, C_2 = symbols('C_0 C_1 C_2')
        t = Symbol('t')

        u_0_expr = Function('u_0')(C_0, C_1, t)
        u_2_expr = Function('u_2')(t)

        X = Matrix([C_0, C_1, C_2])
        t_min, t_max = 0, 10
        u_data_0 = np.array([[ t_min ,  0.1], [ t_max ,  0.2]])
        u_data_2 = np.array([[ t_min ,  0.4], [ t_max ,  0.5]])
        input_fluxes = {0: u_data_0, 2: u_data_2}
        symbolic_input_fluxes = {0: u_0_expr, 2: u_2_expr}
        
        u_0_interp = interp1d(u_data_0[:,0], u_data_0[:,1])
        def u0_func(C_0_val, C_1_val, t_val):
            return C_0_val*0 + C_1_val*0 + u_0_interp(t_val)
        
        u_1_interp = interp1d(u_data_2[:,0], u_data_2[:,1])
        def u2_func(t_val):
            return u_1_interp(t_val)
        
        func_set = {u_0_expr: u0_func, u_2_expr: u2_func}
        
        output_fluxes = {}
        internal_fluxes = {(0,1): 5*C_0, (1,0): 4*C_1**2}
        srm = SmoothReservoirModel(
            X, 
            t, 
            symbolic_input_fluxes, 
            output_fluxes, 
            internal_fluxes
        )

        start_values = np.array([1, 2, 3])
        times = np.linspace(t_min,t_max, 11)
        smr = SmoothModelRun(srm, parameter_dict={}, start_values=start_values, times=times,func_set=func_set)
        
        soln,_ = smr.solve()
        # To be able to check if a stored state_transition_operator cache 
        # is applicable to the SmoothModelRun object it is supposed to speed up
        #print(smr.myhash())  
        smr.myhash()  

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
        
        #ages = np.linspace(-1,1,3)
        # negative ages will be cut off automatically
        #start_age_densities = lambda a: np.exp(-a)*start_values
        smr.build_state_transition_operator_cache()
        ca = smr._state_transition_operator_cache

        filename = 'sto.cache'
        smr.save_state_transition_operator_cache(filename)
        smr.load_state_transition_operator_cache(filename)
    
        self.assertTrue(np.all(ca==smr._state_transition_operator_cache))
        
        # now we change the model run and make sure that the 
        # saved cache is recognized as invalid.
        start_values_2 = np.array([6, 3])
        smr = SmoothModelRun(srm, {}, start_values_2, times)
        with self.assertRaises(Exception):
            smr.load_state_transition_operator_cache(filename)



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
