from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from testinfrastructure.InDirTest import InDirTest
from testinfrastructure.helpers import pp, pe
from sympy import Symbol, symbols, Function, ImmutableMatrix
from CompartmentalSystems import helpers_reservoir as hr
from CompartmentalSystems.model_run import plot_stocks_and_fluxes
from CompartmentalSystems.smooth_model_run import SmoothModelRun 
from CompartmentalSystems.discrete_model_run import DiscreteModelRun, TimeStep, TimeStepIterator
import numpy as np
import matplotlib.pyplot as plt
"""
Disclaimer:
    not a real test yet, no real assertions just an end to end
    example probably full of bugs
    the parts have to be tested to trust the result
"""


class TestDiscreteReservoirModel(InDirTest):
    def test_symbolic(self):
        # We assume that the net fluxes F have a functional form
        # F  = f(i,x)
        # Where:
        # F is some in-,internal or out-flux. F = F_(source,target)
        # x is the pool content (tupel)
        # i is the time bin index.
        # which results in a difference equation for the (tupel) x
        # x_i+1 = g(i,x)
        #       = u(i,x) + B(i,x)*x            (1)
        # In the following example the net fluxes F
        # are the result of:
        # 1.) an euler forward discretization of
        # 2.) a linear model
        # But actually both properties are non essential
        # We could, for instance, also create
        # a Runge-Kutta discretization of a nonlinear ODE
        # or in the most general situation drop the assumption
        # to deal with a discretization of an ODE altogether,
        # since this is only ONE way to create difference equations
        # or discrete dynamical systems.

        it = Symbol('it')
        I_leaf = Function("I_leaf")
        leaf, wood = symbols("leaf wood")
        k_leaf_out, k_wood_out = symbols("k_leaf_out k_wood_out")
        k_leaf_wood, k_wood_leaf = symbols("k_leaf_wood k_wood_leaf")
        delta_t = symbols('delta_t')

        net_in_fluxes_by_symbol = {leaf: I_leaf(it) }
        net_out_fluxes_by_symbol = {
                leaf: k_leaf_out * leaf * delta_t,
                wood: k_wood_out * wood * delta_t
        }
        net_internal_fluxes_by_symbol = {
                (leaf, wood): k_leaf_wood * leaf * delta_t,
                (wood, leaf): k_wood_leaf * wood * delta_t
        }
        timeIndex = it
        state_variable_tuple = (leaf, wood)

        # Our task is to create the right hand side of the difference equation
        x = state_variable_tuple
        R = hr.release_operator_2(
            net_out_fluxes_by_symbol,
            net_internal_fluxes_by_symbol,
            x
        )
        T = hr.transfer_operator_2(
            net_out_fluxes_by_symbol,
            net_internal_fluxes_by_symbol,
            x
        )
        B = T * R
        print(B)
        u = hr.in_or_out_flux_tuple(
            state_variable_tuple,
            net_in_fluxes_by_symbol
        )
        print("##################")
        print('u')
        print(u)
        #rhs = u(it,x)+B(it,x)*x

        # num_rhs
        par_dict = {
                k_leaf_out: 1,
                k_wood_out: 1,
                k_wood_leaf: 1,
                k_leaf_wood: 1,
                delta_t: 1
        }
        i_min = 0
        i_max = 5
        def my_func(ind):
            data_leaf = [
                5*(np.sin(2 * np.pi/12 * i) + 1.1)
                for i in range(i_min, i_max)
            ]
            return data_leaf[ind]

        func_dict = {
                I_leaf(it): my_func
        }
        num_B, num_u = map(
            lambda expr: hr.numerical_array_func(
                state_variable_tuple,
                it,
                expr,
                par_dict,
                func_dict
            ),
            (B, u)
        )
        print(num_u(0,np.array([0,0])))
        print(num_u(0,np.array([1,1])))
        x_0 = np.array([0,0])
        u_0 = num_u(i_min,x_0)
        B_0 = num_B(i_min,x_0)

        tsit=TimeStepIterator(
            # fake matrix
            initial_ts=TimeStep(B=B_0,u=u_0,x=x_0,t=0),
            B_func=num_B,
            u_func=num_u,
            number_of_steps=i_max - i_min,
            delta_t=2
        )
        #steps=[ts for ts in tsit]
        #print(steps)
        # we could also choose to remember only the xs
        xs = [ts.x for ts in tsit]
        print(xs)

        dmr = DiscreteModelRun.from_B_and_u_funcs(
            x_0=x_0,
            B_func=num_B,
            u_func=num_u,
            number_of_steps=i_max - i_min,
            delta_t=2
        )
        print(dmr.solve())


    def test_discrete_symbolic_from_continious_symbolic(self):
        k_leaf, k_root = symbols('k_leaf, k_root')
        beta_leaf, beta_root = symbols('beta_leaf, beta_root')
        delta_t = Symbol('delta_t')
        Npp = Symbol('Npp')
        
        B_cont_sym = ImmutableMatrix([
            [-k_leaf,  0 ],
            [ 0, -k_root ]
        ])
    
        beta = ImmutableMatrix([
            beta_leaf,
            beta_root
        ])
         
        u_cont_sym = Npp*beta
    
        netB_disc_sym = hr.euler_forward_net_B_sym(B_cont_sym,delta_t)
        netu_disc_sym = hr.euler_forward_net_u_sym(u_cont_sym,delta_t)
    

    def test_discrete_model_run_from_smooth_reservoir_model(self):
        x_leaf, x_root = symbols('x_leaf, x_root')
        k_leaf, k_root = symbols('k_leaf, k_root')
        beta_leaf, beta_root = symbols('beta_leaf, beta_root')
        
        Npp = Function('Npp')
        t = Symbol('t')
        it = Symbol('it')
        delta_t = Symbol('delta_t')
        
        B_cont_sym = ImmutableMatrix([
            [-k_leaf,  0 ],
            [ 0, -k_root ]
        ])
    
        beta = ImmutableMatrix([
            beta_leaf,
            beta_root
        ])
        u_cont_sym = Npp(t)*beta
    
        srm = SmoothReservoirModel.from_B_u(
            state_vector=ImmutableMatrix([x_leaf,x_root]),
            time_symbol=Symbol('t'),
            B=B_cont_sym,
            u=u_cont_sym
        )
        par_dict_cont = {
            k_leaf: 1,
            k_root: 1,
            beta_leaf: 0.3,
            beta_root: 0.7
        }
        par_dict_disc = {** par_dict_cont, **{delta_t: 0.25}} 
        i_min = 0
        i_max = 50
        n = i_max - i_min - 1 
        
        def my_cont_func(t):
            return    5*(1+np.sin(2 * np.pi/12 * t))
        
    
        start_values = np.array((1,1))
        
        smr = SmoothModelRun(
            srm,
            parameter_dict = par_dict_cont,
            start_values = start_values,
            times = np.array([par_dict_disc[delta_t]*i for i in range(n)]),
            func_set={Npp(t): my_cont_func}
        )     
        dmr = DiscreteModelRun.from_euler_forward_smooth_reservoir_model(
            srm,
            par_dict=par_dict_disc,
            func_dict={Npp(delta_t * it): my_cont_func},
            delta_t=delta_t,
            number_of_steps=n,
            start_values=start_values
        )     
        fig=plt.figure()
        ax = fig.subplots()
        ax.plot(smr.solve()[:,0],color='b')
        ax.plot(dmr.solve()[:,0],'+',color='r')
        fig.savefig('test.pdf')
    
