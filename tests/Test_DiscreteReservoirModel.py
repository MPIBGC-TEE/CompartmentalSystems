from testinfrastructure.InDirTest import InDirTest
from sympy import Symbol, symbols, Function
from CompartmentalSystems import helpers_reservoir as hr
from CompartmentalSystems.discrete_model_run import DiscreteModelRun, TimeStep, TimeStepIterator
import numpy as np
"""
Disclaimer:
    not a real test yet, no real assertions just an end to end
    example probably full of bugs
    the parts have to be tested to trust the result
"""


class TestDiscreteReservoirModel(InDirTest):
    def test_iterator(self):
        B_0=np.array([
            [ -1, 0.5],
            [0.5,  -1]
        ])
        # fake input
        u_0=np.array([1, 1])
        x_0=np.array([0, 0])
        B_func = lambda it, x: B_0 #fake
        u_func = lambda it, x: u_0 #fake
        tsit=TimeStepIterator(
            # fake matrix
            initial_ts= TimeStep(B=B_0,u=u_0,x=x_0,t=0),
            B_func=B_func,
            u_func=u_func,
            number_of_steps=10,
            delta_t=2
        )
        steps=[ts for ts in tsit]
        #print(steps)
        # we could also choose to remember only the xs
        xs = [ts.x for ts in tsit]
        print(xs)




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

