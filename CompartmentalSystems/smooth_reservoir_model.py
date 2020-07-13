"""Module for symbolical treatment of smooth reservoir models.

This module handles the symbolic treatment of compartmental/reservoir/pool 
models.
It does not deal with numerical computations and model simulations,
but rather defines the underlying structure of the respective model.

All fluxes or matrix entries are supposed to be SymPy expressions.
*Smooth* means that no ``Piecewise`` or ``DiracDelta`` functions should be 
involved in the model description.

Counting of compartment/pool/reservoir numbers starts at zero and the 
total number of pools is :math:`d`.
"""


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from copy import copy, deepcopy
from string import Template
from functools import reduce
from sympy import (zeros, Matrix, simplify, diag, eye, gcd, latex, Symbol, 
                   flatten, Function, solve, limit, oo , ask , Q, assuming
                   ,sympify)
from sympy.printing import pprint
import numpy as np
import multiprocessing

from .helpers_reservoir import factor_out_from_matrix, has_pw, flux_dict_string, jacobian
from .pool import Pool
from typing import TypeVar


class Error(Exception):
    """Generic error occurring in this module."""
    pass


class SmoothReservoirModel(object):
    """General class of smooth reservoir models.

    Attributes:
        state_vector (SymPy dx1-matrix): The model's state vector 
            :math:`x`.
            Its entries are SymPy symbols.
        state_variables (list of str): 
            Names of the variables in the state vector.
            Its entries are of type ``str``.
        time_symbol (SymPy symbol): The model's time symbol.
        input_fluxes (dict): The model's external input fluxes.
            ``{key1: flux1, key2: flux2}`` with ``key`` the pool number and 
            ``flux`` a SymPy expression for the influx.
        output_fluxes (dict): The model's external output fluxes.
            ``{key1: flux1, key2: flux2}`` with ``key`` the pool number and 
            ``flux`` a SymPy expression for the outflux.
        internal_fluxes (dict): The model's internal_fluxes.
            ``{key1: flux1, key2: flux2}`` with ``key = (pool_from, pool_to)``
            and *flux* a SymPy expression for the flux.
    """
    @classmethod
    def from_state_variable_indexed_fluxes(cls,state_vector, time_symbol, 
            input_fluxes, output_fluxes, internal_fluxes)->"SmoothReservoirModel":
        """Return an instance of SmoothReservoirModel.
    
        Args:
            state_vector (SymPy dx1-matrix): The model's state vector 
                :math:`x`.
                Its entries are SymPy symbols.
            time_symbol (SymPy symbol): The model's time symbol.
            input_fluxes (dict): The model's external input fluxes.
                ``{key1: flux1, key2: flux2}`` with ``key`` the symbol of the target pool. (as used in the state vector)
                and ``flux`` a SymPy expression for the influx.
            output_fluxes (dict): The model's external output fluxes.
                ``{key1: flux1, key2: flux2}`` with ``key`` the symbols for the source pool (as used in the state vector)
                and ``flux`` a SymPy expression for the outflux.
            internal_fluxes (dict): The model's internal_fluxes.
                ``{key1: flux1, key2: flux2}`` with 
                ``key = (source pool symbol, target pool symbol)`` and ``flux`` a SymPy expression 
                for the flux.
    
        Returns:
            :class:`SmoothReservoirModel`
        """
        # transform to integer indexed dicts
        int_input={state_vector.index(k):v for k,v in input_fluxes.items()}
        int_output={state_vector.index(k):v for k,v in output_fluxes.items()}
        int_internal={(state_vector.index(k[0]),state_vector.index(k[1])):v for k,v in internal_fluxes.items()}
        # call normal init
        return cls( state_vector, time_symbol, int_input, int_output, int_internal)

    def __init__(self, state_vector, time_symbol, 
                       input_fluxes={}, output_fluxes={}, internal_fluxes={}):
        """Initialize an instance of SmoothReservoirModel.
    
        Args:
            state_vector (SymPy dx1-matrix): The model's state vector 
                :math:`x`.
                Its entries are SymPy symbols.
            time_symbol (SymPy symbol): The model's time symbol.
            input_fluxes (dict): The model's external input fluxes.
                ``{key1: flux1, key2: flux2}`` with ``key`` the pool number 
                and ``flux`` a SymPy expression for the influx.
            output_fluxes (dict): The model's external output fluxes.
                ``{key1: flux1, key2: flux2}`` with ``key`` the pool number 
                and ``flux`` a SymPy expression for the outflux.
            internal_fluxes (dict): The model's internal_fluxes.
                ``{key1: flux1, key2: flux2}`` with 
                ``key = (pool_from, pool_to)`` and ``flux`` a SymPy expression 
                for the flux.
    
        Returns:
            :class:`SmoothReservoirModel`
        """
        self.state_vector = state_vector
        self.state_variables = [sv.name for sv in state_vector]
        self.time_symbol=time_symbol
        self.input_fluxes=input_fluxes
        self.output_fluxes=output_fluxes
        self.internal_fluxes=internal_fluxes
        # fixme mm:
        # this is a kind of circular dependency 
        # or at least a clumsy, duplicating and therefore error prone approach 
        # at a one-to-many relationship
        # there is no need to store SmoothModelRun objects in ReservoirModel 
        # objects since we already have the 
        # attribute model_run_combinations in class Model
        #self.model_runs=[]

        # fixme mm:
        # see the description of the model_runs property
        # def add_model_run(self,mr):
        #     self.model_runs.append(mr)
    

    @property
    def no_input_model(self):
        return SmoothReservoirModel(
            self.state_vector,
            self.time_symbol,
            {},# no input fluxes
            self.output_fluxes,
            self.internal_fluxes
        )
        
    @property
    def function_expressions(self):
        """ Returns the superset of the free symbols of the flux expressions.
        """
        flux_list=self.all_fluxes()
        fun_sets=[ fun_set 
                # the sympify in the next line is only necessary for
                # fluxexpressions that are integers (which have no atoms method) 
                for fun_set in map(lambda
                    flux:sympify(flux).atoms(Function),flux_list)] 
        if len(fun_sets)==0:
            res=set()
        else:
            res=reduce( lambda A,B: A.union(B),fun_sets)
        return res 


    @property
    def free_symbols(self):
        """ Returns the superset of the free symbols of the flux expressions including the state variables.
        """
        flux_exprs=self.all_fluxes()
        free_sym_sets=[ sym_set 
                # the sympification in the next line is only necessary for
                # fluxexpressions that are numbers
                # It does no harm on expressions 
                for sym_set in map(lambda sym:sympify(sym).free_symbols,flux_exprs)] 

        if len(free_sym_sets)==0:
            res=set()
        else:
            res=reduce( lambda A,B: A.union(B),free_sym_sets)
        return res 

 
    def subs(self,parameter_dict):
        """ Returns a new instance of class: `SmoothReservoirModel` with all parameters in the parameter_dict replaced 
            by their values by calling subs on all the flux expressions. 
            Args:
                parameter_dict: A dictionary with the structure {parameter_symbol:parameter_value,....}
        """
        return SmoothReservoirModel(
            self.state_vector,
            self.time_symbol,
            {k:fl.subs(parameter_dict) for k,fl in    self.input_fluxes.items()},
            {k:fl.subs(parameter_dict) for k,fl in   self.output_fluxes.items()},
            {k:fl.subs(parameter_dict) for k,fl in self.internal_fluxes.items()}
        )

    def __str__(self):
        """ This method is called implicitly by print and gives an returns a string that gives an overview over the fluxes 
        """
        s="Object of class "+str(self.__class__)
        indent=2 
        s+="\n Input fluxes:\n"
        s+=flux_dict_string(self.input_fluxes,indent)

        s+="\n Internal fluxes:\n"
        s+=flux_dict_string(self.internal_fluxes,indent)
        
        s+="\n Output fluxes:\n"
        s+=flux_dict_string(self.output_fluxes,indent)

        return s 

    def all_fluxes(self): 
        # since input and output fluxes are indexed by integers they could
        # overload each other in a common dictionary
        # to avoid this we create a list
        return [v for v in self.input_fluxes.values()] + [v for v in self.output_fluxes.values()] + [v for v in self.internal_fluxes.values()]

    @property
    def jacobian(self):
        state_vec=Matrix(self.state_vector)
        vec=Matrix(self.F)
        return jacobian(vec,state_vec)

    
    @property
    def is_compartmental(self):
        """ Returns checks that all fluxes are nonnegative
        at the time of implementation this functionality sympy did not support 
        relations in predicates yet.
        So while the following works:
        
        with assuming(Q.positive(x) & Q.positive(y)):
           print(ask(Q.positive(2*x+y)
        
        it is not possible yet to get a meaningful answer to:
        
        with assuming(Q.is_true(x>0) & Q.is_true(y>0)):
           print(ask(Q.positive(2*x+y)
        
        We therefore cannot implement more elaborate assumptions like k_1-(a_12+a_32)>=0 
        but still can assume all the state_variables and the time_symbol to be nonnegative
        Therefore we can check the compartmental_property best after all paramater value have been substituted.
        At the moment the function throws an exception if this is not the case.
        """
        #check if all free symbols have been removed
        allowed_symbs= set( [sym for sym in self.state_vector])
        if hasattr(self,"time_symbol"):
            allowed_symbs.add(self.time_symbol)

        if not(allowed_symbs.issuperset(self.free_symbols)):
            raise Exception(
                    Template("Sympy can not check the parameters without assumptions. Try to substitute all variables except the state variables and the time symbol. Use the subs methot of the class {c}").subs(c=self__class__)
                )

        def f(expr):
            res= ask(Q.nonnegative(expr))
            if res is None:
                raise Exception(
                        Template("""Sympy can not (yet) check the parameters even with correct assumptions,\ 
since relations (<,>) are not implemented yet. 
It gave up for the following expression: ${e}."""
                        ).substitute(e=expr) 
                    )
            return res

        # making a list of predicated stating that all state variables are nonnegative
        predList=[Q.nonnegative(sym) for sym in self.state_vector]
        if hasattr(self,"time_symbol"):
            predList+=[Q.nonnegative(self.time_symbol)]

        with assuming(*predList):
            # under this assumption eveluate all fluxes
            all_fluxes_nonnegative=all(map(f,self.all_fluxes()))

        return all_fluxes_nonnegative
        
    
    # alternative constructor based on the formulation f=u+Bx
    @classmethod
    def from_B_u(cls, state_vector, time_symbol, B, u)->'SmoothReservoirModel':
        """Construct and return a :class:`SmoothReservoirModel` instance from 
           :math:`\\dot{x}=B\\,x+u`
    
        Args:
            state_vector (SymPy dx1-matrix): The model's state vector 
                :math:`x`.
                Its entries are SymPy symbols.
            time_symbol (SymPy symbol): The model's time symbol.
            B (SymPy dxd-matrix): The model's compartmental matrix.
            u (SymPy dx1-matrix): The model's external input vector.
    
        Returns:
            :class:`SmoothReservoirModel`
        """
#        if not(u):
#           # fixme mm:
#           # make sure that ReservoirModels standard constructor can handle an 
#           # empty dict and produce the empty matrix only if necessary later
#           u=zeros(x.rows,1)
        
        # fixme mm:
        # we do not seem to have a check that makes sure 
        # that the argument B is compartmental
        # maybe the fixme belongs rather to the SmoothModelRun class since 
        # we perhaps need parameters 
        
        input_fluxes = dict()
        for pool in range(u.rows):
            inp = u[pool]
            if inp:
                input_fluxes[pool] = inp
    
        output_fluxes = dict()
        # calculate outputs
        for pool in range(state_vector.rows):
            outp = -sum(B[:, pool]) * state_vector[pool]
            #outp = simplify(outp)
            if outp:
                output_fluxes[pool] = outp
        
        # calculate internal fluxes
        internal_fluxes = dict()
        pipes = [(i,j) for i in range(state_vector.rows) 
                        for j in range(state_vector.rows) if i != j]
        for pool_from, pool_to in pipes:
            flux = B[pool_to, pool_from] * state_vector[pool_from]
            #flux = simplify(flux)
            if flux:
                internal_fluxes[(pool_from, pool_to)] = flux
        
        # call the standard constructor 
        srm = SmoothReservoirModel(state_vector, time_symbol,
                  input_fluxes, output_fluxes, internal_fluxes)
        
        return srm

    @property
    def state_variable_set(self):
        return set(self.state_vector)

    @property
    def F(self):
        """SymPy dx1-matrix: The right hand side of the differential equation 
        :math:`\\dot{x}=B\\,x+u`."""
        v = (self.external_inputs + self.internal_inputs
                - self.internal_outputs - self.external_outputs)
        #for i in range(len(v)):
        #    v[i] = simplify(v[i])
        return v
    
    @property
    def external_inputs(self):
        """SymPy dx1-matrix: Return the vector of external inputs."""
        u = zeros(self.nr_pools, 1)
        for k, val in self.input_fluxes.items():
            u[k] = val
        return u
    
    @property
    def external_outputs(self):
        """SymPy dx1-matrix: Return the vector of external outputs."""
        o = zeros(self.nr_pools, 1)
        for k, val in self.output_fluxes.items():
            o[k] = val
        return o
        
    @property
    def internal_inputs(self):
        """SymPy dx1-matrix: Return the vector of internal inputs."""
        n = self.nr_pools
        u_int = zeros(n, 1)
        for ln in range(n):     
            # find all entries in the fluxes dict that have the target key==ln
            expr = 0
            for k, val in self.internal_fluxes.items():
                if k[1] == ln: #the second part of the tupel is the recipient
                    expr += val
            u_int[ln] = expr
        return u_int
    
    @property
    def internal_outputs(self):
        """SymPy dx1-matrix: Return the vector of internal outputs."""
        n = self.nr_pools
        o_int = zeros(n, 1)
        for ln in range(n):     
            # find all entries in the fluxes dict that have the target key==ln
            expr = 0
            for k, val in self.internal_fluxes.items():
                if k[0] == ln:# the first part of the tupel is the donator
                    expr += val
            o_int[ln] = expr
        return o_int

    @property
    def nr_pools(self):
        """int: Return the number of pools involved in the model."""
        return(len(self.state_variables))

    def port_controlled_Hamiltonian_representation(self):
        """tuple: :math:`J, R, N, x, u` from 
        :math:`\\dot{x} = [J(x)-R(x)] \\frac{\\partial}{\\partial x}H+u`.
    	with :math:`H=\\sum_i x_i \\implies \\frac{\\partial}{\\partial x}H =(1,1,...,1)`

        Returns:
            tuple:
            - J (skew symmetric SymPy dxd-matrix) of internal fluxbalances: 
              :math:`J_{i,j}=r_{j,i}-r_{i,j}`
            - Q (SymPy dxd-matrix): Diagonal matrix describing the dissipation
              rates (outfluxes). 
            - x (SymPy dx1-matrix): The model's state vector.
            - u (SymPy dx1-matrix): The model's external input vector.
        """
        nr_pools = self.nr_pools
        inputs = self.input_fluxes
        outputs = self.output_fluxes
        internal_fluxes = self.internal_fluxes

        C = self.state_vector

        # convert inputs
        u = self.external_inputs

        # calculate decomposition operators
        decomp_fluxes = []
        for pool in range(nr_pools):
            if pool in outputs.keys():
                decomp_flux = outputs[pool]
            else:
                decomp_flux = 0
            decomp_fluxes.append(simplify(decomp_flux))

        Q = diag(*decomp_fluxes)

        # calculate the skewsymmetric matrix J
        J = zeros(nr_pools)
        
        for (i,j), flux in internal_fluxes.items():
            J[j,i] +=flux
            J[i,j] -=flux

        return (J, Q, C, u)

    def xi_T_N_u_representation(self, factor_out_xi=True):
        """tuple: :math:`\\xi, T, N, x, u` from 
        :math:`\\dot{x} = \\xi\\,T\\,N\\,x+u`.

        Args:
            factor_out_xi (bool): If true, xi is extracted from the matrix,
                otherwise :math:`xi=1` will be returned.
                (Defaults to ``True``.)

        Returns:
            tuple:
            - xi (SymPy number): Environmental coefficient.
            - T (SymPy dxd-matrix): Internal fluxes. Main diagonal contains 
                ``-1`` entries.
            - N (SymPy dxd-matrix): Diagonal matrix containing the decomposition
                rates.
            - x (SymPy dx1-matrix): The model's state vector.
            - u (SymPy dx1-matrix): The model's external input vector.
        """
        nr_pools = self.nr_pools
        inputs = self.input_fluxes
        outputs = self.output_fluxes
        internal_fluxes = self.internal_fluxes

        C = self.state_vector

        # convert inputs
        u = self.external_inputs

        # calculate decomposition operators
        decomp_rates = []
        for pool in range(nr_pools):
            if pool in outputs.keys():
                decomp_flux = outputs[pool]
            else:
                decomp_flux = 0
            decomp_flux += sum([flux for (i,j), flux in internal_fluxes.items() 
                                        if i == pool])
            decomp_rates.append(simplify(decomp_flux/C[pool]))

        N = diag(*decomp_rates)

        # calculate transition operator
        T = -eye(nr_pools)
        
        for (i,j), flux in internal_fluxes.items():
            T[j,i] = flux/C[i]/N[i,i]

        # try to extract xi from N and T
        if factor_out_xi:
            xi_N = factor_out_from_matrix(N)
            N = N/xi_N

            xi_T = factor_out_from_matrix(T)
            T = T/xi_T

            xi = xi_N * xi_T
        else:
            xi = 1

        return (xi, T, N, C, u)

    @property
    def compartmental_matrix(self):
        """SymPy Matrix: :math:`B` from 
        :math:`\\dot{x} = B\\,x+u`.

        Returns:
            SymPy dxd-matrix: :math:`B = \\xi\\,T\\,N`
        """
        # could be computed directly from Jaquez
        # but since we need the xi*T*N decomposition anyway
        # we can use it
        xi, T, N, C, u = self.xi_T_N_u_representation(factor_out_xi=False)
        return(xi*T*N)
    
    def age_moment_system(self, max_order):
        """Return the age moment system of the model.

        Args:
            max_order (int): The maximum order up to which the age moment 
                system is created (1 for the mean).

        Returns:
            tuple:
            - extended_state (SymPy d*(max_order+1)x1-matrix): The extended 
                state vector of the age moment system.
            - extended_rhs (SymPy d*(max_order+1)x1-matrix): The extended right 
                hand side of the age moment ODE.
        """
        u = self.external_inputs
        #X = Matrix(self.state_variables)
        X = self.state_vector
        B = self.compartmental_matrix

        n = self.nr_pools
        extended_state = list(X)
        former_additional_states = [1]*n
        extended_rhs = list(self.F)
        for k in range(1, max_order+1):
            additional_states = [Symbol(str(x)+'_moment_'+str(k)) for x in X]
            g = [k*former_additional_states[i]
                    +(sum([(additional_states[j]-additional_states[i])
                        *B[i,j]*X[j] for j in range(n)])
                      -additional_states[i]*u[i])/X[i] for i in range(n)]

            former_additional_states = additional_states
            extended_state.append(additional_states)
            extended_rhs.append(g)

        extended_state = Matrix(flatten(extended_state))
        extended_rhs = Matrix(flatten(extended_rhs))

        return (extended_state, extended_rhs)

    def figure(self, figure_size = (7,7), logo = False, thumbnail = False):
        """Return a figure representing the reservoir model.

        Args:
            figure_size (2-tuple, optional): Width and height of the figure. 
                Defaults to (7,7).
            logo (bool, optional): If True, figure_size set to (3,3), no legend,
                smaller font size. Defaults to False.
            thumbnail (bool, optional): If True, produce a very small version,
                no legend. Defaults to False.

        Returns:
            Matplotlib figure: Figure representing the reservoir model.
        """
        inputs = self.input_fluxes
        outputs =  self.output_fluxes
        internal_fluxes = self.internal_fluxes

        pool_alpha = 0.3
        pool_color = 'blue'
        pipe_colors = {
            'linear': 'blue',
            'nonlinear': 'green', 
            'no state dependence': 'red'
        }

        mutation_scale = 50
        #mutation_scale=20
        arrowstyle = "simple"
        pipe_alpha=0.5
        connectionstyle='arc3, rad=0.1'
        fontsize = 24
        legend = True
        if thumbnail:
            mutation_scale = 10
            legend = False
            arrowstyle = "-"
            figure_size = (0.7,0.7)    

        if logo:
            mutation_scale = 15
            legend = False
            fontsize = 16
            figure_size = (3,3)     
       
        fig = plt.figure(figsize=figure_size, dpi=300)
        if legend:
            #ax = fig.add_axes([0,0,1,0.9])
            ax = fig.add_axes([0,0,0.8,0.8])
        else:
            #ax = fig.add_axes([0,0,1,1])
            ax = fig.add_subplot(1,1,1)
        
        ax.set_axis_off()
         
        nr_pools = self.nr_pools

        base_r = 0.1 + (0.5-0.1)/10*nr_pools
        if nr_pools > 1:
            r = base_r * (1-np.exp(1j*2*np.pi/nr_pools))
            r = abs(r) / 2 * 0.6
            r = min(r, (0.5-base_r)*0.5)
        else:
            r = base_r * 0.5
        
        r = abs(r)

        if thumbnail:
            r = r * 0.7
    

        # plot pools and external fluxes
        pools = []
        for i in range(nr_pools):
            z = base_r * np.exp(i*2*np.pi/nr_pools*1j)
            x = 0.5 - z.real
            y = 0.5 + z.imag

            pool = Pool(
                x, 
                y, 
                r, 
                pool_color, 
                pool_alpha, 
                pipe_alpha, 
                connectionstyle, 
                arrowstyle,
                mutation_scale
            )
            pool.plot(ax)

            if not thumbnail:
                pool_name = self.state_vector[i]
                pool.plot_name(ax, "$"+latex(pool_name)+"$", fontsize)

            if i in inputs.keys():
                pool.plot_input_flux(
                    ax,
                    pipe_colors[self._input_flux_type(i)]
                )

            if i in outputs.keys():
                pool.plot_output_flux(
                    ax,
                    pipe_colors[self._output_flux_type(i)]
                )

            pools.append(pool)

        # plot internal fluxes
        for (i,j) in internal_fluxes.keys():
            pools[i].plot_internal_flux_to(
                ax,
                pools[j],
                pipe_colors[self._internal_flux_type(i,j)]
            )


        if legend:
            legend_descs = []
            legend_colors = []
            for desc, col in pipe_colors.items():
                legend_descs.append(desc)
                legend_colors.append(
                    mpatches.FancyArrowPatch(
                        (0,0),
                        (1,1), 
                        connectionstyle=connectionstyle,
                        arrowstyle=arrowstyle, 
                        mutation_scale=mutation_scale, 
                        alpha=pipe_alpha, 
                        color=col
                    )
                )
            
            ax.legend(legend_colors, legend_descs, loc='upper center', 
                        bbox_to_anchor=(0.5, 1.1), ncol = 3)
        
        return fig
    

    ##### 14C methods #####


#    def to_14C_only(self, decay_symbol_name, Fa_expr_name):
#        """Construct and return a :class:`SmoothReservoirModel` instance that
#           models the 14C component of the original model.
#    
#        Args:
#            decay_symbol_name (str): The name of the 14C decay rate symbol.
#            Fa_expr_name(str): The name of the symbol to be used for the
#                atmospheric C14 fraction function.
#        Returns:
#            :class:`SmoothReservoirModel`
#        """
##        state_vector_14C = Matrix(
##            self.nr_pools,
##            1,
##            [Symbol(sv.name+'_14C') for sv in self.state_vector]
##        )
#        state_vector_14C = self.state_vector
#
#        decay_symbol = Symbol(decay_symbol_name)
#        B_14C = copy(self.compartmental_matrix) - decay_symbol*eye(self.nr_pools)
#        u = self.external_inputs
#        Fa_expr = Function(Fa_expr_name)(self.time_symbol)
#        u_14C = Matrix(self.nr_pools, 1, [expr*Fa_expr for expr in u])
#
#        srm_14C = SmoothReservoirModel.from_B_u(
#            state_vector_14C,
#            self.time_symbol,
#            B_14C,
#            u_14C
#        )
#
#        return srm_14C
#
#    def to_14C_explicit(self, decay_symbol_name, Fa_expr_name):
#        """Construct and return a :class:`SmoothReservoirModel` instance that
#           models the 14C component additional to the original model.
#    
#        Args:
#            decay_symbol_name (str): The name of the 14C decay rate symbol.
#            Fa_expr_name(str): The name of the symbol to be used for the
#                atmospheric C14 fraction function.
#        Returns:
#            :class:`SmoothReservoirModel`
#        """
#        state_vector = self.state_vector
#        B, u = self.compartmental_matrix, self.external_inputs
#        srm_14C = self.to_14C_only(decay_symbol_name, Fa_expr_name)
#        state_vector_14C = srm_14C.state_vector
#        B_C14 = srm_14C.compartmental_matrix
#        u_14C = srm_14C.external_inputs
#
#        nr_pools = self.nr_pools
#
#        state_vector_total = Matrix(nr_pools*2, 1, [1]*(nr_pools*2))
#        state_vector_total[:nr_pools,0] = state_vector
#        state_vector_total[nr_pools:,0] = state_vector_14C
#        
#        B_total = eye(nr_pools*2)
#        B_total[:nr_pools, :nr_pools] = B
#        B_total[nr_pools:, nr_pools:] = B_C14
#
#        u_total = Matrix(nr_pools*2, 1, [1]*(nr_pools*2))
#        u_total[:nr_pools,0] = u
#        u_total[nr_pools:,0] = u_14C
#
#        srm_total = SmoothReservoirModel.from_B_u(
#            state_vector_total,
#            self.time_symbol,
#            B_total,
#            u_total)
#
#        return srm_total
        
    def steady_states(self, par_set = None):
        if par_set is None:
            #compute steady state formulas
            par_set = {}
        # try to calculate the steady states for ten seconds
        # after ten seconds stop it
        q = multiprocessing.Queue()
        def calc_steady_states(q):    
            ss = solve(self.F.subs(par_set), self.state_vector, dict=True)
            q.put(ss)
    
        p = multiprocessing.Process(target=calc_steady_states, args=(q,))
        p.start()
        p.join(10)
        if p.is_alive():
            p.terminate()
            p.join()
            steady_states = []
        else:
            steady_states = q.get()
       
        formal_steady_states = []
        for ss in steady_states:
            result = []
            ss_dict = {}
            for sv_symbol in self.state_vector:
                if sv_symbol in ss.keys():
                    ss[sv_symbol] = simplify(ss[sv_symbol])
                else:
                    ss[sv_symbol] = sv_symbol

                ss_expr = ss[sv_symbol]
                if self.time_symbol in ss_expr.free_symbols:
                    # take limit of time to infinity if steady state still depends on time
                    ss_expr = limit(ss_expr, self.time_symbol, oo)
                ss_dict[sv_symbol.name] = ss_expr

            formal_steady_states.append(ss_dict)

        return formal_steady_states


    def is_state_dependent(self,expr):
        efss=expr.free_symbols
        svs=set([e for e in self.state_vector])
        inter=efss.intersection(svs)
        return not(len(inter)==0)	

    @property
    def is_linear(self):
        """Returns True if we can make SURE that the model is linear by checking that the jacobian is not state dependent.
        Note that external numerical functions of state variables are represented as sympy.Function f(x_1,x_2,...,t)
        Sympy will consider the derivative of math:`df/dx_i` with respect to state variable math:`x_i` as function math:`g(x_1, x_2,...)` too, since it can not exclude this possibility if we know f only numerically. 
        In consequence this method will return False even if the numerical implementation of f IS linear in math:`x_1,x_2,...` . 
        To avoid this situation you can just reformulate linear external functions math:`f(x_1,x_2,...,t)` as linear combinations
        of state independent external functions math:`f(x_1,x_2,...,t)=g_1(t)x_1+g_2(t)x_2+...` so that sympy can detect the linearity.
        

        Returns:
            bool: 'True', 'False'

        """
        return not(self.is_state_dependent(self.jacobian))

    ##### functions for internal use only #####


    # the following two functions are used by the 'figure' method to determine 
    # the color of the respective arrow 

    def _internal_flux_type(self, pool_from, pool_to):
        """Return the type of an internal flux.

        Args:
            pool_from (int): The number of the pool from which the flux starts.
            pool_to (int): The number of the pool to which the flux goes.

        Returns:
            str: 'linear', 'nonlinear', 'no state dependence'

        Raises:
            Error: If unknown flux type is encountered.
        """
        sv = self.state_vector[pool_from]
        flux = self.internal_fluxes[(pool_from, pool_to)]

        if has_pw(flux):
            #print("Piecewise")    
            #print(latex(flux))
            return "nonlinear"
            
        if gcd(sv, flux) == 1:
            return 'no state dependence'

        # now test for dependence on further state variables, 
        # which would lead to nonlinearity
        if (gcd(sv, flux) == sv) or gcd(sv, flux) == 1.0*sv:
            flux /= sv
            free_symbols = flux.free_symbols

            for sv in list(self.state_vector):
                if sv in free_symbols:
                    return 'nonlinear'

            return 'linear'
        else:
            # probably this can never happen
            raise(Error('Unknown internal flux type'))
    
    def _input_flux_type(self, pool_to):
        """Return the type of an external input flux.

        Args:
            pool_to (int): The number of the pool to which the flux contributes.

        Returns:
            str: 'linear', 'nonlinear', 'no state dependence'

        Raises:
            Error: If unknown flux type is encountered.
        """
        sv = self.state_vector[pool_to]
        # we compute the derivative of the appropriate row of the input vector w.r.t. all the state variables
        # (This is a row of the jacobian)  
        u_i=Matrix([self.external_inputs[pool_to]])
        s_v=Matrix(self.state_vector)
        J_i=jacobian(u_i,s_v)
        # an input that does not depend on state variables has a zero derivative with respect 
        # to all state variables
        if all([ j_ij==0 for j_ij in J_i]):
            return 'no state dependence'
        # an input that depends on state variables in a linear way
        # has a constant derivatives with respect to all state variables 
        # (the derivative has no state variables in its free symbols)
        J_ifss=J_i.free_symbols
        svs=set([e for e in self.state_vector])
        inter=J_ifss.intersection(svs)
        if len(inter)==0:
            return 'linear'
        else:
            return 'nonlinear'



    def _output_flux_type(self, pool_from):
        """Return the type of an external output flux.

        Args:
            pool_from (int): The number of the pool from which the flux starts.

        Returns:
            str: 'linear', 'nonlinear', 'no state dependence'

        Raises:
            Error: If unknown flux type is encountered.
        """
        sv = self.state_vector[pool_from]
        flux = self.output_fluxes[pool_from]
        if gcd(sv, flux) == 1:
            return 'no state dependence'

        # now test for dependence on further state variables, 
        # which would lead to nonlinearity
        if (gcd(sv, flux) == sv) or gcd(sv, flux) == 1.0*sv:
            flux /= sv
            free_symbols = flux.free_symbols

            for sv in list(self.state_vector):
                if sv in free_symbols:
                    return 'nonlinear'

            return 'linear'
        else:
            # probably this can never happen
            raise(Error('Unknown internal flux type'))
    


