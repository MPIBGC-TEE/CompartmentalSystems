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

from __future__ import division

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from copy import copy
from sympy import (zeros, Matrix, simplify, diag, eye, gcd, latex, Symbol, 
                   flatten, Function)
import numpy as np

from .helpers_reservoir import factor_out_from_matrix, has_pw


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


    def __init__(self, state_vector, time_symbol, 
                       input_fluxes, output_fluxes, internal_fluxes):
        """Return an instance of SmoothReservoirModel.
    
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
        # or at least a clumsy approach at a one-to-many relationship
        # there is no need to store SmoothModelRun objects in ReservoirModel 
        # objects since we already have the 
        # attribute model_run_combinations in class Model
        #self.model_runs=[]

    # fixme mm:
    # see the description of the model_runs property
   # def add_model_run(self,mr):
   #     self.model_runs.append(mr)

    # alternative constructor based on the formulation f=u+Bx
    @classmethod
    def from_B_u(cls, state_vector, time_symbol, B, u):
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
            outp = simplify(outp)
            if outp:
                output_fluxes[pool] = outp
        
        # calculate internal fluxes
        internal_fluxes = dict()
        pipes = [(i,j) for i in range(state_vector.rows) 
                        for j in range(state_vector.rows) if i != j]
        for pool_from, pool_to in pipes:
            flux = B[pool_to, pool_from] * state_vector[pool_from]
            flux = simplify(flux)
            if flux:
                internal_fluxes[(pool_from, pool_to)] = flux
        
        # call the standard constructor 
        srm = cls(state_vector, time_symbol,
                  input_fluxes, output_fluxes, internal_fluxes)
        return srm

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
	with
	:math:H=\\sum_i x_i \\implies \\frac{\\partial}{\\partial x}H =(1,1,...,1) 

        Returns:
            tuple:
            - J (skew symmetric SymPy dxd-matrix): Internal of internal fluxbalances. 
	    	:math: J_{i,j}=r_{j,i}-r_{i,j) 
            - Q (SymPy dxd-matrix): Diagonal matrix describing the dissipation (outfluxes) 
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

    @property
    # fixme mm:
    # If the method does not return a single value it should not be decorated as a property
    # T could be as well as N or u or xi but not the whole lot.
    # The purpose of the property decorator is not to save parentheses but to make actual properties 
    # indistinguishable from methods that synthesize them to give us flexibility in the implementation
    def xi_T_N_u_representation(self):
        """tuple: :math:`\\xi, T, N, x, u` from 
        :math:`\\dot{x} = \\xi\\,T\\,N\\,x+u`.

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
        xi_N = factor_out_from_matrix(N)
        N = N/xi_N

        xi_T = factor_out_from_matrix(T)
        T = T/xi_T

        xi = xi_N * xi_T

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
        xi, T, N, C, u = self.xi_T_N_u_representation
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
            Matplotlib figure: Figure representing the reservir model.
        """
        inputs = self.input_fluxes
        outputs =  self.output_fluxes
        internal_fluxes = self.internal_fluxes


        pool_alpha = 0.3
        pool_color = 'blue'
        pipe_colors = {'linear': 'blue', 'nonlinear': 'green', 
                        'no substrate dependence': 'red'}
        pipe_alpha=0.5

        mutation_scale = 50
        #mutation_scale=20
        arrowstyle = "simple"
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
       
        fig = plt.figure(figsize=figure_size,dpi=300)
        if legend:
            ax = fig.add_axes([0,0,1,0.9])
        else:
            #ax = fig.add_axes([0,0,1,1])
            ax = fig.add_subplot(1,1,1)
        
        ax.set_axis_off()
         
        class PlotPool():
            def __init__(self, x, y, size, name, 
                               inputs, outputs, nr, reservoir_model):
                self.x = x
                self.y = y
                self.size = size
                self.name = name
                self.inputs = inputs
                self.outputs = outputs
                self.nr = nr
                # fixme:
                # circular dependency
                # actually a reservoir model can have pools and 
                # pipelines (and should initialize them)
                # the pipelines are not properties of a pool
                # but of the model
                # suggestion:
                # new class for Pipe (or PlotPipe) that can plot itself
                # with a color property set on initialization by the 
                # model depending on the linearity
                self.reservoir_model = reservoir_model

            def plot(self, ax):
                # plot the pool itself
                ax.add_patch(mpatches.Circle((self.x, self.y), 
                                              self.size, 
                                              alpha=pool_alpha,
                                              color=pool_color))
                if not thumbnail:
                    ax.text(self.x, self.y, "$"+latex(self.name)+"$", 
                            fontsize = fontsize, 
                            horizontalalignment='center', 
                            verticalalignment='center')
                
                # plot input flux if there is one
                if self.inputs:
                    x1 = self.x
                    y1 = self.y
                    z1 = self.x-0.5 + (self.y-0.5)*1j
                    arg1 = np.angle(z1) - np.pi/6
    
                    z1 = z1 + np.exp(1j*arg1) * self.size
                    x1 = 0.5+z1.real
                    y1 = 0.5+z1.imag
                    
                    z2 = z1 + np.exp(1j * arg1) * self.size * 1.0
                    
                    x2 = 0.5+z2.real
                    y2 = 0.5+z2.imag
        
                    col = pipe_colors['linear']
                    ax.add_patch(mpatches.FancyArrowPatch((x2,y2), (x1,y1), 
                        connectionstyle='arc3, rad=0.1', arrowstyle=arrowstyle, 
                        mutation_scale=mutation_scale, alpha=pipe_alpha, 
                        color=col))

                if self.outputs:
                    x1 = self.x
                    y1 = self.y
                    z1 = self.x-0.5 + (self.y-0.5)*1j
                    arg1 = np.angle(z1) + np.pi/6
    
                    z1 = z1 + np.exp(1j*arg1) * self.size
                    x1 = 0.5+z1.real
                    y1 = 0.5+z1.imag
                    
                    z2 = z1 + np.exp(1j * arg1) * self.size *1.0
                    
                    x2 = 0.5+z2.real
                    y2 = 0.5+z2.imag
    
                    col = pipe_colors[
                        self.reservoir_model._output_flux_type(self.nr)]
                    ax.add_patch(mpatches.FancyArrowPatch((x1,y1), (x2,y2), 
                        arrowstyle=arrowstyle, connectionstyle='arc3, rad=0.1', 
                        mutation_scale=mutation_scale, alpha=pipe_alpha, 
                        color=col))

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
    
        #patches.append(mpatches.Circle((0.5, 0.5), base_r))
        pools = []
        for i in range(nr_pools):
            z = base_r * np.exp(i*2*np.pi/nr_pools*1j)
            x = 0.5 - z.real
            y = 0.5 + z.imag
            if i in inputs.keys():
                inp = inputs[i]
            else:
                inp = None
            if i in outputs.keys():
                outp = outputs[i]
            else:
                outp = None
            pools.append(PlotPool(
                x, y, r, self.state_vector[i], inp, outp, i, self))

        for pool in pools:
            pool.plot(ax)
        pipe_alpha=0.5

        for (i,j) in internal_fluxes.keys():
            z1 = (pools[i].x-0.5) + (pools[i].y-0.5) * 1j
            z2 = (pools[j].x-0.5) + (pools[j].y-0.5) * 1j

            arg1 = np.angle(z2-z1) - np.pi/20
            z1 = z1+np.exp(1j*arg1)*r
           
            arg2 = np.angle(z1-z2)  + np.pi/20
            z2 = z2+np.exp(1j*arg2)*r

            x1 = 0.5+z1.real
            y1 = 0.5+z1.imag

            x2 = 0.5+z2.real
            y2 = 0.5+z2.imag

            col = pipe_colors[self._internal_flux_type(i,j)]

            ax.add_patch(mpatches.FancyArrowPatch((x1,y1),(x2,y2), 
                connectionstyle='arc3, rad=0.1', arrowstyle=arrowstyle, 
                mutation_scale=mutation_scale, alpha=pipe_alpha, color=col))

        if legend:
            legend_descs = []
            legend_colors = []
            for desc, col in pipe_colors.items():
                legend_descs.append(desc)
                legend_colors.append(mpatches.FancyArrowPatch((0,0),(1,1), 
                    connectionstyle='arc3, rad=0.1', arrowstyle=arrowstyle, 
                    mutation_scale=mutation_scale, alpha=pipe_alpha, color=col))
            
            ax.legend(legend_colors, legend_descs, loc='upper center', 
                        bbox_to_anchor=(0.5, 1.1), ncol = 3)
        
        return fig
    

    ##### 14C methods #####


    def to_14C_only(self, decay_symbol_name, Fa_expr_name):
        """Construct and return a :class:`SmoothReservoirModel` instance that
           models the 14C component of the original model.
    
        Args:
            decay_symbol_name (str): The name of the 14C decay rate symbol.
            Fa_expr_name(str): The name of the symbol to be used for the
                atmospheric C14 fraction function.
        Returns:
            :class:`SmoothReservoirModel`
        """
        state_vector_14C = Matrix(
            self.nr_pools,
            1,
            [Symbol(sv.name+'_14C') for sv in self.state_vector])
        decay_symbol = Symbol(decay_symbol_name)
        B_14C = copy(self.compartmental_matrix) - decay_symbol*eye(self.nr_pools)
        u = self.external_inputs
        Fa_expr = Function(Fa_expr_name)(self.time_symbol)
        u_14C = Matrix(self.nr_pools, 1, [expr*Fa_expr for expr in u])

        srm_14C = SmoothReservoirModel.from_B_u(
            state_vector_14C,
            self.time_symbol,
            B_14C,
            u_14C)

        return srm_14C

    def to_14C_explicit(self, decay_symbol_name, Fa_expr_name):
        """Construct and return a :class:`SmoothReservoirModel` instance that
           models the 14C component additional to the original model.
    
        Args:
            decay_symbol_name (str): The name of the 14C decay rate symbol.
            Fa_expr_name(str): The name of the symbol to be used for the
                atmospheric C14 fraction function.
        Returns:
            :class:`SmoothReservoirModel`
        """
        state_vector = self.state_vector
        B, u = self.compartmental_matrix, self.external_inputs
        srm_14C = self.to_14C_only(decay_symbol_name, Fa_expr_name)
        state_vector_14C = srm_14C.state_vector
        B_C14 = srm_14C.compartmental_matrix
        u_14C = srm_14C.external_inputs

        nr_pools = self.nr_pools

        state_vector_total = Matrix(nr_pools*2, 1, [1]*(nr_pools*2))
        state_vector_total[:nr_pools,0] = state_vector
        state_vector_total[nr_pools:,0] = state_vector_14C
        
        B_total = eye(nr_pools*2)
        B_total[:nr_pools, :nr_pools] = B
        B_total[nr_pools:, nr_pools:] = B_C14

        u_total = Matrix(nr_pools*2, 1, [1]*(nr_pools*2))
        u_total[:nr_pools,0] = u
        u_total[nr_pools:,0] = u_14C

        srm_total = SmoothReservoirModel.from_B_u(
            state_vector_total,
            self.time_symbol,
            B_total,
            u_total)

        return srm_total
        


    ##### functions for internal use only #####


    # the following two functions are used by the 'figure' method to determine 
    # the color of the respective arrow 

    def _internal_flux_type(self, pool_from, pool_to):
        """Return the type of an internal flux.

        Args:
            pool_from (int): The number of the pool from which the flux starts.
            pool_to (int): The number of the pool to which the flux goes.

        Returns:
            str: 'linear', 'nonlinear', 'no substrate dependence'

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
            return 'no substrate dependence'

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


    def _output_flux_type(self, pool_from):
        """Return the type of an external output flux.

        Args:
            pool_from (int): The number of the pool from which the flux starts.

        Returns:
            str: 'linear', 'nonlinear', 'no substrate dependence'

        Raises:
            Error: If unknown flux type is encountered.
        """
        sv = self.state_vector[pool_from]
        flux = self.output_fluxes[pool_from]
        if gcd(sv, flux) == 1:
            return 'no substrate dependence'

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



