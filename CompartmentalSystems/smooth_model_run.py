"""Module for numerical treatment of piece-wise continuous reservoir models.

An abstract 
:class:`~.smooth_reservoir_model.SmoothReservoirModel` is 
filled with life by giving initial values, a parameter set, a time grid, 
and potentially additional involved functions to it.

The model can then be run and as long as the model is linear,
based on the state transition operator age and transit time
distributions can be computed.

Nonlinear models can be linearized along a solution trajectory.

Counting of compartment/pool/reservoir numbers start at zero and the 
total number of pools is :math:`d`.
"""

from numbers import Number
from copy import copy, deepcopy
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import matrix_power

import plotly.graph_objs as go

import base64
import hashlib
import mpmath
from frozendict import frozendict

from sympy import lambdify, flatten, latex, Function, sympify, sstr, solve, \
                  ones, Matrix, ImmutableMatrix
from sympy.core.function import UndefinedFunction
from sympy.abc import _clash
from sympy.printing import pprint

import scipy.linalg
from scipy.linalg import inv
from numpy.linalg import pinv
from scipy.special import factorial
from scipy.integrate import odeint, quad 
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import newton, brentq, minimize

from tqdm import tqdm
#from testinfrastructure.helpers import pe

from .smooth_reservoir_model import SmoothReservoirModel
from .model_run import ModelRun
from .helpers_reservoir import (
    deprecation_warning
    ,warning
    ,make_cut_func_set
    ,has_pw
    ,numsol_symbolic_system_old
    ,numsol_symbolical_system 
    ,arrange_subplots
    ,melt
    ,generalized_inverse_CDF
    ,draw_rv 
    ,stochastic_collocation_transform
    ,numerical_rhs
    ,numerical_rhs_old
    ,MH_sampling
    ,save_csv 
    ,load_csv
    ,stride
    ,f_of_t_maker
    ,const_of_t_maker
    ,numerical_function_from_expression
    ,x_phi_ode
    ,phi_tmax
    ,x_tmax
    ,print_quantile_error_statisctics
    ,custom_lru_cache_wrapper
    ,net_Us_from_discrete_Bs_and_xs
    ,net_Fs_from_discrete_Bs_and_xs
    ,net_Rs_from_discrete_Bs_and_xs
    ,check_parameter_dict_complete
)

from .BlockIvp import BlockIvp
from .myOdeResult import solve_ivp_pwc
from .Cache import Cache

class Error(Exception):
    """Generic error occurring in this module."""
    pass


class SmoothModelRun(ModelRun):
    """Class for a model run based on a 
    :class:`~.smooth_reservoir_model.SmoothReservoirModel`.

    Attributes:
        model (:class:`~.smooth_reservoir_model.SmoothReservoirModel`): 
            The reservoir model on which the model run bases.
        parameter_dict (dict): ``{x: y}`` with ``x`` being a SymPy symbol 
            and ``y`` being a numerical value.
        start_values (numpy.array): The vector of start values.
        times (numpy.array): The time grid used for the simulation.
            Typically created by ``numpy.linspace``.
        func_set (dict): ``{f: func}`` with ``f`` being a SymPy symbol and 
            ``func`` being a Python function. Defaults to ``dict()``.

    Pool counting starts with ``0``. In combined structures for pools and 
    system, the system is at the position of a ``(d+1)`` st pool.
    """

    def __init__(self, model, parameter_dict, 
                        start_values, times, func_set=None ):
        """Return a SmoothModelRun instance.

        Args:
            model (:class:`~.smooth_reservoir_model.SmoothReservoirModel`): 
                The reservoir model on which the model run bases.
            parameter_dict (dict): ``{x: y}`` with ``x`` being a SymPy symbol 
                and ``y`` being a numerical value.
            start_values (numpy.array): The vector of start values.
            times (numpy.array): The time grid used for the simulation.
                Typically created by ``numpy.linspace``.
            func_set (dict): ``{f: func}`` with ``f`` being a SymPy symbol and 
                ``func`` being a Python function. Defaults to ``dict()``.

        Raises:
            Error: If ``start_values`` is not a ``numpy.array``.
        """
        # we cannot use dict() as default because the test suite makes weird 
        # things with it! But that is bad style anyways
        if parameter_dict is None: parameter_dict = dict()
        if func_set is None: func_set = dict()
        
        # check parameter_dict + func_set for completeness
        free_symbols = check_parameter_dict_complete(
                            model, 
                            parameter_dict, 
                            func_set)
        if free_symbols != set():
            raise(Error('Missing parameter values for ' + str(free_symbols)))
        

        self.model = model
        self.parameter_dict = frozendict(parameter_dict)
        self.times = times
        # make sure that start_values are an array,
        # even a one-dimensional one
        self.start_values = np.array(start_values).reshape(model.nr_pools,)

        if not(isinstance(start_values, np.ndarray)):
            raise(Error("start_values should be a numpy array"))
        # fixme mm: 
        #func_set = {str(key): val for key, val in func_set.items()}
        # The conversion to string is not desirable here
        # should rather implement a stricter check (which fails at the moment because some tests use the old syntax
        #for f in func_set.keys():
        #    if not isinstance(f,UndefinedFunction):
        #        raise(Error("The keys of the func_set should be of type:  sympy.core.function.UndefinedFunction"))
        self.func_set = frozendict(func_set)


    def __str__(self):
        return str(
                 [ 'id(self)'+str(id(self)), 'id(model)'+str(id(self.model))]
                +["id "+str(key)+" "+str(id(val)) for key,val in   self.func_set.items()]
                +["id "+str(key)+" "+str(id(val)) for key,val in   self.parameter_dict.items()]
                )  
 

        
    @property
    def dts(self):
        """
        The lengths of the time intervals.
        """
        return np.diff(self.times).astype(np.float64)
    
    def B_func(self, vec_sol_func=None):
        # Design comment:
        # Note that the vec_sol_func argument is necessary because
        # we have slight differences in solutions coming from different
        # numerical solutions. If we intended to apply B_func to 
        # values not belonging to the solution, we would not do so
        # in an instace method but a separate function.

        if vec_sol_func == None:
            vec_sol_func = self.solve_func()
        
        # we inject the soltution into B to get the linearized version
        srm = self.model
        tup = (srm.time_symbol,) + tuple(srm.state_vector)
        numfun = numerical_function_from_expression(
            srm.compartmental_matrix,
            tup,
            self.parameter_dict,
            self.func_set
        )

        # we want a function  that accepts a vector argument for x
        
        def B_func(t):
            x = vec_sol_func(t)
            return numfun(t,*x)

        return B_func
    
    def linearize_old(self):
        """Return a linearized SmoothModelRun instance.

        Linearization happens along the solution trajectory. Only for linear 
        systems all functionality is guaranteed,
        this is why nonlinear systems should be linearized first.

        Returns:
            :class:`SmoothModelRun`: A linearized version of the original 
            :class:`SmoothModelRun`, with the solutions now being part 
            of ``func_set``.
        """
        sol_funcs = self.sol_funcs_old()
        
        srm = self.model
        xi, T, N, C, u = deepcopy(srm.xi_T_N_u_representation())
        svec = srm.state_vector

        symbolic_sol_funcs = {sv: Function(sv.name + '_sol')(srm.time_symbol) 
                                for sv in svec}

        # need to define a function_factory to create the function we need to 
        # avoid late binding
        # with late binding pool will always be nr_pools and always the last 
        # function will be used!
        def func_maker(pool):
            def func(t):
                return sol_funcs[pool](t)

            return(func)

        sol_dict = {}
        for pool in range(self.nr_pools):
            key = sstr(symbolic_sol_funcs[svec[pool]])
            sol_dict[key] = func_maker(pool)


        linearized_B = (xi*T*N).subs(symbolic_sol_funcs)
        linearized_u = u.subs(symbolic_sol_funcs)

        func_set=frozendict({key:val for mydict in [self.func_set,sol_dict]
                                     for key,val in mydict.items()})  

        cl=srm.__class__
        linearized_srm = cl.from_B_u(
            srm.state_vector, 
            srm.time_symbol, 
            linearized_B, 
            linearized_u
        )      

        linearized_smr = self.__class__(
            linearized_srm, 
            self.parameter_dict,
            self.start_values, 
            self.times, 
            func_set=func_set
        )
 
        return linearized_smr


    def linearize(self):
        """Return a linearized SmoothModelRun instance.

        Linearization happens along the solution trajectory. Only for linear 
        systems all functionality is guaranteed,
        this is why nonlinear systems should be linearized first.

        Returns:
            :class:`SmoothModelRun`: A linearized version of the original 
            :class:`SmoothModelRun`, with the solutions now being part 
            of ``func_set``.
        """
        #sol_funcs = self.sol_funcs()
        sol_funcs = self.sol_funcs()
        
        srm = self.model
        xi, T, N, C, u = srm.xi_T_N_u_representation()
        svec = srm.state_vector

        symbolic_sol_funcs = {sv: Function(sv.name + '_sol')(srm.time_symbol) 
                                for sv in svec}

        # need to define a function_factory to create the function we need to 
        # avoid late binding
        # with late binding pool will always be nr_pools and always the last 
        # function will be used!
        def func_maker(pool):
            def func(t):
                return sol_funcs[pool](t)

            return(func)

        sol_dict = {}
        for pool in range(self.nr_pools):
            key = sstr(symbolic_sol_funcs[svec[pool]])
            sol_dict[key] = func_maker(pool)


        linearized_B = (xi*T*N).subs(symbolic_sol_funcs)
        linearized_u = u.subs(symbolic_sol_funcs)

        func_set=frozendict({key:val for mydict in [self.func_set,sol_dict] for key,val in mydict.items()})  
        #func_set = self.func_set
        #func_set.update(sol_dict)

        cl = srm.__class__
        linearized_srm = cl.from_B_u(
            srm.state_vector, 
            srm.time_symbol, 
            linearized_B, 
            linearized_u
        )      

        linearized_smr = self.__class__(
            linearized_srm, 
            self.parameter_dict,
            self.start_values, 
            self.times, 
            func_set=func_set
        )
 
        return linearized_smr

    @staticmethod
    #fixme mm 2018-9-5:
    # Why is this is mehtod of class SmoothModelRun?
    # It does not rely on the class definition in any 
    # way. 
    # Is it because the helper module is not exposed in the API?
    def moments_from_densities(max_order, densities):
        """Compute the moments up to max_order of the given densities.

        Args:
            max_order (int): The highest order up to which moments are 
                to be computed.
            densities (numpy.array): Each entry is a Python function of one 
                variable (age) that represents a probability density function.

        Returns:
            numpy.ndarray: moments x pools, containing the moments of the given 
            densities.
        """
        n = densities(0).shape[0]

        def kth_moment(k):
            def kth_moment_pool(k, pool):
                norm = quad(lambda a: densities(a)[pool], 0, np.infty)[0]
                if norm == 0: return np.nan
                return (quad(lambda a: a**k*densities(a)[pool], 0, np.infty)[0] 
                            / norm)

            return np.array([kth_moment_pool(k,pool) for pool in range(n)])

        return np.array([kth_moment(k) for k in range(1, max_order+1)])



    ########## public methods and properties ########## 

    
    @property
    def nr_pools(self):
        """int: Return the number of pools involved in the model."""
        return self.model.nr_pools

    def solve_single_value_old(self, alternative_start_values=None):
        """Solve the model and return a function of time.

        Args:
            alternative_start_values (numpy.array, optional): If not given, the 
                original ``start_values`` are used.

        Returns:
            Python function ``f``: ``f(t)`` is a numpy.array that containts the 
            pool contents at time ``t``.
        """
        return self._solve_age_moment_system_single_value_old(0, None, 
                        alternative_start_values)

    def solve_func(self, alternative_start_values=None):
        """Solve the model and return a function of time.

        Args:
            alternative_start_values (numpy.array, optional): If not given, the 
                original ``start_values`` are used.

        Returns:
            Python function ``f``: ``f(t)`` is a numpy.array that containts the 
            pool contents at time ``t``.
        """
        return self._solve_age_moment_system_func(
                0,				 
                None, 
                alternative_start_values
            )

    def solve_old(self, alternative_times = None, alternative_start_values=None):
        """Solve the model and return a solution grid.

        Args:
            alternative_times (numpy.array): If not given, the original time 
                grid is used.
            alternative_start_values (numpy.array): If not given, 
                the original start_values are used.

        Returns:
            numpy.ndarray: len(times) x nr_pools, contains the pool contents 
            at the times given in the time grid.
        """
        return self._solve_age_moment_system_old(0, None, alternative_times, 
                        alternative_start_values)

    def solve(self, alternative_start_values=None):
        """Solve the model and return a solution grid. If the solution has been computed previously (even by other methods) the cached result will be returned.

        Args:
            alternative_start_values (numpy.array): If not given, 
                the original start_values are used.

        Returns:
            numpy.ndarray: len(times) x nr_pools, contains the pool contents 
            at the times given in the time grid.
        """
        soln, sol_func = self._solve_age_moment_system(
            0, 
            None,
            alternative_start_values
        )
        return soln

    ##### fluxes as functions #####
    
    def sol_funcs_old(self):
        """Return linearly interpolated solution functions.

        Returns:
            Python function ``f``: ``f(t)`` returns a numpy.array containing the
            pool contents at time ``t``.
        """
        times = self.times

        sol = self.solve_old(times)
        sol_funcs = []
        for i in range(self.nr_pools):
            sol_inter = interp1d(times, sol[:,i])
            sol_funcs.append(sol_inter)

        return sol_funcs

    #fixme: test
    def sol_funcs(self):#->List[Callable[float,float]]:
        """Returns list of linearly interpolated solution functions per pool.
        Returns:
            List of Python functions ``[f[i]]``, where ``f[i](t)`` returns 
                pool i's content at time ``t``.
        """
        times = self.times


        #sol = self.solve()
        vec_sol_func = self.solve_func()
        # the factorie is necessary to avoid untstrict evaluation
        def func_maker(pool):
            def func(t):
                return vec_sol_func(t)[pool]
            return(func)
       
        return [func_maker(i) for i in range(self.nr_pools)]

    def sol_funcs_dict_by_symbol(self):
        """
        Return linearly interpolated solution functions as a dictionary 
        indexed by the symbols of the state variables
        """
        #sol_funcs=self.sol_funcs()
        sol_funcs=self.sol_funcs()
        state_vector=self.model.state_vector
        n=len(state_vector)
        sol_dict_by_smybol={state_vector[i]:sol_funcs[i] for i in range(n)}
        return sol_dict_by_smybol

    def sol_funcs_dict_by_name(self):
        """Return linearly interpolated solution functions. as a dictionary indexed by the name (string) of the
        state variables"""
        sol_dict_by_name={k.name:v for k,v in self.sol_funcs_dict_by_symbol().items()}
        return sol_dict_by_name
        
    def external_input_flux_funcs(self):
        """Return a dictionary of the external input fluxes.
        The resulting functions base on sol_funcs and are linear interpolations.

        Returns:
            dict: ``{key: func}`` with ``key`` representing the pool which 
            receives the input and ``func`` a function of time that returns 
            a ``float``.
        """
        return self._flux_funcs(self.model.input_fluxes)

    def internal_flux_funcs(self):
        """Return a dictionary of the internal fluxes.
        

        Returns:
            dict: ``{key: func}`` with ``key=(pool_from, pool_to)`` representing
            the pools involved and ``func`` a function of time that returns 
            a ``float``.
        """
        return self._flux_funcs(self.model.internal_fluxes)

    def output_flux_funcs(self):
        """Return a dictionary of the external output fluxes.
        

        Returns:
            dict: ``{key: func}`` with ``key`` representing the pool from which
            the output comes and ``func`` a function of time that returns a 
            ``float``.
        """
        return self._flux_funcs(self.model.output_fluxes)
    
    def acc_gross_external_output_vector(self, data_times=None):
        """Return the vectors of accumulated external outputs.

        Returns:
            numpy.ndarray: len(times)-1 x nr_pools
        """
        times = self.times if data_times is None else data_times
        nt = len(times)-1
        res = np.zeros((nt,self.nr_pools))
        for k in range(nt):
            for pool_nr, func in self.output_flux_funcs().items():
                res[k,pool_nr] = quad(func,times[k],times[k+1])[0]
        
        return res


    #fixme: here _func indicated that this here is already a function of t
    # on other occasions _func indicated that a function is returned
    def output_vector_func(self, t):
        """Return a vector of the external output fluxes at time ``t``.
        

        Returns:
            numpy.array: The ``i`` th entry is the output from pool ``i`` at 
            time ``t``.
        """
        res = np.zeros((self.nr_pools,))
        for key, value in self.output_flux_funcs().items():
            res[key] = value(t)

        return res


    ##### fluxes as vector-valued functions #####
    

    # Note: This function could be rewritten using a vector valued input function 
    def external_input_vector_func(self, cut_off = True):
        """Return a vector valued function for the external inputs.


        Returns:
            Python function ``u``: ``u(t)`` is a ``numpy.array`` containing the 
            external inputs at time ``t``.
            Note: 
            If the required (future) values for the input exceed the maximum of
            times they are assumed to be zero if ``cut_off`` is ``True``.  
            If ``cut_off`` is ``False`` then the input function is assumed to
            be valid everywhere which might be dangerous if they are
            extrapolated from data.
        """
        if not hasattr(self, '_external_input_vector_func'):
            t0 = self.times[0]
            # cut off inputs until t0 (exclusive)
            if cut_off:
                t_valid = lambda t: True if ((t0<=t) and 
                                (t<=self.times[-1])) else False
            else:
                t_valid = lambda t: True

            input_fluxes = []
            for i in range(self.nr_pools):
                if i in self.external_input_flux_funcs().keys():
                    input_fluxes.append(self.external_input_flux_funcs()[i])
                else:
                    input_fluxes.append(lambda t: 0)
        
            u = lambda t: (np.array([f(t) for f in input_fluxes], 
                            dtype=np.float) 
                                if t_valid(t) else np.zeros((self.nr_pools,)))
            
            self._external_input_vector_func = u
     
        return self._external_input_vector_func

 
    def output_rate_vector_at_t(self, t):
        """Return a vector of output rates at time ``t``.

        Args:
            t (float): The time at which the output rates are computed.

        Returns:
            numpy.array: The ith entry contains the output rate of pool ``i`` 
            at time ``t``.
        """
        n = self.nr_pools

        #sol_funcs = self.sol_funcs()
        vec_sol_func = self.solve_func()
        output_vec_at_t = self.output_vector_func(t)

#        rate_vec = np.zeros((n,))
        x = vec_sol_func(t)
#        for pool in range(n):
#            #x = sol_funcs[pool](t)
#            if x != 0:
#                rate_vec[pool] = output_vec_at_t[pool] / x

        return np.nan_to_num(output_vec_at_t / x)


    ##### fluxes as vector over self.times #####

    def acc_gross_external_input_vector(self, data_times=None):
        """Return the grid of accumulated external input vectors.

        Returns:
            numpy.ndarray: len(times) x nr_pools
        """
        times = self.times if data_times is None else data_times
        nt = len(times)-1
        res = np.zeros((nt, self.nr_pools))
        for k in range(nt):
            for pool_nr, func in self.external_input_flux_funcs().items():
                res[k,pool_nr] = quad(func,times[k],times[k+1])[0]
        
        return res
    

    @property
    #this function should be rewritten using the vector values solution 
    def external_input_vector(self):
        """Return the grid of external input vectors.

        Returns:
            numpy.ndarray: len(times) x nr_pools
        """
        res = self._flux_vector(self.model.external_inputs)
        # no inputs at t0 (only >t0)
        #res[0,:] = np.zeros((self.nr_pools,))
        
        return res

    @property
    #this function should be rewritten using the vector valued solution 
    def external_output_vector(self):
        """Return the grid of external output vectors.

        Returns:
            numpy.ndarray: len(times) x nr_pools
        """
        return(self._flux_vector(self.model.external_outputs))

    @property    
    def output_rate_vector(self):
        """Return the grid of output rate vectors.

        Returns:
            numpy.ndarray: len(times) x nr_pools, ``solution/output_vector``
        """
        soln = self.solve()
        output_vec = self.external_output_vector

        # take care of possible division by zero
        output_vec[soln==0] = 0
        soln[soln==0] = 0

        return output_vec/soln

    #fixme hm: test
    def acc_gross_internal_flux_matrix(self, data_times=None):
        """Return the grid of flux matrices.

        Returns:
            numpy.ndarray: len(times) x nr_pools x nr_pools
        """
        times = self.times if data_times is None else data_times
        nt = len(times)-1
        res = np.zeros((nt, self.nr_pools, self.nr_pools))
        for k in range(nt):
            for key, func in self.internal_flux_funcs().items():
                j, i = key
                res[k,i,j] = quad(func,times[k],times[k+1])[0]
        
        return res


    ##### age density methods #####
    

    def pool_age_densities_single_value(self, start_age_densities=None):
        """Return a function for the pool age densities.

        Args:
            start_age_densities (Python function, optional): A function of age 
                that returns a numpy.array containing the masses with the given 
                age at time :math:`t_0`. Defaults to None, meaning that all 
                initial mass is considered to have zero age.

        Returns:
            Python function ``p_sv``: ``p_sv(a, t)`` returns ``a numpy.array`` 
            containing the pool contents with age ``a`` at time ``t``.
        """
        p1_sv = self._age_densities_1_single_value(start_age_densities)
        p2_sv = self._age_densities_2_single_value()

        p_sv = lambda a, t: p1_sv(a,t) + p2_sv(a,t)
        
        return p_sv

    
    # returns a function p that takes an age array "ages" as argument
    # and gives back a three-dimensional ndarray (ages x times x pools)
    # start_age_densities is a array-valued function of age
    def pool_age_densities_func(self, start_age_densities=None):
        """Return a function that takes an array of ages and returns the 
        pool age densities.

        Args:
            start_age_densities (Python function, optional): A function of age 
                that returns a numpy.array containing the masses with the given 
                age at time :math:`t_0` for every pool. 
                Defaults to None, meaning that all initial mass is considered 
                to have zero age.
        
        Returns:
            Python function ``p``: ``p(ages)`` returns a ``numpy.ndarray`` 
            len(ages) x len(times) x nr_pools containing the pool contents 
            with the respective ages at the respective times, where ``ages`` 
            is a ``numpy.array``.
        """
        p1 = self._age_densities_1(start_age_densities)
        p2 = self._age_densities_2()
        
        def p(ages):
            if hasattr(self, '_computed_age_density_fields'):
                if ((start_age_densities, tuple(ages)) in 
                        self._computed_age_density_fields.keys()):
                    #print('using cached result')
                    return self._computed_age_density_fields[
                                (start_age_densities, tuple(ages))]
            else:
                self._computed_age_density_fields = {}
        
            field_list = []
            for a in tqdm(ages):
                field_list.append(p1(np.array([a])) + p2(np.array([a])))

            field = np.array(field_list)[:,0,:,:]
            
            self._computed_age_density_fields[
                (start_age_densities, tuple(ages))] = field
            return field
                
        return p

    
    def system_age_density_single_value(self, start_age_densities=None):
        """Return a function for the system age density.

        Args:
            start_age_densities (Python function, optional): A function of age 
                that returns a numpy.array containing the masses with the given 
                age at time :math:`t_0`. 
                Defaults to None, meaning that all initial mass is considered 
                to have zero age.

        Returns:
            Python function ``sys_p_sv``: ``sys_p_sv(a, t)`` returns the system 
            content with age ``a`` at time ``t``.
        """
        p_sv = self.pool_age_densities_single_value(start_age_densities)
        sys_p_sv = lambda a, t: sum(p_sv(a,t))

        return sys_p_sv


    # return array ages x times with ages based on pool_age_densities
    def system_age_density(self, pool_age_densities):
        """Return the system age density based on the given pool age densities.

        Args:
            pool_age_densites (numpy.ndarray len(ages) x len(times) x nr_pools):
                The pool age density values.

        Returns:
            numpy.ndarray: (len(ages) x len(times)) The sum of the pool age 
            contents over all pools.
        """
        return pool_age_densities.sum(2)


    # combine pool and system age densities to one numpy array
    def age_densities(self, pool_age_densities, system_age_density):
        """Combine pool and system age densities to one numpy.array.

        Args:
            pool_age_densites (numpy.ndarray len(ages) x len(times) x nr_pools):
                The pool age density values.
            system_age_density (numpy.ndarray len(ages) x len(times)): 
                The system age density values.

        Returns:
            numpy.ndarray: (len(ages) x len(times) x (nr_pools+1)).
            The system age density values are appended to the end of the 
            pool density values (system = pool ``d+1`` with ``d = nr_pools``).
        """
        n = self.nr_pools
        nr_ages = pool_age_densities.shape[0]
        nr_times = pool_age_densities.shape[1]

        _age_densities = np.zeros((nr_ages, nr_times, n+1))
        _age_densities[:,:,:n] = pool_age_densities
        _age_densities[:,:,n] = system_age_density

        return _age_densities


    ##### age moment methods #####


    def age_moment_vector_from_densities(self, order, start_age_densities):
        """Compute the ``order`` th moment of the pool ages by integration.

        This function is extremely slow, since for each pool the integral over 
        the density is computed based on the singe-valued functions. It is 
        implemented only for the sake of completeness and to test the results 
        obtained by faster methods.

        Args:
            order (int): The order of the moment to be computed.
            start_age_densities (Python function, optional): 
                A function of age that returns a numpy.array containing the 
                masses with the given age at time :math:`t_0`.

        Returns:
            numpy.ndarray: len(times) x nr_pools. 
            Contains the ``order`` th moment 
            of the pool ages over the time grid.
        """
        p_sv = self.pool_age_densities_single_value(start_age_densities)
        times = self.times
        #x = self.solve_old()
        x = self.solve()
        n   = self.nr_pools
        k   = order

        def am_at_time_index_for_pool(ti, pool):
            def integrand(a):
                return (a**k) * p_sv(a, times[ti])[pool]
            
            return x[ti, pool]**(-1) * quad(integrand, 0, np.inf)[0]        

        def age_moment_at_time_index(ti):
            return np.array([am_at_time_index_for_pool(ti, pool) 
                                for pool in range(n)])

        am_arr = np.array([age_moment_at_time_index(ti) 
                            for ti in range(len(times))]) 
        am = np.ndarray((len(times), n), np.float, am_arr)

        return am


    def age_moment_vector_semi_explicit(self, order, 
                                        start_age_moments=None, times=None):
        """Compute the ``order`` th moment of the pool ages by a semi-explicit 
        formula.

        This function bases on a semi-explicit formula such that no improper 
        integrals need to be computed.
        
        Args:
            order (int): The order of the age moment to be computed.
            start_age_moments (numpy.ndarray order x nr_pools, optional): 
                Given initial age moments up to the order of interest. 
                Can possibly be computed by :func:`moments_from_densities`. 
                Defaults to ``None`` assuming zero initial ages.
            times (numpy.array, optional): Time grid. 
                Defaults to ``None`` and the original time grid is used.

        Returns:
            numpy.ndarray: len(times) x nr_pools.
            The ``order`` th pool age moments over the time grid.
        """
            
        if times is None: times = self.times
        t0 = times[0]
        n = self.nr_pools
        k = order
        
        if start_age_moments is None:
            start_age_moments = np.zeros((order, n))

        start_age_moments[np.isnan(start_age_moments)] = 0

        p2_sv = self._age_densities_2_single_value()

        def binomial(n, k):
            return 1 if k==0 else (0 if n==0 
                                    else binomial(n-1, k) + binomial(n-1, k-1))

        Phi = lambda t, t0, x: self._state_transition_operator(t, t0, x)

        def x0_a0_bar(j):
            if j == 0: 
                return self.start_values
                
            return np.array(self.start_values) * start_age_moments[j-1,:]

        def both_parts_at_time(t):
            def part2_time(t):
                def part2_time_index_pool(ti, pool):
                    return quad(lambda a: a**k * p2_sv(a, t)[pool], 0, t-t0)[0]

                return np.array([part2_time_index_pool(t, pool) 
                                    for pool in range(n)])

            def part1_time(t):
                def summand(j):
                    return binomial(k, j)*(t-t0)**(k-j)*Phi(t, t0, x0_a0_bar(j))

                return sum([summand(j) for j in range(k+1)])

            return part1_time(t) + part2_time(t)

        #soln = self.solve_old()
        soln = self.solve()

        def both_parts_normalized_at_time_index(ti):
            t = times[ti]
            bp = both_parts_at_time(t)
            diag_values = np.array([x if x>0 else np.nan for x in soln[ti,:]])
            X_inv = np.diag(diag_values**(-1))

            #return (np.mat(X_inv) * np.mat(bp).transpose()).A1
            return (np.matmul(X_inv, bp).transpose()).flatten()

        return np.array([both_parts_normalized_at_time_index(ti) 
                            for ti in range(len(times))])
        

    def age_moment_vector(self, order, start_age_moments = None):
        """Compute the ``order`` th pool age moment vector over the time grid 
        by an ODE system.

        This function solves an ODE system to obtain the pool age moments very
        fast. If the system has empty pools at the beginning, the semi-explicit 
        formula is used until all pools are non-empty. Then the ODE system 
        starts.

        Args:
            order (int): The order of the pool age moments to be computed.
            start_age_moments (numpy.ndarray order x nr_pools, optional): 
                Given initial age moments up to the order of interest. 
                Can possibly be computed by :func:`moments_from_densities`. 
                Defaults to None assuming zero initial ages.

        Returns:
            numpy.ndarray: len(times) x nr_pools.
            The ``order`` th pool age moments over the time grid.
        """
        n = self.nr_pools
        times = self.times

        
        if start_age_moments is None:
            start_age_moments = np.zeros((order, n))
        
        max_order=start_age_moments.shape[0]
        if order>max_order:
            raise Error("""
                To solve the moment system with order{0}
                start_age_moments up to (at least) the same order have to be
                provided. But the start_age_moments.shape was
                {1}""".format(order,start_age_moments.shape)
            )
        if order<max_order:
            warning("""
                Start_age_moments contained higher order values than needed.
                start_age_moments order was {0} while the requested order was
                {1}. This is no problem but possibly unintended. The higer
                order moments will be clipped """.format(max_order,order)
            )
            # make sure that the start age moments are clipped to the order
            # (We do not need start values for higher moments and the clipping
            # avoids problems with recasting if higher order moments are given 
            # by the user)
            start_age_moments=start_age_moments[0:order,:]

        if not (0 in self.start_values):
            #ams = self._solve_age_moment_system_old(order, start_age_moments)
            ams,_ = self._solve_age_moment_system(order, start_age_moments)
            return ams[:,n*order:]
        else:
            # try to start adapted mean_age_system once no pool 
            # has np.nan as mean_age (empty pool)

            # find last time index that contains an empty pool --> ti
            #soln = self.solve_old()
            soln = self.solve()
            ti = len(times)-1
            content = soln[ti,:]
            while not (0 in content) and (ti>0): 
                ti = ti-1
                content = soln[ti,:]

            # not forever an empty pool there?
            if ti+1 < len(times):
                # compute moment with semi-explicit formula 
                # as long as there is an empty pool
                amv1_list = []
                amv1 = np.zeros((ti+2, order*n))
                for k in range(1, order+1):
                    amv1_k = self.age_moment_vector_semi_explicit(
                        k, start_age_moments, times[:ti+2])
                    amv1[:,(k-1)*n:k*n] = amv1_k

                # use last values as start values for moment system 
                # with nonzero start values
                new_start_age_moments = amv1[-1,:].reshape((n, order))
                start_values = soln[ti+1]
                #ams = self._solve_age_moment_system_old(
                #    order, new_start_age_moments, times[ti+1:], start_values)
                ams,_ = self._solve_age_moment_system(
                    order, new_start_age_moments, start_values, times[ti+1:])
                amv2 = ams[:,n*order:]

                # put the two parts together
                part1 = amv1[:,(order-1)*n:order*n][:-1]
                amv = np.ndarray((len(times), n))
                amv[:part1.shape[0], :part1.shape[1]] = part1
                amv[part1.shape[0]:, :amv2.shape[1]] = amv2
                return amv
            else:
                # always an empty pool there
                return self.age_moment_vector_semi_explicit(
                        order, start_age_moments)


    # requires start moments <= order
    def system_age_moment(self, order, start_age_moments=None):
        """Compute the ``order`` th system age moment vector over the time grid 
        by an ODE system.

        The pool age moments are computed by :func:`age_moment_vector` and then 
        weighted corresponding to the pool contents.

        Args:
            order (int): The order of the pool age moments to be computed.
            start_age_moments (numpy.ndarray order x nr_pools, optional): 
                Given initial age moments up to the order of interest. 
                Can possibly be computed by :func:`moments_from_densities`. 
                Defaults to None assuming zero initial ages.

        Returns:
            numpy.array: The ``order`` th system age moment over the time grid.
        """
        n = self.nr_pools
        age_moment_vector = self.age_moment_vector(order, start_age_moments)
        age_moment_vector[np.isnan(age_moment_vector)] = 0
        #soln = self.solve_old()
        soln = self.solve()
         
        total_mass = soln.sum(1) # row sum
        total_mass[total_mass==0] = np.nan

        system_age_moment = (age_moment_vector*soln).sum(1)/total_mass

        return system_age_moment
        

    ##### transit time density methods #####


    def backward_transit_time_density_single_value_func(
            self, start_age_densities
        ):
        """Return a function that returns a single value for the 
        backward transit time density.

        Args:
            start_age_densities (Python function, optional): 
                A function of age that returns a numpy.array containing the 
                masses with the given age at time :math:`t_0`.

        Returns:
            Python function ``p_sv``: ``p_sv(a, t)`` returns the mass that 
            leaves the system at time ``t`` with age ``a``.
        """
        n = self.nr_pools
        p_age_sv = self.pool_age_densities_single_value(start_age_densities)

        def p_sv(a, t):
            p = p_age_sv(a, t)
            r = self.output_rate_vector_at_t(t)
            return (r*p).sum() 
            
        return p_sv


    # return an array ages x times with ages based on pool_age_densities
    def backward_transit_time_density(self, pool_age_densities):
        """Compute the backward transit time based on given pool age densities.

        The result is obtained by computing a weighted sum of the pool age 
        densities according to output rates.

        Args:
            pool_age_densites (numpy.ndarray len(ages) x len(times) x nr_pools):
                The pool age density values.
    
        Returns:
            numpy.ndarray: len(ages) x len(times). Mass leaving the system with 
            the respective age at the respective time.
        """
        r = self.output_rate_vector
        return (pool_age_densities*r).sum(2)

    
    def forward_transit_time_density_single_value_func(self, cut_off=True, my_B_func=None):
        """Return a function that returns a single value for the 
        forward transit time density.

        Args:
            cut_off (bool, optional): If ``True``, no density values are going to 
                be computed after the end of the time grid, instead 
                ``numpy.nan`` will be returned. 
                Defaults to ``True`` and ``False`` might lead to unexpected behavior.
        
        Returns:
            Python function ``p_sv``: ``p_sv(a, t)`` returns how much mass will 
            leave the system with age ``a`` when it came in at time ``t``.
        """
        if my_B_func is None:
            my_B_func = self.B_func(self.x_solve_func_skew())

        n = self.nr_pools
        times = self.times
        Phi = self._state_transition_operator
        input_func = self.external_input_vector_func()
        t0 = times[0]   
        t_max = times[-1] 
        def p_ftt_sv(a, t):
            # nothing leaves before t0
            if (t+a < t0): return 0.0

            #fixme: for Metropolis-Hastings we might need the density 
            #very far away...
            # we cannot compute the density if t+a is out of bounds
            if cut_off and (t+a > t_max): return np.nan

            u = input_func(t)
            if sum(u) == 0: return np.nan
            if (a < 0): return 0.0
            
            return -np.matmul(my_B_func(t+a),Phi(t+a, t, u)).sum()

        return p_ftt_sv


    #fixme: return value not consistent with backward_transit_time_density
    # not that easy to resolve, since here we do not use age_densities,
    # instead ages is really needed to be able to make the shift or call 
    # the state_transition_operator
    def forward_transit_time_density_func(self, cut_off=True, times=None):
        """Return a function based on an age array for the forward transit time 
        density.

        Args:
            cut_off (bool, optional): If True, no density values are going to 
                be computed after the end of the time grid, instead 
                ``numpy.nan`` will be returned. 
                Defaults to True and False might lead to unexpected behavior.
            times (numpy.array, optional): Time grid. 
                Defaults to ``None`` and the original time grid is used.
        
        Returns:
            Python function ``p``: ``p(ages)`` is a ``numpy.ndarray`` 
            len(ages) x len(times) that gives the mass that will leave the
            system with the respective age when it came in at time ``t``, 
            where ``ages`` is a ``numpy.array``.
        """
        wrapper = custom_lru_cache_wrapper(maxsize=len(self.times))
        cached_B_func = wrapper(self.B_func(self.x_solve_func_skew()))

        if times is None:
            times = self.times

        p_sv = self.forward_transit_time_density_single_value_func(
            cut_off, 
            my_B_func=cached_B_func
        )

        pp = lambda a: np.array([p_sv(a,t) for t in times], np.float)
        #p = lambda ages: np.array([pp(a) for a in ages], np.float)
        def p(ages):
            field_list = []
            for a in tqdm(ages):
                field_list.append(pp(a))

            field = np.array(field_list)

            return field

        return p


    ##### transit time moment methods #####

    
    def backward_transit_time_moment_from_density(self, 
            order, start_age_densities):
        """Compute the ``order`` th backward transit time moment based on an 
        improper integral over the density.

        This function is extremely slow and implemented only for the sake of 
        completeness and for testing results from faster approaches.

        Args:
            order (int): The order of the backward transit time moment that is 
                to be computed.
            start_age_densities (Python function, optional): A function of age 
                that returns a numpy.array containing the masses with the given 
                age at time :math:`t_0`.
        
        Returns:
            numpy.array: The ``order`` th backward transit time moment over the 
            time grid.
        """
        p_sv = self.backward_transit_time_density_single_value_func(
                    start_age_densities)
        times = self.times
        k = order

        ext_outp_vec = self.external_output_vector
        ext_outp = ext_outp_vec.sum(1)
     
        def btt_moment_at_time_index(ti):
            def integrand(a):
                return (a**k) * p_sv(a, times[ti])
            
            return ext_outp[ti]**(-1) * quad(integrand, 0, np.inf)[0]        

        bttm = np.array([btt_moment_at_time_index(ti) 
                            for ti in range(len(times))]) 
        return bttm


    def backward_transit_time_moment(self, order, start_age_moments=None):
        """Compute the ``order`` th backward transit time moment based on the 
        :func:`age_moment_vector`.

        Args:
            order (int): The order of the backward transit time moment that is 
                to be computed.
            start_age_moments (numpy.ndarray order x nr_pools, optional): 
                Given initial age moments up to the order of interest. 
                Can possibly be computed by :func:`moments_from_densities`. 
                Defaults to None assuming zero initial ages.
       
        Returns:
            numpy.array: The ``order`` th backward transit time moment over the 
            time grid.
        """ 
        age_moment_vector = self.age_moment_vector(order, start_age_moments)
        r = self.external_output_vector
        
        return (r*age_moment_vector).sum(1)/r.sum(1)


#    def forward_transit_time_moment(self, order, epsrel=1e-2):
#        """Compute the ``order`` th forward transit time moment.
#
#        Attention! This function integrates over the state transition operator 
#        until infinite time.
#        The results are going to be weird, since at the end of the time grid 
#        some cut- off will happen which biases the result.
#        Be also aware that additionally - to avoid convergence issues in quad -
#        the relative tolerance is set to 1e-2 by default.
#
#        Args:
#            order (int): The order of the forward transit time moment to be 
#                computed.
#
#        Returns:
#            numpy.array: The ``order`` th forward transit time moment over the 
#            time grid.
#        """
#        k = order
#        times = self.times
#        Phi = self._state_transition_operator
#        input_vector = self.external_input_vector
#
#        #import warnings
#        #from scipy.integrate import IntegrationWarning
#        #warnings.simplefilter("error", IntegrationWarning)
#        def moment_at_ti(ti):
#            u = input_vector[ti] 
#            
#            # if we have no inputs, there cannot be a transit(time)
#            if u.sum() == 0:    
#                return np.nan
#
#            def integrand(a):
#                res = (k*a**(k-1)*Phi(times[ti]+a, times[ti], u).sum())/u.sum()
#                #print(a, Phi(times[ti]+a, times[ti], u), res)
#                return res
#            
#            return quad(integrand, 0, np.infty,epsrel=epsrel)[0]
#            
#            # Remark: 
#            # We want to compute an inproper integral 
#            # instead of calling res=quad(integrand, 0, np.infty)[0]
#            # we could apply a variable transformation z=a/(c+a) # with an arbitrary c (possibly c=1 but we can optimize the choice  for better performance) 
#            # so we have \int_0^\infty f(a) dx= \int_0^z(a=\infty) f(a(z))*da/dz *dz  =\int_0^1  f(a(z)) c/(1-z**2) dz
#            # to do:
#            # To have the best accuracy we try to find c so that the peak of the integrand is projected to the middle of the new integration interval [0,1]
#            # 1.) find the maximum of the integrand
#            # 2.) find the c that projects this x to z=1/2
#            #c =1000
#            #def a(z):
#            #    return c*z/(1-z) 
#            #def transformed_integrand(z):
#            #    res = integrand(a(z))*c/(1-z**2) 
#            #    return res
#            #
#            #return quad(transformed_integrand, 0, 1)[0]
#
#        #res = np.array([moment_at_ti(ti) for ti in range(len(times))])
#        res = []
#        for ti in tqdm(range(len(times))):
#            res.append(moment_at_ti(ti))
#        res = np.array(res)
#
#        return res


    #fixme: split into two functions for SCCS and MH
    # do not use dict as default value
    def apply_to_forward_transit_time_simulation(self, 
            f_dict={'mean': np.mean}, N=10000, M=2, k=5, MH=False):
        """This is just a tentative approach.

        To be honest, the problem of an unkown infinite future cannot be solved 
        this way, since the densities used to simulate forward transit time 
        random variables are biased by the cut-off at the end of the time grid.
        """
        # f is a Python function, for the mean, take f = np.mean
        # N is the number of simulations per each time step
        # M is the number of collocation points for 
        # stochastic collocation sampling
        # allowed values for M are 2, 3, 4, ..., 11
        # other values lead to inverse transform sampling (slow)
        # k is the order of the smoothing and interpolating spline
        # 'smoothing_spline' is best used for inverse transform sampling, 
        # because of additional smoothing for low
        # number of random variates
        # for SCMCS (M in [2,...,11]), 'interpolation' is better, 
        # because the higher number of random variates 
        # (because of faster sampling) makes their mean already quite precise 
        # (in the framework of what is possible with SCMCS)
  
        times = self.times
        Phi = self._state_transition_operator
        input_func = self.external_input_vector_func()

        if not MH:
            self.n = 0
            def F_FTT(a, t):
                u = input_func(t)
                if u.sum() == 0: 
                    return np.nan
                
                if (a <= 0): return 0.0

                self.n += 1
                return 1 - Phi(t+a, t, u).sum()/u.sum()
    
            
            def simulate(n, CDF):
                # compute lagrange polynomial p if M is in [2, ..., 11]
                g = stochastic_collocation_transform(M, CDF)
                if g is None: 
                    # inverse transform sampling
                    print('inverse transform sampling')
                    rvs = np.array([draw_rv(CDF) for _ in range(n)])
                else:
                    norms = np.random.normal(size = n)
                    rvs = g(norms)
        
                return rvs

        else:
            self.m = 0
            p_sv = self.forward_transit_time_density_single_value_func(
                cut_off=False
            )
            def f_FTT(a, t):
                self.m += 1
                return p_sv(a, t)


        res = {f_name: {'values': [], 
                        'smoothing_spline': None, 
                        'interpolation': None} for f_name in f_dict.keys()}
        for t in times:
            print('time', t)
            # no iput means no forward transit time
            u = input_func(t)
            if u.sum() == 0: 
                rvs = np.nan
            else:
                if not MH:
                    rvs = simulate(N, lambda a: F_FTT(a, t))
                    print(self.n, 'calls of state transition operator')
                else:
                    rvs = MH_sampling(N, lambda a: f_FTT(a, t))
                    print(self.m, 'calls of forward transit time density')

            for f_name, f in f_dict.items():
                value = f(rvs)
                res[f_name]['values'].append(value)
                print(f_name, value)
                
        for f_name in res.keys():
            y = np.array(res[f_name]['values'])
            z = y.copy()
            res[f_name]['values'] = y.copy()

            # give weight zero to nan values fo compting the spline
            w = np.isnan(y)
            y[w] = 0.
            res[f_name]['smoothing_spline'] = UnivariateSpline(
                times, y, w=~w, k=k, check_finite=True)
            res[f_name]['interpolation'] = interp1d(times[~w], z[~w], kind=k)

        return res

    # use inverse transform sampling
    def apply_to_forward_transit_time_simulation_its(self, 
            f_dict, times, N=1000, k=5):
        """This is just a tentative approach.

        To be honest, the problem of an unkown infinite future cannot be solved 
        this way, since the densities used to simulate forward transit time 
        random variables are biased by the cut-off at the end of the time grid.
        """
        # f is a Python function, for the mean, take f = np.mean
        # N is the number of simulations per each time step
        # times is an np.array of interpolation points
        # k is the order of the smoothing and interpolating spline
        # 'smoothing_spline' is best used for inverse transform sampling, 
        # because of additional smoothing for low
        # number of random variates
  
        Phi = self._state_transition_operator
        input_func = self.external_input_vector_func()

        def F_FTT(a, t):
            u = input_func(t)
            if u.sum() == 0: 
                return np.nan
            
            if (a <= 0): return 0.0

            return 1 - Phi(t+a, t, u).sum()/u.sum()

        res = {f_name: {'values': [], 
                        'smoothing_spline': None, 
                        'interpolation': None} for f_name in f_dict.keys()}
        for t in times:
            print('time', t)
            # no iput means no forward transit time
            u = input_func(t)
            if u.sum() == 0: 
                rvs = np.nan
            else:
                CDF = lambda a: F_FTT(a, t)
                rvs = np.array([draw_rv(CDF) for _ in range(N)])

            for f_name, f in f_dict.items():
                value = f(rvs)
                res[f_name]['values'].append(value)
                print(f_name, value)
                
        def compute_splines(res, times):
            for f_name in res.keys():
                y = np.array(res[f_name]['values'])
                z = y.copy()
                res[f_name]['values'] = y.copy()

                # give weight zero to nan values fo compting the spline
                w = np.isnan(y)
                y[w] = 0.
                res[f_name]['smoothing_spline'] = UnivariateSpline(
                    times, y, w=~w, k=k, check_finite=True)
                res[f_name]['interpolation'] = interp1d(times[~w],z[~w],kind=k)

            return res

        return compute_splines(res, times)


    ##### comma separated values output methods #####


    def save_pools_and_system_density_csv(self, filename, pool_age_densities, 
            system_age_density, ages):
        """Save the pool and system age densities to a csv file.

        The system value will be coded into pool number -1.
        
        Args:
            filename (str): The name of the csv file to be written.
            pool_age_densites (numpy.ndarray len(ages) x len(times) x nr_pools):
                The pool age density values.
            system_age_density (numpy.ndarray len(ages) x len(times)): 
                The system age density values.
            ages (numpy.array): The ages that correspond to the indices in the
                zeroth dimension of the density arrays.

        Returns:
            None
        """
        n = self.nr_pools
        times = self.times
    
        ndarr = np.zeros((system_age_density.shape[0], len(times), n+1))
        ndarr[:,:,:n] = pool_age_densities
        ndarr[:,:,n] = system_age_density

        pool_entries = [i for i in range(n)] + [-1]
        melted = melt(ndarr, [ages, times, pool_entries])
        header = '"age", "time", "pool", "value"'
        save_csv(filename, melted, header)


    def save_pools_and_system_value_csv(self, filename, pools_ndarr, 
            system_arr):
        """Save pool and system values to a csv file.

        Values could be the mean age, for example. One dimension less than a
        density.
        The system value will be coded into pool number -1.

        Args:
            filename (str): The name of the csv file to be written.
            pools_ndarr (numpy.ndarray len(times) x nr_pools): The values to be
                saved over the time-pool grid.
            system_arr (numpy.array): The values over the time grid 
                corresponding to the system.
    
        Returns:
            None:
        """
        n = self.nr_pools
        times = self.times
    
        ndarr = np.concatenate(
            (pools_ndarr, system_arr.reshape((len(times), 1))), axis=1)

        pool_entries = [i for i in range(n)] + [-1]
        melted = melt(ndarr, [times, pool_entries])
        header = '"time", "pool", "value"'
        save_csv(filename, melted, header)


    ## helping methods ##

    def density_values_for_pools(self, pool_densities_sv, pool_age_values):
        """Compute the pool age densities over the time grid at ages given in 
        pool_age_values.

        This function can be used to obtain the density values at mean or median
        values to draw a curve on top of the density surface. But actually this
        is now implemented in a much faster way based on the surface itself.

        Args:
            pool_age_densites_sv (Python function): A function that takes 
                ``a``, ``t`` as arguments and returns a vector of pool contents 
                with mass a at time t. Potentially coming from 
                :func:`pool_age_densities_single_value`.
            pool_age_values (numpy.ndarray len(times) x nr_pools): The ages over
                the time-pool grid at which the density values are to be 
                computed.

        Returns:
            numpy.ndarray: (len(times) x nr_pools) The pool density values over
            the time-pool grid based on the given age values.
        """
        n = self.nr_pools
        times = self.times
    
        # for each pool we have a different age value 
        z = []
        for pool in range(n):
            val = pool_age_values[:,pool]
            #z.append(np.array([pool_densities_sv(val[i], times[i])[pool] 
            #                        for i in range(len(times))]))
            new_z_list = []
            for i in tqdm(range(len(times))):
                new_z_list.append(pool_densities_sv(val[i], times[i])[pool])

            z.append(np.array(new_z_list))

            z = np.array(z).T

        return z

    # return density values for mean, median, etc.
    #fixme: test
    def density_values(self, density_sv, values):
        """Compute the density value over the time grid at ages given in values.

        This function can be used to obtain the density values at mean or median
        values to draw a curve on top of the density surface. But actually this
        is now implemented in a much faster way based on the surface itself.

        Args:
            density_sv (Python function): A function that takes ``a``, ``t`` 
                as arguments and returns density value with age a at time ``t``.
                Potentially coming from :func:`system_age_density_single_value`.
            values (numpy.array): The ages over the time grid at which the 
                density values are to be computed.

        Returns:
            numpy.array: The density values over the time grid based 
            on the given age values.
        """
        times = self.times
        def f(i):
            if np.isnan(values[i]): return np.nan
            return density_sv(values[i], times[i])

        #dv_list = [f(i) for i in range(len(times))]

        dv_list = []
        for i in tqdm(range(len(times))):
            dv_list.append(f(i))

        return np.array(dv_list)


    def save_value_csv(self, filename, arr, times=None):
        """Save values over the time grid to a csv file.

        Args:
            filename (str): The name of the csv file to be written.
            arr (numpy.array): The values to be saved over the time grid.
            times (np.array, optional): The time grid to be used.
                Defaults to ``None`` in which case the orginal time grid 
                is used.
        Returns:
        """
        if times is None:
            times = self.times
        
        melted = melt(arr, [times])
        header = '"time", "value"'
        save_csv(filename, melted, header)

    def save_density_csv(self, filename, density, ages, times=None):
        """Save density values over age-time grid to csv file.

        Args:
            filename (str): The name of the csv file to be written.
            density (numpy.ndarray len(ages) x len(times)): The density values
                to be saved over the age-time grid.
            ages (numpy.array): The ages corresponding to the indices in the
                zeroth dimension of the density ndarray.
            times (numpy.array, optional): An alternative time grid to be used.
                Defaults to ``None`` which means that the original time grid is 
                going to be used.

        Returns:
        """
        if times is None: times = self.times
        melted = melt(density, [ages, times])
        header = '"age", "time", "value"'
        save_csv(filename, melted, header)
        

    ##### comma separated values input methods #####

    
    def load_value_csv(self, filename):
        melted = load_csv(filename)
        return (melted[:,1]).copy()


    ## combining pool and system structures ##


    def combine_pools_and_system_values(self, pools_values, system_values):
        """Append the system values to the pool values as if they belonged to 
        another pool.

        Args:
            pools_values (numpy.ndarray len(times) x nr_pools): The values to be
                saved over the time-pool grid.
            system_values (numpy.array): The system values to be saved over the
                time grid.

        Returns:
            numpy.ndarray: len(times) x (nr_pools+1) The pool and system values
            over the time-pool grid with the system added at the end as another
            pool.
        """
        n = self.nr_pools
        times = self.times
        values = np.zeros((len(times), n+1))
        values[:,:n] = pools_values
        values[:, n] = system_values
    
        return values


    ## age ##

    
    def load_pools_and_system_densities_csv(self, filename, ages):
        """Load pool and system age densities from a csv file.

        Attention: It is assumed that the data were saved before with the very 
        same ages, times, and pools.
        Furthermore, it is assumed that the system value always follows the 
        pool values.

        Args:
            filename (str): The csv file from which the data are to be read.
            ages (numpy.array): The ages corresponding to the age indices. 
                What is needed here is in fact only the length of the age grid.

        Returns:
            numpy.ndarray: len(ages) x len(times) x (nr_pools+1) The density 
            values for the pools and the system over the 
            ages-times-(pools+system) grid.
        """
        melted = load_csv(filename)
        n = self.nr_pools
        
        return np.ndarray((len(ages), len(self.times), n+1), 
                            buffer=(melted[:,3]).copy())


    def load_density_csv(self, filename, ages, times=None):
        """Load density values from a csv file.

        Attention: It is assumed that the data were saved before with the very
        same ages, times, and pools.

        Args:
            filename (str): The csv file from which the data are to be read.
            ages (numpy.array): The ages corresponding to the age indices. 
                What is needed here is in fact only the length of the age grid.
            times (numpy.array, optional): The ages corresponding to the time
                indices. 
                What is needed here is in fact only the length of the time grid.
                Defaults to ``None`` and the original times are being used.
        Returns:
            numpy.ndarray: len(ages) x len(times) The density values over the 
            ages-times grid.
        """
        if times is None:
            times = self.times

        melted = load_csv(filename)
        return np.ndarray((len(ages), len(times)), 
                            buffer=(melted[:,2]).copy())

    def load_pools_and_system_value_csv(self, filename):
        """Load pool and system values from a csv file.

        Values could be the mean/median age, for example. One dimension less 
        than a density.

        Attention: It is assumed that the data were saved before with the very
        same ages, times, and pools.
        Furthermore, it is assumed that the system value always follows the 
        pool values.

        Args:
            filename (str): The csv file from which the data are to be read.

        Returns:
            numpy.ndarray: len(times) x (nr_pools+1) The values for the pools 
            and the system over the times-(pools+system) grid.
        """
        melted = load_csv(filename)

        n = self.nr_pools
        values_lst = []
        for pool in range(n):
            indices = melted[:,1] == pool
            values_lst.append(melted[np.ix_(indices),2][0])
        pool_values = np.array(values_lst).transpose()

        indices = melted[:,1] == -1
        system_values = melted[np.ix_(indices),2][0]

        return (pool_values, system_values)


    ##### plotting methods #####

    
    ## solutions ##


    def plot_solutions(self, fig, fontsize = 10):
        """Plot the solution trajectories.
    
        For each trajectory (nr_pools+1) a new subplot is created.
    
        Args:
            fig (Matplotlib figure): The fig to which the subplots are added.
            fontsize (float, optional): Defaults to 10.
    
        Returns:
            None.
            Instead ``fig`` is changed in place.
        """
    #fixme:
    # since time units and units are related to those
    # of the other fluxes it would be more consistent
    # to make them a property of SmoothModelRun and use
    # them in the other plots as well

        times = self.times
        n = self.nr_pools
        #soln = self.solve_old()
        soln = self.solve()


        def make_ax_nice(ax, title):
            ax.set_title(title, fontsize = fontsize)
            ax.set_xlabel(self._add_time_unit(latex(self.model.time_symbol)), 
                            fontsize=fontsize)
            ax.set_ylabel(self._add_content_unit('content'), fontsize=fontsize)
            ax.set_xlim(times[0], times[-1])
            ax.set_ylim(ax.get_ylim()[0]*0.9, ax.get_ylim()[1]*1.1)
        

        ax = fig.add_subplot(n+1, 1, 1)
        ax.plot(times, soln.sum(1))
        make_ax_nice(ax, 'System')

        for pool in range(n):
            ax = fig.add_subplot(n+1, 1, 2+pool)
            ax.plot(times, soln[:,pool])
            make_ax_nice(
                ax, "$" + latex(self.model.state_variables[pool]) + "$")

        fig.tight_layout()
   
 
    def plot_phase_plane(self, ax, i, j, fontsize = 10):
        """Plot one single phase plane.

        Args:
            ax (Matplotlib axis): The axis onto which the phase plane is 
                plotted.
            i, j (int): The numbers of the pools for which the phase plane is 
                plotted.
            fontsize (float, optional): Defaults to 10.

        Returns:
            None.
            Instead ``ax`` is changed in place.
        """
        times = self.times
        #soln = self.solve_old()
        soln = self.solve()
        ax.plot(soln[:, i], soln[:, j])
        
        x0 = soln[0, i]
        y0 = soln[0, j]
        ax.scatter([x0],[y0], s=60)

        x1 = soln[[len(times)//2-1], i][0]
        y1 = soln[[len(times)//2-1], j][0]
        x2 = soln[[len(times)//2+1], i][0]
        y2 = soln[[len(times)//2+1], j][0]
        ax.add_patch(mpatches.FancyArrowPatch((x1,y1), (x2,y2), 
                    arrowstyle='simple', mutation_scale=20, alpha=1))

        ax.set_xlabel(self._add_content_unit(
            "$"+latex(sympify(self.model.state_variables[i]))+"$"), fontsize=fontsize)
        ax.set_ylabel(self._add_content_unit(
            "$"+latex(sympify(self.model.state_variables[j]))+"$"), fontsize=fontsize)


    def plot_phase_planes(self, fig, fontsize = 10):
        """Plot all phase planes.

        For each (i,j)-phase plane a new subplot is added.

        Args:
            fig (Matplotlib figure): The fig to which the subplots are added.
            fontsize (float, optional): Defaults to 10.
    
        Returns:
            None.
            Instead ``fig`` is changed in place.
        """
        n = self.nr_pools
        
        if n>=2:
#            planes = [(i,j) for i in range(n) for j in range(i)]
#            rows, cols = arrange_subplots(len(planes))
            k = 0
            for i in range(n):
                for j in range(n):
                    k += 1
                    if i > j:
                        ax = fig.add_subplot(n, n, k)
                        self.plot_phase_plane(ax, i, j, fontsize)
                        ax.get_xaxis().set_ticks([])
                        ax.get_yaxis().set_ticks([])

            fig.tight_layout()
    

    ## fluxes ##
    

    def plot_internal_fluxes(self, fig, fontsize = 10):
        """Plot all internal fluxes.

        For each internal flux a new subplot is added.

        Args:
            fig (Matplotlib figure): The fig to which the subplots are added.
            fontsize (float, optional): Defaults to 10.
    
        Returns:
            None.
            Instead ``fig`` is changed in place.
        """
        internal_flux_funcs = self.internal_flux_funcs()
        n = len(internal_flux_funcs.keys())
        times = self.times
        #n=self.nr_pools
        i = 1
        for key, value in internal_flux_funcs.items():
            ax = fig.add_subplot(n,1,i)
            ax.plot(times, [internal_flux_funcs[key](t) for t in times])
    
            ax.set_title(
                'Flux from $' 
                + latex(self.model.state_variables[key[0]]) 
                + '$ to $'
                + latex(self.model.state_variables[key[1]]) 
                + '$',
                fontsize=fontsize)
            ax.set_xlabel(self._add_time_unit(
                '$' + latex(self.model.time_symbol) + '$'), fontsize=fontsize)
            ax.set_ylabel(self._add_flux_unit('flux'), fontsize=fontsize)
            i += 1

        fig.tight_layout()


    def plot_external_output_fluxes(self, fig, fontsize = 10):
        """Plot all external output fluxes.

        For each external output flux a new subplot is added.

        Args:
            fig (Matplotlib figure): The fig to which the subplots are added.
            fontsize (float, optional): Defaults to 10.
    
        Returns:
            None.
            Instead ``fig`` is changed in place.
        """
        times = self.times
        output_flux_funcs = self.output_flux_funcs()
        n = len(output_flux_funcs.keys())
        
        i = 1
        for key, value in output_flux_funcs.items():
            ax = fig.add_subplot(n,1,i)
            ax.plot(times, [output_flux_funcs[key](t) for t in times])
            ax.set_title(
                'External outflux from $' 
                + latex(self.model.state_variables[key]) 
                + '$', 
                fontsize=fontsize)
            ax.set_xlabel(
                self._add_time_unit('$' + latex(self.model.time_symbol) + '$'), 
                fontsize=fontsize)
            ax.set_ylabel(self._add_flux_unit('flux'), fontsize=fontsize)
            i += 1

        fig.tight_layout()
                
    
    def plot_external_input_fluxes(self, fig, fontsize = 10):
        """Plot all external inpput fluxes.

        For each external input flux a new subplot is added.

        Args:
            fig (Matplotlib figure): The fig to which the subplots are added.
            fontsize (float, optional): Defaults to 10.
    
        Returns:
            None.
            Instead ``fig`` is changed in place.
        """
        times = self.times
        input_flux_funcs = self.external_input_flux_funcs()
        n = len(input_flux_funcs.keys())
        i = 1
        for key, value in input_flux_funcs.items():
            ax = fig.add_subplot(n,1,i)
            ax.plot(times, [input_flux_funcs[key](t) for t in times])
            ax.set_title(
                'External influx to $' 
                + latex(self.model.state_variables[key]) 
                + '$', 
                fontsize=fontsize)
            ax.set_xlabel(
                self._add_time_unit('$' + latex(self.model.time_symbol) + '$'), 
                fontsize=fontsize)
            ax.set_ylabel(self._add_flux_unit('flux'), fontsize=fontsize)
            i += 1

        fig.tight_layout()


    # means # 


    def plot_mean_ages(self, fig, start_mean_ages):
        """Plot the time evolution of the mean ages for all pools and the 
        system.

        For each pool and the system a separate subplot is created.

        Args:
            fig (Matplotlib figure): The fig to which the subplots are added.
            start_mean_ages (numpy.array): Contains the start mean ages of the 
                pools at time :math:`t_0`.
    
        Returns:
            None.
            Instead ``fig`` is changed in place.
        """
        times = self.times
        n = self.nr_pools
        start_age_moments = np.ndarray(
                                (1,n), np.float, np.array(start_mean_ages))
        time_symbol = self.model.time_symbol
        states = self.model.state_variables

        ma_vector = self.age_moment_vector(1, start_age_moments)
        sma = self.system_age_moment(1, start_age_moments)

        def make_ax_nice(ax, title):
            ax.set_title(title)
            ax.set_xlabel(self._add_time_unit("$" + latex(time_symbol) + "$"))
            ax.set_ylabel(self._add_time_unit("mean age"))

            ax.set_xlim([times[0], times[-1]])

        ax = fig.add_subplot(n+1, 1, 1)
        ax.plot(times, sma)
        make_ax_nice(ax, "System")

        for i in range(n):
            ax = fig.add_subplot(n+1, 1, 2+i)
            ax.plot(times, ma_vector[:,i])
            make_ax_nice(ax, "$" + latex(states[i]) + "$")
                
        fig.tight_layout()


    def plot_mean_backward_transit_time(self, ax, start_mean_ages):
        """Plot the time evolution of the mean backward transit time.

        For each pool and the system a separate subplot is created.

        Args:
            ax (Matplotlib axis): The ax onto which the plot is done.
            start_mean_ages (numpy.array): Contains the start mean ages of the 
                pools at time :math:`t_0`.
    
        Returns:
            None.
            Instead ``ax`` is changed in place.
        """
        times = self.times
        n = self.nr_pools
        start_age_moments = np.ndarray(
                                (1,n), np.float, np.array(start_mean_ages))
        time_symbol = self.model.time_symbol
        tr_val = self.backward_transit_time_moment(1, start_age_moments)
        ax.plot(times, tr_val)
        
        ax.set_title("Mean backward transit time")

        ax.set_xlabel(self._add_time_unit("$" + latex(time_symbol) + "$"))
        ax.set_ylabel(self._add_time_unit("mean BTT"))

        ax.set_xlim([times[0], times[-1]])


    ## densities ##


    # age #

    
    def add_line_to_density_plot_plotly(self, fig, data, color, name, 
            time_stride=1, width=5, on_surface=True, bottom=True, 
            legend_on_surface=False, legend_bottom=False):
        """Add a line to an existing Plotly density plot.

        Args:
            fig (Plotly figure): Contains already a density plot to which the 
                new line is added.
            data (numpy.array len(times)): The age data of the new line.
            color (#RRGGBB): The color of the new line.
            name (str): The name of the new line for the legend.
            time_stride (int, optional): Coarsity of the plot in the time 
                direction to save memory. 
                Defaults to 1 meaning that all times are plotted and no memory 
                is saved.
            width (int, optional): Width of the new line. Defaults to 5.
            on_surface (bool, optional): If True, a new line with the given age
                data is plotted on top of the given density.
                Defaults to True.
            bottom (bool optional): If True, a new line with the given age data
                is plotted in the xy-plane. 
                Defaults to True.
            legend_on_surface (bool, optional): If True, the line on the surface
                is mentioned in the legend.
                Has no effect if on_surface is False.
                Defaults to False.
            legend_bottom (bool, optional): If True, the line in the xy-plane is
                mentioned in the legend.
                Has no effect if bottom is False.
                Defaults to False.

        Returns:
            None.
            Instead ``fig`` is changed in place.
        """
        times = self.times
        strided_data = stride(data, time_stride)
        strided_times = stride(times, time_stride)
    
        if bottom:
            #trace_bottom = go.Scatter3d(
            fig.add_scatter3d(
            name=name,
            x=-strided_times, y=strided_data, z=0*strided_times,
            mode = 'lines',
            line=dict(
                color=color,
                width=width
                ),
            showlegend = legend_bottom
            )
            #fig['data'] += [trace_bottom]

        if on_surface:
            # compute the density values on the surface
            #strided_times = -fig['data'][0]['x']
            strided_ages = fig['data'][0]['y']
            density_data = fig['data'][0]['z']

            strided_z = []
            for ti in range(len(strided_times)):
                time = strided_times[ti]
                age = strided_data[ti]

                if ((np.isnan(age)) or (age < strided_ages[0]) or 
                    (age > strided_ages[-1])):
                    strided_z.append(np.nan)
                else:
                    ti_lower = strided_times.searchsorted(time)-1
                    ti_upper = (ti_lower+1 if ti_lower+1<len(strided_times) 
                                            else ti_lower)
                    time_lower = strided_times[ti_lower]
                    time_upper = strided_times[ti_upper]
    
                    ai_lower = strided_ages.searchsorted(age)-1
                    ai_upper = (ai_lower+1 if ai_lower+1<len(strided_ages) 
                                            else ai_lower)
                    age_lower = strided_ages[ai_lower]
                    age_upper = strided_ages[ai_upper]
    
                    bl_density_value = density_data[ai_lower, ti_lower]
                    br_density_value = density_data[ai_lower, ti_upper]
                    bottom_density_value = (bl_density_value + (time-time_lower)
                                            /(time_upper-time_lower)*
                                            (br_density_value-bl_density_value))
    
                    tl_density_value = density_data[ai_upper, ti_lower]
                    tr_density_value = density_data[ai_upper, ti_upper]
                    top_density_value = (tl_density_value + (time-time_lower)/
                                            (time_upper-time_lower)*
                                            (tr_density_value-tl_density_value))
    
                    density_value = (bottom_density_value + 
                                    (age-age_lower)/(age_upper-age_lower)*
                                    (top_density_value-bottom_density_value))
                    strided_z.append(density_value)

            #trace_on_surface = go.Scatter3d(
            #    name=name,
            #    x=-strided_times, y=strided_data, z=strided_z,
            #    mode = 'lines',
            #    line=dict(
            #        color=color,
            #        width=width
            #        ),
            #    showlegend = legend_on_surface
            #)
            #fig['data'] += [trace_on_surface]
            fig.add_scatter3d(
                name=name,
                x=-strided_times, y=strided_data, z=strided_z,
                mode = 'lines',
                line=dict(
                    color=color,
                    width=width
                    ),
                showlegend = legend_on_surface
            )

    def plot_3d_density_plotly(self, title, density_data, ages, 
            age_stride=1, time_stride=1):
        """Create a 3-dimendional density plot with Plotly.

        The colors are created such that they are constant along the age-time 
        diagonal.
        Thus, equal colors mark equal entry time into the system.

        Args:
            title (str): The title of the new figure.
            density_data (numpy.ndarray len(ages) x len(times)): 
                The density data to be plotted.
            age_stride (int, optional): Coarsity of the plot in the age 
                direction to save memory. 
                Defaults to 1 meaning that all times are plotted and no memory
                is saved.
            time_stride (int, optional): Coarsity of the plot in the time 
                direction to save memory. 
                Defaults to 1 meaning that all times are plotted and no memory 
                is saved.

        Returns:
            Plotly figure.
        """
        data, layout = self._density_plot_plotly(
                                density_data, ages, age_stride, time_stride)
        layout['title'] = title
        fig = go.Figure(data=data, layout=layout)
        
        return fig

#    def add_equilibrium_surface_plotly(self, fig, opacity=0.7, index=0):
#        """
#        The function has been renamed since 
#            1. It is not certain that the system has an equilibrium at all. 
#            2. The age distribution at the beginning of a model run does not have to 
#               represent an equilibrium age distribution
#               (even if the system was in equilibrium at t0 in the sense that the pool contents do not change any more the age distribution still could.)
#               
#            please call add_constant_age_distribution_surface_plotly instead! 
#        """
#        txt=self.add_equilibrium_surface_plotly.__doc__
#        deprecation_warning(txt)
#        self.add_constant_age_distribution_surface_plotly(fig, opacity, index)

    def add_constant_age_distribution_surface_plotly(self, fig, opacity=0.7, index=0):
        """Add a grey and transparent density surface to an existing
        Plotly density plot.

        If index is not specified it is assumed to be 0 and the values correspond to the first time in the times porperty of the model run (the age distribution at the beginning) 
        and repeated for all times.
        The plotted surface represents an age distribution that is constant in time.
        It is intended to increase the visibility of changes in the age distribution with time.
        Note that this constant age distribution does NOT necessarily correspond to a 
        possible (constant) development of the system. 
        This would only be true if the system was in equilibrium and the age distribution 
        was the equilibrium age distribution.
        While this special case is a very interesting application this function does not 
        assertain that such an equlibrium situation is even possible.

        Args:
            fig (Plotly figure): The existing density plot to which the 
                surface is added.
            opacity (between 0 and 1, optional): The opacity of the new surface.
                Defaults to 0.9.
                Unfortunately, the opacity option does not seem to work 
                properly.
            index (int, optional): The time index from which the age distribution 
                data is taken.
                Defaults to 0 such that the constant distribution is computed  at time :math:`t_0`.
    
        Returns:
            None.
            Instead ``fig`` is changed in place.
        """
        data = fig['data'][0]
        x = data['x']
        y = data['y']
        z = data['z'].copy()
        for ti in range(z.shape[1]):
            z[:,ti] = z[:,index]
        #eq_surface_data = go.Surface(
        fig.add_surface(
            x=x, 
            y=y, 
            z=z, 
            showscale=False,
            opacity = opacity,
            surfacecolor=np.zeros_like(z))
        #fig['data'].append(eq_surface_data)


    ##### cumulative distribution methods #####


    def cumulative_pool_age_distributions_single_value(self, 
            start_age_densities=None, F0=None):
        """Return a function for the cumulative pool age distributions.

        Args:
            start_age_densities (Python function, optional): A function of age 
                that returns a numpy.array containing the masses with the given 
                age at time :math:`t_0`. Defaults to None.
            F0 (Python function): A function of age that returns a numpy.array 
                containing the masses with age less than or equal to the age at 
                time :math:`t_0`. Defaults to None.

        Raises:
            Error: If both ``start_age_densities`` and ``F0`` are ``None``. 
                One must be given.
                It is fastest to provide ``F0``, otherwise ``F0`` will be 
                computed by numerical integration of ``start_age_densities``.

        Returns:
            Python function ``F_sv``: ``F_sv(a,t)`` is the vector of pool 
            masses (``numpy.array``) with age less than or equal to ``a`` at 
            time ``t``.
        """
        n = self.nr_pools
        #soln = self.solve_old()
        soln = self.solve()
        if soln[0,:].sum() == 0:
            start_age_densities = lambda a: np.zeros((n,))

        if F0 is None and start_age_densities is None:
            raise(Error('Either F0 or start_age_densities must be given.'))

        times = self.times
        t0 = times[0]
        #sol_funcs = self.sol_funcs()
        #sol_funcs_array = lambda t: np.array([sol_funcs[pool](t) 
        #                                           for pool in range(n)])
        #sol_funcs_array = self.solve_single_value_old()
        sol_funcs_array = self.solve_func()

        if F0 is None:
            p0 = start_age_densities
            F0 = lambda a: np.array([quad(lambda s: p0(s)[pool], 0, a)[0] 
                                        for pool in range(n)])

        Phi = self._state_transition_operator

        def G_sv(a, t):
            if a < t-t0: return np.zeros((n,))
            #print(t, t0, a-(t-t0))
            res = Phi(t, t0, F0(a-(t-t0)))
            c = hasattr(self, '_state_transition_operator_cache')
            #print(c, 'G', res, t, t0, a, a-(t-t0))
            return res


        def H_sv(a, t):
            # count everything from beginning?
            if a >= t-t0: a = t-t0

            # mass at time t
            #x_t_old = np.array([sol_funcs[pool](t) for pool in range(n)])
            x_t = sol_funcs_array(t)
            # mass at time t-a
            #x_tma_old = [np.float(sol_funcs[pool](t-a)) for pool in range(n)]
            x_tma = sol_funcs_array(t-a)
            # what remains from x_tma at time t
            m = Phi(t, t-a, x_tma)
            c = hasattr(self, '_state_transition_operator_cache')
            #print(c, 'H', m, t, a, x_t, x_tma)
            # difference is not older than t-a
            res = x_t-m
            # cut off accidental negative values
            return np.maximum(res, np.zeros(res.shape))

        def F(a, t):
            res = G_sv(a,t) + H_sv(a,t)
            #print(a, t, res)
            #print('G', G_sv(a,t), 'H', H_sv(a,t))
            return res

        return F

    def cumulative_system_age_distribution_single_value(self, 
            start_age_densities=None, F0=None):
        """Return a function for the cumulative system age distribution.

        Args:
            start_age_densities (Python function, optional): A function of age 
                that returns a numpy.array containing the masses with the given 
                age at time :math:`t_0`. Defaults to None.
            F0 (Python function): A function of age that returns a numpy.array 
                containing the masses with age less than or equal to the age at 
                time :math:`t_0`. Defaults to None.

        Raises:
            Error: If both ``start_age_densities`` and ``F0`` are None. 
                One must be given.
                It is fastest to provide ``F0``, otherwise ``F0`` will be 
                computed by numerical integration of ``start_age_densities``.

        Returns:
            Python function ``F_sv``: ``F_sv(a, t)`` is the mass in the system 
            with age less than or equal to ``a`` at time ``t``.
        """
        n = self.nr_pools
        #soln = self.solve_old()
        soln = self.solve()
        if soln[0,:].sum() == 0:
            start_age_densities = lambda a: np.zeros((n,))
        
        if F0 is None and start_age_densities is None:
            raise(Error('Either F0 or start_age_densities must be given.'))

        F_sv = self.cumulative_pool_age_distributions_single_value(
                start_age_densities=start_age_densities, F0=F0)
        
        return lambda a, t: F_sv(a,t).sum()

    #fixme: test
    def cumulative_backward_transit_time_distribution_single_value(self,
            start_age_densities=None, F0=None):
        """Return a function for the cumulative backward transit time 
        distribution.

        Args:
            start_age_densities (Python function, optional): A function of age
                that returns a numpy.array containing the masses with the given
                age at time :math:`t_0`. Defaults to None.
            F0 (Python function): A function of age that returns a numpy.array
                containing the masses with age less than or equal to the age at
                time :math:`t_0`. Defaults to None.

        Raises:
            Error: If both ``start_age_densities`` and ``F0`` are ``None``. 
                One must be given.
                It is fastest to provide ``F0``, otherwise ``F0`` will be 
                computed by numerical integration of ``start_age_densities``.

        Returns:
            Python function ``F_sv``: ``F_sv(a, t)`` is the mass leaving the 
            system at time ``t`` with age less than or equal to ``a``.
        """
        if F0 is None and start_age_densities is None:
            raise(Error('Either F0 or start_age_densities must be given.'))

        F_sv = self.cumulative_pool_age_distributions_single_value(
                start_age_densities=start_age_densities, F0=F0)
        rho = self.output_rate_vector_at_t

        def F_btt_sv(a, t):
            res = (rho(t)*F_sv(a, t)).sum()
            #print(a, t, res)
            return res

        return F_btt_sv

    #fixme: test
    def cumulative_forward_transit_time_distribution_single_value_func(
            self, cut_off=True):
        """Return a function for the cumulative forward transit time 
        distribution.

        Args:
            cut_off (bool, optional): If ``True``, no density values are going 
                to be computed after the end of the time grid, instead 
                ``numpy.nan`` will be returned. 
                Defaults to ``True``.
                ``False`` might lead to unexpected behavior.

        Returns:
            Python function ``F_sv``: ``F_sv(a, t)`` is the mass leaving the 
            system at time ``t+a`` with age less than or equal to ``a``.
        """
        times = self.times
        t_max = times[-1]
        Phi = self._state_transition_operator
        u_func = self.external_input_vector_func()

        def F_ftt_sv(a, t):
            #print(a, t, a+t>t_max)
            if cut_off and a+t>t_max: return np.nan
            u = u_func(t)
            res = u.sum() - Phi(t+a, t, u).sum()
            #print(a, t, u, res)
            return res

        return F_ftt_sv


    ##### quantiles #####


    def pool_age_distributions_quantiles(
            self,
            quantile,
            start_values        = None, 
            start_age_densities = None,
            F0                  = None,
            method              = 'brentq',
            tol                 = 1e-8
        ):
        """Return pool age distribution quantiles over the time grid.

        The compuation is done by computing the generalized inverse of the 
        respective cumulative distribution using a nonlinear root search 
        algorithm. Depending on how slowly the cumulative distribution can be 
        computed, this can take quite some time.

        Args:
            quantile (between 0 and 1): The relative share of mass that is 
                considered to be left of the computed value. A value of ``0.5`` 
                leads to the computation of the median of the distribution.
            start_values (numpy.ndarray, len(times) x nr_pools, optional): 
                For each pool an array over the time grid of start values for 
                the nonlinear search.
                Good values are slighty greater than the solution values.
                Defaults to an array of zeros for each pool
            start_age_densities (Python function, optional): A function of age 
                that returns a ``numpy.array`` containing the masses with the 
                given age at time :math:`t_0`. 
                Defaults to ``None``.
            F0 (Python function): A function of age that returns a 
                ``numpy.array`` containing the masses with age less than or
                equal to the age at time :math:`t_0`. 
                Defaults to ``None``.
            method (str): The method that is used for finding the roots of a 
                nonlinear function. Either 'brentq' or 'newton'. 
                Defaults to 'brentq'.
            tol (float): The tolerance used in the numerical root search 
                algorithm. A low tolerance decreases the computation speed 
                tremendously, so a value of ``1e-01`` might already be fine. 
                Defaults to ``1e-08``.

        Raises:
            Error: If both ``start_age_densities`` and ``F0`` are ``None``. 
                One must be given.
                It is fastest to provide ``F0``, otherwise ``F0`` will be 
                computed by numerical integration of ``start_age_densities``.

        Returns:
            numpy.ndarray: (len(times) x nr_pools)
            The computed quantile values over the time-pool grid.
        """
        n = self.nr_pools
        soln = self.solve()
        if soln[0,:].sum() == 0:
            start_age_densities = lambda a: np.zeros((n,))

        if F0 is None and start_age_densities is None:
            raise(Error('Either F0 or start_age_densities must be given.'))

        times = self.times

        if start_values is None:
            start_values = np.ones((len(times), n))

        F_sv = self.cumulative_pool_age_distributions_single_value(
            start_age_densities = start_age_densities,
            F0                  = F0
        )

        res = []
        for pool in range(n):
            print('Pool:', pool)
            F_sv_pool = lambda a, t: F_sv(a,t)[pool]
            res.append(
                self.distribution_quantiles(
                    quantile,
                    F_sv_pool,
                    norm_consts  = soln[:,pool],
                    start_values = start_values[:,pool],
                    method       = method,
                    tol          = tol
                )
        )

        return np.array(res).transpose()
    
    def system_age_distribution_quantiles(
            self,
            quantile,
            start_values        = None, 
            start_age_densities = None,
            F0                  = None,
            method              = 'brentq',
            tol                 = 1e-8
        ):
        """Return system age distribution quantiles over the time grid.

        The compuation is done by computing the generalized inverse of the 
        respective cumulative distribution using a nonlinear root search 
        algorithm. Depending on how slowly the cumulative distribution can be 
        computed, this can take quite some time.

        Args:
            quantile (between 0 and 1): The relative share of mass that is 
                considered to be left of the computed value. A value of ``0.5`` 
                leads to the computation of the median of the distribution.
            start_values (numpy.array, optional): An array over the time grid of
                start values for the nonlinear search.
                Good values are slighty greater than the solution values.
                Must have the same length as ``times``.
                Defaults to an array of zeros.
            start_age_densities (Python function, optional): A function of age 
                that returns a ``numpy.array`` containing the masses with the 
                given age at time :math:`t_0`. 
                Defaults to ``None``.
            F0 (Python function): A function of age that returns a 
                ``numpy.array`` containing the masses with age less than or 
                equal to the age at time :math:`t_0`. 
                Defaults to ``None``.
            method (str): The method that is used for finding the roots of a 
                nonlinear function. Either 'brentq' or 'newton'. 
                Defaults to 'brentq'.
            tol (float): The tolerance used in the numerical root search 
                algorithm. A low tolerance decreases the computation speed 
                tremendously, so a value of ``1e-01`` might already be fide. 
                Defaults to ``1e-08``.

        Raises:
            Error: If both ``start_age_densities`` and ``F0`` are ``None``. 
                One must be given.
                It is fastest to provide ``F0``, otherwise ``F0`` will be 
                computed by numerical integration of ``start_age_densities``.

        Returns:
            numpy.array: The computed quantile values over the time grid.
        """
        n = self.nr_pools
        soln = self.solve()
        if soln[0,:].sum() == 0:
            start_age_densities = lambda a: np.zeros((n,))

        if F0 is None and start_age_densities is None:
            raise(Error('Either F0 or start_age_densities must be given.'))
        
        F_sv = self.cumulative_system_age_distribution_single_value(
            start_age_densities = start_age_densities,
            F0                  = F0
        )
        #soln = self.solve_old()
        start_age_moments = self.moments_from_densities(1, start_age_densities)
        
        if start_values is None: 
            start_values = self.system_age_moment(1, start_age_moments)

        a_star = self.distribution_quantiles(
            quantile, 
            F_sv, 
            norm_consts  = soln.sum(1), 
            start_values = start_values, 
            method       = method,
            tol          = tol
        )

        return a_star


    def distribution_quantiles(self, quantile, F_sv, norm_consts=None, 
                start_values=None, times=None, method='brentq', tol=1e-8):
        """Return distribution quantiles over the time grid of a given 
        distribution.

        The compuation is done by computing the generalized inverse of the 
        respective cumulative distribution using a nonlinear root search 
        algorithm. Depending on how slowly the cumulative distribution can be 
        computed, this can take quite some time.

        Args:
            quantile (between 0 and 1): The relative share of mass that is 
                considered to be left of the computed value. A value of ``0.5`` 
                leads to the computation of the median of the distribution.
            F_sv (Python function): A function of age ``a`` and time ``t`` that 
                returns the mass that is of age less than or equal to ``a`` at 
                time ``t``.
            norm_consts (numpy.array, optional): An array over the time grid of
                total masses over all ages. 
                Defaults to an array of ones assuming given probability 
                distributions.
            start_values (numpy.array, optional): An array over the time grid of
                start values for the nonlinear search.
                Good values are slighty greater than the solution values.
                Must have the same length as ``times``.
                Defaults to an array of zeros.
            times (numpy.array, optional): Time grid on which to compute the
                quantiles.
                Defaults to ``None`` in which case the orignal time grid 
                is used.
            method (str): The method that is used for finding the roots of a 
                nonlinear function. Either 'brentq' or 'newton'. 
                Defaults to 'brentq'.
            tol (float): The tolerance used in the numerical root search 
                algorithm. A low tolerance decreases the computation speed 
                tremendously, so a value of ``1e-01`` might already be fine. 
                Defaults to ``1e-08``.

        Returns:
            numpy.array: The computed quantile values over the time grid.
        """
        if times is None:
            times = self.times
        
        if start_values is None:
            start_values = np.zeros((times,))

        if norm_consts is None:
            norm_consts = np.ones((times,))

        def quantile_at_ti(ti):
            #print('ti', ti)
            if norm_consts[ti] == 0: return np.nan

            def g(a):
                if np.isnan(a): return np.nan
                res =  quantile*norm_consts[ti] - F_sv(a, times[ti])
                #print('a:', a,'t', times[ti], 'g(a):', res, 'nc', 
                #           norm_consts[ti], 'F_sv', F_sv(a, times[ti]))
                return res

            start_age = start_values[ti]
            
            if method == 'newton': 
                a_star = newton(g, start_age, maxiter=500, tol=tol)
            if method == 'brentq': 
                a_star = generalized_inverse_CDF(lambda a: F_sv(a, times[ti]), 
                                                 quantile*norm_consts[ti], 
                                                 start_dist=start_age, 
                                                 tol=tol)

            return a_star

        m = len(times)
        #q_lst = [quantile_at_ti(ti) for ti in range(len(times))]

        q_lst = []
        for ti in tqdm(range(len(times))):
            q_lst.append(quantile_at_ti(ti))

        return np.array(q_lst)

    @staticmethod
    def distribution_quantile(quantile, F, 
            norm_const=None, start_value=None, method='brentq', tol=1e-8):
        """Return distribution quantile (one single value) of a given distribution.

        The compuation is done by computing the generalized inverse of the 
        respective cumulative distribution using a nonlinear root search 
        algorithm.

        Args:
            quantile (between 0 and 1): The relative share of mass that is 
                considered to be left of the computed value. A value of ``0.5`` 
                leads to the computation of the median of the distribution.
            F (Python function): A function of age ``a`` that 
                returns the mass that is of age less than or equal to ``a``.
            norm_const (numpy.array, optional): The amount of total mass of the
                distribution. 
                Defaults to one assuming a given probability distribution.
            start_value (float, optional): A start value for the nonlinear
                search.
                A good values is slighty greater than the solution value.
                Defaults to zero.
            method (str): The method that is used for finding the roots of a 
                nonlinear function. Either 'brentq' or 'newton'. 
                Defaults to 'brentq'.
            tol (float): The tolerance used in the numerical root search 
                algorithm. A low tolerance decreases the computation speed 
                tremendously, so a value of ``1e-01`` might already be fine. 
                Defaults to ``1e-08``.

        Returns:
            float: The computed quantile value of the distribution.
        """
        if start_value is None:
            start_value = 0

        if norm_const is None:
            norm_const = 1

        def quantile_f():
            if norm_const == 0: return np.nan

            def g(a):
                if np.isnan(a): return np.nan
                res =  quantile*norm_const - F(a)
                #print('a:', a,'t', times[ti], 'g(a):', res, 'nc', 
                #           norm_consts[ti], 'F', F(a, times[ti]))
                return res

            if method == 'newton': 
                a_star = newton(g, start_value, maxiter=500, tol=tol)
            if method == 'brentq': 
                a_star = generalized_inverse_CDF(
                           lambda a: F(a), 
                           quantile*norm_const, 
                           start_dist=start_value, 
                           tol=tol
                        )

            return a_star

        q = quantile_f()
        return q


    ## by ode ##


    def pool_age_distributions_quantiles_by_ode(self, quantile, 
            start_age_densities, F0=None, check_time_indices=None, **kwargs):
        """Return pool age distribution quantiles over the time grid.

        The compuation is done by solving an ODE for each pool as soon as the 
        pool is nonempty.
        The initial value is obtained by computing the generalized inverse of 
        the pool age distribution by a numerical root search algorithm.

        Args:
            quantile (between 0 and 1): The relative share of mass that is 
                considered to be left of the computed value. A value of ``0.5`` 
                leads to the computation of the median of the distribution.
            start_age_densities (Python function): A function of age 
                that returns a ``numpy.array`` containing the masses with the 
                given age at time :math:`t_0`. 
            F0 (Python function, optional): A function of age that returns a 
                ``numpy.array`` containing the masses with age less than or 
                equal to the age at time :math:`t_0`. 
                Defaults to ``None``.
                It is fastest to provide ``F0``, otherwise ``F0`` will be 
                computed by numerical integration of ``start_age_densities``.
            check_time_indices (numpy.array, optional): Indices of the tiime
                 grid on which the ODE result are checked against an explicit
                 solution computed by the pseudo-inverse of the cumulative 
                distribution function.
                Defaults to ``None`` in which case no check is performed.
            kwargs: Passed to the ``solve_ivp``, e.g., ``method`` 
                or ``max_step``.

        Returns:
            numpy.ndarray: (len(times) x nr_pools) The computed quantile values 
            over the time-pool grid.
        """
        res = []
        for pool in range(self.nr_pools):
            print('Pool:', pool)
            res.append(
                self.pool_age_distribution_quantiles_pool_by_ode(
                    quantile, 
                    pool,
                    start_age_densities,
                    F0=F0,
                    check_time_indices=check_time_indices,
                    **kwargs
                )
            )

        return np.array(res).transpose()
    
    def x_solve_func_skew(self):
        block_ode,x_block_name,phi_block_name=self._x_phi_block_ode()
        return x_tmax(
            self.times[0],
            self.times[-1],
            block_ode,
            tuple(self.start_values),
            x_block_name,
            phi_block_name
        )


    def pool_age_distribution_quantiles_pool_by_ode(self, quantile, pool, 
            start_age_densities, F0=None, check_time_indices=None, **kwargs):
        """Return pool age distribution quantile over the time grid for one 
        single pool.

        The compuation is done by solving an ODE as soon as the pool is 
        nonempty.
        The initial value is obtained by computing the generalized inverse of 
        the pool age distribution by a numerical root search algorithm.

        Args:
            quantile (between 0 and 1): The relative share of mass that is 
                considered to be left of the computed value. A value of ``0.5`` 
                leads to the computation of the median of the distribution.
            pool (int): The number of the pool for which the age quantile is to 
                be computed.
            start_age_densities (Python function): A function of age 
                that returns a ``numpy.array`` containing the masses with the 
                given age at time :math:`t_0`. 
            F0 (Python function, optional): A function of age that returns a 
                ``numpy.array`` containing the masses with age less than or 
                equal to the age at time :math:`t_0`. 
                It is fastest to provide ``F0``, otherwise ``F0`` will be 
                computed by numerical integration of ``start_age_densities``.
                Defaults to ``None``.
            check_time_indices (numpy.array, optional): Indices of the tiime
                 grid on which the ODE result are checked against an explicit
                 solution computed by the pseudo-inverse of the cumulative 
                distribution function.
                Defaults to ``None`` in which case no check is performed.
            kwargs: Passed to the ``solve_ivp``, e.g., ``method`` 
                    or ``max_step``.

        Raises:
            Error: If ``start_age_densities`` is ``None``. 

        Returns:
            numpy.ndarray: (len(times)) The computed quantile values over the 
            time grid.
        """
        #soln = self.solve_old()
        vec_sol_func = self.x_solve_func_skew()
        soln = vec_sol_func(self.times)
        empty = soln[0, pool] == 0

        if not empty and start_age_densities is None:
            raise(Error('start_age_densities must be given'))
        
        times = self.times
        n = self.nr_pools

        if not empty and F0 is None:
            p0 = start_age_densities
            F0 = lambda a: np.array([quad(lambda s: p0(s)[i], 0, a)[0] 
                                        for i in range(n)])
        
        p = self.pool_age_densities_single_value(start_age_densities)
        u = self.external_input_vector_func()
        F = self.cumulative_pool_age_distributions_single_value(
                start_age_densities=start_age_densities, F0=F0)
        #sol_funcs = self.solve_single_value_old()
        #vec_sol_func = self.solve_func()

        # find last time index such that the pool is empty --> ti
        ti = len(times)-1
        content = soln[ti, pool]
        while (content > 0) and (ti > 0): 
            ti = ti-1
            content = soln[ti, pool]
        
        if content == 0: ti += 1
        if (ti == len(times)): return np.nan*np.ones((len(times),))
  
        if ti == 0:
            sv = generalized_inverse_CDF(lambda a: F0(a)[pool], 
                                         quantile*self.start_values[pool])
        else:
            #if start_age_densities is None:
            #    raise(Error('Cannot start delayed quantile computation,'
            #                    'since start_age_densities are missing.'))
            CDFs = self.cumulative_pool_age_distributions_single_value(
                        start_age_densities)
            CDF = lambda a: CDFs(a, times[ti])
            sv = generalized_inverse_CDF(lambda a: CDF(a)[pool], 
                                         quantile*soln[ti, pool])

        times = times[ti:]

        t_max = times[-1]
        t_min = times[0]
        pb = tqdm(total = t_max-t_min)

        global last_t, last_res
        last_t = -1
        last_res = -1.0

        def rhs(y, t_val):
            y = np.float(y)
            global last_t, last_res
            
            t_val = min(t_val, t_max)
            
            # rhs will be called twice with the same value apparently,  
            # we can use this to speed it up
            if t_val == last_t: return last_res
            #print('y', y, 't', t_val)

            if (t_val <= t_max) and (t_val-t_min-pb.n > 0):
                #pb.n = t_val-t_min
                #pb.update(0)
                pb.update(t_val-t_min-pb.n)

#            print('Quantile, line 2866')
#            print('y', y, 't', t_val)
        
            p_val = p(y, t_val)[pool]
            u_val = u(t_val)[pool]
            F_vec = F(y, t_val).reshape((n,1))
            x_vec = vec_sol_func(t_val)#.reshape((n,1))
            B = self.B_func(vec_sol_func)(t_val)

#            print('B', B)
#            print('x', x_vec)
#            print('B*x', B.dot(x_vec))
#            print('p', p_val)
#            print('u', u_val)
#            print('F', F_vec)
#            print('B*F', B.dot(F_vec))
#            print(B.dot(F_vec)[pool])
#            print(B.dot(F_vec)[1])

            #if p_val == 0:
                #raise(Error('Division by zero during quantile computation.'))
            #else:
            res = 1 + 1/p_val*(u_val*(quantile-1.0)
                        +quantile*(np.matmul(B,x_vec)[pool])-(np.matmul(B,F_vec)[pool]))
            #print('res', res)
            #print('---')

            last_t = t_val
            last_res = res
            return np.array(res).reshape(1,)

        #short_res = odeint(rhs, sv, times, atol=tol, mxstep=10000)
        rhs2 = lambda t_val, y: rhs(y, t_val)
        short_res = solve_ivp_pwc(
            rhss   = (rhs2,),
            y0     = np.array([sv]).reshape(1,),
            t_span = (times[0], times[-1]),
            t_eval = times,
            **kwargs
        ).y
        short_res = np.rollaxis(short_res, -1, 0)

        pb.close()

        res = np.ndarray((len(self.times),))
        res[:ti] = np.nan
        res[ti:] = short_res.reshape((len(times),))

        if check_time_indices is not None:
            qs_ode = res[check_time_indices]
            qs_pi = np.zeros_like(qs_ode)
            for nr, ct_index in enumerate(check_time_indices):
                qs_pi[nr] = self.__class__.distribution_quantile(
                    quantile,
                    lambda a: F(a, self.times[ct_index])[pool], 
                    norm_const=soln[ct_index, pool],
                    start_value=qs_ode[nr]
                )
                #, method='brentq', tol=1e-8):

            print_quantile_error_statisctics(qs_ode, qs_pi)

        #print(res)
        return res


    def system_age_distribution_quantiles_by_ode(self, quantile, 
            start_age_densities, F0=None, check_time_indices=None, **kwargs):
        """Return system age distribution quantile over the time grid.

        The compuation is done by solving an ODE as soon as the system is 
        nonempty.
        The initial value is obtained by computing the generalized inverse of 
        the system age distribution by a numerical root search algorithm.

        Args:
            quantile (between 0 and 1): The relative share of mass that is 
                considered to be left of the computed value. A value of ``0.5`` 
                leads to the computation of the median of the distribution.
            pool (int): The number of the pool for which the age quantile is to 
                be computed.
            start_age_densities (Python function): A function of age 
                that returns a ``numpy.array`` containing the masses with the 
                given age at time :math:`t_0`. 
            F0 (Python function, optional): A function of age that returns a 
                ``numpy.array`` containing the masses with age less than or 
                equal to the age at time :math:`t_0`. 
                It is fastest to provide ``F0``, otherwise ``F0`` will be 
                computed by numerical integration of ``start_age_densities``.
                Defaults to ``None``.
            check_time_indices (numpy.array, optional): Indices of the tiime
                 grid on which the ODE result are checked against an explicit
                 solution computed by the pseudo-inverse of the cumulative 
                distribution function.
                Defaults to ``None`` in which case no check is performed.
            kwargs: Passed to the ``solve_ivp``, e.g., ``method`` 
                    or ``max_step``.

        Raises:
            Error: If ``start_age_densities`` is ``None``. 

        Returns:
            numpy.ndarray: The computed quantile values over the time grid.
        """
        #soln = self.solve_old()
        #soln = self.solve()
        vec_sol_func = self.x_solve_func_skew()
        soln = vec_sol_func(self.times)
        # check if system is empty at the beginning,
        # if so, then we use 0 as start value, otherwise
        # we need to compute it from F0 (preferably) or start_age_density
        empty = soln[0,:].sum() == 0

        if not empty and start_age_densities is None:
            raise(Error('start_age_densities must be given'))
        
        times = self.times
        original_times = copy(times)
        n = self.nr_pools

        if not empty and F0 is None:
            p0 = start_age_densities
            F0 = lambda a: np.array([quad(lambda s: p0(s)[pool], 0, a)[0] 
                                        for pool in range(n)])
        
        p = self.system_age_density_single_value(start_age_densities)
        u = self.external_input_vector_func()
        F = self.cumulative_pool_age_distributions_single_value(
                start_age_densities=start_age_densities, F0=F0)
        #sol_funcs = self.solve_single_value_old()
        #vec_sol_func = self.solve_func()

        # find last time index such that the system is empty --> ti
        ti = len(times)-1
        content = soln[ti,:]
        while (content.sum() > 0) and (ti > 0): 
            ti = ti-1
            content = soln[ti,:]
        
        if content.sum() == 0: ti += 1
        if (ti == len(times)): return np.nan*np.ones((len(times),))
  
        if ti == 0:
            sv = generalized_inverse_CDF(lambda a: F0(a).sum(), 
                                         quantile*self.start_values.sum())
        else:
            #if start_age_densities is None:
            #    raise(Error('Cannot start delayed quantile computation,'
            #                    'since start_age_Densities are missing.'))
            CDFs = self.cumulative_system_age_distribution_single_value(
                            start_age_densities)
            CDF = lambda a: CDFs(a, times[ti])
            sv = generalized_inverse_CDF(CDF, quantile*soln[ti,:].sum())

        times = times[ti:]

        t_max = times[-1]
        t_min = times[0]
        pb = tqdm(total = t_max-t_min)

        global last_t, last_res
        last_t = -1
        last_res = -1.0

        def rhs(y, t_val):
            y = np.float(y)
            global last_t, last_res
            
            t_val = min(t_val, t_max)

            # rhs will be called twice with the same value apparently,  
            # we can use this to speed it up
            if t_val == last_t: return last_res

            if (t_val <= t_max) and (t_val-t_min-pb.n > 0):
                #pb.n = t_val-t_min
                #pb.update(0)
                pb.update(t_val-t_min-pb.n)

            #pb.update(t_val-t_min, n=0)
            #print()
            #print('y', y, 't', t_val)
        
            p_val = p(y, t_val)
            u_vec = u(t_val)
            F_vec = F(y, t_val).reshape((n,1))
            x_vec = vec_sol_func(t_val)#.reshape((n,1))
            B=self.B_func(vec_sol_func)(t_val)

            #print('B', B)
            #print('x', x_vec)
            #print('B*x', B.dot(x_vec))
            #print('y', y)
            #print('t', t_val)
            #print('p', p_val)
            #print('u', u_vec)
            #print('F', F_vec)
            #print('B*F', B.dot(F_vec))

            #print(F_val/x_val.sum()*((B*x_val).sum()-(B*F_val).sum()))
            #if p_val == 0:
            #    raise(Error('Division by zero during quantile computation.'))
            #else:
            res = 1 + 1/p_val*(u_vec.sum()*(quantile-1.0)+
                            quantile*(np.matmul(B,x_vec)).sum()-(np.matmul(B,F_vec)).sum())
            #print('res', res)

            last_t = t_val
            last_res = res
            return np.array(res).reshape(1,)

        #short_res = odeint(rhs, sv, times, atol=tol, mxstep=10000)
        rhs2 = lambda t_val, y: rhs(y, t_val)
        short_res = solve_ivp_pwc(
            rhss   = (rhs2,),
            y0     = np.array([sv]).reshape(1,),
            t_span = (times[0], times[-1]),
            t_eval = times,
            **kwargs
        ).y
        short_res = np.rollaxis(short_res, -1, 0)

        pb.close()

        res = np.ndarray((len(original_times),))
        res[:ti] = np.nan
        res[ti:] = short_res.reshape((len(times),))

        if check_time_indices is not None:
            qs_ode = res[check_time_indices]
            qs_pi = np.zeros_like(qs_ode)
            for nr, ct_index in enumerate(check_time_indices):
                qs_pi[nr] = self.__class__.distribution_quantile(
                    quantile,
                    lambda a: F(a, self.times[ct_index]).sum(), 
                    norm_const=soln[ct_index,:].sum(),
                    start_value=qs_ode[nr]
                )
                #, method='brentq', tol=1e-8):

            print_quantile_error_statisctics(qs_ode, qs_pi)

        #print(res)
        return res


    ########## 14C methods #########


    def to_14C_explicit(self, start_values_14C, Fa_func, decay_rate=0.0001209681):
        """Construct and return a :class:`SmoothModelRun` instance that
           models the 14C component additional to the original model run.
    
        Args:
            start_values_14C (numpy.nd_array, nr_pools): 14C start values.
            Fa_func (func(t)): returns atmospheric fraction to be multiplied with the input vector 
            decay rate (float, optional): The decay rate to be used, defaults to
                ``0.0001209681`` (daily).
        Returns:
            :class:`SmoothModelRun`
        """
        srm_14C = self.model.to_14C_explicit('lamda_14C', 'Fa_14C')

        # create SmoothModelRun for 14C
        par_set_14C = {k:v for k, v in self.parameter_dict.items()}
        par_set_14C['lamda_14C'] = decay_rate

        nr_pools = self.nr_pools
        start_values_14C_cb = np.ones(nr_pools*2)
        start_values_14C_cb[:nr_pools] = self.start_values
        start_values_14C_cb[nr_pools:] = start_values_14C
        times_14C = self.times

        #Fa_atm = copy(atm_delta_14C)
        #Fa_atm[:,1] = Fa_atm[:,1]/1000 + 1
        #Fa_func = interp1d(Fa_atm[:,0], Fa_atm[:,1])
        func_set_14C = {k:v for k,v in self.func_set.items()}

        function_string = 'Fa_14C(' + srm_14C.time_symbol.name + ')'
        func_set_14C[function_string] = Fa_func

        smr_14C = SmoothModelRun(
            srm_14C, 
            par_set_14C,
            start_values_14C_cb,
            times_14C,
            func_set_14C,
        )

        return smr_14C


    ########## private methods #########


    def _solve_age_moment_system_single_value_old(self, max_order, 
            start_age_moments=None, start_values=None):
        t0 = self.times[0]
        t_max = self.times[-1]

        def func(t):
            if t < t0:
                # times x pools 
                res = np.zeros((1, self.nr_pools))
                res[res==0] = np.nan
                return res
            
            # fixme: do we really want to cut off here? 
            # This could be dangerous
            if t > t_max: t = t_max

            new_times = [t0, t]
            soln = self._solve_age_moment_system_old(max_order, 
                                                 start_age_moments, 
                                                 times=new_times, 
                                                 start_values=start_values)

            return soln[-1]

        return func 

    def _solve_age_moment_system_func(
            self, 
            max_order, 
            start_age_moments=None, 
            start_values=None
        ):

        t0 = self.times[0]
        t_max = self.times[-1]
        soln, func = self._solve_age_moment_system(
            max_order, 
            start_age_moments, 
            #times=new_times, 
            start_values=start_values
        )

        def save_func(times):
            if isinstance(times,np.ndarray):
                if times[0]<t0 or times[-1]>t_max:
                    raise Exception("""
                        times[0]<t0 or times[-1]>t_max: solve_ivp returns an interpolated
                        function, which does not check if the functions is called
                        with arguments outside the computed range, but we do. 
                        """
                    )
                else:
                    return np.rollaxis(func(times),-1,0)
            else:
                if (times < t0) or (times > t_max):
                    raise Exception("""
                        t<t0 or t>t_max: solve_ivp returns an interpolated
                        function, which does not check if the functions is called
                        with arguments outside the computed range, but we do. 
                        """
                    )
                else:
                    return func(times)

        return save_func 

    def _solve_age_moment_system(self, max_order, 
            start_age_moments=None, start_values=None, times=None, store=True):
        # this function caches the interpolation function instead of the values
        
        #if max_order < 1:
        #    raise(ValueError("For numerical consistency we use the age moment system only for order >=1 (mean). Use solve instead!"))

        if not  ((times is None) and (start_values is None)): store = False

        if times is None:
            times = self.times

        if start_values is None: start_values = self.start_values

        if not(isinstance(start_values, np.ndarray)):
            #print(start_values)
            raise(Error("start_values should be a numpy array"))

        n = self.nr_pools
        if start_age_moments is None:
            start_age_moments = np.zeros((max_order, n))
        
        start_age_moments_list = flatten([a.tolist() for a in 
                            [start_age_moments[i,:] 
                                for i in range(start_age_moments.shape[0])]])
       
        storage_key = tuple(start_age_moments_list) + ((max_order,),)

        # return cached result if possible
        if store:
            if hasattr(self, "_previously_computed_age_moment_sol"):
                if storage_key in self._previously_computed_age_moment_sol:
                    #print('using cached age moment system:', storage_key)
                    #print(
                    #   self._previously_computed_age_moment_sol[storage_key])
                    return self._previously_computed_age_moment_sol[storage_key]
            else:
                self._previously_computed_age_moment_sol = {}

        srm = self.model
        state_vector, rhs = srm.age_moment_system(max_order)
#        print('---')
#        print(state_vector)
#        print(rhs)
#        input() 
        # compute solution
        new_start_values = np.zeros((n*(max_order+1),))
        new_start_values[:n] = np.array(start_values)#.reshape(n,) 
        new_start_values[n:] = np.array(start_age_moments_list)

        soln, sol_func = numsol_symbolical_system(
            state_vector,
            srm.time_symbol,
            rhs,
            [self.parameter_dict],
            [self.func_set],
            new_start_values, 
            times,
            #dense_output=True,
            #disc_times=self.disc_times
        )
        def restrictionMaker(order):
            #pe('soln[:,:]',locals())
            restrictedSolutionArr=soln[:,:(order+1)*n]
            def restrictedSolutionFunc(t):
                return sol_func(t)[:(order+1)*n]

            return (restrictedSolutionArr,restrictedSolutionFunc)
            
        # save all solutions for order <= max_order
        if store:
            # as it seems, if max_order is > 0, the solution (solved with
            # max_order=0) is sligthly different from the part of first part
            # of the higher order system that corresponds als to the solution.
            # The difference is very small ( ~1e-5 ), but big
            # enough to cause numerical problems in functions depending on
            # the consistency of the solution and the state transition
            # operator.

            #consequently we do not save the solution
            # for orders less than max_order separately
            for order in [max_order]:
                shorter_start_age_moments_list = (
                    start_age_moments_list[:order*n])
                #print(start_age_moments_list)
                #print(shorter_start_age_moments_list)
                storage_key = (tuple(shorter_start_age_moments_list) 
                                + ((order,),))
                #print('saving', storage_key)

                self._previously_computed_age_moment_sol[storage_key] = restrictionMaker(order)
                
                #print(self._previously_computed_age_moment_sol[storage_key])

        return (soln, sol_func)

    def _solve_age_moment_system_old(self, max_order, 
            start_age_moments=None, times=None, start_values=None, store=True):
        #store = True
        if not ((times is None) and (start_values is None)): store = False

        if times is None: 
            times = self.times

        if start_values is None: start_values = self.start_values

        if not(isinstance(start_values, np.ndarray)):
            #print(start_values)
            raise(Error("start_values should be a numpy array"))

        n = self.nr_pools
        if start_age_moments is None:
            start_age_moments = np.zeros((max_order, n))
        
        start_age_moments_list = flatten([a.tolist() for a in 
                            [start_age_moments[i,:] 
                                for i in range(start_age_moments.shape[0])]])
       
        storage_key = tuple(start_age_moments_list) + ((max_order,),)

        # return cached result if possible
        if store:
            if hasattr(self, "_previously_computed_age_moment_sol_old"):
                if storage_key in self._previously_computed_age_moment_sol_old:
                    #print('using cached age moment system:', storage_key)
                    #print(
                    #   self._previously_computed_age_moment_sol_old[storage_key])
                    return self._previously_computed_age_moment_sol_old[storage_key]
            else:
                self._previously_computed_age_moment_sol_old = {}

        srm = self.model
        state_vector, rhs = srm.age_moment_system(max_order)
       
        # compute solution
        new_start_values = np.zeros((n*(max_order+1),))
        new_start_values[:n] = np.array((start_values)).reshape((n,)) 
        new_start_values[n:] = np.array((start_age_moments_list))

        soln= numsol_symbolic_system_old(
            state_vector,
            srm.time_symbol,
            rhs,
            self.parameter_dict,
            self.func_set,
            new_start_values, 
            times
        )
        
        # save all solutions for order <= max_order
        if store:
            for order in range(max_order+1):
                shorter_start_age_moments_list = (
                    start_age_moments_list[:order*n])
                #print(start_age_moments_list)
                #print(shorter_start_age_moments_list)
                storage_key = (tuple(shorter_start_age_moments_list) 
                                + ((order,),))
                #print('saving', storage_key)

                self._previously_computed_age_moment_sol_old[storage_key] = (
                    soln[:,:(order+1)*n])
                #print(self._previously_computed_age_moment_sol_old[storage_key])

        return soln


    @property
    def no_input_model(self):
        m=self.model
        return m.no_input_model
        #SmoothReservoirModel(
        #    m.state_vector,
        #    m.time_symbol,
        #    {},
        #    m.output_fluxes,
        #    m.internal_fluxes
        #)


    @property
    def _no_input_sol(self):
        # note that the solution of the no input system 
        # only coincides with the (application of) 
        # the statetransition operator if the system is linear
        # so this function can only compute the state transition operatro 
        # for a linear(ized) system


        if not hasattr(self, '_saved_no_input_sol'):
            m = self.model
            m_no_inputs=self.no_input_model
            
            no_inputs_num_rhs = numerical_rhs_old(
                m_no_inputs.state_vector, 
                m_no_inputs.time_symbol, 
                m_no_inputs.F, 
                self.parameter_dict,
                self.func_set,
                self.times)
    
            def no_input_sol(times, start_vector):
                ('nos', times, start_vector)
                # Start and end time too close together? Do not integrate!
                if abs(times[0]-times[-1]) < 1e-14: 
                    return np.array(start_vector)
                sv = np.array(start_vector).reshape((self.nr_pools,))

                return odeint(no_inputs_num_rhs, sv, times, mxstep = 10000)[-1]
        
            self._saved_no_input_sol = no_input_sol

        return self._saved_no_input_sol


    def initialize_state_transition_operator_cache(self, lru_maxsize, lru_stats=False, size=1):
        custom_lru_cache = custom_lru_cache_wrapper(
        	maxsize=lru_maxsize, # variable maxsize now for lru cache
        	typed=False,
        	stats=lru_stats # use custom statistics feature
        )
        
        nr_pools = self.nr_pools
        times = self.times
        t_min = times[0]
        t_max = times[-1]
        cache_times = np.linspace(t_min, t_max, size+1)
        ca = np.zeros((size, nr_pools, nr_pools)) 
        cache = Cache(cache_times, ca, self.myhash())
        cache._cached_phi_tmax = custom_lru_cache(phi_tmax) 

        self._state_transition_operator_cache = cache 


    #fixme: 
    # this method is not yet aware of the Cache class
    def save_state_transition_operator_cache(self, filename):
        self._state_transition_operator_cache.save(filename)

    def load_state_transition_operator_cache(self, filename):
        tmpCache = Cache.from_file(filename)
        if self.myhash()==tmpCache.myhash:
            self._state_transition_operator_cache=tmpCache
        else:
            raise Exception('State transition operator cache hash is different from the hash of the present model run and cannot be used. Please REMOVE THE CACHE FILE:'+filename)

    def myhash(self):
        """ 
        Compute a hash considering SOME but NOT ALL properties of a
        model run. The function's main use is to detect saved state transition
        operator cashes that are no longer compatible with the model run object
        that wants to use them. This check is useful but NOT COMPREHENSIVE.
        """
        times=self.times

        def make_hash_sha256(o):
            hasher = hashlib.sha256()
            #hasher.update(repr(make_hashable(o)).encode())
            hasher.update(repr(o).encode())
            return base64.b64encode(hasher.digest()).decode()
        
        return make_hash_sha256(
            (
                frozendict(self.model.input_fluxes),    
                frozendict(self.model.internal_fluxes),    
                frozendict(self.model.output_fluxes),    
                ImmutableMatrix(self.model.state_vector),
                # to compute a hash of an arbitrary function object is difficult
                # in particular if the function depends on data.
                frozendict(self.parameter_dict),
                self.start_values,
                (times[0],times[-1])
            )   
        )


    def _x_phi_block_ode(self):
        x_block_name = 'x'
        phi_block_name = 'phi'
        if not(hasattr(self, '_x_phi_block_ode_cache')):
            nr_pools = self.nr_pools
            
            block_ode = x_phi_ode(
                self.model,
                self.parameter_dict,
                self.func_set,
                x_block_name,
                phi_block_name
            )
            self._x_phi_block_ode_cache = block_ode
        return self._x_phi_block_ode_cache, x_block_name, phi_block_name

    def _state_transition_operator(self, t, t0, x):
        return np.matmul(self.Phi(t, t0), x).reshape((self.nr_pools,))



    def _state_transition_operator_for_linear_systems(self, t, t0, x):
        # this function could be used in a "linear smooth model run class"
        # At the moment it is only used by the tests to show
        # why a replacement was necessary for the general case
        
        srm = self.model
        if not srm.is_linear:
            raise Exception("This method can only be applied to linear systems. Maybe you have to linearize along a solution first?  ( Consider using the linearize method )"  )

        if t0 > t:
            raise(Error("Evaluation before t0 is not possible"))
        if t0 == t:
            return x.flatten() 
       
        n = self.nr_pools
        no_input_sol = self._no_input_sol

        soln = (no_input_sol([t0, t], x)).reshape((n,))        
        
        # avoid small negative values
        return np.maximum(soln, np.zeros_like(soln))
        
    #if self._state_transition_operator_cache is None:
        #    # do not use the cache, it has not yet been created
        #    #self.build_state_transition_operator_cache()
        #    soln = (no_input_sol([t0, t], x)).reshape((n,))        
        #else:
        #    # use the already created cache
        #    times = self.times
        #    t_min = times[0]
        #    t_max = times[-1]
        #    nc = self._cache_size
    
        #    cached_times = np.linspace(t_min, t_max, nc)
        #    ca = self._state_transition_operator_cache
    
        #    # find tm1
        #    tm1_ind = cached_times.searchsorted(t0)
        #    tm1 = cached_times[tm1_ind]
    
        #    # check if next cached time is already behind t
        #    if t <= tm1: return no_input_sol([t0, t], x)
    
        #    # first integrate x to tm1: y = Phi(tm1, t_0)x
        #    y = (no_input_sol([t0, tm1], x)).reshape((n,1))
    
        #    step_size = (t_max-tm1)/(nc-1)
        #    if step_size > 0:
        #        tm2_ind = np.int(np.min([np.floor((t-tm1)/step_size), nc-1]))
        #        tm2 = tm1 + tm2_ind*step_size
    
        #        #print(t, t0, t==t0, tm1_ind, tm1, tm2_ind, tm2, step_size) 
        #        B = ca[tm1_ind,tm2_ind,:,:]
        #        #print(t, t0, tm1, tm2, step_size, B)
        #        
        #        z = np.dot(B, y)
        #    else:
        #        tm2 = tm1
        #        z = y
        #    #z = (no_input_sol([tm1, tm2], y)[-1]).reshape((n,))
    
        #    # integrate z to t: sol=Phi(t,tm2)*z
        #    soln = (no_input_sol([tm2, t],z)).reshape((n,))
        #
        #return np.maximum(soln, np.zeros_like(soln))

    #this function should be rewritten using the vector valued solution 
    def _flux_vector(self, flux_vec_symbolic):
        #sol = self.solve_old()
        sol = self.solve()
        srm = self.model
        n = self.nr_pools
        times = self.times
        
        tup = tuple(srm.state_vector) + (srm.time_symbol,)
        res = np.zeros((len(times), n))
        
        flux_vec_symbolic = sympify(flux_vec_symbolic, locals = _clash)
        flux_vec_symbolic = flux_vec_symbolic.subs(self.parameter_dict)
        #cut_func_set = {key[:key.index('(')]: val 
        #                    for key, val in self.func_set.items()}
        cut_func_set=make_cut_func_set(self.func_set)
        flux_vec_fun = lambdify(tup, 
                                flux_vec_symbolic, 
                                modules=[cut_func_set, 'numpy'])

        res = np.zeros((len(times), n))
        for ti in range(len(times)):
            args = [sol[ti, pool] for pool in range(n)] + [times[ti]]
            val = flux_vec_fun(*args)
            res[ti,:] = val.reshape((n,))

        return res


    ##### age density methods #####


    def _age_densities_1_single_value(self, start_age_densities = None):
        # for part that comes from initial value
        if start_age_densities is None:
            # all mass is assumed to have age 0 at the beginning
            def start_age_densities(a):
                if a != 0: return np.array((0,)*self.nr_pools)
                return np.array(self.start_values)

        # cut off negative ages in start_age_densities
        def p0(a):
            if a >= 0: 
                return start_age_densities(a)
            else:
                return np.zeros((self.nr_pools,))

        Phi = self._state_transition_operator#_for_linear_systems
 
        t0 = self.times[0]

        #ppp = lambda a, t: self._state_transition_operator(t,t0,p0(a-(t-t0)))
        def ppp(a, t):
            #print('iv: ', a, t)

            #fixme: cut off accidental negative values
            #print('Y', a-(t-t0), p0(a-t-t0))
            #print('smr 3821 ppp', t, t0, a, a-(t-t0))
            res = np.maximum(Phi(t, t0, p0(a-(t-t0))), 0)
            #print('ppp:', res)
            return res

        return ppp

    # return a function p1 that takes an age np.array
    # and gives back an nd array (age, time, pool)
    def _age_densities_1(self, start_age_densities = None):
        # for part that comes from initial value

        ppp = self._age_densities_1_single_value(start_age_densities)
        pp = lambda a: np.array([ppp(a,t) for t in self.times], np.float)
        p1 = lambda ages: np.array([pp(a) for a in ages], np.float)
        
        return p1
        
    def _age_densities_2_single_value(self):
        # for part that comes from the input function u
       
        t0 = self.times[0]
        u = self.external_input_vector_func()
        #u = lambda x: np.array([1,2])

        def ppp(a, t):
            #print('input', a, t)
            if (a < 0) or (t-t0 <= a):
                val = np.zeros((1,self.nr_pools))[-1]
            else:
                u_val = u(t-a)
                #print('u_val', u_val)
                val = self._state_transition_operator(t, t-a, u_val)

            #fixme: cut off accidental negative values
            res = np.maximum(val, 0)
            #print('ppp:', res)
            return res

        return ppp

    # returns a function p2 that takes an age array "ages" as argument
    # and gives back a three-dimensional ndarray (ages x times x pools)
    def _age_densities_2(self):
        # for part that comes from the input function u
        ppp = self._age_densities_2_single_value()
        pp = lambda a: np.array([ppp(a,t) for t in self.times], np.float)
        p2 = lambda ages: np.array([pp(a) for a in ages], np.float)

        return p2


    ##### plotting methods #####
    
    
    def _density_plot_plotly(self, field, ages, age_stride=1, time_stride=1):
        times = self.times

        strided_field = stride(field, (age_stride, time_stride))
        strided_ages = stride(ages, age_stride)
        strided_times = stride(times, time_stride)
 
        surfacecolor = strided_field.copy()
        for ai in range(strided_field.shape[0]):
            for ti in range(strided_field.shape[1]):
                surfacecolor[ai,ti] = - (ai - ti)
        
        data = [go.Surface(x = -strided_times, 
                           y = strided_ages, 
                           z = strided_field, 
                           showscale = False, 
                           surfacecolor = surfacecolor, 
                           colorscale = 'Rainbow')]
        
        tickvals = np.linspace(strided_times[0], strided_times[-1], 5)
        ticktext = [str(v) for v in tickvals]
        tickvals = -tickvals
        
        layout = go.Layout(
            width = 800,
            height = 800,
            scene = dict(
                xaxis = dict(
                    title = 'Time',
                    tickmode = 'array',
                    tickvals = tickvals,
                    ticktext = ticktext
                    #range = [-times[0], -times[-1]]
                ),
                yaxis = dict(
                    title = 'Age',
                    range = [ages[0], ages[-1]]
                ),
                zaxis = dict(
                    title = 'Mass',
                    range = [0, np.amax(strided_field)]
                )
            )
        )

        return data, layout


    ## plot helper methods ##

    #fixme: unit treatment disabled
    def _add_time_unit(self, label):
        #if self.model.time_unit:
        #    label += r"$\quad(\mathrm{" + latex(self.model.time_unit) + "})$"

        return label

    def _add_content_unit(self, label):
        #if self.model.content_unit:
        #    label +=r"$\quad(\mathrm{" + latex(self.model.content_unit) + "})$"

        return label

    def _add_flux_unit(self, label):
        #if self.model.content_unit and self.model.time_unit:
        #    label += r"$\quad(\mathrm{" + latex(self.model.content_unit) 
        #    label += "/" + latex(self.model.time_unit) + "})$"
        
        return label


    ## flux helper functions ##

 
    #fixme: test and move
    

    def _flux_funcs(self, expr_dict):
        m = self.model
        #sol_funcs = self.sol_funcs()
        sol_funcs = self.sol_funcs()
        flux_funcs = {}
        tup = tuple(m.state_variables) + (m.time_symbol,)
        for key, expression in expr_dict.items():
            if isinstance(expression, Number):
                # in this case (constant flux) lambdify for some reason 
                # does not return a vectorized function but one that
                # allways returns a number even when it is called with 
                # an array argument. We therfore create such a function 
                # ourselves
                flux_funcs[key] = const_of_t_maker(expression)
            else:
                # fixme mm 11-5-2018 
                # the sympify in the next line should be unnecesary since 
                # the expressions are already expressions and not strings
                # and now also not Numbers
                #o_par = sympify(expression, locals=_clash).subs(self.parameter_dict)
                o_par = expression.subs(self.parameter_dict)
                cut_func_set = make_cut_func_set(self.func_set)
                ol = lambdify(tup, o_par, modules = [cut_func_set, 'numpy'])
                #ol = numerical_function_from_expression(expression,tup,self.parameter_dict,self.func_set) 
                flux_funcs[key] = f_of_t_maker(sol_funcs, ol)

        return flux_funcs


    ## temporary ##


    def _FTTT_lambda_bar(self, end, s, u):
        u_norm = u.sum()
        if u_norm == 0:
            return 0

        Phi = self._state_transition_operator
        t1 = end
        result = -np.log(Phi(t1, s, u).sum()/u_norm)/(t1-s)
        
        return result


    def _FTTT_lambda_bar_R(self, start, end):
        if (start < self.times[0]) or (end > self.times[-1]):
            raise(Error('Interval boundaries out of bounds'))
        if start > end:
            raise(Error('Starting time must not be later then ending time'))

        t0 = start
        t1 = end
        u_func = self.external_input_vector_func()
        #soln_func = self.solve_single_value_old()
        vec_soln_func = self.solve_func()
        x0 = vec_soln_func(t0)
        x0_norm = x0.sum()
        
        A = x0_norm*(t1-t0)*self._FTTT_lambda_bar(t1, t0, x0)
        
        #print('A', A)

        def B_integrand(s):
            u = u_func(s)
            u_norm = u.sum()

            return u_norm*(t1-s)*self._FTTT_lambda_bar(t1, s, u)

        B = quad(B_integrand, t0, t1)[0]
        #print('B', B)

        C = x0_norm*(t1-t0)
        #print('C', C)

        def D_integrand(s):
            u_norm = u_func(s).sum()
            return u_norm*(t1-s)

        D = quad(D_integrand, t0, t1)[0]
        #print('D', D)

        return (A+B)/(C+D)

  
    def _FTTT_T_bar_R(self, start, end):
        if (start < self.times[0]) or (end > self.times[-1]):
            raise(Error('Interval boundaries out of bounds'))
        if start > end:
            raise(Error('Starting time must not be later then ending time'))

        t0 = start
        t1 = end
        u_func = self.external_input_vector_func()
        Phi = self._state_transition_operator

        #soln_func = self.solve_single_value_old()
        vec_soln_func = self.solve_func()
        x0 = vec_soln_func(t0)
        x0_norm = x0.sum()
        
        if x0_norm > 0:
            A = x0_norm*(t1-t0)*1/self._FTTT_lambda_bar(t1, t0, x0)
        else:
            A = 0
        #print('A', A)

        def B_integrand(s):
            u = u_func(s)
            u_norm = u.sum()
            if u_norm > 0:
                return u_norm*(t1-s)*1/self._FTTT_lambda_bar(t1, s, u)
            else:
                return 0

        B = quad(B_integrand, t0, t1)[0]
        #print('B', B)

        C = x0_norm*(t1-t0)
        #print('C', C)

        def D_integrand(s):
            u_norm = u_func(s).sum()
            return u_norm*(t1-s)

        D = quad(D_integrand, t0, t1)[0]
        #print('D', D)

        return (A+B)/(C+D)


    def _FTTT_lambda_bar_S(self, start, end):
        # for Martin Rasmussens surrogate system

        if (start < self.times[0]) or (end > self.times[-1]):
            raise(Error('Interval boundaries out of bounds'))
        if start > end:
            raise(Error('Starting time must not be later than ending time'))

        if start == end:
            return np.nan

        t0, t1 = start, end 
        #soln_func = self.solve_single_value_old()
        vec_soln_func = self.solve_func()
        x0 = vec_soln_func(t0)
        x1 = vec_soln_func(t1)

        z0 = x0.sum()
        z1 = x1.sum()

        u_func = self.external_input_vector_func()

        # function to minimize during Newton to find lambda_bar_S
        # g seems to have huge numerical issues
        def g(lamda):
            def f(z, t):
                # RHS in the surrogate system
                return -lamda*z+sum(u_func(t))
        
            # solve the system with current lambda
            sol = odeint(f, z0, [t0, t1])

            # return the distance of the current final time value
            # from the desired z1
            res = sol[-1]-z1
            return res

        # g2 seems to work much better
        def g2(lamda):
            if lamda <= 0:
                return 137

            def f(s):
                res = np.exp(-lamda*(t1-s))*sum(u_func(s))
                #print(lamda, res, u_func(s), t1, s)
                return res

            int_res = quad(f, t0, t1)[0]
            z0_remaining = np.exp(-lamda*(t1-t0))*z0
            if (z0_remaining<1e-08) or np.isnan(z0_remaining):
                z0_remaining = 0
            res = z0_remaining-z1+int_res
            #print(lamda, z0_remaining, z1, int_res, res)
            return res

        # return lambda_bar_S after optimization
        try:
            #res = newton(g, 0.5, maxiter=5000)
            #res = newton(g2, 1.5, maxiter=500)
            res = brentq(g2, 0, 5, maxiter=500)
        except RuntimeError:
            print('optimization aborted')
            return np.nan
        
        if res <= 0:
            return np.nan

        if not isinstance(res, float):
            res = res[0]
        return res
            

    def _calculate_steady_states(self):
    #fixme: should be possible only for autonomous, possibly nonlinear,
    # models
    #fixme: test?
        ss = solve(self.model.F.subs(self.parameter_dict), 
                   self.model.state_vector, 
                   dict=True)
        
        return_ss = []
        for ss_i in ss:
            add = True
            for key, val in ss_i.items():
                if self.model.time_symbol in val.free_symbols:
                    add = False

            if add:
                return_ss.append(ss_i)

        return return_ss


    def _FTTT_lambda_bar_R_left_limit(self, t0):
        #B0 = self.B(t0)
        vec_sol_funcs = self.solve_func()
        B0 = self.B_func()(t0)
        iv = Matrix(self.start_values) # column vector
        z = (-ones(1, len(iv))*B0).T
        
        return (z.T*iv/mpmath.norm(iv, 1))[0]

    ## new FTTT approach ##

    def _alpha_s_i(self, s, i, t1):
        Phi = self._state_transition_operator
        e_i = np.zeros(self.nr_pools)
        e_i[i] = 1
        
        return 1 - Phi(t1,s,e_i).sum()
            
    def _alpha_s(self, s, t1, vec):
        Phi = self._state_transition_operator
        vec_norm = vec.sum()

        return 1 - Phi(t1,s,vec).sum()/vec_norm

    def _EFFTT_s_i(self, s, i, t1, alpha_s_i = None):
        Phi = self._state_transition_operator
        if alpha_s_i is None:
            alpha_s_i = self._alpha_s_i(s, i, t1)
       
        e_i = np.zeros(self.nr_pools)
        e_i[i] = 1
        def F_FTT_i(a):
            return 1 - Phi(s+a,s,e_i).sum()

        def integrand(a):
            return 1 - F_FTT_i(a)/alpha_s_i

        result = quad(integrand, 0, t1-s, epsabs=1.5e-03, epsrel=1.5e-03)[0]
        return result

    def _TR(self, s, t1, v): # v is the remaining vector, not normalized
        Phi = self._state_transition_operator

        n = self.nr_pools
        Phi_matrix = np.zeros((n,n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1
            Phi_matrix[:,i] = Phi(t1,s,e_i)

        A = scipy.linalg.logm(Phi_matrix)/(t1-s)
        A_inv = scipy.linalg.inv(A)
        o = np.ones(n)
        v_normed = v/v.sum()
   
        return (t1-s) + (-o @ A_inv @ v_normed)


    def _FTTT_finite_plus_remaining(self, s, t1, t0):
        if s == t0:
            #soln_func = self.solve_single_value_old()
            vec_soln_func = self.solve_func()
            vec = vec_soln_func(s)
        else:
            u_func = self.external_input_vector_func()
            vec = u_func(s)
        vec_norm = vec.sum()

        if vec_norm > 0 :
            Phi = self._state_transition_operator

            # the finite time part
            finite = 0
            for i in range(self.nr_pools):
                alpha_s_i = self._alpha_s_i(s, i, t1)
                EFFTT_s_i = self._EFFTT_s_i(s, i, t1, alpha_s_i)
                finite += vec[i] * alpha_s_i * EFFTT_s_i
            
            # the part for the remaining mass
            if s < t1:
                v = Phi(t1,s,vec) # remaining mass at time t1
                alpha_s = self._alpha_s(s, t1, vec)
                remaining = (1-alpha_s) * vec_norm * self._TR(s, t1, v)
            else:
                remaining = 0

            return finite + remaining
        else:
            return 0


    def _FTTT_conditional(self, t1, t0):
        if (t0 < self.times[0]) or (t1 > self.times[-1]):
            raise(Error('Interval boundaries out of bounds'))
        if t0 >= t1:
            raise(Error('Starting time must be earlier then ending time'))
        
        A = (t1-t0) * self._FTTT_finite_plus_remaining(t0, t1, t0)

        def B_integrand(s):
            return (t1-s) * self._FTTT_finite_plus_remaining(s, t1, t0)

        B = quad(B_integrand, t0, t1, epsabs=1.5e-03, epsrel=1.5e-03)[0]

        #soln_func = self.solve_single_value_old()
        vec_soln_func = self.solve_func()
        x0 = vec_soln_func(t0)
        x0_norm = x0.sum()
        C = x0_norm*(t1-t0)
        
        u_func = self.external_input_vector_func()
        def D_integrand(s):
            u_norm = u_func(s).sum()
            return u_norm*(t1-s)

        D = quad(D_integrand, t0, t1)[0]

        return (A+B)/(C+D)


        #### Important again ####


    def Phi_func(self):
        # note that the functions used to produce the matrix 
        # self.Phi are cached (if the cache is initialized)
        # so that the repeated calls by the following lambda
        # are actually cheap
        return lambda T, S: self.Phi(T, S)

    def Phi(self, T, S):
        nr_pools=self.nr_pools
        start_Phi_2d=np.identity(nr_pools)
        
        if S > T:
            raise(Error("Evaluation before S is not possible"))
        if S == T:
            return start_Phi_2d
        
        solve_func = self.solve_func()
        block_ode, x_block_name, phi_block_name = self._x_phi_block_ode()
        
        if hasattr(self,'_state_transition_operator_cache'):
            cache = self._state_transition_operator_cache
            cache_times = cache.keys
            S_phi_ind = cache.phi_ind(S)
            T_phi_ind = cache.phi_ind(T)
            my_phi_tmax = cache._cached_phi_tmax 
            def phi(t, s, t_max):
                x_s = tuple(solve_func(s))
                return my_phi_tmax(
                    s,
                    t_max,
                    block_ode,
                    x_s,
                    x_block_name,
                    phi_block_name
                )(t)
            S_phi_ind = cache.phi_ind(S)
            T_phi_ind  = cache.phi_ind(T)

#            t_max = self.times[-1]
            t_max = cache.end_time_from_phi_ind(T_phi_ind)

            # catch the corner cases where the cache is useless.
            if (T_phi_ind-S_phi_ind) < 1:
                return phi(T, S, t_max=cache.end_time_from_phi_ind(T_phi_ind))
            tm1 = cache.end_time_from_phi_ind(S_phi_ind)
        
            ## first integrate to tm1: 
            if tm1 != S:
                phi_tm1_S = phi(tm1, S, tm1)
            else: 
                phi_tm1_S = start_Phi_2d

            phi_T_tm1 = phi(T, tm1, self.times[-1])
            return np.matmul(phi_T_tm1, phi_tm1_S)

        else:
            def phi(t, s):
                x_s = solve_func(s)
                start_Phi_2d = np.identity(nr_pools)
                start_blocks = [
                    (x_block_name, x_s),
                    (phi_block_name, start_Phi_2d) 
                ]
                blivp = block_ode.blockIvp(start_blocks)
                
                return blivp.block_solve(t_span=(s, t))[phi_block_name][-1,...]
            return phi(T, S)

    def fake_discretized_Bs(self, data_times=None): 
        if data_times is None:
            data_times = self.times
        
        nr_pools = self.nr_pools
        n = len(data_times)
        Bs = np.zeros((n-1, nr_pools, nr_pools))
        
        for k in range(n-1):
            Bs[k,:,:] = self.Phi(data_times[k+1], data_times[k])

        return Bs

    def acc_net_internal_flux_matrix(self, data_times=None):    
        if data_times is None:
            data_times = self.times

        x_func = self.solve_func()
        xs = x_func(data_times)

        nt = len(data_times)-1
        nr_pools = self.nr_pools
        Bs = self.fake_discretized_Bs(data_times)
        
        return net_Fs_from_discrete_Bs_and_xs(Bs, xs)

    def acc_net_external_output_vector(self, data_times=None):
        if data_times is None:
            data_times = self.times

        x_func = self.solve_func()
        xs = x_func(data_times)
        
        nt = len(data_times)-1
        nr_pools = self.nr_pools
        Bs = self.fake_discretized_Bs(data_times)
        
        return net_Rs_from_discrete_Bs_and_xs(Bs, xs)

    def acc_net_external_input_vector(self, data_times=None):    
        if data_times is None:
            data_times = self.times

        x_func = self.solve_func()
        xs = x_func(data_times)
        
        nt = len(data_times)-1
        nr_pools = self.nr_pools
        Bs = self.fake_discretized_Bs(data_times)
        
        return net_Us_from_discrete_Bs_and_xs(Bs, xs)
    
    def fake_net_discretized_output(self, data_times):
        x_func = self.solve_func()
        xs = x_func(data_times)
    
        net_Fs = self.acc_net_internal_flux_matrix(data_times)
        net_Rs = self.acc_net_external_output_vector(data_times)
        net_Us = self.acc_net_external_input_vector(data_times)

        return xs, net_Us, net_Fs, net_Rs

    def fake_gross_discretized_output(self, data_times):
        ## prepare some fake output data
        #x = self.solve_single_value_old()
        x_func = self.solve_func()
        xs = x_func(data_times)
    
        gross_Fs = self.acc_gross_internal_flux_matrix(data_times)
        gross_Rs = self.acc_gross_external_output_vector(data_times)
        gross_Us = self.acc_gross_external_input_vector(data_times)
    
        return xs, gross_Us, gross_Fs, gross_Rs

        

 
