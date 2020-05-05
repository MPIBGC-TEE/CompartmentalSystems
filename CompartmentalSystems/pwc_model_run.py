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

#from numbers import Number
#from copy import copy, deepcopy
#from matplotlib import cm
#import matplotlib.patches as mpatches
#import matplotlib.pyplot as plt

import numpy as np
#from numpy.linalg import matrix_power

#import plotly.graph_objs as go

#import base64
#import hashlib
#import mpmath
from frozendict import frozendict

from sympy import flatten#, lambdify, latex, Function, sympify, sstr, solve, \
#                  ones, Matrix, ImmutableMatrix
#from sympy.core.function import UndefinedFunction
#from sympy.abc import _clash
#from sympy.printing import pprint

#import scipy.linalg
#from scipy.linalg import inv
#from numpy.linalg import pinv
#from scipy.special import factorial
#from scipy.integrate import odeint, quad 
#from scipy.interpolate import interp1d, UnivariateSpline
#from scipy.optimize import newton, brentq, minimize

#from tqdm import tqdm
#from functools import reduce
#from testinfrastructure.helpers import pe

from .smooth_reservoir_model import SmoothReservoirModel
from .model_run import ModelRun
from .helpers_reservoir import (
#    deprecation_warning
#    ,warning
#    ,make_cut_func_set
#    ,has_pw
     numsol_symbolical_system,
#    ,arrange_subplots
#    ,melt
#    ,generalized_inverse_CDF
#    ,draw_rv 
#    ,stochastic_collocation_transform
#    ,numerical_rhs
#    ,MH_sampling
#    ,save_csv 
#    ,load_csv
#    ,stride
#    ,f_of_t_maker
#    ,const_of_t_maker
#    ,numerical_function_from_expression
#    ,x_phi_ode
#    ,phi_tmax
#    ,x_tmax
#    ,print_quantile_error_statisctics
#    ,custom_lru_cache_wrapper
#    ,net_Us_from_discrete_Bs_and_xs
#    ,net_Fs_from_discrete_Bs_and_xs
#    ,net_Rs_from_discrete_Bs_and_xs
    check_parameter_dict_complete
)

#from .BlockIvp import BlockIvp
#from .myOdeResult import solve_ivp_pwc
#from .Cache import Cache


class Error(Exception):
    """Generic error occurring in this module."""
    pass


class PWCModelRun(ModelRun):
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

    def __init__(self, model, parameter_dicts, 
                        start_values, times, disc_times, func_dicts=None):
        """Return a PWCModelRun instance.

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
            disc_times (list): ``known discontinuities (times) for any of the fluxes``.

        Raises:
            Error: If ``start_values`` is not a ``numpy.array``.
        """
        assert len(disc_times) > 0

        self.disc_times = disc_times

        if parameter_dicts is None: 
            parameter_dicts = (dict()) * (len(disc_times)+1)
        if func_dicts is None: 
            func_dicts = (dict()) * (len(disc_times)+1)
        
        # check parameter_dicts + func_dicts for completeness
        for pd, fd in zip(parameter_dicts, func_dicts):
            free_symbols = check_parameter_dict_complete(
                model, 
                pd,
                fd
            )
            if free_symbols != set():
                raise(Error('Missing parameter values for ' + str(free_symbols)))


        self.model = model
        self.parameter_dicts = (frozendict(pd) for pd in parameter_dicts)
        self.times = times
        # make sure that start_values are an array,
        # even a one-dimensional one
        self.start_values = np.array(start_values).reshape(model.nr_pools,)

        if not(isinstance(start_values, np.ndarray)):
            raise(Error("start_values should be a numpy array"))
        self.func_dicts = (frozendict(fd) for fd in func_dicts)


#    @property
#    def nr_intervls(self):
#        return len(self.disc_times)+1
        

    @property
    def nr_pools(self):
        return self.model.nr_pools


    @property
    def dts(self):
        return np.diff(self.times).astype(np.float64)

    def solve(self, alternative_start_values=None): 
        soln, sol_func = self._solve_age_moment_system(
            0, 
            None,
            alternative_start_values
        )
        return soln
    
    def acc_gross_external_input_vector(self):
        pass

    def acc_gross_internal_flux_matrix(self):
        pass
    
    def acc_gross_external_output_vector(self) :
        pass
    
    def acc_net_external_input_vector(self):
        pass

    def acc_net_internal_flux_matrix(self):
        pass
    
    def acc_net_external_output_vector(self) :
        pass


################################################################################


    def _solve_age_moment_system(
            self,
            max_order, 
            start_age_moments = None,
            start_values      = None,
            times             = None,
            store             = True
        ):
        if not ((times is None) and (start_values is None)): store = False
        if times is None: times = self.times
        if start_values is None: start_values = self.start_values

        if not(isinstance(start_values, np.ndarray)):
            raise(Error("start_values should be a numpy array"))

        n = self.nr_pools
        if start_age_moments is None:
            start_age_moments = np.zeros((max_order, n))
        
        start_age_moments_list = flatten(
            [
                a.tolist() for a in 
                [
                    start_age_moments[i,:] for i in range(start_age_moments.shape[0])
                ]
            ]
        )
        storage_key = tuple(start_age_moments_list) + ((max_order,),)

        # return cached result if possible
        if store:
            if hasattr(self, "_previously_computed_age_moment_sol"):
                if storage_key in self._previously_computed_age_moment_sol:
                    return self._previously_computed_age_moment_sol[storage_key]
            else:
                self._previously_computed_age_moment_sol = {}

        srm = self.model
        state_vector, rhs = srm.age_moment_system(max_order)
        
        # compute solution
        new_start_values = np.zeros((n*(max_order+1),))
        new_start_values[:n] = np.array(start_values) 
        new_start_values[n:] = np.array(start_age_moments_list)

        soln, sol_func = numsol_symbolical_system(
            state_vector,
            srm.time_symbol,
            rhs,
            self.parameter_dicts,
            self.func_dicts,
            new_start_values,
            times,
            disc_times = self.disc_times
        )

        def restrictionMaker(order):
            restrictedSolutionArr = soln[:,:(order+1)*n]
            def restrictedSolutionFunc(t):
                return sol_func(t)[:(order+1)*n]

            return (restrictedSolutionArr, restrictedSolutionFunc)
            
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

    
