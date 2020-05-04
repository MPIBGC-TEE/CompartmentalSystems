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
from functools import reduce
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


#class PWCModelRun2(ModelRun):
class PWCModelRun2:
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
        # we cannot use dict() as default because the test suite makes weird 
        # things with it! But that is bad style anyways
       
        assert len(disc_times) > 0

        self.disc_times = disc_times

        if parameter_dicts is None: 
            parameter_dicts = (dict()) * (len(disc_times)+1)
        if func_dicts is None: 
            func_dicts = (dict()) * (len(disc_times)+1)
        
        # check parameter_dicts + func_dicts for completeness
        for pd,fd in zip(parameter_dicts,func_dicts):
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
        # fixme mm: 
        #func_set = {str(key): val for key, val in func_set.items()}
        # The conversion to string is not desirable here
        # should rather implement a stricter check (which fails at the moment because some tests use the old syntax
        #for f in func_set.keys():
        #    if not isinstance(f,UndefinedFunction):
        #        raise(Error("The keys of the func_set should be of type:  sympy.core.function.UndefinedFunction"))
        self.func_dicts = (frozendict(fd) for fd in func_dicts)
        #self._state_transition_operator_cache = None
        self._external_input_vector_func = None


    @property
    def nr_intervls(self):
        return len(self.disc_times)+1
        
    @property
    def dts(self):
        """
        The lengths of the time intervals.
        """
        return np.diff(self.times).astype(np.float64)
    
