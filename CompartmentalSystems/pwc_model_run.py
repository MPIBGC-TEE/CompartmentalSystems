from numbers import Number
import numpy as np
from frozendict import frozendict
from sympy import flatten, lambdify, ImmutableMatrix
from scipy.integrate import quad
import hashlib
import base64

from .model_run import ModelRun
from .helpers_reservoir import (
    numsol_symbolical_system,
    check_parameter_dict_complete,
    make_cut_func_set,
    f_of_t_maker,
    const_of_t_maker,
    x_phi_ode,
    net_Us_from_discrete_Bs_and_xs,
    net_Fs_from_discrete_Bs_and_xs,
    net_Rs_from_discrete_Bs_and_xs,
    custom_lru_cache_wrapper,
    phi_tmax,
    numerical_function_from_expression
)
from .Cache import Cache


class Error(Exception):
    """Generic error occurring in this module."""
    pass


class PWCModelRun(ModelRun):
    def __init__(
        self,
        model,
        parameter_dicts,
        start_values,
        times,
        disc_times,
        func_dicts=None
    ):
        if len(disc_times) == 0:
            raise(Error("No 'disc_times' given"))

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
                raise(
                    Error(
                        "Missing parameter values for {}".format(free_symbols)
                    )
                )

        self.model = model
        self.parameter_dicts = tuple(frozendict(pd) for pd in parameter_dicts)
        self.times = times
        # make sure that start_values are an array,
        # even a one-dimensional one
        self.start_values = np.array(start_values).reshape(model.nr_pools,)

        if not(isinstance(start_values, np.ndarray)):
            raise(Error("'start_values' should be a numpy array"))
        self.func_dicts = tuple(frozendict(fd) for fd in func_dicts)

    @property
    def nr_intervals(self):
        return len(self.disc_times)+1

    @property
    def boundaries(self):
        times = self.times
        return np.array([times[0]] + self.disc_times + [times[-1]])

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

    def B_func(self, vec_sol_func=None):
        if vec_sol_func is None:
            vec_sol_func = self.solve_func()

        srm = self.model
        tup = (srm.time_symbol,) + tuple(srm.state_vector)

        def func_maker(pd, fd):
            # we inject the solution into B to get the linearized version
            numfun = numerical_function_from_expression(
                srm.compartmental_matrix,
                tup,
                pd,
                fd
            )

            # we want a function  that accepts a vector argument for x
            def B_func_k(t):
                x = vec_sol_func(t)
                return numfun(t, *x)

            return B_func_k

        L = []
        for pd, fd in zip(self.parameter_dicts, self.func_dicts):
            L.append(func_maker(pd, fd))

        return self.join_functions_rc(L)

    def acc_gross_external_input_vector(self, data_times=None):
        times = self.times if data_times is None else data_times
        nt = len(times) - 1

        flux_funcss = self.external_input_flux_funcss()
        res = np.zeros((nt, self.nr_pools))
        for pool_nr in range(self.nr_pools):
            flux_func = self.join_flux_funcss_rc(flux_funcss, pool_nr)

            for k in range(nt):
                res[k, pool_nr] = quad(
                    flux_func,
                    times[k],
                    times[k+1]
                )[0]

        return res

    def acc_gross_internal_flux_matrix(self, data_times=None):
        times = self.times if data_times is None else data_times
        nt = len(times) - 1
        nr_pools = self.nr_pools

        flux_funcss = self.internal_flux_funcss()
        res = np.zeros((nt, nr_pools, nr_pools))
        for pool_from in range(nr_pools):
            for pool_to in range(nr_pools):
                key = (pool_from, pool_to)
                flux_func = self.join_flux_funcss_rc(flux_funcss, key)

                for k in range(nt):
                    res[k, pool_to, pool_from] = quad(
                        flux_func,
                        times[k],
                        times[k+1]
                    )[0]

        return res

    def acc_gross_external_output_vector(self, data_times=None):
        times = self.times if data_times is None else data_times
        nt = len(times) - 1

        flux_funcss = self.external_output_flux_funcss()
        res = np.zeros((nt, self.nr_pools))
        for pool_nr in range(self.nr_pools):
            flux_func = self.join_flux_funcss_rc(flux_funcss, pool_nr)

            for k in range(nt):
                res[k, pool_nr] = quad(
                    flux_func,
                    times[k],
                    times[k+1]
                )[0]

        return res

    def acc_net_external_input_vector(self, data_times=None):
        if data_times is None:
            data_times = self.times

        x_func = self.solve_func()
        xs = x_func(data_times)
        Bs = self.fake_discretized_Bs(data_times)

        return net_Us_from_discrete_Bs_and_xs(Bs, xs)

    def acc_net_internal_flux_matrix(self, data_times=None):
        if data_times is None:
            data_times = self.times

        x_func = self.solve_func()
        xs = x_func(data_times)

        Bs = self.fake_discretized_Bs(data_times)

        return net_Fs_from_discrete_Bs_and_xs(Bs, xs)

    def acc_net_external_output_vector(self, data_times=None):
        if data_times is None:
            data_times = self.times

        x_func = self.solve_func()
        xs = x_func(data_times)
        Bs = self.fake_discretized_Bs(data_times)

        return net_Rs_from_discrete_Bs_and_xs(Bs, xs)

###############################################################################

    def _solve_age_moment_system(
        self,
        max_order,
        start_age_moments=None,
        start_values=None,
        times=None,
        store=True
    ):

        if not ((times is None) and (start_values is None)):
            store = False
        if times is None:
            times = self.times
        if start_values is None:
            start_values = self.start_values

        if not(isinstance(start_values, np.ndarray)):
            raise(Error("start_values should be a numpy array"))

        n = self.nr_pools
        if start_age_moments is None:
            start_age_moments = np.zeros((max_order, n))

        start_age_moments_list = flatten(
            [
                a.tolist() for a in
                [
                    start_age_moments[i, :]
                    for i in range(start_age_moments.shape[0])
                ]
            ]
        )
        storage_key = tuple(start_age_moments_list) + ((max_order,),)

        # return cached result if possible
        if store:
            if hasattr(self, "_previously_computed_age_moment_sol"):
                if storage_key in self._previously_computed_age_moment_sol:
                    return(
                        self._previously_computed_age_moment_sol[storage_key])
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
            disc_times=self.disc_times
        )

        def restrictionMaker(order):
            restrictedSolutionArr = soln[:, :(order+1)*n]

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

            # consequently we do not save the solution
            # for orders less than max_order separately
            for order in [max_order]:
                shorter_start_age_moments_list = (
                    start_age_moments_list[:order*n])
                storage_key = (
                    tuple(shorter_start_age_moments_list)
                    + ((order,),)
                )
                self._previously_computed_age_moment_sol[storage_key]\
                    = restrictionMaker(order)

        return (soln, sol_func)

    def external_input_flux_funcss(self):
        return self._flux_funcss(self.model.input_fluxes)

    def internal_flux_funcss(self):
        return self._flux_funcss(self.model.internal_fluxes)

    def external_output_flux_funcss(self):
        return self._flux_funcss(self.model.output_fluxes)

    def _flux_funcss(self, expr_dict):
        model = self.model
        sol_funcs = self.sol_funcs()
        tup = tuple(model.state_variables) + (model.time_symbol,)

        flux_funcss = []
        for pd, fd in zip(self.parameter_dicts, self.func_dicts):
            flux_funcs = {}
            for key, expression in expr_dict.items():
                f_par = expression.subs(pd)
                if isinstance(f_par, Number):
                    # in this case (constant flux) lambdify for some reason
                    # does not return a vectorized function but one that
                    # allways returns a number even when it is called with
                    # an array argument. We therfore create such a function
                    # ourselves
                    flux_funcs[key] = const_of_t_maker(f_par)
                else:
                    cut_func_set = make_cut_func_set(fd)
                    fl = lambdify(tup, f_par, modules=[cut_func_set, 'numpy'])
                    flux_funcs[key] = f_of_t_maker(sol_funcs, fl)

            flux_funcss.append(flux_funcs)

        return tuple(flux_funcss)

    def sol_funcs(self):
        vec_sol_func = self.solve_func()

        # the factory is necessary to avoid unstrict evaluation
        def func_maker(pool):
            def func(t):
                return vec_sol_func(t)[pool]
            return(func)

        return [func_maker(i) for i in range(self.nr_pools)]

    def solve_func(self, alternative_start_values=None):
        return self._solve_age_moment_system_func(
            0,
            None,
            alternative_start_values
        )

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
            start_values=start_values
        )

        def save_func(times):
            if isinstance(times, np.ndarray):
                if times[0] < t0 or times[-1] > t_max:
                    raise Exception(
                        """
                        times[0] < t0 or times[-1] > t_max: solve_ivp returns
                        an interpolated function which does not check if the
                        function is called with arguments outside the
                        computed range, but we do.
                        """
                    )
                else:
                    return np.rollaxis(func(times), -1, 0)
            else:
                if (times < t0) or (times > t_max):
                    raise Exception(
                        """
                        t < t0 or t > t_max: solve_ivp returns an
                        interpolated function, which does not check if the
                        functions is called with arguments outside the computed
                        range, but we do.
                        """
                    )
                else:
                    return func(times)

        return save_func

    def join_functions_rc(self, funcs):
        def func(t):
            index = np.where(self.boundaries[1:] > t)[0][0]
            index = min(index, len(self.boundaries)-2)
            res = funcs[index](t)
            return res

        return func

    def join_flux_funcss_rc(self, flux_funcss, key):
        L = []
        for f in flux_funcss:
            L.append(f.get(key, lambda t: 0))

        return self.join_functions_rc(L)

    def fake_discretized_Bs(self, data_times=None):
        if data_times is None:
            data_times = self.times

        nr_pools = self.nr_pools
        ldt = len(data_times)-1
        Bs = np.zeros((ldt, nr_pools, nr_pools))

        for k in range(ldt):
            Bs[k, :, :] = self.Phi(data_times[k+1], data_times[k])

        return Bs

    def Phi(self, T, S):
        nr_pools = self.nr_pools
        start_Phi_2d = np.identity(nr_pools)

        if S > T:
            raise(Error("Evaluation before S is not possible"))
        if S == T:
            return start_Phi_2d

        solve_func = self.solve_func()
        block_ode, x_block_name, phi_block_name = self._x_phi_block_ode()

        if hasattr(self, '_state_transition_operator_cache'):
            cache = self._state_transition_operator_cache
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
            T_phi_ind = cache.phi_ind(T)

            # catch the corner cases where the cache is useless.
            if (T_phi_ind-S_phi_ind) < 1:
                return phi(T, S, t_max=cache.end_time_from_phi_ind(T_phi_ind))
            tm1 = cache.end_time_from_phi_ind(S_phi_ind)

            # first integrate to tm1:
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

                return blivp.block_solve(
                    t_span=(s, t)
                )[phi_block_name][-1, ...]

            return phi(T, S)

    def _x_phi_block_ode(self):
        x_block_name = 'x'
        phi_block_name = 'phi'
        if not(hasattr(self, '_x_phi_block_ode_cache')):
            block_ode = x_phi_ode(
                self.model,
                self.parameter_dicts,
                self.func_dicts,
                x_block_name,
                phi_block_name,
                disc_times=self.disc_times
            )
            self._x_phi_block_ode_cache = block_ode
        return self._x_phi_block_ode_cache, x_block_name, phi_block_name

    def initialize_state_transition_operator_cache(
        self,
        lru_maxsize,
        lru_stats=False,
        size=1
    ):
        custom_lru_cache = custom_lru_cache_wrapper(
            maxsize=lru_maxsize,  # variable maxsize now for lru cache
            typed=False,
            stats=lru_stats  # use custom statistics feature
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

    def myhash(self):
        """
        Compute a hash considering SOME but NOT ALL properties of a
        model run. The function's main use is to detect saved state transition
        operator cashes that are no longer compatible with the model run object
        that wants to use them. This check is useful but NOT COMPREHENSIVE.
        """
        times = self.times

        def make_hash_sha256(o):
            hasher = hashlib.sha256()
#            hasher.update(repr(make_hashable(o)).encode())
            hasher.update(repr(o).encode())
            return base64.b64encode(hasher.digest()).decode()

        return make_hash_sha256(
            (
                frozendict(self.model.input_fluxes),
                frozendict(self.model.internal_fluxes),
                frozendict(self.model.output_fluxes),
                ImmutableMatrix(self.model.state_vector),
                self.parameter_dicts,
                self.start_values,
                (times[0], times[-1]),
                tuple(self.disc_times)
            )
        )
