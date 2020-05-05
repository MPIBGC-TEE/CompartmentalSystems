from numbers import Number
import numpy as np
from frozendict import frozendict
from sympy import flatten, lambdify
from scipy.integrate import quad

from .model_run import ModelRun
from .helpers_reservoir import (
    numsol_symbolical_system,
    check_parameter_dict_complete,
    make_cut_func_set,
    f_of_t_maker,
    const_of_t_maker
)


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
        if not disc_times:
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
        pass
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

    def acc_net_external_input_vector(self):
        pass

    def acc_net_internal_flux_matrix(self):
        pass

    def acc_net_external_output_vector(self):
        pass

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
            index = np.where(self.boundaries >= t)[0][0] - 1
            index = max(0, index)
            res = funcs[index](t)
            return res

        return func

    def join_flux_funcss_rc(self, flux_funcss, key):
        L = []
        for f in flux_funcss:
            L.append(f.get(key, lambda t: 0))

        return self.join_functions_rc(L)
