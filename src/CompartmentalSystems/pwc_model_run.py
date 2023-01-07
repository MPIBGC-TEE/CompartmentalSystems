from numbers import Number
import numpy as np
from frozendict import frozendict
from sympy import flatten, lambdify, ImmutableMatrix
from scipy.integrate import quad
import hashlib
import base64
from tqdm import tqdm

from .model_run import ModelRun
from .smooth_model_run import SmoothModelRun
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
    numerical_function_from_expression,
    warning
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
        func_dicts=None,
        no_check=False
    ):
        if len(disc_times) == 0:
            raise(Error("No 'disc_times' given"))

        self.disc_times = disc_times

        if parameter_dicts is None:
            parameter_dicts = (dict()) * (len(disc_times)+1)

        if func_dicts is None:
            func_dicts = (dict()) * (len(disc_times)+1)

        # check parameter_dicts + func_dicts for completeness
        if not no_check:
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
        dt_list = [t for t in self.disc_times]
        res = np.array([times[0]] + dt_list + [times[-1]])
        return res

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
            B_k = func_maker(pd, fd)
            L.append(B_k)

        return self.join_functions_rc(L)

    def external_input_vector_func(self, cut_off=True):
        if not hasattr(self, '_external_input_vector_func'):
            t0 = self.times[0]
            # cut off inputs until t0 (exclusive)
            if cut_off:
                t_valid = lambda t: True if ((t0 <= t) and  # noqa: E731
                                (t <= self.times[-1])) else False
            else:
                t_valid = lambda t: True  # noqa: E731

            L = []

            def func_maker(external_input_flux_funcs):
                input_fluxes = []
                for i in range(self.nr_pools):
                    if i in external_input_flux_funcs.keys():
                        input_fluxes.append(
                            external_input_flux_funcs[i]
                        )
                    else:
                        input_fluxes.append(lambda t: 0)

                u = lambda t: (  # noqa: #731
                    np.array(
                        [f(t) for f in input_fluxes],
                        dtype=np.float
                    )
                    if t_valid(t) else np.zeros((self.nr_pools,))
                )
                return u

            for external_input_flux_funcs in self.external_input_flux_funcss():
                L.append(func_maker(external_input_flux_funcs))

            self._external_input_vector_func = self.join_functions_rc(L)

        return self._external_input_vector_func


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
                tmp = quad(flux_func, times[k], times[k+1])
#                print(times[k], times[k+1], tmp)
#                input()
                res[k, pool_nr] = tmp[0]

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

    @staticmethod
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

#    def age_moment_vector_semi_explicit(
#        self,
#        order,
#        start_age_moments=None,
#        times=None
#    ):
#        """Compute the ``order`` th moment of the pool ages by a semi-explicit 
#        formula.
#
#        This function bases on a semi-explicit formula such that no improper 
#        integrals need to be computed.
#        
#        Args:
#            order (int): The order of the age moment to be computed.
#            start_age_moments (numpy.ndarray order x nr_pools, optional): 
#                Given initial age moments up to the order of interest. 
#                Can possibly be computed by :func:`moments_from_densities`. 
#                Defaults to ``None`` assuming zero initial ages.
#            times (numpy.array, optional): Time grid. 
#                Defaults to ``None`` and the original time grid is used.
#
#        Returns:
#            numpy.ndarray: len(times) x nr_pools.
#            The ``order`` th pool age moments over the time grid.
#        """
#            
#        if times is None: times = self.times
#        t0 = times[0]
#        n = self.nr_pools
#        k = order
#        
#        if start_age_moments is None:
#            start_age_moments = np.zeros((order, n))
#
#        start_age_moments[np.isnan(start_age_moments)] = 0
#
#        p2_sv = self._age_densities_2_single_value()
#
#        def binomial(n, k):
#            return 1 if k==0 else (0 if n==0 
#                                    else binomial(n-1, k) + binomial(n-1, k-1))
#
#        def Phi_x(t, t0, x):
#            return np.matmul(self.Phi(t, t0), x)
#
#        def x0_a0_bar(j):
#            if j == 0: 
#                return self.start_values
#                
#            return np.array(self.start_values) * start_age_moments[j-1,:]
#
#        def both_parts_at_time(t):
#            def part2_time(t):
#                def part2_time_index_pool(ti, pool):
#                    return quad(lambda a: a**k * p2_sv(a, t)[pool], 0, t-t0)[0]
#
#                return np.array([part2_time_index_pool(t, pool) 
#                                    for pool in range(n)])
#
#            def part1_time(t):
#                def summand(j):
#                    return binomial(k, j)*(t-t0)**(k-j)*Phi_x(t, t0, x0_a0_bar(j))
#
#                return sum([summand(j) for j in range(k+1)])
#
#            return part1_time(t) + part2_time(t)
#
#        #soln = self.solve_old()
#        soln = self.solve()
#
#        def both_parts_normalized_at_time_index(ti):
#            t = times[ti]
#            bp = both_parts_at_time(t)
#            diag_values = np.array([x if x>0 else np.nan for x in soln[ti,:]])
#            X_inv = np.diag(diag_values**(-1))
#
#            #return (np.mat(X_inv) * np.mat(bp).transpose()).A1
#            return (np.matmul(X_inv, bp).transpose()).flatten()
#
#        return np.array([both_parts_normalized_at_time_index(ti) 
#                            for ti in range(len(times))])
        
    def age_moment_vector_up_to(self, up_to_order, start_age_moments=None):
        """Compute the pool age moment vectors up to ``up_to_order`` over
        the time grid by an ODE system.

        This function solves an ODE system to obtain the pool age moments very
        fast. If the system has empty pools at the beginning, the semi-explicit 
        formula is used until all pools are non-empty. Then the ODE system 
        starts.

        Args:
            up_to_order (int): The order up to which the pool age moments
                are to be computed.
            start_age_moments (numpy.ndarray order x nr_pools, optional): 
                Given initial age moments up to the order of interest. 
                Can possibly be computed by :func:`moments_from_densities`. 
                Defaults to None assuming zero initial ages.

        Returns:
            numpy.ndarray: len(times) x order x nr_pools.
            The orders up to ``up_to_order`` of the  pool age moments
            over the time grid.
        """
        n = self.nr_pools
        times = self.times
        
        if start_age_moments is None:
            start_age_moments = np.zeros((up_to_order, n))
        
        max_order=start_age_moments.shape[0]
        if up_to_order>max_order:
            raise Error("""
                To solve the moment system with order{0}
                start_age_moments up to (at least) the same order have to be
                provided. But the start_age_moments.shape was
                {1}""".format(up_to_order, start_age_moments.shape)
            )
        if up_to_order<max_order:
            warning("""
                Start_age_moments contained higher order values than needed.
                start_age_moments order was {0} while the requested order was
                {1}. This is no problem but possibly unintended. The higer
                order moments will be clipped """.format(max_order, up_to_order)
            )
            # make sure that the start age moments are clipped to the order
            # (We do not need start values for higher moments and the clipping
            # avoids problems with recasting if higher order moments are given 
            # by the user)
            start_age_moments=start_age_moments[0:order,:]

        if not (0 in self.start_values):
            #ams = self._solve_age_moment_system_old(order, start_age_moments)
            ams, _ = self._solve_age_moment_system(up_to_order, start_age_moments)
            return ams.reshape((-1, up_to_order+1, n))
        else:
            raise(ValueError('At least one pool is empty at the beginning.'))
#            # try to start adapted mean_age_system once no pool 
#            # has np.nan as mean_age (empty pool)
#
#            # find last time index that contains an empty pool --> ti
#            #soln = self.solve_old()
#            soln = self.solve()
#            ti = len(times)-1
#            content = soln[ti,:]
#            while not (0 in content) and (ti>0): 
#                ti = ti-1
#                content = soln[ti,:]
#
#            # not forever an empty pool there?
#            if ti+1 < len(times):
#                # compute moment with semi-explicit formula 
#                # as long as there is an empty pool
#                amv1_list = []
#                amv1 = np.zeros((ti+2, order*n))
#                for k in range(1, order+1):
#                    amv1_k = self.age_moment_vector_semi_explicit(
#                        k, start_age_moments, times[:ti+2])
#                    amv1[:,(k-1)*n:k*n] = amv1_k
#
#                # use last values as start values for moment system 
#                # with nonzero start values
#                new_start_age_moments = amv1[-1,:].reshape((n, order))
#                start_values = soln[ti+1]
#                #ams = self._solve_age_moment_system_old(
#                #    order, new_start_age_moments, times[ti+1:], start_values)
#                ams,_ = self._solve_age_moment_system(
#                    order, new_start_age_moments, start_values, times[ti+1:])
#                amv2 = ams[:,n*order:]
#
#                # put the two parts together
#                part1 = amv1[:,(order-1)*n:order*n][:-1]
#                amv = np.ndarray((len(times), n))
#                amv[:part1.shape[0], :part1.shape[1]] = part1
#                amv[part1.shape[0]:, :amv2.shape[1]] = amv2
#                return amv
#            else:
#                # always an empty pool there
#                return self.age_moment_vector_semi_explicit(
#                        order, start_age_moments)


    def age_moment_vector(self, order, start_age_moments=None):
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
        amvs_up_to_order = self.age_moment_vector_up_to(order, start_age_moments)
        return amvs_up_to_order[:, order, :]


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
        soln = self.solve()
         
        total_mass = soln.sum(1) # row sum
        total_mass[total_mass==0] = np.nan

        system_age_moment = (age_moment_vector*soln).sum(1)/total_mass

        return system_age_moment

    @property
    def external_output_vector(self):
        """Return the grid of external output vectors.

        Returns:
            numpy.ndarray: len(times) x nr_pools
        """
        if not hasattr(self, "_external_output_vector"):
            times = self.times
            nt = len(times)

            flux_funcss = self.external_output_flux_funcss()
            res = np.zeros((nt, self.nr_pools))
            for pool_nr in range(self.nr_pools):
                flux_func = self.join_flux_funcss_rc(flux_funcss, pool_nr)
                for k in range(nt):
                    res[k, pool_nr] = flux_func(times[k])

            self._external_output_vector = res

        return self._external_output_vector

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

    def cumulative_pool_age_distributions_single_value(
        self, 
        start_age_densities=None,
        F0=None
    ):
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

        def G_sv(a, t):
            if a < t-t0: return np.zeros((n,))
            res = np.matmul(self.Phi(t, t0), F0(a-(t-t0))).reshape((self.nr_pools,))
            return res

        def H_sv(a, t):
            # count everything from beginning?
            if a >= t-t0: 
                a = t-t0

            # mass at time t
            #x_t_old = np.array([sol_funcs[pool](t) for pool in range(n)])
            x_t = sol_funcs_array(t)
            # mass at time t-a
            #x_tma_old = [float(sol_funcs[pool](t-a)) for pool in range(n)]
            x_tma = sol_funcs_array(t-a)
            # what remains from x_tma at time t
            m = np.matmul(self.Phi(t, t-a), x_tma).reshape((self.nr_pools,))
            # difference is not older than t-a
            res = x_t-m
            # cut off accidental negative values
            return np.maximum(res, np.zeros(res.shape))

        def F(a, t):
            res = G_sv(a,t) + H_sv(a,t)
            return res

        return F

    def cumulative_backward_transit_time_distribution_single_value_func(
        self,
        start_age_densities=None,
        F0=None
    ):
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
            start_age_densities=start_age_densities,
            F0=F0
        )
#        rho = self.output_rate_vector_at_t
        
        B_func = self.B_func()
        rho = lambda t: -B_func(t).sum(0)

        def F_btt_sv(a, t):
            res = (rho(t)*F_sv(a, t)).sum()
            #print(a, t, res)
            return res

        return F_btt_sv

    def backward_transit_time_quantiles(
        self,
        quantile,
        F0, # cumulative start age distribution (not normalized)
#        start_values=None,
        time_indices=None,
        method="brentq",
        tol=1e-08
    ):
        times = self.times
        norm_consts = self.external_output_vector.sum(axis=1)
        if time_indices is not None:
            times = times[time_indices]
            norm_consts = norm_consts[time_indices]
        
#        if start_values is None:
#            start_values = np.zeros_like(times)

        F_btt_sv = self.cumulative_backward_transit_time_distribution_single_value_func(
            F0=F0
        )

        res = []
        xi = 0.0
        for k in tqdm(range(len(times))):
            xi = SmoothModelRun.distribution_quantile(
                quantile, 
                lambda a: F_btt_sv(a, times[k]), 
                norm_const=norm_consts[k],
#                start_value=start_values[k],
                start_value=xi,
                method=method,
                tol=tol
            )
            res.append(xi)

        return np.array(res)

    def pool_age_distributions_quantiles(
        self,
        quantile,
        start_values=None, 
        start_age_densities=None,
        F0=None,
        method='brentq',
        tol=1e-8
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
            start_age_densities=start_age_densities,
            F0=F0
        )

        res = []
        for pool in range(n):
            print('Pool:', pool)
            F_sv_pool = lambda a, t: F_sv(a,t)[pool]
            res.append(
                SmoothModelRun.distribution_quantiles(
                    self,
                    quantile,
                    F_sv_pool,
                    norm_consts=soln[:,pool],
                    start_values=start_values[:,pool],
                    method=method,
                    tol=tol
                )
        )

        return np.array(res).transpose()
    
    def cumulative_system_age_distribution_single_value(
        self, 
        start_age_densities=None, F0=None
    ):
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
            start_age_densities=start_age_densities,
            F0=F0
        )
        
        return lambda a, t: F_sv(a,t).sum()

    def system_age_distribution_quantiles(
            self,
            quantile,
            start_values=None, 
            start_age_densities=None,
            F0=None,
            method='brentq',
            tol=1e-8
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
        if soln[0, :].sum() == 0:
            start_age_densities = lambda a: np.zeros((n,))

        if F0 is None and start_age_densities is None:
            raise(Error('Either F0 or start_age_densities must be given.'))
        
        F_sv = self.cumulative_system_age_distribution_single_value(
            start_age_densities=start_age_densities,
            F0=F0
        )

        if start_values is None:
            start_values = np.ones((len(self.times), ))

        a_star = SmoothModelRun.distribution_quantiles(
            self,
            quantile, 
            F_sv, 
            norm_consts=soln.sum(1), 
            start_values=start_values, 
            method=method,
            tol=tol
        )

        return a_star


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
            if t == self.boundaries[-1]:
                return funcs[-1](t)

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
