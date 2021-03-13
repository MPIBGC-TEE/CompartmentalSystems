import numpy as np

from scipy.integrate import solve_ivp, quad
from scipy.linalg import expm
from scipy.optimize import root, least_squares
from sympy import Matrix, symbols, zeros, Symbol
from tqdm import tqdm

from .helpers_reservoir import ALPHA_14C
from .model_run import ModelRun
from .pwc_model_run import PWCModelRun
from .smooth_reservoir_model import SmoothReservoirModel

from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel as LAPM


###############################################################################


class PWCModelRunFDError(Exception):
    """Generic error occurring in this module."""
    pass


###############################################################################


class PWCModelRunFD(ModelRun):
#    def __init__(
#        self,
#        time_symbol,
#        data_times,
#        start_values,
#        gross_Us,
#        gross_Fs,
#        gross_Rs
#    ):
#
#        self.data_times = data_times
#        cls = self.__class__
#        disc_times = data_times[1:-1]
#
#        print('reconstructing us')
##        us = cls.reconstruct_us(data_times, gross_Us)
##        self.dts = np.diff(self.data_times).astype(np.float64)
#        us = gross_Us / self.dts.reshape(-1, 1)
#
#        print('reconstructing Bs')
#        Bs = cls.reconstruct_Bs(
#            data_times,
#            start_values,
#            gross_Us,
#            gross_Fs,
#            gross_Rs
#        )
#
#        
#
#        nr_pools = len(start_values)
#        strlen = len(str(nr_pools))
#
#        def pool_str(i): return ("{:0"+str(strlen)+"d}").format(i)
#        par_dicts = [dict()] * len(us)
#
#        srm_generic = cls.create_srm_generic(
#            time_symbol,
#            Bs,
#            us
#        )
#        time_symbol = srm_generic.time_symbol
#
#        par_dicts = []
#        for k in range(len(us)):
#            par_dict = dict()
#            for j in range(nr_pools):
#                for i in range(nr_pools):
#                    sym = Symbol('b_'+pool_str(i)+pool_str(j))
#                    if sym in srm_generic.F.free_symbols:
#                        par_dict[sym] = Bs[k, i, j]
#
#            for i in range(nr_pools):
#                sym = Symbol('u_'+pool_str(i))
#                if sym in srm_generic.F.free_symbols:
#                    par_dict[sym] = us[k, i]
#
#            par_dicts.append(par_dict)
#
#        func_dicts = [dict()] * len(us)
#
#        self.pwc_mr = PWCModelRun(
#            srm_generic,
#            par_dicts,
#            start_values,
#            data_times,
#            func_dicts=func_dicts,
#            disc_times=disc_times
#        )
#        self.us = us
#        self.Bs = Bs

    def __init__(self, Bs, us, pwc_mr):
        self.data_times = pwc_mr.times
        self.pwc_mr = pwc_mr
        self.Bs = Bs
        self.us = us

    @classmethod
    def from_Bs_and_us(cls, time_symbol, data_times, start_values, Bs, us):
        isnan = np.isnan(start_values).sum() + np.isnan(Bs).sum() + np.isnan(us).sum()
        if isnan > 0:
            raise(PWCModelRunFDError("Nans detected"))
        
        disc_times = data_times[1:-1]

        nr_pools = len(start_values)
        strlen = len(str(nr_pools))

        def pool_str(i): 
            return ("{:0"+str(strlen)+"d}").format(i)

        par_dicts = [dict()] * len(us)

        srm_generic = cls.create_srm_generic(
            time_symbol,
            Bs,
            us
        )

        par_dicts = []
        for k in range(len(us)):
            par_dict = dict()
            for j in range(nr_pools):
                for i in range(nr_pools):
                    sym = Symbol('b_'+pool_str(i)+pool_str(j))
                    if sym in srm_generic.F.free_symbols:
                        par_dict[sym] = Bs[k, i, j]

            for i in range(nr_pools):
                sym = Symbol('u_'+pool_str(i))
                if sym in srm_generic.F.free_symbols:
                    par_dict[sym] = us[k, i]

            par_dicts.append(par_dict)

        func_dicts = [dict()] * len(us)

#        print("loading PWCModelRun", flush=True) # leaves a message to scheduler, prevents timeout
        pwc_mr = PWCModelRun(
            srm_generic,
            par_dicts,
            start_values,
            data_times,
            disc_times=disc_times,
            func_dicts=func_dicts,
            no_check=True
        )
        
        return cls(Bs, us, pwc_mr)

    @classmethod
    def from_gross_fluxes(
        cls,
        time_symbol,
        data_times,
        start_values,
        gross_Us,
        gross_Fs,
        gross_Rs,
        xs=None,
        integration_method='solve_ivp',
        nr_nodes=None,
        check_success=True
    ):
        us = cls.reconstruct_us(
            data_times,
            gross_Us
        )

#        print(
#            'reconstructing Bs, using integration_method =',
#            integration_method
#        )
        Bs, _, _ = cls.reconstruct_Bs(
            data_times,
            start_values,
            gross_Us,
            gross_Fs,
            gross_Rs,
            xs,
            integration_method,
            nr_nodes,
            check_success
        )

        return cls.from_Bs_and_us(
            time_symbol,
            data_times,
            start_values,
            Bs,
            us
        )

    @property
    def model(self):
        return self.pwc_mr.model

    @property
    def nr_pools(self):
        return self.pwc_mr.nr_pools

    @property
    def dts(self):
        return np.diff(self.data_times).astype(np.float64)

    @property
    def start_values(self):
        return self.pwc_mr.start_values

    def solve(self, alternative_start_values=None):
        return self.pwc_mr.solve(alternative_start_values)

    def B_func(self, vec_sol_func=None):
        return self.pwc_mr.B_func(vec_sol_func)

    @property
    def times(self):
        return self.data_times

    def external_input_vector_func(self, cut_off=True):
        return self.pwc_mr.external_input_vector_func(cut_off)

    def acc_gross_external_input_vector(self):
        return self.us*self.dts.reshape(-1, 1)

    def acc_gross_internal_flux_matrix(self):
        return self.pwc_mr.acc_gross_internal_flux_matrix()

    def acc_gross_external_output_vector(self):
        return self.pwc_mr.acc_gross_external_output_vector()

    def acc_net_external_input_vector(self):
        return self.pwc_mr.acc_net_external_input_vector()

    def acc_net_internal_flux_matrix(self):
        return self.pwc_mr.acc_net_internal_flux_matrix()

    def acc_net_external_output_vector(self):
        return self.pwc_mr.acc_net_external_output_vector()

    def fake_discretized_Bs(self, data_times=None):
        return self.pwc_mr.fake_discretized_Bs(data_times)

#    def age_moment_vector_semi_explicit(
#        self,
#        order,
#        start_age_moments=None,
#        times=None
#    ):
#        return self.pwc_mr.age_moment_vector_semi_explicit(
#            order,
#            start_age_moments
#            times
#        )

    def fake_eq_model(self, nr_time_steps):
        # average the first `nr_time_steps` us and Bs
        mean_u = Matrix(self.us[:nr_time_steps, ...].mean(axis=0))
        mean_B = Matrix(self.Bs[:nr_time_steps, ...].mean(axis=0))
                            
        eq_model = LAPM(
            mean_u,
            mean_B,
            force_numerical=True
        )
        return eq_model

    def fake_eq_14C(
        self,
        nr_time_steps,
        F_atm,
        decay_rate,
        lim=np.inf,
        alpha=None
    ):
        if alpha is None:
            alpha = ALPHA_14C

        eq_model = self.fake_eq_model(nr_time_steps)
#        E_a = np.array(eq_model.a_expected_value).reshape(-1)
#        for i in range(len(E_a)):
#            print(E_a[i], E_a[i]/365.25, F_atm(np.float(E_a[i])))
        f0_normalized = lambda a: np.array(
            eq_model.a_density(a),
            dtype=np.float64).reshape((-1,)
        )
        f0 = lambda a: f0_normalized(a) * self.start_values

        eq_14C = np.nan * np.ones((self.nr_pools, ))
        for pool in range(self.nr_pools):
            f0_pool = lambda a: f0(a)[pool]
            def f0_pool_14C(a):
                res = (
                    (F_atm(a)/1000.0 + 1) * 
                    f0_pool(a)
#                    * alpha # makes integration imprecise
                     * np.exp(-decay_rate*a)
                )
                return res

            # integration via solve_ivp is fast and successful
            res = solve_ivp(
                lambda a, y: f0_pool_14C(a),
                (0, lim),
                np.array([0])
            )
#            print("c", res.y.reshape(-1)[-1])
#            print(E_a[pool]/365.25, F_atm(np.float(E_a[pool])))

            eq_14C[pool] = res.y.reshape(-1)[-1] * alpha
        
#        print(eq_14C/self.start_values/alpha)
        return eq_14C


    def fake_start_age_moments(self, nr_time_steps, up_to_order):
        eq_model = self.fake_eq_model(nr_time_steps)

        start_age_moments = np.stack(
            [
                np.array(eq_model.a_nth_moment(order), dtype=np.float64).reshape((-1,))
                for order in range(1, up_to_order+1)
            ],
            axis=0
        )
        return start_age_moments

    def fake_cumulative_start_age_distribution(self, nr_time_steps):
        eq_model = self.fake_eq_model(nr_time_steps)

        F0_normalized = lambda a: np.array(eq_model.a_cum_dist_func(a), dtype=np.float64).reshape((-1,))
        F0 = lambda a: F0_normalized(a) * self.start_values

        return F0

    def age_moment_vector_up_to(self, up_to_order, start_age_moments=None):
        return self.pwc_mr.age_moment_vector_up_to(
            up_to_order,
            start_age_moments
        )

    def age_moment_vector(self, order, start_age_moments=None):
        return self.pwc_mr.age_moment_vector(
            order,
            start_age_moments
        )

    def system_age_moment(self, order, start_age_moments=None):
        return self.pwc_mr.system_age_moment(
            order,
            start_age_moments
        )

    def backward_transit_time_moment(self, order, start_age_moments=None):
        return self.pwc_mr.backward_transit_time_moment(
            order,
            start_age_moments
        )

    @property
    def external_output_vector(self):
        return self.pwc_mr.external_output_vector

    def cumulative_backward_transit_time_distribution_single_value_func(
        self,
        start_age_densities=None,
        F0=None
    ):
        return self.pwc_mr.cumulative_backward_transit_time_distribution_single_value_func(
            start_age_densities,
            F0
        )

    def backward_transit_time_quantiles(
        self,
        quantile,
        F0, # cumulative star age distribution (not normalized)
        time_indices=None,
        method="brentq",
        tol=1e-08
    ):
        return self.pwc_mr.backward_transit_time_quantiles(
            quantile,
            F0,
            time_indices,
            method,
            tol
        )

    def initialize_state_transition_operator_cache(
        self,
        lru_maxsize,
        lru_stats=False,
        size=1
    ):
        self.pwc_mr.initialize_state_transition_operator_cache(
            lru_maxsize,
            lru_stats,
            size
        )
        self._state_transition_operator_cache = self.pwc_mr._state_transition_operator_cache

    def pool_age_distributions_quantiles(
        self,
        quantile,
        start_values=None, 
        start_age_densities=None,
        F0=None,
        method='brentq',
        tol=1e-8
    ):
        return self.pwc_mr.pool_age_distributions_quantiles(
            quantile,
            start_values, 
            start_age_densities,
            F0,
            method,
            tol
        )

    def system_age_distribution_quantiles(
        self,
        quantile,
        start_values=None, 
        start_age_densities=None,
        F0=None,
        method='brentq',
        tol=1e-8
    ):
        return self.pwc_mr.system_age_distribution_quantiles(
            quantile,
            start_values, 
            start_age_densities,
            F0,
            method,
            tol
        )

#    moved to ModelDataObject
#    def to_netcdf(self, mdo, file_path):
#        """Return a netCDF dataset that contains stocks and fluxes."""
#
#        data_vars = {}
#        model_ds = xr.Dataset(
#            data_vars=data_vars,
#            coords=coords,
#            attr=attrs
#        }
#        model_ds.to_netcdf(file_path)
#        model_ds.close()
#
#    @classmethod
#    def load_from_file(cls, filename):
#        pwc_mr_fd_dict = picklegzip.load(filename)
#
#        return cls.load_from_dict(pwc_mr_fd_dict)
#
#
#    def save_to_file(self, filename):
#        soln = self.solve()
#        self.soln = soln
##        pwc_mr = self.to_smooth_model_run()
##        soln1 = pwc_mr.solve()
##        self.soln1 = soln1
##        soln2, soln2_func = pwc_mr.solve_2()
##        self.soln2 = soln2
#        pwc_mr_fd_dict = {
#            'start_values': self.start_values,
#            'time_symbol':  self.model.time_symbol,
#            'Bs':           self.Bs,
#            'us':           self.us,
#            'data_times':   self.data_times,
##            'times':        self.times
#            'soln':         soln,
##            'soln1':        soln1,
##            'soln2':        soln2,
#            'location':     getattr(self, 'location', None)
#        }
#
#        picklegzip.dump(pwc_mr_fd_dict, filename)
#
#
#    def to_smooth_model_run(self):
#        smr = SmoothModelRun(
#            self.model,
#            self.parameter_set,
#            self.start_values,
#            self.data_times,
#            self.func_set
#        )
#
#        return smr
#
#
    
    @classmethod
    def reconstruct_us(cls, data_times, gross_Us):
        dts = np.diff(data_times).astype(np.float64)
        us = gross_Us / dts.reshape(-1, 1)
        
        return us


    @classmethod
    def reconstruct_B_surrogate(
        cls,
        dt,
        x0,
        gross_U,
        gross_F,
        gross_R,
        B0,
        integration_method,
        nr_nodes,
        check_success=True
    ):
        # reconstruct a B that meets F and r possibly well,
        # B0 is some initial guess for the optimization
        nr_pools = len(x0)
        gross_u = gross_U/dt

        def x_trapz(tr_times, B):
            def x_of_tau(tau):
                M = expm(tau*B)
                if np.abs(np.linalg.det(B)) > 1e-16:
                    return M @ x0 + np.linalg.inv(B) @ (-np.identity(nr_pools)+M) @ gross_u
                else:
#                    tr_sub_times = np.linspace(0, tau, nr_nodes)
#                    xs_sub = np.array(
#                        list(
#                            map(lambda s: expm((tau-s)*B), tr_sub_times)
#                        )
#                    )
#                    return M @ x0 + np.trapz(xs_sub, tr_sub_times, axis=0) @ gross_u

                    x = solve_ivp(
#                        lambda s, _: expm((tau-s)*B) @ gross_u,
                        lambda _, xi: (B @ xi) + gross_u,
                        t_span=(0, tau),
                        y0=x0,
                        method="Radau"
                    ).y[..., -1]
                    return x
    
            x_vals = np.array(list(map(x_of_tau, tr_times)))
            x1 = x_of_tau(tr_times[-1])
            return np.trapz(x_vals, tr_times, axis=0), x1

        def x_tau(tau, B):
            if np.abs(np.linalg.det(B)) > 1e-16:
                M = expm(tau*B)
                return M @ x0 + np.linalg.inv(B) @ (-np.identity(nr_pools)+M) @ gross_u
            else:
                x = solve_ivp(
                    #lambda s, _: expm((tau-s)*B) @ gross_u,
                    lambda _, xi: (B @ xi) + gross_u,
                    t_span=(0, tau),
                    y0=x0,
                    method="Radau"
                ).y[..., -1]
            return x

        # integrate x
        def integrate_x(tr_times, B):
            def rhs(tau, X):
                return x_tau(tau, B)

#            xs = np.array(list(map(x, tr_times))), x(tr_times[-1])
#            int_x np.trapz(xs, tr_times, axis=0)
#            int_x = solve_ivp_pwc(
#                rhss=[rhs],
#                t_span=(tr_times[0], tr_times[-1]),
#                y0=np.zeros_like(x0)
#            ).y[..., -1]
            int_x = solve_ivp(
                rhs,
                t_span=(tr_times[0], tr_times[-1]),
                y0=np.zeros_like(x0),
                method="Radau"
            ).y[..., -1]
            return int_x, x_tau(tr_times[-1], B)

        # convert parameter vector to compartmental matrix
        def pars_to_matrix(pars):
            B = np.zeros((nr_pools**2))
            B[int_indices.tolist()] = pars[:len(int_indices)]
            B = B.reshape((nr_pools, nr_pools))
            d = B.sum(0)
            #print("d", d)
            d[(ext_indices-nr_pools**2).tolist()] += pars[len(int_indices):]
#            if (len(ext_indices) > 3) and (pars[len(int_indices):][3] < 0):
#                print(551, pars)
#                print(pars[len(int_indices):], pars[len(int_indices):].shape)
            B[np.diag_indices(nr_pools)] = -d

            #print("=====================")
            #print(pars)
            #print(d)
            #print(B)
            #print("nnnnnnnnnnnnnnnnnnnn")
            return B

        # function to minimize difference vector of
        # internal fluxes F and outfluxes r
        def g_tr(pars, integration_method, nr_nodes):
            B = pars_to_matrix(pars)

            if integration_method == 'solve_ivp':
                tr_times = (0, dt)
                int_x, x_end = integrate_x(tr_times, B)
            elif integration_method == 'trapezoidal':
                if nr_nodes is None:
                    raise PWCModelRunFDError(
                        'For trapezoidal rule nr_nodes is obligatory'
                    )
                tr_times = np.linspace(0, dt, nr_nodes)
                int_x, x_end = x_trapz(tr_times, B)
            else:
                raise PWCModelRunFDError('Invalid integration_method')

            res1_int = gross_F.reshape((nr_pools**2,))[int_indices.tolist()]
            res2_int = pars[:len(int_indices)] \
                * int_x[(int_indices % nr_pools).tolist()]
            res_int = np.abs(res1_int-res2_int)
            
            tmp_F = np.zeros_like(gross_F).reshape(nr_pools**2)
            tmp_F[int_indices.tolist()] = res2_int
            tmp_F = tmp_F.reshape(nr_pools, nr_pools)
#            print(gross_F-tmp_F)

            res1_ext = gross_R[(ext_indices-nr_pools**2).tolist()]
            res2_ext = pars[len(int_indices):]\
                * int_x[(ext_indices-nr_pools**2).tolist()]
            res_ext = np.abs(res1_ext-res2_ext)
            
            tmp_R = np.zeros_like(gross_R)
            tmp_R[(ext_indices-nr_pools**2).tolist()] = res2_ext
#            print()
#            print(gross_R)
#            print(tmp_R)
#            print(gross_R-tmp_R)

            res = np.append(res_int, res_ext)
            #print(B.sum(0))
            #print(pars)
            #print("B")
            #print(B)
            #res += sum(B.sum(0) > 0) * 1e10
#            res = res ** 2

#            print(np.sum(res))
#            print("res.shape", res.shape)
            return res

        # wrapper around g_tr that keeps parameter within boundaries
        def constrainedFunction(
            x,
            f,
            lower,
            upper,
            integration_method,
            nr_nodes,
            minIncr=0.001
        ):
            x = np.asarray(x)
            s = pars_to_matrix(x).sum(0)
            penalty = np.sum(np.where(s > 0, s, 0)) * 1e5

            lower = np.asarray(lower)
            upper = np.asarray(upper)

            xBorder = np.where(x < lower, lower, x)
            xBorder = np.where(x > upper, upper, xBorder)
            fBorder = f(xBorder, integration_method, nr_nodes)
            distFromBorder = (
                np.sum(np.where(x < lower, lower-x, 0.))
                + np.sum(np.where(x > upper, x-upper, 0.))
            )
            res = (fBorder + (
                        fBorder + np.where(fBorder > 0, minIncr, -minIncr)
                    ) * distFromBorder
                    ) + penalty
#            print(sum(res))
            return res

        lbounds = [0]*(nr_pools**2 + nr_pools)
        for i in range(nr_pools):
            lbounds[i*nr_pools+i] = -10

        ubounds = [10]*(nr_pools**2 + nr_pools)
#        ubounds = [1]*(nr_pools**2 + nr_pools)
        for i in range(nr_pools):
            ubounds[i*nr_pools+i] = 0

        par_indices = []
        for i in range(nr_pools):
            for j in range(nr_pools):
                if (gross_F[i, j] > 0):
                    par_indices.append(i*nr_pools+j)
        for i in range(nr_pools):
            if gross_R[i] > 0:
                par_indices.append(nr_pools**2+i)

        par_indices = np.array(par_indices)
        int_indices = par_indices[par_indices < nr_pools**2]
        ext_indices = par_indices[par_indices >= nr_pools**2]

        lbounds = np.array(lbounds)[par_indices.tolist()]
        ubounds = np.array(ubounds)[par_indices.tolist()]
        A0 = np.append(B0.reshape((nr_pools**2,)), -B0.sum(0))
        pars0 = A0[par_indices.tolist()]

        methods = ["hybr", "lm"]
        for nr, method in enumerate(methods):
            y = root(
                constrainedFunction,
                x0=pars0,
                args=(g_tr, lbounds, ubounds, integration_method, nr_nodes),
                method=method,
    #            options={"xtol": 1e-08}
            )
    #        print(np.max(y.fun))
    
    #        y = least_squares(
    #            g_tr,
    #            x0      = pars0,
    ##            verbose = 2,
    ##            xtol    = 1000,
    #            args=(integration_method, nr_nodes),
    #            bounds  = (lbounds, ubounds)
    #        )
    
    #        import scipy.optimize as opt
    #
    #        bounds = [(lbounds[i], ubounds[i]) for i in range(len(lbounds))]
    #        y = opt.minimize(
    #            lambda x, im, n: np.sum(np.abs(g_tr(x, im, n))),
    #            x0=pars0,
    #            args=(integration_method, nr_nodes),
    #            bounds=bounds,
    #            options={'disp': True},
    #            method='trust-constr'
    #        )
    
    #        print('y', y)
            if check_success and (not y.success):
                if nr == len(methods)-1:
                    msg = y.message.replace("\n", " ")
                    raise(PWCModelRunFDError(msg))
                else:
                    print("switching root method from '%s' to '%s'" % (method, methods[nr+1]))
    
            else:
                x = y.x
        #        print(x)
                xBorder = np.where(x < lbounds, lbounds, x)
                xBorder = np.where(x > ubounds, ubounds, xBorder)
        #        print("708", xBorder)
                B = pars_to_matrix(xBorder)
                #print(B)
        
                x1 = x_tau(dt, B)
        #        x1 = x0 + gross_U + gross_F.sum(1) - gross_F.sum(0) - gross_R
    
                return B, x1

    @classmethod
    def reconstruct_Bs(
        cls,
        times,
        start_values,
        gross_Us,
        gross_Fs,
        gross_Rs,
        xs=None,
        integration_method='solve_ivp',
        nr_nodes=None,
        check_success=True
    ):
        print(
            "reconstructing Bs using 'integration_method' =",
            integration_method,
            "and 'nr_nodes' =",
            nr_nodes
        )

        nr_pools = len(start_values)

        def guess_B0(dt, x_approx, F, r):
            nr_pools = len(x_approx)
            B = np.identity(nr_pools)

            # construct off-diagonals
            for j in range(nr_pools):
                if x_approx[j] != 0:
                    B[:, j] = F[:, j] / x_approx[j] / dt
                else:
                    B[:, j] = 0

            # construct diagonals
            for j in range(nr_pools):
                if x_approx[j] != 0:
                    B[j, j] = - (sum(B[:, j]) - B[j, j] + r[j]/x_approx[j]/dt)
                else:
                    B[j, j] = -1

            return B

        x = start_values
        Bs = np.zeros((len(times)-1, nr_pools, nr_pools))
        computed_xs = np.nan * np.ones((len(times), nr_pools))
        computed_xs[0] = start_values
        for k in tqdm(range(len(times)-1)):
            dt = times[k+1] - times[k]

            if xs is not None:
                B0 = guess_B0(dt, (x+xs[k+1])/2, gross_Fs[k], gross_Rs[k])
                #x = xs[k]
            else:
                B0 = guess_B0(dt, x, gross_Fs[k], gross_Rs[k])

            B, x = cls.reconstruct_B_surrogate(
                dt,
                x,
                gross_Us[k],
                gross_Fs[k],
                gross_Rs[k],
                B0,
                integration_method,
                nr_nodes,
                check_success
            )

#            print(
#                k, 
#                np.max(np.abs(xs[k+1]-x)),
#                np.max((np.abs(xs[k+1]-x)/xs[k+1])*100),
#                #xs[k+1],
#                #x
#            )
            if sum(B.sum(0)>0) > 0:
                msg = "reconstructed matrix is not compartmental in time step %d\n" % k
#                msg += str(B) + "\n"
                msg += str(B.sum(0))
                raise(PWCModelRunFDError(msg))

            Bs[k, :, :] = B
            computed_xs[k+1] = x

        if xs is not None:
            max_abs_err = np.max(np.abs(xs-computed_xs))
            max_rel_err = np.max(np.abs(xs-computed_xs)/xs*100)
        else:
            max_abs_err = None
            max_rel_err = None

        return Bs, max_abs_err, max_rel_err

    @classmethod
    def B_pwc(cls, times, Bs):
        def func_maker(i, j):
            def func(t):
                index = np.where(times <= t)[0][-1]
                index = min(index, Bs.shape[0]-1)
                return Bs[index, i, j]
            return func

        nr_pools = Bs[0].shape[0]
        B_funcs = dict()
        for j in range(nr_pools):
            for i in range(nr_pools):
                B_funcs[(i, j)] = func_maker(i, j)

        return B_funcs

#    @classmethod
#    def reconstruct_u(cls, dt, data_U):
#        return data_U / dt
#
#    @classmethod
#    def reconstruct_us(cls, times, gross_Us):
#        us = np.zeros_like(gross_Us)
#        for k in range(len(times)-1):
#            #dt = times[k+1] - times[k]
#
#            #dt = dt.item().days ##### to be removed or made safe
#            us[k,:] = cls.reconstruct_u(self.dts[k], gross_Us[k])
#
#        return us

    @classmethod
    def u_pwc(cls, times, us):
        def func_maker(i):
            def func(t):
                index = np.where(times <= t)[0][-1]
                index = min(index, us.shape[0]-1)
                return us[index, i]
            return func

        nr_pools = us[0].shape[0]
        u_funcs = []
        for i in range(nr_pools):
            u_funcs.append(func_maker(i))

        return u_funcs

    @classmethod
#    def create_B_generic(cls, Bs, time_symbol):
    def create_B_generic(cls, Bs):
        nr_pools = Bs.shape[1]
        strlen = len(str(nr_pools))

        def pool_str(i): return ("{:0"+str(strlen)+"d}").format(i)

        def is_constant_Bs(i, j):
            c = Bs[0, i, j]
#            print("pwc_mr_fd 647", c, Bs[:, i, j], flush=True)
            diff = np.float64(Bs[:, i, j] - c)

            if len(diff[diff == 0.0]) == len(Bs):
                res = True
            else:
                res = False

            return res

        B_generic = zeros(nr_pools, nr_pools)
        for j in range(nr_pools):
            for i in range(nr_pools):
                if not is_constant_Bs(i, j):
                    # B_generic[i,j] = \
                    #     Function('b_'+pool_str(i)+pool_str(j))(time_symbol)
                    B_generic[i, j] = Symbol('b_'+pool_str(i)+pool_str(j))
                else:
                    B_generic[i, j] = Bs[0, i, j]

        return B_generic

    @classmethod
#    def create_u_generic(cls, us, time_symbol):
    def create_u_generic(cls, us):
        nr_pools = us.shape[1]
        strlen = len(str(nr_pools))

        def pool_str(i): return ("{:0"+str(strlen)+"d}").format(i)

        def is_constant_us(i):
            c = us[0, i]
            diff = np.float64(us[:, i] - c)

            if len(diff[diff == 0.0]) == len(us):
                res = True
            else:
                res = False

            return res

        u_generic = zeros(nr_pools, 1)
        for i in range(nr_pools):
            if not is_constant_us(i):
                # u_generic[i] = Function('u_'+pool_str(i))(time_symbol)
                u_generic[i] = Symbol('u_'+pool_str(i))
            else:
                u_generic[i] = us[0, i]

        return u_generic

    @classmethod
    def create_srm_generic(cls, time_symbol, Bs, us):
        nr_pools = Bs.shape[1]
        strlen = len(str(nr_pools))

        def pool_str(i): return ("{:0"+str(strlen)+"d}").format(i)

        state_vector_generic = Matrix(
            nr_pools,
            1,
            [symbols('x_'+pool_str(i)) for i in range(nr_pools)]
        )

#        B_generic = cls.create_B_generic(Bs, time_symbol)
#        u_generic = cls.create_u_generic(us, time_symbol)
        B_generic = cls.create_B_generic(Bs)
        u_generic = cls.create_u_generic(us)

        srm_generic = SmoothReservoirModel.from_B_u(
            state_vector_generic,
            time_symbol,
            B_generic,
            u_generic)

        return srm_generic

#    def solve(self):
#        if not hasattr(self, 'soln'):
#            soln = self.pwc_mrs[0].solve()
#            for k in range(1, len(self.data_times)-1):
#                soln = soln[:-1]
#                k_soln = self.pwc_mrs[k].solve()
#                soln = np.concatenate((soln, k_soln), axis=0)
#
#
#
#            soln = custom_solve_ivp
#            self.soln = soln
#
#        return self.soln
#
#
#    ## which model would be reasonable??
#    ## here we try to find B and u to match xss
#    ## potentiallypotentially  very ill-conditioned
#
##    def find_equilibrium_model(self, xss, B0, u0):
##        B = self.model.compartmental_matrix
##        u = self.model.external_inputs
##
##        nr_pools = self.model.nr_pools
##
##        ## convert parameter vector to compartmental matrix
##        def pars_to_B_and_u(pars):
##            B_tmp = np.zeros((nr_pools**2))
##            B_tmp[int_indices.tolist()] = pars[:len(int_indices)]
##            B_tmp = B_tmp.reshape((nr_pools,nr_pools))
##            for i in range(nr_pools):
##                for j in range(nr_pools):
##                    if not B[i,j].is_Function:
##                       B_tmp[i,j] = B[i,j]
##
##            u_tmp = np.zeros((nr_pools))
##            u_tmp[(ext_indices-nr_pools**2).tolist()]\
##                = pars[len(int_indices):]
##
##            for i in range(nr_pools):
##                if not u[i].is_Function:
##                    u_tmp[i] = u[i]
##
##            return B_tmp, u_tmp
##
##        ## function to minimize difference vector of Bx und -u
##        def g_eq(pars):
##            B, u = pars_to_B_and_u(pars)
###            print(B.sum(), u.sum())
##
##            res1 = np.abs(B @ xss + u)
###            res1 = np.abs(xss + pinv(B) @ u)
##            res2 = np.maximum(B.sum(0),0) # B compartmental?
##
##            print(res1.sum(), res2.sum()*1e12, (res1 + res2*1e12).sum())
##            return res1 + res2 * 1e12
##
##        lbounds = [0]*(nr_pools**2 + nr_pools)
##        for i in range(nr_pools):
##            lbounds[i*nr_pools+i] = -10
##
##        ubounds = [10]*(nr_pools**2 + nr_pools)
##        for i in range(nr_pools):
##            ubounds[i*nr_pools+i] = 0
##
##        par_indices = []
##        for i in range(nr_pools):
##            for j in range(nr_pools):
##                if B[i,j].is_Function:
##                    par_indices.append(i*nr_pools+j)
##        for i in range(nr_pools):
##            if u[i].is_Function:
##                par_indices.append(nr_pools**2+i)
##
##        par_indices = np.array(par_indices)
##        int_indices = par_indices[par_indices<nr_pools**2]
##        ext_indices = par_indices[par_indices>=nr_pools**2]
##
##        lbounds = np.array(lbounds)[par_indices.tolist()]
##        ubounds = np.array(ubounds)[par_indices.tolist()]
##
##        A0 = np.append(B0.reshape((nr_pools**2,)), u0)
##        pars0 = A0[par_indices.tolist()]
##
##        y = least_squares(
##            g_eq,
##            x0       = pars0,
##            verbose  = 2,
###            xtol     = 1e-03,
##            bounds   = (lbounds, ubounds)
##        )
##
###        print(y)
##        B, u = pars_to_B_and_u(y.x)
##        return B, u
#
