import numpy as np
from numpy.linalg import matrix_power, pinv
from scipy.integrate import quad, solve_ivp
from scipy.linalg import inv
from scipy.special import factorial, binom
from tqdm import tqdm
from functools import lru_cache
from typing import List, Callable, Union, Tuple

from .helpers_reservoir import (
    net_Us_from_discrete_Bs_and_xs,
    net_Fs_from_discrete_Bs_and_xs,
    net_Rs_from_discrete_Bs_and_xs,
    p0_maker,
    custom_lru_cache_wrapper,
    generalized_inverse_CDF,
    ALPHA_14C
)
from . import picklegzip

##############################################################################


class DMRError(Exception):
    """Generic error occurring in this module."""
    pass


##############################################################################


class DiscreteModelRun():
    def __init__(self, times, Bs, xs):
        """
        Bs State transition operators for one time step
        """
        self.times = times
        self.Bs = Bs
        self.xs = xs

    def acc_net_internal_flux_matrix(self):
        Bs = self.Bs
        xs = self.xs

        return net_Fs_from_discrete_Bs_and_xs(Bs, xs)

    def acc_net_external_output_vector(self):
        xs = self.xs
        Bs = self.Bs

        return net_Rs_from_discrete_Bs_and_xs(Bs, xs)

    def acc_net_external_input_vector(self):
        xs = self.xs
        Bs = self.Bs

        return net_Us_from_discrete_Bs_and_xs(Bs, xs)

    @property
    def start_values(self):
        return self.xs[0, :]

    @property
    def nr_pools(self):
        return len(self.start_values)

    @classmethod
    def from_Bs_and_net_Us(cls, start_values, times, Bs, net_Us):
        """
        Bs State transition operators for one time step
        """
        xs = cls._solve(start_values, Bs, net_Us)
        return cls(times, Bs, xs)

    @classmethod
    def from_Bs_and_Us_2(cls, start_values, times, Bs, Us):
        """
        Bs State transition operators for one time step
        """
        xs = cls._solve_2(start_values, Bs, Us)
        dmr = cls(times, Bs, xs)
        dmr.Us = Us

        return dmr

    @classmethod
    def from_fluxes(cls, start_values, times, net_Us, net_Fs, net_Rs):
        Bs = cls.reconstruct_Bs_without_xs(
            start_values,
            net_Us,
            net_Fs,
            net_Rs
        )
        return cls.from_Bs_and_net_Us(
            start_values,
            times,
            Bs,
            net_Us
        )

    @classmethod
    def from_fluxes_2(cls, start_values, times, Us, Fs, Rs):
        Bs = cls.reconstruct_Bs_without_xs_2(
            start_values,
            Us,
            Fs,
            Rs
        )
        return cls.from_Bs_and_Us_2(
            start_values,
            times,
            Bs,
            Us
        )

    @classmethod
    def from_fluxes_and_solution(cls, data_times, xs, net_Fs, net_Rs):
        Bs = cls.reconstruct_Bs(xs, net_Fs, net_Rs)
        dmr = cls(data_times, Bs, xs)
        return dmr

    @property
    @lru_cache()
    def net_Us(self):
        n = len(self.Bs)
        return np.array(
            [
                self.xs[k+1]-np.matmul(self.Bs[k], self.xs[k])
                for k in range(n)
            ]
        )

    @property
    def dts(self):
        """
        The lengths of the time intervals.
        """
        return np.diff(self.times).astype(np.float64)
    
    @property
    def dt(self):
        """
        The length of the time intervals.
        At the moment we assume equidistance without checking
        """
        return self.dts[0]

    def time_bin_index(
            self,
            t: float
    ) -> int:
        """
        The index of the bin enclosing the given time
        """
        return int(np.floor(t/self.dt)) 

    @classmethod
    def from_SmoothModelRun(cls, smr, nr_bin):

        # we discard the inner spacing
        # of smr.times since it is potentially
        # not equidistant
        data_times=np.linspace(
            smr.times[0],
            smr.times[-1],
            nr_bin+1 
        )
        return cls(
            data_times,
            smr.fake_discretized_Bs(data_times),
            smr.solve_func()(data_times)
        )

    @classmethod
    def reconstruct_Fs_and_Rs(cls, xs, Bs):
        Fs = np.nan * np.ones_like(Bs)
        Rs = np.nan * np.ones(Bs.shape[:-1])
        for k in range(Bs.shape[0]):
            for j in range(Bs.shape[2]):
                Fs[k, :, j] = Bs[k, :, j] * xs[k, j]
                Rs[k, j] = (1 - Bs[k, :, j].sum()) * xs[k,j]
            for j in range(Bs.shape[2]):
                Fs[k, j, j] = 0

        return Fs, Rs

    @classmethod
    def reconstruct_Bs(cls, xs, Fs, Rs):
        Bs = np.nan * np.ones_like(Fs)
        for k in range(len(Rs)):
            try:
                B = cls.reconstruct_B(xs[k], Fs[k], Rs[k])
                Bs[k, :, :] = B
            except DMRError as e:
                msg = str(e) + 'time step %d' % k
                raise(DMRError(msg))

        return Bs

    @classmethod
    def reconstruct_Bs_without_xs(cls, start_values, Us, Fs, Rs):
        x = start_values
        Bs = np.nan * np.ones_like(Fs)
        for k in tqdm(range(len(Rs))):
            try:
                B = cls.reconstruct_B(x, Fs[k], Rs[k])
                Bs[k, :, :] = B
                x = B @ x + Us[k]
            except DMRError as e:
                msg = str(e) + 'time step %d' % k
                raise(DMRError(msg))

        return Bs

    @classmethod
    def reconstruct_Bs_without_xs_2(cls, start_values, Us, Fs, Rs):
        x = start_values
        Bs = np.nan * np.ones_like(Fs)
        for k in range(len(Rs)):
            try:
                B = cls.reconstruct_B_2(x, Fs[k], Rs[k], Us[k])
                Bs[k, :, :] = B
                x = B @ (x + Us[k])
            except DMRError as e:
                msg = str(e) + 'time step %d' % k
                raise(DMRError(msg))

        return Bs

    @classmethod
    def reconstruct_B(cls, x, F, R):
        nr_pools = len(x)

        B = np.identity(nr_pools)
        if len(np.where(F < 0)[0]) > 0:
            raise(DMRError('Negative flux: '))

        # construct off-diagonals
        for j in range(nr_pools):
            if x[j] < 0:
                raise(DMRError('Content negative: pool %d, ' % j))
            if x[j] != 0:
                B[:, j] = F[:, j] / x[j]
            else:
                B[:, j] = 0

        # construct diagonals
        for j in range(nr_pools):
            if x[j] != 0:
                B[j, j] = 1 - (sum(B[:, j]) - B[j, j] + R[j] / x[j])
                if B[j, j] < 0:
                    if np.abs(B[j, j]) < 1e-03: # TODO: arbitrary value
                        B[j, j] = 0
                    else:
                        pass                     
#                        print(B[j, j])
#                        print(x[j], R[j], F[:, j].sum(), F[j, :].sum()) 
                        raise(DMRError('Diag. val < 0: pool %d, ' % j))
            else:
                B[j, j] = 1

#        # correct for negative diagonals
#        neg_diag_idx = np.where(np.diag(B)<0)[0]
#        for idx in neg_diag_idx:
#            print("'repairing' neg diag in pool", idx)
#            # scale outfluxes down to empty pool
#            col = B[:, idx]
#            d = col[idx]
#            s = 1-d
#            B[:, idx] = B[:, idx] / s
#            r = R[idx] / x[idx] / s
#            B[idx, idx] = 1 - (sum(B[:, idx]) - B[idx, idx] + r)

        return B

    @classmethod
    def reconstruct_B_2(cls, x, F, R, U):
        nr_pools = len(x)

        B = np.identity(nr_pools)
        if len(np.where(F < 0)[0]) > 0:
            raise(DMRError('Negative flux: '))

        # construct off-diagonals
        for j in range(nr_pools):
            if x[j] < 0:
                raise(DMRError('Content negative: pool %d, ' % j))
            if x[j] + U[j] != 0:
                B[:, j] = F[:, j] / (x[j] + U[j])
            else:
                B[:, j] = 0

        # construct diagonals
        for j in range(nr_pools):
            if x[j] + U[j] != 0:
#                B[j, j] = 1 - (sum(B[:, j]) - B[j, j] + R[j] / (x[j] + U[j]))
                B[j, j] =  ((x[j] + U[j]) * (1 - sum(B[:, j]) + B[j, j]) - R[j]) / (x[j] + U[j])
                if B[j, j] < 0:
#                    B[j, j] = 0
#                    y = np.array([B[i, j] * (x[j] + U[j]) for i in range(nr_pools)])
#                    print(y)
#                    print()
#                    print(F[:, j])
#                    print(y - F[:, j])
#                    print(sum(B[:, j]))
#                    print((1-sum(B[:, j])) * (x[j] + U[j]), R[j])
#                    print(x[j] + U[j], (sum(F[:, j]) + R[j]) / 0.15)
#                    raise
                    if np.abs(B[j, j]) < 1e-08:
                        B[j, j] = 0.0
                    else:
#                        pass
                        print(B[j, j])
                        print(x[j], U[j], R[j], F[:, j].sum(), F[j, :].sum()) 
                        print(U[j] - R[j] - F[:, j].sum() + F[j, :].sum())
                        print(B[:, j]) 
                        raise(DMRError('Diag. val < 0: pool %d, ' % j))
            else:
                B[j, j] = 1

#        # correct for negative diagonals
#        neg_diag_idx = np.where(np.diag(B)<0)[0]
#        for idx in neg_diag_idx:
##            print("'repairing' neg diag in pool", idx)
#            # scale outfluxes down to empty pool
#            col = B[:, idx]
#            d = col[idx].sum()
#            s = 1-d
##            print(s)
#            B[:, idx] = B[:, idx] / s
#            r = R[idx] / (x[idx] + U[idx]) / s
#            B[idx, idx] = 1 - (sum(B[:, idx]) - B[idx, idx] + r)
#            if np.abs(B[idx, idx]) < 1e-08:
#                B[idx, idx] = 0
#
#            print(B[idx, idx], (B @ (x + U)))

        return B

#    @classmethod
#    def reconstruct_Bs(cls, data_times, start_values, Fs, rs, net_Us):
#        nr_pools = len(start_values)
#        Bs = np.zeros((len(data_times)-1, nr_pools, nr_pools))
#
#        x = start_values
#        for k in range(len(data_times)-1):
#    #        B = cls.reconstruct_B(xs[k], Fs[k+shift], rs[k+shift])
#            B = cls.reconstruct_B(x, Fs[k], rs[k], k)
#            x = B @ x + net_Us[k]
#            Bs[k,:,:] = B
#        return Bs

    def solve(self):
        return self.xs

    @classmethod
    def _solve(cls, start_values, Bs, net_Us):
        xs = np.nan*np.ones((len(Bs)+1, len(start_values)))
        xs[0, :] = start_values
        for k in range(0, len(net_Us)):
            xs[k+1] = Bs[k] @ xs[k] + net_Us[k]

        return xs

    @classmethod
    def _solve_2(cls, start_values, Bs, net_Us):
        xs = np.nan*np.ones((len(Bs)+1, len(start_values)))
        xs[0, :] = start_values
        for k in range(0, len(net_Us)):
            xs[k+1] = Bs[k] @ (xs[k] + net_Us[k])

        return xs

    def acc_external_output_vector(self):
        n = self.nr_pools
        rho = np.array([1-B.sum(0).reshape((n,)) for B in self.Bs])
        soln = self.solve()[:-1]
        r = rho * soln

        return r

    def acc_external_output_vector_2(self):
        n = self.nr_pools
        rho = np.array([1-B.sum(0).reshape((n,)) for B in self.Bs])
        soln = self.solve()
        r = rho * (soln[:-1] + self.Us)

        return r

    def acc_internal_flux_matrix(self):
        # fixme mm 20-04-2020:
        # potential gain by use of sparse matrices
        Bs = self.Bs
        soln = self.solve()[:-1]
        return np.array([Bs[k] * soln[k] for k in range(len(Bs))])

    # return value in unit "time steps"
    def compute_start_m_factorial_moment(self, order, time_index=0):
        Id = np.identity(self.nr_pools)
        B = self.Bs[time_index]
        x = self.solve()[time_index]
        X = x * Id
        n = order

        fm = factorial(n) * pinv(X) @ matrix_power(B, n)
        fm = fm @ matrix_power(pinv(Id-B), n) @ x

        return fm

    # return value in unit "time steps x dt[0]"
    def compute_start_age_moments(self, max_order, time_index=0):
        def stirling(n, k):
            n1 = n
            k1 = k
            if n <= 0:
                return 1
            elif k <= 0:
                return 0
            elif (n == 0 and k == 0):
                return -1
            elif n != 0 and n == k:
                return 1
            elif n < k:
                return 0
            else:
                temp1 = stirling(n1-1, k1)
                temp1 = k1*temp1

            return (k1*(stirling(n1-1, k1)))+stirling(n1-1, k1-1)

        nr_pools = self.nr_pools
#        Id = np.identity(nr_pools)
#        B0 = self.Bs[time_index]
#        x0 = self.solve()[time_index]
#        X0 = x0 * Id
        start_age_moments = []
        dt = self.dts[0]
        for n in range(1, max_order+1):
            # the old formula is not correct for higher moments
            # in discrete time
            # start_age_moment = factorial(n) * inv(X0)
            # start_age_moment @= matrix_power(inv(Id-B0), n) @ x0
            start_m_moment = np.zeros(nr_pools)
            for k in range(n+1):
                start_m_moment += stirling(n, k) * \
                    self.compute_start_m_factorial_moment(k, time_index)

            start_age_moments.append(start_m_moment*dt**n)

        return np.array(start_age_moments)

    def fake_xss(self, nr_time_steps):
        Id = np.identity(self.nr_pools)

        if np.all(self.net_Us == 0):
            raise(DMRError("Cannot fake xss, because there are no inputs to the systems"))
        mean_U = self.net_Us[:nr_time_steps, ...].mean(axis=0)
        mean_B = self.Bs[:nr_time_steps, ...].mean(axis=0)

        # fake equilibrium
        fake_xss = pinv(Id-mean_B) @ mean_U

        return fake_xss

    def fake_eq_14C(self, nr_time_steps, F_atm, decay_rate, lim, alpha=None):
        if alpha is None:
            alpha = ALPHA_14C

        # input in age steps ai
        p0 = self.fake_start_age_masses(nr_time_steps)
#        import matplotlib.pyplot as plt
#        fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(18, 18))
#        times = np.linspace(0, 1000, 50)
#        z = np.array([p0_ai(int(t)) for t in times])
#        y = np.array([p0(t) for t in times])
#        for k, ax in zip(range(self.nr_pools), axes.flatten()):
#            ax.plot(times, y[:, k], label="c")
#            ax.plot(times, z[:, k])
#            ax.legend()
#        fig.show()

#        E_a = self.fake_start_age_moments(nr_time_steps, 1).reshape(-1)

        eq_14C = np.nan * np.ones((self.nr_pools, ))
        for pool in range(self.nr_pools):
#            print(np.float(E_a[pool])/365.25, F_atm(np.float(E_a[pool])))

            # input in age steps ai, output as mass, not density
            p0_pool = lambda ai: p0(ai)[pool]
#            def p0_pool_14C(ai):
#                res = (
##                    (F_atm(ai)/1000.0 + 1) *
#                    ai *  p0_pool(ai)
##                    * np.exp(-decay_rate*ai)
#                    )
#                return res

            # input in age (not age indices)
            def p0_pool_14C_quad(a):
                res = (
                    (F_atm(a)/1000.0 + 1) * 
                    p0_pool(int(a/self.dt)) / self.dt # make masses to density
#                    * alpha # makes integration imprecise
                    * np.exp(-decay_rate*int(a))
                    )
#                print(a, res)
                return res

            # integration via solve_ivp is fast and successful
            res_quad = solve_ivp(
                lambda a, y: p0_pool_14C_quad(a),
                (0, lim),
                np.array([0])
            )
#            print("quad", res_quad.y.reshape(-1)[-1])#/365.25/self.start_values[pool])
            res = res_quad.y.reshape(-1)[-1]
##            res = res_quad[0]
#            ai = 0
#            res = 0
#            res2 = 0
#            while ai <= 2*lim_ai:
#                res += p0_pool_14C(ai)
#                res2 += p0_pool(ai)
##                print(res, res2)
#                ai += 1
#            print(res, res2)
            eq_14C[pool] = res * alpha

        return eq_14C


    # return value in unit "time steps"
    def fake_start_m_factorial_moment(self, order, nr_time_steps):
        Id = np.identity(self.nr_pools)

        # fake equilibrium
        fake_xss = self.fake_xss(nr_time_steps)
        mean_B = self.Bs[:nr_time_steps, ...].mean(axis=0)

        B = mean_B
        x = fake_xss
        X = x * Id
        n = order

        fm = factorial(n) * pinv(X) @ matrix_power(B, n)
        fm = fm @ matrix_power(pinv(Id-B), n) @ x

        return fm

    # return value in unit "time steps x dt[0]"
    def fake_start_age_moments(self, nr_time_steps, up_to_order):
        def stirling(n, k):
            n1 = n
            k1 = k
            if n <= 0:
                return 1
            elif k <= 0:
                return 0
            elif (n == 0 and k == 0):
                return -1
            elif n != 0 and n == k:
                return 1
            elif n < k:
                return 0
            else:
                temp1 = stirling(n1-1, k1)
                temp1 = k1*temp1

            return (k1*(stirling(n1-1, k1)))+stirling(n1-1, k1-1)

        nr_pools = self.nr_pools
#        Id = np.identity(nr_pools)
#        B0 = self.Bs[time_index]
#        x0 = self.solve()[time_index]
#        X0 = x0 * Id
        start_age_moments = []
        dt = self.dts[0]
        for n in range(1, up_to_order+1):
            # the old formula is not correct for higher moments
            # in discrete time
            # start_age_moment = factorial(n) * inv(X0)
            # start_age_moment @= matrix_power(inv(Id-B0), n) @ x0
            start_m_moment = np.zeros(nr_pools)
            for k in range(n+1):
                start_m_moment += stirling(n, k) * \
                    self.fake_start_m_factorial_moment(k, nr_time_steps)

            start_age_moments.append(start_m_moment*dt**n)

        return np.array(start_age_moments)

    def age_moment_vector_up_to(self, up_to_order, start_age_moments):
        soln = self.solve()
        ams = self._solve_age_moment_system(up_to_order, start_age_moments)

        res = np.nan * np.ones((ams.shape[0], ams.shape[1]+1, ams.shape[2]))
        res[:, 0, :] = soln
        res[:, 1:, :] = ams
        
        return res

    def age_moment_vector(self, order, start_age_moments):
        ams = self._solve_age_moment_system(order, start_age_moments)
        amv = ams[:, order-1, :]

        return amv

    def system_age_moment(self, order, start_age_moments):
        age_moment_vector = self.age_moment_vector(order, start_age_moments)
        age_moment_vector[np.isnan(age_moment_vector)] = 0
        soln = self.solve()

        total_mass = soln.sum(1)  # row sum
        total_mass[total_mass == 0] = np.nan

        system_age_moment = (age_moment_vector*soln).sum(1)/total_mass

        return system_age_moment

    def _solve_age_moment_system(self, max_order, start_age_moments):
        n = self.nr_pools
        Id = np.identity(n)
        ones = np.ones(n)
        soln = self.solve()
        soln[soln < 1e-12] = 0
#        dts = self.dts

        def diag_inv_with_zeros(A):
            res = np.zeros_like(A)
            for k in range(A.shape[0]):
                if np.abs(A[k, k]) != 0:
                    res[k, k] = 1/A[k, k]
                else:
#                    res[k, k] = np.nan
                    res[k, k] = 0

            return res

        age_moments = [start_age_moments]
        for i in tqdm(range(len(self.times)-1)):
            vec = np.zeros((max_order, n))
            X_np1 = soln[i+1] * Id
            X_n = soln[i] * Id
            B = self.Bs[i]
            for k in range(1, max_order+1):
                moment_sum = np.zeros(n)
                for j in range(1, k+1):
                    moment_sum += age_moments[-1][j-1, :].reshape((n,)) \
                                  * binom(k, j) #* dts[i]**(k-j)

#                vec[k-1, :] = inv(X_np1) @ B @\
                vec[k-1, :] = diag_inv_with_zeros(X_np1) @ B @\
                        X_n @ (moment_sum + ones)#*dts[i]**k)

            age_moments.append(vec)

        return np.array(age_moments)

    def backward_transit_time_moment(
            self,
            order: int,
            start_age_moments: np.ndarray
        )-> np.ndarray:
        """Compute the ``order`` th backward transit time moment based on the 
        This is done by computing a weighted sum of of the pool wise 
        age moments. 
        For every pool the weight is givem by the fraction of the 
        of this pools output of the combined output of all pools.
        :func:`age_moment_vector`.

        Args:
            order (int): The order of the backward transit time moment that is 
                to be computed.
            start_age_moments (numpy.ndarray order x nr_pools, optional): 
                Given initial age moments up to the order of interest. 
                Can possibly be computed by :func:`moments_from_densities`. 
                Defaults to None assuming zero initial ages.

        Returns:
            numpy.array: shape (nr_bins,nr_pools) 
            The ``order`` th backward transit time moment over the time grid.
        """ 

        # the shape of the age moment vector is (nr_bins,nr_pools)
        r=self.acc_net_external_output_vector()

        # the shape of the age moment vector is (nr_bins+1,nr_pools)
        # so we have to cut it
        age_moment_vector = self.age_moment_vector(
                order,
                start_age_moments
        )[:-1,:] 
        pool_axis=1
        return (
                    (r*age_moment_vector).sum(axis=pool_axis)/
                    r.sum(axis=pool_axis)
               )

    def start_age_densities_func(self):
        B = self.Bs[0]
        u = self.net_Us[0]
        dt = self.dts[0]

        # assuming constant time step before times[0]
        def p(a):
            n = int(a // dt)
            if a <= 0:
                return np.zeros_like(u)
            return matrix_power(B, n) @ u  # if age zero exists

        return p

    def initialize_state_transition_operator_matrix_cache(
        self,
        lru_maxsize,
        lru_stats=False,
    ):
        custom_lru_cache = custom_lru_cache_wrapper(
            maxsize=lru_maxsize,  # variable maxsize now for lru cache
            typed=False,
            stats=lru_stats  # use custom statistics feature
        )

        self._state_transition_operator_matrix_cache = custom_lru_cache(
            self._state_transition_operator_matrix
        )

    # or initialize by function above with adaptable size
#    @lru_cache(maxsize=200)
    def _state_transition_operator_matrix(self, k1, k0):
        if hasattr(self, "_state_transition_operator_matrix_cache"):
            phi = self._state_transition_operator_matrix_cache
        else:
            phi = self._state_transition_operator_matrix

        if k0 > k1:
            raise(ValueError("k0 > k1 in state_transition_operator_matrix"))
        elif k0 == k1:
            return np.eye(self.nr_pools)
        elif k1 == k0+1:
            return self.Bs[k0]
        else:
             im=int((k1+k0)/2)
            #im = k1-1
        #return phi(im, k0) @ phi(k1, im)
        return phi(k1, im) @ phi(im, k0)

        #if (hasattr(self, '_sto_recent') and
        #   (self._sto_recent['k0'] == k0) and
        #   (self._sto_recent['k1'] == k1)):
        #    Phi = self.Bs[k1-1] @ self._sto_recent['Phi']
        #elif (hasattr(self, '_sto_recent') and
        #        (self._sto_recent['k0'] == k0+1) and
        #        (self._sto_recent['k1'] == k1)):
        #    Phi = self._sto_recent['Phi'] * self.Bs[k0]
        #else:
        #Phi = np.identity(self.nr_pools)
        #for k in range(k0, k1):
        #    Phi = self.Bs[k] @ Phi

        ##self._sto_recent = {'k0': k0, 'k1': k1, 'Phi': Phi}

        #return Phi 
  
    def _state_transition_operator(self, k1, k0, x):
        # fixme mm 12-3-2020
        # This code implicitly assumes that 
        # t0 and t1 are elements of self.times
        # So the actual arguments are the indices k0 and k1
        # If we restrict the ages to integral multiples of 
        # dt too, we could perform all computations 
        # on an integer grid, and scale the results later
        # by multiplying ages and times by dt
        # This would avoid the 'np.where' calls
        #
        # There is an argument for using equidistant age
        # distributions:
        # While it is perfectly possible to have mass with
        # arbitrary age there is usually an influx of mass
        # with original age 0 that aquires over time an 
        # age that is in integral multiple of dt. So every
        # (original) age that is not such a multiple will eventually        # be stradled by two integral multiples
        

        ## grid
        #if k0 > k1:
        #    raise(DMRError('Evaluation before t0 not possible'))

        #if k1 == k0:
        #    return x

       # #k0 = np.where(self.times == t0)[0][0]
       # k0 = np.where(np.abs(self.times - t0) < 1e-09)[0][0]
       # #k1 = np.where(self.times == t1)[0][0]
       # k1 = np.where(np.abs(self.times - t1) < 1e-09)[0][0]

       # if (hasattr(self, '_sto_recent') and
       #    (self._sto_recent['k0'] == k0) and
       #    (self._sto_recent['k1'] == k1)):
       #     Phi = self.Bs[k1-1] @ self._sto_recent['Phi']
       # elif (hasattr(self, '_sto_recent') and
       #         (self._sto_recent['k0'] == k0+1) and
       #         (self._sto_recent['k1'] == k1)):
       #     Phi = self._sto_recent['Phi'] * self.Bs[k0]
       # else:
       #     Phi = np.identity(self.nr_pools)
       #     for k in range(k0, k1):
       #         Phi = self.Bs[k] @ Phi

       # self._sto_recent = {'k0': k0, 'k1': k1, 'Phi': Phi}
        Phi = self._state_transition_operator_matrix(k1, k0 )
        return Phi @ x

    def age_densities_1_single_value_func(
            self,
            start_age_densities_of_bin: Callable[[int], np.ndarray]
        ) -> Callable[[int, int], float]:# not a float but an np.array (nr_pools)
        """
        Return a function f(ia, it) that computes
        the quotient delta_m(ia, it)/delta_a where delta_m
        it the remainder of the initial mass distribution that
        has age ia*da at time it*dt.
        """

        t0 = self.times[0]

        #def p0(ai):
        #    if ai >= 0:
        #        return start_age_densities_of_bin(ai)
        #    else:
        #        return np.zeros((self.nr_pools,))
        p0 = p0_maker(start_age_densities_of_bin)

        Phi = self._state_transition_operator

        def p1_sv(ia, kt):
            #res = Phi(t, t0, p0(a-(t-t0)))
            kt0 = 0
            res = Phi(kt, kt0, p0(ia-kt))
            return res

        return p1_sv

    def _age_densities_1_func(
            self,
            start_age_densities_of_bin_index
        ):

        #p1_sv = self.age_densities_1_single_value_func(start_age_densities)
        #times = self.times
        #t0 = times[0]
        #dt = self.dts[0]
        #def p1(age_bin_indices):

        #    vals = []
        #    if len(age_bin_indices) > 0:
        #        for ia in tqdm(age_bin_indices):
        #            vals.append(
        #                np.stack(
        #                    [p1_sv(ia, it) for it in range(len(self.times[:-1]))],
        #                    axis=0
        #                )
        #            )
        #            print(vals[-1].shape)
        #            #vals.append(p1_sv(a, t))
        #    vals = np.array(vals)

        #    return vals

        #return p1

        p0 = p0_maker(start_age_densities_of_bin_index)

        def p1(age_bin_indices):
            nt = len(self.times[:-1])
            na = len(age_bin_indices)
            nrp = self.nr_pools
            if len(age_bin_indices) > 0:
                vals = np.zeros((na,nt,nrp))
                for it in range(nt):
                    phi = self._state_transition_operator_matrix(it,0)
                    sais= np.stack(
                        [
                            p0(ai - it) 
                            for ai in age_bin_indices
                        ],
                        axis=1
                    )
                    #print(sais.shape)
                    #print((phi @ sais).shape)
                    #print(vals.shape)
                    vals[:,it,:] = (phi @ sais).transpose()
                return vals

        return p1

    def age_densities_2_single_value_func(self):
        times = self.times
        t0 = times[0]
        Phi = self._state_transition_operator
        kt0=0
        def p2_sv(ia, kt):
            if (ia < 0) or (kt-kt0 <= ia):
                return np.zeros((self.nr_pools,))
            #k = np.where(times == t-a)[0][0]
            #kt = np.where(np.abs(times - (t-a)) < 1e-09)[0][0]
#            U = self.net_Us[kt] # wrong!
            U = self.net_Us[kt-ia-1]
            res = Phi(kt, kt-ia, U)  # age arrived at end of last time step

            # the density returned by the smooth model run has
            # dimension mass*time^-1 for every point in the age,time plane
            # whereas the discrete model run
            # returns a mass for every (da x dt) bin in the age,time plane
            # Therefore we have to divide by dt here
            return res / self.dts[0] 
            #return res 

        return p2_sv

    def _age_densities_2_func(self):
        p2_sv = self.age_densities_2_single_value_func()

        times = self.times
        t0 = times[0]

        #def p2(a_min, a_max, t, coarsity):
        #    if a_min > t-t0:
        #        a_min = t-t0
        #    a_max = min(t-t0, a_max)

        #    k_t = np.where(times == t)[0][0]
        #    try:
        #        k_a_min = np.where(t-times[:(k_t+1)] >= a_min)[0][-1]
        #        k_a_max = np.where(t-times[:(k_t+1)] <= a_max)[0][0]
        #    except IndexError:
        #        return np.array([]), np.array([])

        #    ages = np.flip(t-times[k_a_max:(k_a_min+1)], 0)
        #    ages = ages[np.arange(0, len(ages), coarsity)]
#       #     vals = np.array([p2_sv(a,t) for a in ages])

        #    vals = []
        #    if len(ages) > 0:
        #        for a in tqdm(ages):
        #            vals.append(p2_sv(a, t))
        #    vals = np.array(vals)

        #    return ages, vals.reshape((len(ages), self.nr_pools))

        #return p2
        def p2(age_bin_indices):
            #if a_min > t-t0:
            #    a_min = t-t0
            #a_max = min(t-t0, a_max)

            #k_t = np.where(times == t)[0][0]
            #try:
            #    k_a_min = np.where(t-times[:(k_t+1)] >= a_min)[0][-1]
            #    k_a_max = np.where(t-times[:(k_t+1)] <= a_max)[0][0]
            #except IndexError:
            #    return np.array([]), np.array([])

            #ages = np.flip(t-times[k_a_max:(k_a_min+1)], 0)
            #ages = ages[np.arange(0, len(ages), coarsity)]
#            vals = np.array([p2_sv(a,t) for a in ages])

            #vals = []
            #if len(age_bin_indices) > 0:
            #    for ia in tqdm(age_bin_indices):
            #        vals.append(
            #            np.stack(
            #                [p2_sv(ia, it) for it in range(len(self.times[:-1]))],
            #                axis=0
            #            )    
            #        )    
            #        #vals.append(p2_sv(a, t))
            #vals = np.array(vals)
            nt = len(self.times[:-1])
            na = len(age_bin_indices)
            nrp = self.nr_pools
            if len(age_bin_indices) > 0:
                vals = np.stack(
                    [
                        np.stack(
                            [p2_sv(ia, it) for it in range(nt)],
                            axis=0
                        )
                        for ia in tqdm(age_bin_indices)
                    ]
                )

            return vals

        return p2

    def age_densities_single_value_func(self, start_age_densities):
        p1_sv = self.age_densities_1_single_value_func(
            start_age_densities
        )
        p2_sv = self.age_densities_2_single_value_func()

        def p_sv(a, t):
            return p1_sv(a, t) + p2_sv(a, t)

        return p_sv

    def pool_age_densities_func(self, start_age_densities_bin):#, coarsity=1):
        p1 = self._age_densities_1_func(start_age_densities_bin)
        p2 = self._age_densities_2_func()

        def p(ages):
            vals_1 = p1(ages)
            vals_2 = p2(ages)
            vals = vals_2 + vals_1
            return vals
        return p
    
    # Old version is commented because we now avoid real valued arguments for times
    # and ages and replaced them by and index denoting the age or time
    # bin. 
    # This ensures the predicatabiliyt of the size of the returned arrays
    # for more than one time.
    #def age_quantiles_at_time(self, q, t, pools, start_age_densities):
    #    dts = self.dts

    #    k = np.where(self.times == t)[0][0]
    #    x = self.soln[k, pools].sum()

    #    prev_age = 0
    #    age = 0
    #    mass = 0

    #    p_sv = self.age_densities_single_value(start_age_densities)
    #    while mass <= q*x:
    #        prev_age = age
    #        if k == 0:
    #            age += dts[0]
    #        else:
    #            age += dts[k-1]
    #            k -= 1

    #        mass += p_sv(age, t)[pools].sum()

    #    return prev_age
    def age_quantile_at_time(
            self,
            q,
            t,
            pools,
            start_age_densities_of_bin
        )-> float:
        """Returns pool age distribution quantiles for the time.
        This is a wrapper providing an interface similar to the continuous
        model runs.
        Internally it will compute the index of the bin containing the given
        time, and call the indexed version, whose result is the index of
        that age bin. The boundary of this age bin will be returned.

        For internal use the indexed version func:age_quantile_bin_at_time_bin
        is usually preferred since it avoids the repeated computation of the
        indices of age and time bins which are the more natural variables for
        this discrete model run.


        Args:
            q:
                    quantile (between 0 and 1): The relative share of mass that is
                    considered to be left of the computed value. A value of ``0.5``
                    leads to the computation of the median of the distribution.
            it:
                    index of the age bin for which the quantile is to be computed.
            pools:
                    the indices of the pools that contribute to the mass
                    that the quantile computation is refering to.
                    If for example a list containing a single index is given
                    then the quantile of mass in the respective pool will be
                    computed.
                    If the list contains all the pool indices the result
                    will be the system age.
            start_age_densities_of_bins:
                    A function that takes a single integer (denoting the
                    age bin) and returns the value of mass per age in that bin.
                    (piecewise constant approximation of a density).


        Returns: an age (will be a multiple of the binsize of the age bins)
        """
        it = self.time_bin_index(t)
        ia = self.age_quantile_bin_at_time_bin(q,it,pools,start_age_densities_of_bin)
        a = self.dt*ia
        return a

    def age_quantile_bin_at_time_bin(
            self,
            q: float,
            it: int,
            pools: List[int],
            start_age_densities_of_bin: Callable[[int],np.ndarray]
        ) -> int:
        """Returns pool age bin index for the  the quantile of the combined
        mass of the mentioned pools for the provided time bin index.

        Args:
            q:
                quantile (between 0 and 1): The relative share of mass that is 
                considered to be left of the computed value. A value of ``0.5`` 
                leads to the computation of the median of the distribution.
            it:
                index of the time bin for which the quantile is to be computed.
            pools:
                the indices of the pools that contribute to the mass
                that the quantile computation is refering to.
                If for example a list containing a single index is given
                then the quantile of mass in the respective pool will be
                computed.
                If the list contains all the pool indices the result
                will be the system age.
            start_age_densities_of_bin:
                A function that takes a single integer (denoting the
                age bin) and returns the value of mass per age in that bin.
                (piecewise constant approximation of a density).

          Returns:
                An integer array of shape (nr_pools,) containing the bin
                number for every pool.
          """

        # x = self.soln[k, pools].sum()
        x = self.solve()[it,pools].sum()

        prev_ai = 0
        ai = 0
        mass = 0

        p_sv = self.age_densities_single_value_func(
            start_age_densities_of_bin # really a density per bin, not a mass (--> *dt=da below)
        )
        while mass <= q*x:
            prev_ai = ai
            ai += 1
            mass += p_sv(ai, it)[pools].sum()*self.dt

        print(prev_ai)
        return prev_ai

    def fake_start_age_masses(self, nr_time_steps):
        Id = np.identity(self.nr_pools)

        mean_u = self.net_Us[:nr_time_steps, ...].mean(axis=0)
        mean_B = self.Bs[:nr_time_steps, ...].mean(axis=0)

        # assuming constant time step before times[0]
        def p0_fake_eq(ai): # ai = age bin index
            if ai < 0: 
                return np.zeros_like(mean_u)
            return matrix_power(mean_B, ai) @ mean_u  # if age zero exists

        fake_xss = self.fake_xss(nr_time_steps)
        renorm_vector = self.start_values / fake_xss 
        p0 = lambda ai: p0_fake_eq(ai) * renorm_vector

        return p0

    def _G_sv(self, P0):
        nr_pools = self.nr_pools
        Phi = self._state_transition_operator_matrix

        def g(ai, ti):
            if ai < ti: 
                return np.zeros((nr_pools,))
            res = np.matmul(Phi(ti, 0), P0(ai-ti)).reshape((self.nr_pools,))
            return res

        return g

    def _H_sv(self):
        nr_pools = self.nr_pools
        Phi = self._state_transition_operator_matrix
        soln = self.solve()
    
        def h(ai, ti):
            # count everything from beginning?
            if ai >= ti:
                ai = ti-1
    
            if ai < 0:
                return np.zeros((nr_pools,))
    
            # mass at time index ti
            x_ti = soln[ti]
            # mass at time index ti-(ai+1)
            x_ti_minus_ai_plus_1 = soln[ti-(ai+1)]
            # what remains from x_ti_minus_ai_plus1 at time index ti
            m = np.matmul(Phi(ti, ti-(ai+1)), x_ti_minus_ai_plus_1).reshape((self.nr_pools,))
            # difference is not older than ti-ai
            res = x_ti-m
            # cut off accidental negative values
            return np.maximum(res, np.zeros(res.shape))

        return h

    def cumulative_pool_age_masses_single_value(self, P0):
        G_sv = self._G_sv(P0)
        H_sv = self._H_sv()
        def P_sv(ai, ti):
            res = G_sv(ai, ti) + H_sv(ai, ti)
            return res

        return P_sv

    def fake_cumulative_start_age_masses(self, nr_time_steps):
        Id = np.identity(self.nr_pools)

        mean_u = self.net_Us[:nr_time_steps, ...].mean(axis=0)
        mean_B = self.Bs[:nr_time_steps, ...].mean(axis=0)

        IdmB_inv = pinv(Id-mean_B)
        # assuming constant time step before times[0]
        def P0_fake_eq(ai): # ai = age bin index
            if ai < 0:
                return np.zeros_like(mean_u)
            return IdmB_inv @ (Id-matrix_power(mean_B, ai+1)) @ mean_u

        # rescale from fake equilibrium pool contents to start_vector contents
        fake_xss = self.fake_xss(nr_time_steps)
        renorm_vector = self.start_values / fake_xss 
        P0 = lambda ai: P0_fake_eq(ai) * renorm_vector

        return P0

    def pool_age_quantiles(self, q, P0):
        P_sv = self.cumulative_pool_age_masses_single_value(P0)
        soln = self.solve()

        res = np.nan * np.ones((len(self.times), self.nr_pools))
        for pool_nr in range(self.nr_pools):
            print('Pool:', pool_nr)
            quantile_ai = 0
            for ti in tqdm(range(len(self.times))):
                quantile_ai = generalized_inverse_CDF(
                    lambda ai: P_sv(int(ai), ti)[pool_nr],
                    q * soln[ti, pool_nr],
                    x1=quantile_ai
                )
            
                if P_sv(int(quantile_ai), ti)[pool_nr] > q * soln[ti, pool_nr]:
                    if quantile_ai > 0:
                        quantile_ai = quantile_ai - 1

                res[ti, pool_nr] = int(quantile_ai)

        return res * self.dt
        
    def system_age_quantiles(self, q, P0):
        P_sv = self.cumulative_pool_age_masses_single_value(P0)
        P_sys_sv = lambda ai, ti: P_sv(ai, ti).sum()
        soln_sum = self.solve().sum(axis=1)

        res = np.nan * np.ones(len(self.times))
        quantile_ai = 0
        for ti in tqdm(range(len(self.times))):
            quantile_ai = generalized_inverse_CDF(
                lambda ai: P_sys_sv(int(ai), ti),
                q * soln_sum[ti],
                x1=quantile_ai
            )
            
            if P_sys_sv(int(quantile_ai), ti) > q * soln_sum[ti]:
                if quantile_ai > 0:
                    quantile_ai = quantile_ai - 1

            res[ti] = int(quantile_ai)

        return res * self.dt
        
    def backward_transit_time_quantiles(self, q, P0):
        P_sv = self.cumulative_pool_age_masses_single_value(P0)
        rho = 1 - self.Bs.sum(1)
        P_btt_sv = lambda ai, ti: (rho[ti] * P_sv(ai, ti)).sum() 
        R = self.acc_net_external_output_vector()

        res = np.nan * np.ones(len(self.times[:-1]))

        quantile_ai = 0
        for ti in tqdm(range(len(self.times[:-1]))):
            quantile_ai = generalized_inverse_CDF(
                lambda ai: P_btt_sv(int(ai), ti),
                q * R[ti, ...].sum(),
                x1=quantile_ai
            )
            
            if P_btt_sv(int(quantile_ai), ti) > q * R[ti, ...].sum():
                if quantile_ai > 0:
                    quantile_ai = quantile_ai - 1

            res[ti] = int(quantile_ai)

        return res * self.dt

    def backward_transit_time_quantiles_inputs_only(self, q):
        H_sv = self._H_sv()
        rho = 1 - self.Bs.sum(1)
        H_btt_sv = lambda ai, ti: (rho[ti] * H_sv(ai, ti)).sum() 
        
        R = rho * np.array([H_sv(ti, ti) for ti in self.times[:-1]])

        res = np.nan * np.ones(len(self.times[:-1]))

        quantile_ai = 0
        for ti in tqdm(range(len(self.times[:-1]))):
            quantile_ai = generalized_inverse_CDF(
                lambda ai: H_btt_sv(int(ai), ti),
                q * R[ti, ...].sum(),
                x1=quantile_ai
            )
            
            if H_btt_sv(int(quantile_ai), ti) > q * R[ti, ...].sum():
                if quantile_ai > 0:
                    quantile_ai = quantile_ai - 1

            res[ti] = int(quantile_ai)

        return res * self.dt

    def CS(self, k0, n):
        Phi = self._state_transition_operator_matrix
        return sum([(Phi(n, k) @ self.net_Us[k]).sum() for k in range(k0, n+1, 1)])

#    # return value in unit "time steps x dt[0]"
#    def backward_transit_time_quantiles_from_masses(self, q, start_age_masses_at_age_bin):
#        R = self.acc_net_external_output_vector()
#
#        # pool age mass vector based on age and time indices
#        p_sv = self.age_densities_single_value_func(
#            start_age_masses_at_age_bin
#        )
#
#        rho = 1 - self.Bs.sum(1)
#        p_btt_sv = lambda ai, ti: (rho[ti] * p_sv(ai, ti)).sum() # outflow mass vector at ai, ti
#
#        res = np.nan * np.ones(len(self.times[:-1]))
#        for ti in tqdm(range(len(self.times[:-1]))):
#            prev_ai = 0
#            ai = 0
#            mass = 0
#            total_outflow_mass_ti = R[ti, ...].sum()
#
#            while mass <= q * total_outflow_mass_ti:
#                if ai % 10000 == 0:
#                    print(
#                        "%04d" % ti, 
#                        "%2.2f" % (ti/len(self.times[:-1])*100), "%",
#                        "%05d" % ai,
#                        "%02.2f" % (mass/(q*total_outflow_mass_ti)*100), "%",
#                        "%05.2f" % mass,
#                        flush=True
#                    )
#                prev_ai = ai
#                ai += 1
#                mass += p_btt_sv(ai, ti)
#
#            res[ti] = prev_ai
#        
#        return res * self.dt # from age index to age
        
    @classmethod
    def load_from_file(cls, filename):
        cmr = picklegzip.load(filename)
        return cmr

    def save_to_file(self, filename):
        picklegzip.dump(self, filename)

#    ########## 14C methods #########
#
#    def to_14C_only(self, start_values_14C, us_14C, decay_rate=0.0001209681):
#        times_14C = self.times
#
#        Bs = self.Bs
#        dts = self.dts
#
#        Bs_14C = np.zeros_like(Bs)
#        for k in range(len(Bs)):
#            # there seems to be no difference
#            Bs_14C[k] = Bs[k] * np.exp(-decay_rate*dts[k])
##            Bs_14C[k] = Bs[k] * (1.0-decay_rate*dts[k])  ## noqa
#
#        dmr_14C = DiscreteModelRun_14C(
#            start_values_14C,
#            times_14C,
#            Bs_14C,
#            us_14C,
#            decay_rate)
#
#        return dmr_14C
