import numpy as np

from numpy.linalg import matrix_power, pinv
from scipy.linalg import inv
from scipy.misc import factorial
from scipy.special import binom
from sympy import Matrix

from tqdm import tqdm

################################################################################


class Error(Exception):
    """Generic error occurring in this module."""
    pass


################################################################################


class DiscreteModelRun(object):
    def __init__(self, start_values, times, Bs, us):
        self.nr_pools = len(us[0])
        self.start_values = start_values.reshape((self.nr_pools,))
        self.times = times
        self.Bs = Bs
        self.us = us
        self.dts = np.diff(self.times).astype(np.float64)

    @classmethod
    def from_SmoothModelRun(cls,smr): 
        #fake something for the test
        nr_pools=smr.nr_pools
        data_times=smr.times
        n=len(data_times)
        Bs = np.zeros((n-1, nr_pools, nr_pools)) 
        us = np.zeros((n-1, nr_pools)) 
        start_index = 0
        for k in range(start_index, n-1):
            delta_t=data_times[k+1]-data_times[k]

            #fixme mm replace with Phi(data_times[k])
            B=np.identity(nr_pools)*(np.exp(-delta_t))
            Bs[k,:,:] = B
            #fixme: replace with integral over phi*u(t)
            u=np.ones(nr_pools)*delta_t
            us[k,:] = u

        return cls(smr.start_values,data_times,Bs,us)


    @classmethod
    def reconstruct_from_data(cls, times, start_values, xs, Fs, rs, us):
        Bs = cls.reconstruct_Bs(times, xs, Fs, rs, us)
    
        dmr = cls(start_values, times, Bs, us)
        return dmr

    @classmethod
    def reconstruct_B(cls, x, F, r, k):
        nr_pools = len(x)
        B = np.identity(nr_pools)
        if len(np.where(F<0)[0]) > 0:
            print('\n\n', np.where(F<0), '\n\n')
            raise(Error('Negative flux detected: time step %d' % k))
    
        # construct off-diagonals
        for j in range(nr_pools):
            if x[j] < 0:
                raise(Error('Reconstructed compartment content negative: pool %d, time %d ' % (j,k)))
            if x[j] != 0:
                B[:,j] = F[:,j] / x[j]
            else:
                B[:,j] = 0

        # construct diagonals
        for j in range(nr_pools):
            if x[j] != 0:
                B[j,j] = 1 - (sum(B[:,j]) - B[j,j] + r[j] / x[j])
                if B[j,j] < 0:
                    #print(j, x, F, r, '\n\n')
                    print('diagonal value = ', B[j,j])
                    print('pool content = ', x[j])
                    raise(Error('Reconstructed diagonal value is negative: pool %d, time %d' % (j,k)))
            else:
                B[j,j] = 1
   
#        for i in range(nr_pools):
#            for j in range(nr_pools):
#                if i != j:
#                    d = B[i,j]*x[j] - F[i,j]
#                    if d != 0: print(d)

        return B

    @classmethod   
    def reconstruct_Bs(cls, data_times, xs, Fs, rs, us):
        nr_pools = len(xs[0])
        Bs = np.zeros((len(data_times)-1, nr_pools, nr_pools)) 
    
        start_index = 0
        x = xs[start_index]
        for k in range(start_index, len(data_times)-1):
    #        B = cls.reconstruct_B(xs[k], Fs[k+shift], rs[k+shift])
            B = cls.reconstruct_B(x, Fs[k], rs[k], k)
            x = B @ x + us[k]
            Bs[k,:,:] = B
    
        return Bs

    def solve(self):
        if not hasattr(self, 'soln'):
            start_values = self.start_values
            soln = [start_values]
            for k in range(len(self.times)-1):
                x_old = soln[k]
                x_new = self.Bs[k] @ x_old + self.us[k]
                soln.append(x_new)
    
            self.soln = np.array(soln)        

        return self.soln
    
    @property
    def external_output_vector(self):
        n = self.nr_pools
        rho = np.array([1-B.sum(0).reshape((n,)) for B in self.Bs])
        soln = self.solve()[:-1]
        r = rho * soln

        return r

    # return value in unit "time steps"
    def compute_start_m_factorial_moment(self, order):
        Id = np.identity(self.nr_pools)
        B = self.Bs[0]
        x = self.solve()[0]
        X = x * Id
        n = order

        fm = factorial(n) * pinv(X) @ matrix_power(B, n) @ \
                matrix_power(pinv(Id-B), n) @ x

        return fm

    # return value in unit "time steps x dt[0]"
    def compute_start_age_moments(self, max_order):
        def stirling(n, k):
            n1=n
            k1=k
            if n<=0:
                return 1
            elif k<=0:
                return 0   
            elif (n==0 and k==0):
                return -1     
            elif n!=0 and n==k:
                return 1
            elif n<k:
                return 0
            else:
                temp1=stirling(n1-1,k1)
                temp1=k1*temp1

            return (k1*(stirling(n1-1,k1)))+stirling(n1-1,k1-1)

        nr_pools = self.nr_pools
        Id = np.identity(nr_pools)
        B0 = self.Bs[0]
        x0 = self.solve()[0]
        X0 = x0 * Id
        start_age_moments = []
        dt = self.dts[0]
        for n in range(1, max_order+1):
            # the old formula is not correct for higher moments
            # in discrete time
            #start_age_moment = factorial(n) * inv(X0) \
            #                    @ matrix_power(inv(Id-B0), n) @ x0
            start_m_moment = np.zeros(nr_pools)
            for k in range(n+1):
                start_m_moment += stirling(n, k) * \
                    self.compute_start_m_factorial_moment(k) 
                    
            start_age_moments.append(start_m_moment*dt**n)

        return np.array(start_age_moments)
        
    def age_moment_vector(self, order, start_age_moments):
        ams = self._solve_age_moment_system(order, start_age_moments)
        amv = ams[:,order-1,:]
        
        return amv

    def system_age_moment(self, order, start_age_moments):
        n = self.nr_pools
        age_moment_vector = self.age_moment_vector(order, start_age_moments)
        age_moment_vector[np.isnan(age_moment_vector)] = 0
        soln = self.solve()
         
        total_mass = soln.sum(1) # row sum
        total_mass[total_mass==0] = np.nan

        system_age_moment = (age_moment_vector*soln).sum(1)/total_mass

        return system_age_moment
        
    def _solve_age_moment_system(self, max_order, start_age_moments):
        n = self.nr_pools
        Id = np.identity(n)
        ones = np.ones(n)
        soln = self.solve()
        dts = self.dts

        age_moments = [start_age_moments]
        for i in range(len(self.times)-1):
            vec = np.zeros((max_order,n))
            X_np1 = soln[i+1] * Id
            X_n = soln[i] * Id
            B = self.Bs[i]
            for k in range(1, max_order+1):
                moment_sum = np.zeros(n)
                for j in range(1, k+1):
                    moment_sum += age_moments[-1][j-1,:].reshape((n,)) \
                                    * binom(k,j) * dts[i]**(k-j)
                    
                vec[k-1,:] = inv(X_np1) @ B @ X_n @ \
                                    (moment_sum + ones*dts[i]**k)

            age_moments.append(vec)

        return np.array(age_moments)

    def backward_transit_time_moment(self, order, start_age_moments):
        age_moment_vector = self.age_moment_vector(order, start_age_moments)
        r = self.external_output_vector

        return (r*age_moment_vector[:-1]).sum(1)/r.sum(1)

    def start_age_densities_func(self):
        B = self.Bs[0]
        u = self.us[0]
        dt = self.dts[0]

        # assuming constant time step before times[0]
        def p(a):
            n = int(a // dt)
            if a <= 0: return np.zeros_like(u)
            return matrix_power(B, n) @ u # if age zero exists

        return p

    def _state_transition_operator(self, t1, t0, x):
        if t0 > t1:
            raise(Error('Evaluation before t0 not possible'))
    
        if t1 == t0: return x

        k0 = np.where(self.times==t0)[0][0]
        k1 = np.where(self.times==t1)[0][0]

        if hasattr(self, '_sto_recent') and \
            (self._sto_recent['k0'] == k0) and \
            (self._sto_recent['k1'] == k1):
                Phi = self.Bs[k1-1] @ self._sto_recent['Phi']
        elif hasattr(self, '_sto_recent') and \
            (self._sto_recent['k0'] == k0+1) and \
            (self._sto_recent['k1'] == k1):
                Phi = self._sto_recent['Phi'] * self.Bs[k0]
        else:
            Phi = np.identity(self.nr_pools)
            for k in range(k0, k1):
                Phi = self.Bs[k] @ Phi

        self._sto_recent = {'k0': k0, 'k1': k1, 'Phi': Phi}

        return Phi @ x

    def age_densities_1_single_value_func(self, start_age_densities):
        t0 = self.times[0]
        
        def p0(a):
            if a >= 0:
                return start_age_densities(a)
            else:
                return np.zeros((self.nr_pools,))

        Phi = self._state_transition_operator

        def p1_sv(a, t):
            res = Phi(t, t0, p0(a-(t-t0)))
            return res

        return p1_sv

    def _age_densities_1_func(self, start_age_densities):
        p1_sv = self.age_densities_1_single_value_func(start_age_densities)

        times = self.times
        t0 = times[0]
        dt = self.dts[0]
        def p1(a_min, a_max, t, coarsity):
            a_min = max(t-t0+dt, a_min)

            k_t = np.where(times==t)[0][0]
            rest_a_max = a_max - (t-t0)
            rest_a_min = a_min - (t-t0)

            a_min_nr = np.ceil(rest_a_min/dt)
            a_max_nr = np.floor(rest_a_max/dt)

            ages = t-t0 + np.arange(a_min_nr*dt, (a_max_nr+1)*dt, dt)
            ages = ages[np.arange(0, len(ages), coarsity)]
            #vals = np.array([p1_sv(a,t) for a in ages])
            
            vals = []
            if len(ages) > 0:
                for a in tqdm(ages):
                    vals.append(p1_sv(a,t))
            vals =  np.array(vals)

            return ages, vals.reshape((len(ages), self.nr_pools))

        return p1
    
    def age_densities_2_single_value_func(self):
        t0 = self.times[0]
        Phi = self._state_transition_operator
        times = self.times

        def p2_sv(a, t):
            if (a < 0) or (t-t0 < a): return np.zeros((self.nr_pools,))
            k = np.where(self.times==t-a)[0][0]
            u = self.us[k]
            res = Phi(t, t-a, u) # age 0 just arrived

            return res

        return p2_sv
 
    def _age_densities_2_func(self):
        p2_sv = self.age_densities_2_single_value_func()
        
        times = self.times
        t0 = times[0]
        def p2(a_min, a_max, t, coarsity):
            if a_min > t-t0:
                a_min = t-t0
            a_max = min(t-t0, a_max)

            k_t = np.where(times==t)[0][0]
            try:
                k_a_min = np.where(t-times[:(k_t+1)] >= a_min)[0][-1]
                k_a_max = np.where(t-times[:(k_t+1)] <= a_max)[0][0]
            except IndexError:
                return np.array([]), np.array([])

            ages = np.flip(t-times[k_a_max:(k_a_min+1)], 0)
            ages = ages[np.arange(0, len(ages), coarsity)]
            #vals = np.array([p2_sv(a,t) for a in ages])

            vals = []
            if len(ages) > 0:
                for a in tqdm(ages):
                    vals.append(p2_sv(a,t))
            vals =  np.array(vals)

            return ages, vals.reshape((len(ages), self.nr_pools))

        return p2

    def age_densities_single_value_func(self, start_age_densities):
        p1_sv = self.age_densities_1_single_value_func(
            start_age_densities
        )
        p2_sv = self.age_densities_single_value_func()

        def p_sv(a,t):
            return p1_sv(a,t) + p2_sv(a,t)

        return p_sv

    def pool_age_densities_func(self, start_age_densities, coarsity=1):
        p1 = self._age_densities_1_func(start_age_densities)
        p2 = self._age_densities_2_func()

        def p(a_min, a_max, t):
            ages_1, vals_1 = p1(a_min, a_max, t, coarsity)
            ages_1 = ages_1.tolist()
            vals_1 = vals_1.tolist()
            ages_2, vals_2 = p2(a_min, a_max, t, coarsity)
            ages_2 = ages_2.tolist()
            vals_2 = vals_2.tolist()

            ages = np.array(ages_2+ages_1)
            vals = np.array(vals_2+vals_1)

            return ages, vals

        return p
 

    def age_quantiles_at_time(self, q, t, pools, start_age_densities):
        dts = self.dts

        k = np.where(self.times==t)[0][0]
        x = self.soln[k,pools].sum()

        prev_age = 0
        age = 0
        mass = 0

        p_sv = self.age_densities_single_value(start_age_densities)
        while mass <= q*x:
            prev_age = age
            if k == 0:
                age += dts[0]
            else:
                age += dts[k-1]
                k -= 1

            mass += p_sv(age, t)[pools].sum()

        return prev_age


    ########## 14C methods #########


    def to_14C_only(self, start_values_14C, us_14C, decay_rate=0.0001209681):
        times_14C = self.times

        n = self.nr_pools
        Bs = self.Bs
        dts = self.dts

        Bs_14C = np.zeros_like(Bs)
        for k in range(len(Bs)):
            # there seems to be no difference
            Bs_14C[k] = Bs[k] * np.exp(-decay_rate*dts[k])
            #Bs_14C[k] = Bs[k] * (1.0-decay_rate*dts[k])

        dmr_14C = DiscreteModelRun_14C(
            start_values_14C,
            times_14C,
            Bs_14C,
            us_14C,
            decay_rate)

        return dmr_14C


################################################################################


class DiscreteModelRun_14C(DiscreteModelRun):

    def __init__(self, start_values, times, Bs, us, decay_rate):
        DiscreteModelRun.__init__(self, start_values, times, Bs, us)
        self.decay_rate = decay_rate

    @property
    def external_output_vector(self):
        n = self.nr_pools
        Bs = self.Bs
        dts = self.dts
        decay_rate = self.decay_rate
        rho = np.array([(1-Bs[k].sum(0).reshape((n,))) 
                          * np.exp(-decay_rate*dts[k]) for k in range(len(Bs))])
        soln = self.solve()
        r = rho * soln[:-1]

        return r


#    def solve(self):
#        if not hasattr(self, 'soln2'):
#            dts = self.dts
#            decay_rate = self.decay_rate
#            start_values = self.start_values
#            soln2 = [start_values]
#            for k in range(len(self.times)-1):
#                x_new = soln2[k].copy()
#                #x_new = x_new + self.us[k]
#                B = self.Bs[k] * np.exp(decay_rate*dts[k])
#                x_new = B @ x_new + self.us[k]
#                x_new = x_new * np.exp(-decay_rate*dts[k])
#                soln2.append(x_new)
#    
#            self.soln2 = np.array(soln2)        
#
#        return self.soln2
#        
