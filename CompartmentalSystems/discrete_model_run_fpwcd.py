import numpy as np

from numpy.linalg import matrix_power, pinv
from scipy.linalg import inv
from scipy.special import factorial, binom
from sympy import Matrix

from tqdm import tqdm

from . import picklegzip
from .helpers_reservoir import x_phi_ivp
from .discrete_model_run import DiscreteModelRun
from .model_run import ModelRun


class DiscreteModelRunFPWCD(DiscreteModelRun, ModelRun):
    def __init__(self, start_values, times, Bs, gross_Us):
        """
        gross_Us accumulated fluxes (flux u integrated over the time step)
        Bs State transition operators for one time step
        """
        self.nr_pools = len(start_values)
        self.start_values = start_values.reshape((self.nr_pools,))
        self.times = times
        self.Bs = Bs
        self.gross_Us = gross_Us

    @property
    def net_Us(self):
        us=self.gross_Us/self.dts
        n=len(self.gross_Us)
        #raise (Exception('Not implemented yet'))
        # we want to compute int_t[k]^t[k+1] Phi(t[k+1],tau) u(tau) d tau
        # using (4) of the PNAS paper
        # with Phi(t[k+1],tau)= exp((t[k+1]-tau)*B[k])

        times = self.times

        # replace xs by start_values and adapt 'guess_B0'
        Bs_pwc = PWCModelRunFD.reconstruct_Bs(times, xs, Fs, rs, us)
        Phi = 

        np.linalg.inv(Bs_pwc[k]) @ (Phi_pwc(times[k+1], times[k]) - np.eye(self.nr_pools)) @ us[k]


        #F_ij = B_ij * x_j
        #F_ij = B_ij * (x_j + U_j + sum(F_jk, k))
        #x[n+1]_j = x[n]_j + u[n]_j - sum(F_ij, i)+sum(F_ji, i) -r [n]_j
        #x[n+1] = B[n] @ x[n] + u[n]


        #return np.array([np.matmul(self.Bs[k],self.gross_Us[k]) for k in range(n)]
        )

    @property
    def acc_external_input_vector(self):
        return self.gross_Us

    @classmethod
    def reconstruct_from_fluxes(cls, data_times, start_values, Fs, rs, gross_Us):
        nr_pools = len(start_values)
        Bs = np.zeros((len(data_times)-1, nr_pools, nr_pools)) 
  
        x = start_values
        net_Us = np.nan * np.ones_like(gross_Us)
        for k in range(len(data_times)-1):
            B = cls.reconstruct_B(x, Fs[k], rs[k], k)
            net_Us[k] = B @ gross_Us[k]
            x = B @ x + net_Us[k]
            Bs[k,:,:] = B

        dmr = cls(start_values, data_times, Bs, net_Us)
        return dmr

    def solve(self):
        if not hasattr(self, 'soln'):
            start_values = self.start_values
            soln = [start_values]
            for k in range(len(self.times)-1):
                x_old = soln[k]
                x_new = self.Bs[k] @ x_old + self.net_Us[k]
                soln.append(x_new)
    
            self.soln = np.array(soln)        

        return self.soln
