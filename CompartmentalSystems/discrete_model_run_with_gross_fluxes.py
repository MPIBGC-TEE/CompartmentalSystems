import numpy as np

from numpy.linalg import matrix_power, pinv
from scipy.linalg import inv
from scipy.special import factorial, binom
from sympy import Matrix

from tqdm import tqdm

from . import picklegzip
from .helpers_reservoir import x_phi_ivp
from .model_run import ModelRun
from .discrete_model_run import DiscreteModelRun


################################################################################


class DMRError(Exception):
    """Generic error occurring in this module."""
    pass


################################################################################


class DiscreteModelRunWithGrossFluxes(DiscreteModelRun, ModelRun):
    def __init__(self, times, Bs, xs, gross_Us, gross_Fs, gross_Rs):
        """
        Note: The net_Us, net_Fs and net_Rs can be computed from the solution and the Bs but there
        is no way to guess the gross fluxes (gross_Us, gross_Fs, gross_Rs)  
        without assumptions about the state transition operator in the         intervals induced by the times argument. 
        Therefore we have to provide gross fluxes separately if we want to be able to return them later as the other ModelRun sub classes.

        gross_Us accumulated influxes (flux u integrated over the time step)
        gross_Fs accumulated internal fluxes (fluxes F_ij integrated over the time step)

        gross_Rs accumulated outfluxes (flux r integrated over the time step)
        Bs State transition operators for one time step
        """
        self.times = times
        self.Bs = Bs
        self.xs = xs
        self.gross_Us = gross_Us
        self.gross_Fs = gross_Fs
        self.gross_Rs = gross_Rs

        # we use the initialization of the superclass 
        # (wich automatically creates an object of the correct sub
        # class, because the sub classes new method is (invisibly)
        # called before) 
        #super().__init__(times, Bs, xs)
        #self.gross_Us = gross_Us
   
    def acc_gross_external_input_vector(self):
        return self.gross_Us

    def acc_gross_internal_flux_matrix(self):
        return self.gross_Fs

    def acc_gross_external_output_vector(self):
        return self.gross_Rs

    @classmethod
    def from_PWCModelRun(cls, pwc_mr, data_times=None): 
        if data_times is None:
            data_times = pwc_mr.times
       
        f_solve = pwc_mr.solve_func()
        xs = f_solve(data_times)
        return cls(
            data_times,
            pwc_mr.fake_discretized_Bs(data_times),
            xs,
            pwc_mr.acc_gross_external_input_vector(data_times),
            pwc_mr.acc_gross_internal_flux_matrix(data_times),
            pwc_mr.acc_gross_external_output_vector(data_times)
        )

    @classmethod
    def reconstruct_from_fluxes_and_solution(cls, data_times, xs, net_Us, net_Fs, net_Rs, gross_Us, gross_Fs, gross_Rs):
        Bs = cls.reconstruct_Bs(xs, net_Fs, net_Rs)
        dmr = cls(data_times, Bs, xs, gross_Us, gross_Fs, gross_Rs)
        
        return dmr

    def acc_external_input_vector(self):
        return self.gross_Us


    def to_14C_only(self, start_values_14C, us_14C, decay_rate=0.0001209681):
        times_14C = self.times

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

