from pathlib import Path
from copy import deepcopy
from sympy import symbols, Function
import numpy as np
from scipy.integrate import quad

from .smooth_reservoir_model_14C import SmoothReservoirModel_14C
from .smooth_model_run import SmoothModelRun
from .helpers_reservoir import (
    net_Rs_from_discrete_Bs_and_xs,
    F_Delta_14C,
    DECAY_RATE_14C_DAILY
)


def pfile_C14Atm_NH():
    p = Path(deepcopy(__file__))
    pfile = p.parents[1].joinpath(
        'CompartmentalSystems',
        'Data',
        'C14Atm_NH.csv'
    )

    return pfile


class Error(Exception):
    """Generic error occurring in this module."""
    pass


class SmoothModelRun_14C(SmoothModelRun):

    """Construct and return a :class:`SmoothModelRun_14C` instance that
       models the 14C component of the original model run.

    Args:
        smr (SmoothModelRun): original model run
        start_values_14C (numpy.nd_array, nr_pools): 14C start values.
        Fa_func (func(t)): returns atmospheric fraction to be
            multiplied with the input vector
        decay rate (float, optional): The decay rate to be used,
            defaults to ``0.0001209681`` (daily).
    """
    def __init__(
        self,
        smr,
        start_values_14C,
        Fa_func,
        decay_rate=DECAY_RATE_14C_DAILY
    ):
        decay_symbol = symbols('lamda_14C')
        Fa_14C = Function('Fa_14C')(smr.model.time_symbol)
        srm_14C = SmoothReservoirModel_14C(smr.model, decay_symbol, Fa_14C)

        # create SmoothModelRun for 14C
        parameter_dict_14C = {k: v for k, v in smr.parameter_dict.items()}
        parameter_dict_14C['lamda_14C'] = decay_rate

#        Fa_atm = copy(atm_delta_14C)
#        Fa_atm[:,1] = Fa_atm[:,1]/1000 + 1
#        Fa_func = interp1d(Fa_atm[:,0], Fa_atm[:,1])

        func_dict_14C = {k: v for k, v in smr.func_set.items()}
        function_string = 'Fa_14C(' + srm_14C.time_symbol.name + ')'
        func_dict_14C[function_string] = Fa_func

        super().__init__(
            srm_14C,
            parameter_dict_14C,
            start_values_14C,
            smr.times,
            func_dict_14C,
        )

        self.smr = smr
        self.Fa_func = Fa_func
        self.decay_rate = decay_rate

    def acc_gross_external_output_vector(self, data_times=None):
        times = self.times if data_times is None else data_times
        nt = len(times)-1
        res = np.zeros((nt, self.nr_pools))
        for k in range(nt):
            for pool_nr, func in self.external_output_flux_funcs().items():
                res[k, pool_nr] = quad(func, times[k], times[k+1])[0]

        return res

    def acc_net_external_output_vector(self, data_times=None):
        if data_times is None:
            data_times = self.times

        x_func = self.solve_func()
        xs = x_func(data_times)
        Bs = self.fake_discretized_Bs(data_times)

        dts = np.diff(data_times)
        decay_corr = np.array([np.exp(self.decay_rate*dt) for dt in dts])
        return net_Rs_from_discrete_Bs_and_xs(
            Bs,
            xs,
            decay_corr=decay_corr
        )

    # Delta 14C methods

    def solve_Delta_14C(self, alpha=None):
        return F_Delta_14C(self.smr.solve(), self.solve(), alpha)

    def acc_gross_external_input_vector_Delta_14C(
        self,
        data_times=None,
        alpha=None
    ):
        return F_Delta_14C(
            self.smr.acc_gross_external_input_vector(data_times),
            self.acc_gross_external_input_vector(data_times),
            alpha
        )

    def acc_net_external_input_vector_Delta_14C(
        self,
        data_times=None,
        alpha=None
    ):
        return F_Delta_14C(
            self.smr.acc_net_external_input_vector(data_times),
            self.acc_net_external_input_vector(data_times),
            alpha
        )

    def acc_gross_external_output_vector_Delta_14C(
        self,
        data_times=None,
        alpha=None
    ):
        return F_Delta_14C(
            self.smr.acc_gross_external_output_vector(data_times),
            self.acc_gross_external_output_vector(data_times),
            alpha
        )

    def acc_net_external_output_vector_Delta_14C(
        self,
        data_times=None,
        alpha=None
    ):
        return F_Delta_14C(
            self.smr.acc_net_external_output_vector(data_times),
            self.acc_net_external_output_vector(data_times),
            alpha
        )

    def acc_gross_internal_flux_matrix_Delta_14C(
        self,
        data_times=None,
        alpha=None
    ):
        return F_Delta_14C(
            self.smr.acc_gross_internal_flux_matrix(data_times),
            self.acc_gross_internal_flux_matrix(data_times),
            alpha
        )

    def acc_net_internal_flux_matrix_Delta_14C(
        self,
        data_times=None,
        alpha=None
    ):
        return F_Delta_14C(
            self.smr.acc_net_internal_flux_matrix(data_times),
            self.acc_net_internal_flux_matrix(data_times),
            alpha
        )

###############################################################################

    def external_output_flux_funcs(self):
        return self._flux_funcs(self.model.output_fluxes_corrected_for_decay)

    @property
    def external_output_vector(self):
        raise(Error('Not implemented'))
#        r = super().external_output_vector
#        # remove the decay because it is not part of respiration
#        correction_rates = - np.ones_like(r) * self.decay_rate
#        soln = self.solve()
#        correction = correction_rates * soln
#        r += correction
#
#        return r
