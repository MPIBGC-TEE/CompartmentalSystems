from pathlib import Path
from copy import deepcopy
from sympy import symbols, Function
import numpy as np
from scipy.integrate import quad

from .smooth_reservoir_model_14C import SmoothReservoirModel_14C
from .pwc_model_run import PWCModelRun
from .helpers_reservoir import net_Rs_from_discrete_Bs_and_xs


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


class PWCModelRun_14C(PWCModelRun):

    def __init__(
        self,
        pwc_mr,
        start_values_14C,
        Fa_func,
        decay_rate=0.0001209681
    ):
        """Construct and return a :class:`PWCModelRun_14C` instance that
           models the 14C component of the original model run.

        Args:
            pwc_mr (PWCModelRun): original model run
            start_values_14C (numpy.nd_array, nr_pools): 14C start values.
            Fa_func (func(t)): returns atmospheric fraction to be
                multiplied with the input vector
            decay rate (float, optional): The decay rate to be used,
                defaults to ``0.0001209681`` (daily).
        Returns:
            :class:`PWCModelRun_14C`
        """
        decay_symbol = symbols('lamda_14C')
        Fa_14C = Function('Fa_14C')(pwc_mr.model.time_symbol)
        srm_14C = SmoothReservoirModel_14C(pwc_mr.model, decay_symbol, Fa_14C)

        # create PWCModelRun for 14C
        parameter_dicts_14C = []
        for pd in pwc_mr.parameter_dicts:
            # pd is a frozendict
            pd_14C = {k: v for k, v in pd.items()}
            pd_14C['lamda_14C'] = decay_rate

            parameter_dicts_14C.append(pd_14C)

#        par_set_14C = {k: v for k, v in pwc_mr.parameter_dict.items()}
#        par_set_14C['lamda_14C'] = decay_rate

#        Fa_atm = copy(atm_delta_14C)
#        Fa_atm[:,1] = Fa_atm[:,1]/1000 + 1
#        Fa_func = interp1d(Fa_atm[:,0], Fa_atm[:,1])

        function_string = 'Fa_14C(' + srm_14C.time_symbol.name + ')'
        func_dicts_14C = []
        for fd in pwc_mr.func_dicts:
            # fd is a frozendict
            fd_14C = {k: v for k, v in fd.items()}
            fd_14C[function_string] = Fa_func

            func_dicts_14C.append(fd_14C)

#        func_set_14C = {k: v for k, v in pwc_mr.func_set.items()}
#        function_string = 'Fa_14C(' + srm_14C.time_symbol.name + ')'
#        func_set_14C[function_string] = Fa_func

        super().__init__(
            srm_14C,
            parameter_dicts_14C,
            start_values_14C,
            pwc_mr.times,
            pwc_mr.disc_times,
            func_dicts_14C,
        )
        self.Fa_func = Fa_func
        self.decay_rate = decay_rate

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

###############################################################################

    def external_output_flux_funcss(self):
        return self._flux_funcss(self.model.output_fluxes_corrected_for_decay)

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
