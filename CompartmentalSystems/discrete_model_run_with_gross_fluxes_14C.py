import numpy as np

from .discrete_model_run_14C import DiscreteModelRun_14C
from .helpers_reservoir import net_Rs_from_discrete_Bs_and_xs


class Error(Exception):
    """Generic error occurring in this module."""
    pass


class DiscreteModelRunWithGrossFluxes_14C(DiscreteModelRun_14C):

    """Construct and return a :class:`DiscreteModelRunWithGrossFluxes_14C`
       instance that models the 14C component of the original model run.

    Args:
        dmrwgf (DiscreteModelRunWithGrossFluxes): original model run
        start_values_14C (numpy.nd_array, nr_pools): 14C start values.
        Fa_func (func(t)): returns atmospheric fraction to be
            multiplied with the input vector
        decay rate (float, optional): The decay rate to be used,
            defaults to ``0.0001209681`` (daily).
    """
    def __init__(
        self,
        dmrwgf,
        start_values_14C,
        #Fa_func,
        net_Us_14C,
        decay_rate=0.0001209681
    ):
        super().__init__(dmrwgf, start_values_14C, net_Us_14C, decay_rate)
        self.dmrwgf = dmrwgf

    def acc_gross_external_input_vector(self):
        return self.dmrwgf.gross_Us

    def acc_gross_internal_flux_matrix(self):
        return self.gross_Fs

    def acc_gross_external_output_vector(self):
        nt = len(times)-1
        res = np.zeros((nt, self.nr_pools))
        for k in range(nt):
            for pool_nr, func in self.external_output_flux_funcs().items():
                res[k, pool_nr] = quad(func, times[k], times[k+1])[0]

        return res

###############################################################################

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
