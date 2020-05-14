import numpy as np

from .discrete_model_run import DiscreteModelRun
from .helpers_reservoir import net_Rs_from_discrete_Bs_and_xs


class Error(Exception):
    """Generic error occurring in this module."""
    pass


class DiscreteModelRun_14C(DiscreteModelRun):

    """Construct and return a :class:`DiscreteModelRun_14C` instance that
       models the 14C component of the original model run.

    Args:
        dmr (DiscreteModelRun): original model run
        start_values_14C (numpy.nd_array, nr_pools): 14C start values.
        Fa_func (func(t)): returns atmospheric fraction to be
            multiplied with the input vector
        decay rate (float, optional): The decay rate to be used,
            defaults to ``0.0001209681`` (daily).
    """
    def __init__(
        self,
        dmr,
        start_values_14C,
        # Fa_func,
        net_Us_14C,
        decay_rate=0.0001209681
    ):

        # compute Bs_14C
        Bs = dmr.Bs
        dts = dmr.dts
        Bs_14C = np.zeros_like(Bs)
        for k in range(len(Bs)):
            Bs_14C[k] = Bs[k] * np.exp(-decay_rate*dts[k])

#        # compute net_Us_14C
#        net_Us = dmr.net_Us
#        net_Us_14C = np.array(
#            [Fa_func(t) * net_U * np.exp(-decay_rate*dt)
#            for t, net_U, dt in zip(dmr.times[:-1], net_Us, dts)]
#        )

        # compute xs_14C
        xs_14C = self._solve(start_values_14C, Bs_14C, net_Us_14C)

        super().__init__(
            dmr.times,
            Bs_14C,
            xs_14C
        )
        # self.Fa_func = Fa_func
        self.decay_rate = decay_rate

    def acc_net_external_output_vector(self):
        decay_corr = np.array([np.exp(self.decay_rate*dt) for dt in self.dts])
        return net_Rs_from_discrete_Bs_and_xs(
            self.Bs,
            self.xs,
            decay_corr=decay_corr
        )

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
