import numpy as np

from .discrete_model_run import DiscreteModelRun

class DiscreteModelRun_14C(DiscreteModelRun):

    def __init__(self, times, Bs, xs, decay_rate):
        super().__init__(self, times, Bs, xs)
        self.decay_rate = decay_rate

    @property
    def external_output_vector(self):
        n = self.nr_pools
        Bs = self.Bs
        dts = self.dts
        decay_rate = self.decay_rate

        # this computation seems to be wrong!
        # see helpers_reservoir.net_Rs_from_discrete_Bs_and_xs
        rho = np.array([(1-Bs[k].sum(0).reshape((n,))) 
                          * np.exp(-decay_rate*dts[k]) for k in range(len(Bs))])
        soln = self.solve()
        r = rho * soln[:-1]

        return r


