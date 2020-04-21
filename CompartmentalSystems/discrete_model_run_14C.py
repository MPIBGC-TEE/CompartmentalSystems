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
#                #x_new = x_new + self.net_Us[k]
#                B = self.Bs[k] * np.exp(decay_rate*dts[k])
#                x_new = B @ x_new + self.net_Us[k]
#                x_new = x_new * np.exp(-decay_rate*dts[k])
#                soln2.append(x_new)
#    
#            self.soln2 = np.array(soln2)        
#
#        return self.soln2
#        
