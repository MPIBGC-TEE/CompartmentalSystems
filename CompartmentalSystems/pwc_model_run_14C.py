import numpy as np
from pathlib import Path
from copy import deepcopy

from .pwc_model_run import PWCModelRun

def pfile_C14Atm_NH():
    p=Path(deepcopy(__file__))
    pfile = p.parents[1].joinpath('CompartmentalSystems','Data','C14Atm_NH.csv')
    return pfile

class PWCModelRun_14C(PWCModelRun):

    def __init__(self, srm, par_set, start_values, times, func_set, decay_rate):
        PWCModelRun.__init__(
            self, 
            srm, 
            par_set,
            start_values,
            times,
            func_set,
        )
        self.decay_rate = decay_rate

    @property 
    def external_output_vector(self):
        r = super().external_output_vector
        # remove the decay because it is not part of respiration
        correction_rates = - np.ones_like(r) * self.decay_rate
        #soln = self.solve_old()
        soln = self.solve()
        correction = correction_rates * soln
        r += correction

        return r

#    def to_14C_only(self, start_values_14C, Fa_func, decay_rate=0.0001209681):
    @classmethod
    def from_PWCModelRun(cls, pwc_mr, start_values_14C, Fa_func, decay_rate=0.0001209681):
        """Construct and return a :class:`PWCModelRun_14C` instance that
           models the 14C component of the original model run.
    
        Args:
            pwc (PWCModelRun): original model run
            start_values_14C (numpy.nd_array, nr_pools): 14C start values.
            Fa_func (func(t)): returns atmospheric fraction to be multiplied with the input vector 
            decay rate (float, optional): The decay rate to be used, defaults to
                ``0.0001209681`` (daily).
        Returns:
            :class:`PWCModelRun_14C`
        """
        srm_14C = pwc_mr.model.to_14C_only('lamda_14C', 'Fa_14C')

        # create PWCModelRun for 14C
        par_set_14C = {k:v for k, v in pwc_mr.parameter_dict.items()}
        par_set_14C['lamda_14C'] = decay_rate
        #fixme: use 14C equilibrium start values
        times_14C = pwc_mr.times

        #Fa_atm = copy(atm_delta_14C)
        #Fa_atm[:,1] = Fa_atm[:,1]/1000 + 1
        #Fa_func = interp1d(Fa_atm[:,0], Fa_atm[:,1])

        func_set_14C = {k:v for k,v in pwc_mr.func_set.items()}
        function_string = 'Fa_14C(' + srm_14C.time_symbol.name + ')'
        func_set_14C[function_string] = Fa_func

        pwc_mr_14C = cls(
            srm_14C, 
            par_set_14C,
            start_values_14C,
            times_14C,
            func_set_14C,
            decay_rate
        )

        return pwc_mr_14C


