# vim:set ff=unix expandtab ts=4 sw=4
from typing import List, Tuple, Dict, Callable, Tuple, Any 
from sympy import sin, exp, symbols, Matrix, Symbol, Function, solve, Eq, log, Expr, lambdify
from .TimeStep import TimeStep
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from ..smooth_reservoir_model import SmoothReservoirModel
from ..smooth_model_run import SmoothModelRun
from ..helpers_reservoir import make_cut_func_set, numerical_function_from_expression

from .TsTpMassFields import TsTpMassFieldsPerPool, TsTpMassFieldsPerPipe
from .CompatibleTsTpMassFieldsPerPool import CompatibleTsTpMassFieldsPerPool
from .TsTpMassField import TsTpMassField
from .TsTpDeathRateField import TsTpDeathRateField
from .TsTpMassFieldsPerPoolPerTimeStep import TsTpMassFieldsPerPoolPerTimeStep



def zero_input(rectangles,t):
    return 0

#def external_death_rate_maker(sender, func, solfs):
#    def wrapper(field, t):
#        tss = field.tss
#        loss = quad(func, t, t + tss)[0]
#        stock = solfs[sender](t)
#        if stock != 0:
#            relative_loss = loss / stock
#        else:
#            relative_loss = 0
#        dr = TsTpDeathRateField(relative_loss * np.ones(field.shape), tss)
#        return dr
#
#    return wrapper
#
#
#def internal_death_rate_maker(key, func, solfs):
#    def wrapper(field, t):
#        sender = key[0]
#        tss = field.tss
#        loss = quad(func, t, t + tss)[0]
#        stock = solfs[sender](t)
#
#        if stock != 0:
#            relative_loss = loss / stock
#        else:
#            relative_loss = 0
#
#        dr = TsTpDeathRateField(relative_loss * np.ones(field.shape), tss)
#        return dr
#
#    return wrapper
#
#
#def external_input_maker(tss, receiver, func):
#    def wrapper(t):
#        return quad(func, t, t + tss)[0]
#
#    return wrapper

def external_input_func_maker(
        srm: SmoothReservoirModel,
        receiver_pool_ind: int,
        parameter_dict: Dict[Symbol,Any],
        func_dict: Dict[Symbol,Callable],
        tss: float
) -> Callable[
            [
                Tuple[TsTpMassField],
                float
            ],
            float
        ]:

    flux_expr = srm.input_fluxes[receiver_pool_ind]
    expr_par = flux_expr.subs(parameter_dict)
    significant_pools = expr_par.free_symbols
    def input_flux_func(
        age_dist_list: Tuple[TsTpMassField],
        t: float
        )-> float:
        # age_dist_list is like a generalized state vector.
        # In theory the influxes 
        # can depend not only on the receiving pool age distribution
        # but on the age distributions of all the other pools as well.
        #
        # This is also the case when there is no age dependence
        # and the system reduces to a compartmental system
        # where in the nonlinear case contents of all pools
        # can potentially influence a flux 
        # (even an influx e.g. when the size of a leaf pool determines how fast carbon can be allocated).
        #
        # However for computational efficiency 
        # we pick out only those pools (age distributions) that 
        # appear in the flux_expr 
        # We set the unused pool contents to NaN for the fluxcomputation since they will be ignored by
        # the fluxfunction anyway.

        # to simulate a (nonlinear) well mixed system we
        # have to sum up the mass in the bins of each pool
        contents = [
                age_dist_list[i].total_content if srm.state_vector[i] in significant_pools else np.nan
                for i in range(len(age_dist_list))
        ]

        cut_func_set = make_cut_func_set(func_dict)
        tup= (srm.time_symbol,)+tuple(srm.state_vector)
        flux_func = lambdify(tup, expr_par, modules=[cut_func_set, 'numpy'])
        res = flux_func(t,*contents)*tss
        #print(res)
        return res

    return input_flux_func

def input_func_maker_from_net_fluxes(
    net_flux_vals: np.ndarray,
    donor_pool_index: int,
    tss: float
) -> Callable[
    [
        Tuple[TsTpMassField],
        float
    ],
    float
]:
    def input_func(
        age_dist_list: Tuple[TsTpMassField],
        t: float
    ) -> TsTpDeathRateField:
        index = int(np.floor(t/tss))
        gain = net_flux_vals[index] 
        return gain

    return input_func

def well_mixed_death_rate_func_maker_from_net_fluxes(
    net_flux_vals: np.ndarray,
    donor_pool_contents: np.ndarray,
    donor_pool_index: int,
    tss: float
) -> Callable[
    [
        Tuple[TsTpMassField],
        float
    ],
    TsTpDeathRateField
]:
    def death_rate_func(
        age_dist_list: Tuple[TsTpMassField],
        t: float
    ) -> TsTpDeathRateField:
        index = int(np.floor(t/tss))
        loss_factor = net_flux_vals[index] / donor_pool_contents[index]
        age_dist = age_dist_list[donor_pool_index]
        dr = TsTpDeathRateField(
            loss_factor * np.ones(age_dist.arr.shape), age_dist.tss
        )
        return dr

    return death_rate_func

def well_mixed_death_rate_func_maker_from_expression(
        state_vector: Tuple[Symbol],
        time_symbol: Symbol,
        flux_expr: Expr,
        donor_pool_ind: int,
        parameter_dict: Dict[Symbol,Any],
        func_dict: Dict[Symbol,Callable],
        tss: float
) -> Callable[
            [
                Tuple[TsTpMassField],
                float
            ],
            TsTpDeathRateField
        ]:

    expr_par = flux_expr.subs(parameter_dict)
    donor_pool_name=state_vector[donor_pool_ind]
    significant_pools = (expr_par.free_symbols).union([donor_pool_name])



    def death_rate_func(
        age_dist_list: Tuple[TsTpMassField],
        t: float
    )-> TsTpDeathRateField:
        # age_dist_list is like a generalized state vector.
        # In theory the fluxes (and consequently the deathrates)
        # can depend not only on the donor pool age distribution
        # but also on the age distributions of 
        # all the other pools as well.
        #
        # This is also the case when there is no age dependence
        # and the system reduces to a compartmental system
        # where in the nonlinear case contents of all pools
        # can potentially influence a flux.
        #
        # However for computational efficiency 
        # we pick out only those pools (age distributions) that 
        # appear in the rate_expr 
        # We set the unused pool contents to NaN for the fluxcomputation since they will be ignored by
        # the fluxfunction anyway.

        # to simulate a (nonlinear) well mixed system we
        # have to sum up the mass in the bins of each pool
        contents = [
                age_dist_list[i].total_content if state_vector[i] in significant_pools else np.nan
                for i in range(len(age_dist_list))
        ]

        cut_func_set = make_cut_func_set(func_dict)
        tup= (time_symbol,)+tuple(state_vector)
        flux_func = lambdify(tup, expr_par, modules=[cut_func_set, 'numpy'])
        # flux_func = numerical_function_from_expression(flux_expr, tup, parameter_dict, func_dict)
        loss = flux_func(t,*contents)*tss # Euler forward
        # in this example we could even cheat by using an ode solver that 
        # integrates the state transition  operator over the timestep and computes the loss
        # as the difference 
        # but this would obscure the more general case where
        # the loss is actually pool and system age dependent (and we cannot use the state transition operator)
        # the new age distribution of mass in a pool (and thereby the pool content as its integral) 
        # is the RESULT of the bin calculation.
        donor_mass = contents[donor_pool_ind]
        
        loss_factor = loss/contents[donor_pool_ind]
        #
        # These functions must be able to define a field eta
        # of the same size as the age distribution of the donor
        # pool 
        # for all the ages present in age_dist it must
        # be able to compute the deathrate
        age_dist = age_dist_list[donor_pool_ind] 
        dr = TsTpDeathRateField(
            loss_factor * np.ones(age_dist.arr.shape), age_dist.tss
        )
        return dr

    return death_rate_func


#########################################################################
#########################################################################
#########################################################################
#########################################################################
class TimeStepIterator:
    """iterator for looping over the results of a difference equation"""

    def __init__(
        self,
        initial_plains,
        external_input_funcs=None,
        internal_death_rate_funcs=dict(),
        external_death_rate_funcs=dict(),
        t0=0,
        number_of_steps=10,
    ):
        self.t0 = t0
        self.initial_plains = initial_plains
        self.number_of_steps = number_of_steps

        self.external_input_funcs = {
            pool_ind: zero_input 
            for pool_ind in range(len(initial_plains))
        } if external_input_funcs is None else external_input_funcs
        self.internal_death_rate_funcs = internal_death_rate_funcs
        self.external_death_rate_funcs = external_death_rate_funcs
        self.reset()

    ######################################################################
    @classmethod
    def piecewise_constant_from_SmoothReservoirModel(
        cls,
        srm: SmoothReservoirModel,
        parameter_dict: Dict[Symbol,Any],
        func_dict: Dict[Function,Callable],
        initial_plains: List[TsTpMassField] = None,
        t_0: float = 0,
        number_of_steps: int = 5,
        tss: float =0.5
    ) -> 'TimeStepIterator':
        ############################################################
        # get the components ready
        # - initial age distributions
        if initial_plains is None:
            x, y = 1, 1 # all the mass in one bin
            s = (x, y)
            initial_plains = CompatibleTsTpMassFieldsPerPool([TsTpMassField(np.zeros(s), tss) for i,v in enumerate(smr.statevector)])

        ############################################################
        # - deathrate functions
        external_death_rate_funcs = {
            src: well_mixed_death_rate_func_maker_from_expression(
                srm.state_vector,
                srm.time_symbol,
                srm.output_fluxes[src],
                src,
                parameter_dict,
                func_dict,
                tss
            )
            for src in srm.output_fluxes.keys()
        }

        internal_death_rate_funcs = {
            (src, d): well_mixed_death_rate_func_maker_from_expression(
                srm.state_vector,
                srm.time_symbol,
                srm.internal_fluxes[(src, d)],
                src,
                parameter_dict,
                func_dict,
                tss
            )
            for src,d in srm.internal_fluxes.keys()
        }
        # - input functions

        external_input_funcs = {
            d: external_input_func_maker(
                srm=srm,
                receiver_pool_ind=d,
                parameter_dict=parameter_dict,
                func_dict=func_dict,
                tss=tss
            )
            for d in srm.input_fluxes.keys()
        }
        #external_input_funcs[0] = zero_input

        #############################################################
        # initialize the Iterator
        it = cls(
            initial_plains,
            external_input_funcs=external_input_funcs,
            internal_death_rate_funcs=internal_death_rate_funcs,
            external_death_rate_funcs=external_death_rate_funcs,
            t0=t_0,
            number_of_steps=number_of_steps,
        )
        return it

    @classmethod
    def from_SmoothModelRun(
        cls,
        smr: SmoothModelRun,
        initial_plains=None,
        t_0: float = 0,
        number_of_steps: int = 5,
        tss: float =0.5
    ) -> 'TimeStepIterator':
        if initial_plains is None:
            x, y = 1, 1 # all the mass in one bin
            s = (x, y)
            initial_plains = CompatibleTsTpMassFieldsPerPool(
                [
                    TsTpMassField(np.zeros(s), tss) 
                    for i, v in enumerate(smr.statevector)
                ]
            )

        xs, net_Us, net_Fs, net_Rs = smr.fake_net_discretized_output(smr.times)
        external_input_funcs = {
            src: input_func_maker_from_net_fluxes(
                net_Us[:, src],
                src,
                tss
            )
            for src in smr.model.output_fluxes.keys()
        }
        external_death_rate_funcs = {
            src: well_mixed_death_rate_func_maker_from_net_fluxes(
                net_Rs[:, src],
                xs[:, src],
                src,
                tss
            )
            for src in smr.model.output_fluxes.keys()
        }
        internal_death_rate_funcs = {
            (src, dest): well_mixed_death_rate_func_maker_from_net_fluxes(
                net_Fs[:, dest, src],
                xs[:, src],
                src,
                tss
            )
            for (src, dest) in smr.model.internal_fluxes.keys()
        }
        # initialize the Iterator
        it = cls(
            initial_plains,
            external_input_funcs=external_input_funcs,
            internal_death_rate_funcs=internal_death_rate_funcs,
            external_death_rate_funcs=external_death_rate_funcs,
            t0=t_0,
            number_of_steps=number_of_steps,
        )
        return it
    #    obj = cls.__new__(cls)
    #    number_of_pools = mr.nr_pools
    #    start_values = mr.start_values
    #    # to avoid excess of numerical cost we limit to 100 time steps here
    #    obj.number_of_steps = 100
    #    # and adapt the time step size accordingly
    #    # holger: change to //4+1 and find out what goes wrong
    #    # with bare fallow in ICBM
    #    times = mr.times[: len(mr.times) // 4]
    #    #        times=mr.times[:obj.number_of_steps]
    #    # print(times)
    #    tss = (times[-1] - times[0]) / obj.number_of_steps
    #    #        tss=(times[1]-times[0])
    #    #        print(times)
    #    #        print(tss)
    #    # fixme: find right times

    #    if not (initial_plains):
    #        obj.initial_plains = CompatibleTsTpMassFieldsPerPool(
    #            [
    #                TsTpMassField(start_values[i] * np.ones((1, 1)), tss)
    #                for i in range(number_of_pools)
    #            ]
    #        )

    #        # holger: added initial distr
    #    #            init_list = []
    #    #            for i in range(number_of_pools):
    #    #                k=20
    #    #                pool_field = np.zeros((k,1))
    #    #                pool_field[:k,0]=[start_values[i]/k for j in range(k)]
    #    ##                pool_field[:50,0] = [0.028*(1-4/5*tss)**Ts for Ts in range(50)]
    #    #                print(sum(pool_field))
    #    #                init_list.append(pool_field)
    #    #
    #    #            obj.initial_plains=CompatibleTsTpMassFieldsPerPool(
    #    #                [
    #    #                    TsTpMassField(init_list[i],tss)
    #    #                    for i in range(number_of_pools)
    #    #                ]
    #    #            )

    #    else:  # adjust tss of the plains
    #        for plane in initial_planes:
    #            plane.tss = tss

    #    ## we now build the deathrate functions
    #    ## note that the factories depend
    #    ## on the solution funtions

    #    # produce the output deathrate functions

    #    obj.external_death_rate_funcs = dict()
    #    solfs = mr.sol_funcs()
    #    for sender, func in mr.external_output_flux_funcs().items():
    #        obj.external_death_rate_funcs[sender] = external_death_rate_maker(
    #            sender, func, solfs
    #        )

    #    ## produce the internal deathrate functions
    #    obj.internal_death_rate_funcs = dict()
    #    for key, func in mr.internal_flux_funcs().items():
    #        obj.internal_death_rate_funcs[key] = internal_death_rate_maker(
    #            key, func, solfs
    #        )

    #    # produce the external inputs
    #    obj.external_input_funcs = dict()
    #    for receiver, func in mr.external_input_flux_funcs().items():
    #        obj.external_input_funcs[receiver] = external_input_maker(
    #            tss, receiver, func
    #        )

    #    obj.t0 = times[0]
    #    obj.reset()
    #    return obj

    ######################################################################
    @property
    def tss(self):
        return self.initial_plains[0].tss

    def reset(self):
        self.i = 0
        self.time = self.t0
        self.rectangles = self.initial_plains

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        number_of_steps = self.number_of_steps
        if self.i == number_of_steps:
            raise StopIteration
        # compute deathrate fields
        t = self.t0 + self.i * self.tss
        internal_death_rate_fields = {
            # in general the fluxes can depend on all statevariables 
            # not only on the donating and receiving pools
            # so f takes a list of all age distributions
            pipe_key: f(self.rectangles, t)
            for pipe_key, f in self.internal_death_rate_funcs.items()
        }

        external_death_rate_fields = {
            # in general the fluxes can depend on all statevariables 
            # not only on the donation pool
            # so in general f depends on  
            # all the other age distributions
            pool_key: f(self.rectangles, t)
            for pool_key, f in self.external_death_rate_funcs.items()
        }
        # compute external inputs
        external_input_numbers = {
            # in general the fluxes can depend on all statevariables 
            # not only on the receiving pools
            # so f takes a list of all age distributions
            key: f(self.rectangles,t) for key, f in self.external_input_funcs.items()
        }

        ts = TimeStep(
            t,
            self.rectangles,
            internal_death_rate_fields,
            external_death_rate_fields,
            external_input_numbers,
        )
        self.rectangles = ts.updated_content
        # print(t, "%0.9f" % self.rectangles[0].total_content)
        # holger: external losses were not removed,
        # they still seem to be at least a little wrong
        # print(self.rectangles[0].total_content)
        self.i += 1
        return ts
