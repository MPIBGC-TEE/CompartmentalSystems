# vim:set ff=unix expandtab ts=4 sw=4
from typing import List, Dict, Callable, Tuple, Any 
import unittest
import numpy as np
from sympy import sin, exp, symbols, Matrix, Symbol, solve, Eq, log, Expr, lambdify
from scipy.integrate import quad
import matplotlib

matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from CompartmentalSystems.bins.CompatibleTsTpMassFieldsPerPool import (
    CompatibleTsTpMassFieldsPerPool,
)
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.bins.TimeStepIterator import TimeStepIterator
from CompartmentalSystems.bins.TimeStep import TimeStep
from CompartmentalSystems.bins.TsTpMassField import TsTpMassField
from CompartmentalSystems.bins.TsTpDeathRateField import TsTpDeathRateField
from CompartmentalSystems.helpers_reservoir import make_cut_func_set, numerical_function_from_expression

from testinfrastructure.InDirTest import InDirTest


def zero_input(rectangles,t):
    return 0


def const_input(rectangles,t):
    return 5


class TestTimeStepIterator(InDirTest):
    def test_list_comprehension(self):
        ############################################################
        # get the components ready
        # - initial age distributions
        tss = 1
        x, y = 9, 9
        s = (x, y)
        age_dist_0 = TsTpMassField(np.zeros(s), tss)
        age_dist_0[2, 2] = 100

        x1, y1 = 4, 4
        s = (x1, y1)
        age_dist_1 = TsTpMassField(np.zeros(s), tss)
        age_dist_1[3, 3] = 100
        # the output from pool_1 has a bigger age_span than pool_0 can encompass
        # the code has to initialize pool_0 regarding the size of pool_1
        initial_plains = CompatibleTsTpMassFieldsPerPool([age_dist_0, age_dist_1])

        ############################################################
        # - deathrate functions
        loss_factor = 0.1
        external_death_rate_funcs = dict()
        internal_death_rate_funcs = dict()
        def func_maker(donor_pool_ind):
            def constant_well_mixed_death_rate(
                    age_dist_list: List[TsTpMassField],
                    t: float
                ) -> TsTpDeathRateField:
                # these functions must be able to define a field eta
                # of the same size as the age distribution of the donor_pool
                # for all the ages present in age_dist it must
                # be able to compute the deathrate
                age_dist = age_dist_list[donor_pool_ind]
                return TsTpDeathRateField(
                    loss_factor * np.ones(age_dist.arr.shape), age_dist.tss
                )
            return constant_well_mixed_death_rate

        external_death_rate_funcs[0] = func_maker(0) 
        external_death_rate_funcs[1] = func_maker(1)
        internal_death_rate_funcs[(0, 1)] = func_maker(0)
        # - input functions
        external_input_funcs = dict()
        external_input_funcs[0] = const_input
        external_input_funcs[1] = zero_input

        #drf = lambda t: 0.2
        ############################################################
        # initialize the Iterator
        it = TimeStepIterator(
            initial_plains,
            external_input_funcs,
            internal_death_rate_funcs,
            external_death_rate_funcs,
            t0=0,
            number_of_steps=100,
        )

        ############################################################
        ############################################################
        ############################################################
        # start testing
        # extract the complete information
        #steps = [ts for ts in it]

        # or only the part one is interested in
        rectangles_for_first_pool = [ts.rectangles[0] for ts in it]
        # print("\n#####################################\nrectangles[0]",rectangles_for_first_pool)
        # or some parts
        tuples = [(ts.time, ts.rectangles[0].total_content) for ts in it]
        x = [t[0] for t in tuples]
        y = [t[1] for t in tuples]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, "x")
        fig.savefig("plot.pdf")
        plt.close(fig.number)


    def test_exponential_decay(self):
        # We take a compartmental model (with age selection functions that do not depend
        # on either pool- or system age and reproduce the results. 
        x_0, t, k, u = symbols("x_0 k t u")
        time_symbol = t
        inputs = {}
        outputs = { 0: x_0*k }
        internal_fluxes = {}
        state_vector =[ x_0 ]
        srm = SmoothReservoirModel(
            state_vector,
            t,
            inputs,
            outputs,
            internal_fluxes
        )

        nr_t_bins = 10
        tss = 5
        t_max = nr_t_bins*tss
        t_min = 0
        times = np.linspace(t_min, t_max, nr_t_bins + 1)
        x0 = float(100)
        start_values = np.array([x0])
        parameter_dict = { k: .1}
        func_dict = {}
        smr = SmoothModelRun(srm, parameter_dict, start_values, times)
        sol=smr.solve()
        age_dist_0 = TsTpMassField(x0*np.ones((1,1)), tss)

        initial_plains = CompatibleTsTpMassFieldsPerPool([age_dist_0])
        it_srm = TimeStepIterator.piecewise_constant_from_SmoothReservoirModel(
            srm,
            parameter_dict,
            func_dict,
            initial_plains,
            t_0=t_min,
            number_of_steps=nr_t_bins,
            tss=tss
         )
        it_smr = TimeStepIterator.from_SmoothModelRun(
            smr,
            initial_plains,
            t_0=t_min,
            number_of_steps=nr_t_bins,
            tss=tss
         )

        ############################################################
        # start testing
        # extract the complete information
        # steps = [ts for ts in it_smr]

        # or some parts
        times = [ts.time for ts in it_srm]
        total_mass_srm = [
            ts.rectangles[0].total_content
            for ts in it_srm
        ]
        total_mass_smr = [
            ts.rectangles[0].total_content
            for ts in it_smr
        ]
        total_mass_cont = sol[:-1, 0]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(times, total_mass_srm, "x", color='b')
        ax.plot(times, total_mass_smr, "o", color='b')
        ax.plot(times, total_mass_cont, color='b')
        fig.savefig("plot.pdf")
        plt.close(fig.number)
        death_rates = [
            ts.external_death_rate_fields[0] for ts in it_srm
        ]
        fig = plt.figure()
        n = len(death_rates)
        for i in range(n):
            ax = fig.add_subplot(n, 1, i+1,projection='3d')
            death_rates[i].plot_bins(ax)
        fig.savefig("deathrate.pdf")
        plt.close(fig.number)

    def test_two_pool(self):
        # We take a compartmental model (with age selection functions that do not depend
        # on either pool- or system age and reproduce the results. 
        x_0, x_1, t, k, u, z = symbols("x_0 x_1 k t u z")
        inputs = {
            0: u*(1-sin(t/5)),
            1: u/10
        }
        #inputs = {
        #    0: z,
        #    1: z
        #}
        outputs = {
            0: 0.01*x_0*k,
            1: 0.01*x_1**2*k
        }
        internal_fluxes = {
            (0, 1): 2*k*x_0,
            (1, 0): .0005*k*x_1
        }
        state_vector =[ x_0, x_1]
        srm = SmoothReservoirModel(
            state_vector,
            t,
            inputs,
            outputs,
            internal_fluxes
        )

        nr_t_bins = 20
        tss = 5
        t_max = nr_t_bins*tss
        t_min = 0
        times = np.linspace(t_min, t_max, nr_t_bins + 1)
        x0 = float(100)
        x1 = float(100)
        start_values = np.array([x0,x1])
        parameter_dict = {
            k: 0.001,
            u: 6,
            z: 0
        }
        func_dict = {}
        smr = SmoothModelRun(srm, parameter_dict, start_values, times)
        sol=smr.solve()
        age_dist_0 = TsTpMassField(x0*np.ones((1,1)), tss)
        age_dist_1 = TsTpMassField(x1*np.ones((1,1)), tss)

        initial_plains = CompatibleTsTpMassFieldsPerPool([age_dist_0, age_dist_1])
        it_srm = TimeStepIterator.piecewise_constant_from_SmoothReservoirModel(
            srm,
            parameter_dict,
            func_dict,
            initial_plains,
            t_0=t_min,
            number_of_steps=nr_t_bins,
            tss=tss
        ) 
        it_smr = TimeStepIterator.from_SmoothModelRun(
            smr,
            initial_plains,
            t_0=t_min,
            number_of_steps=nr_t_bins,
            tss=tss
         )

        ############################################################
        ############################################################
        ############################################################
        # start testing
        # extract the complete information
        # steps = [ts for ts in it_srm]

        # or some parts
        times = [ts.time for ts in it_srm]
        total_mass_srm_0 = [ ts.rectangles[0].total_content for ts in it_srm]
        total_mass_srm_1 = [ ts.rectangles[1].total_content for ts in it_srm]
        total_mass_smr_0 = [ ts.rectangles[0].total_content for ts in it_smr]
        total_mass_smr_1 = [ ts.rectangles[1].total_content for ts in it_smr]
        total_mass_cont_0 = sol[:-1,0]
        total_mass_cont_1 = sol[:-1,1]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(times, total_mass_srm_0, 'x', color='r')
        ax.plot(times, total_mass_smr_0, 'o', color='r')
        ax.plot(times, total_mass_cont_0, color='r')
        ax.plot(times, total_mass_srm_1, 'x', color='b')
        ax.plot(times, total_mass_smr_1, 'o', color='b')
        ax.plot(times, total_mass_cont_1,  color='b')
        fig.savefig("plot.pdf")
        plt.close(fig.number)
        death_rates_0 = [ ts.internal_death_rate_fields[(0,1)] for ts in it_srm]
        #death_rates_1 = [ ts.internal_death_rate_fields[1] for ts in it]
        fig = plt.figure()
        #n = len(death_rates)
        n = 1
        for i in range(n):
            ax = fig.add_subplot(n, 1, i+1,projection='3d')
            death_rates_0[i].plot_bins(ax)
            #ax = fig.add_subplot(n, 2, i+1,projection='3d')
            #death_rates_1[i].plot_bins(ax)
        fig.savefig("deathrate.pdf")
        plt.close(fig.number)
if __name__ == "__main__":
    unittest.main()
