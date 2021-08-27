#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:

import sys
import unittest
import numpy as np
from typing import List
from sympy import sin, exp, symbols, Matrix, Symbol, solve, Eq, log
from scipy.integrate import quad
from testinfrastructure.InDirTest import InDirTest
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
from CompartmentalSystems.model_run import plot_stocks_and_fluxes
import CompartmentalSystems.helpers_reservoir as hr


class TestDiscreteModelRun(InDirTest):
    def test_from_SmoothModelRun(self):
        x_0, x_1, t, k, u = symbols("x_0 x_1 k t u")
        inputs = {
            0: u*(2-2*sin(2*t)),
            1: u
        }
        outputs = {
            0: x_0*k,
            1: x_1**2*k
        }
        internal_fluxes = {
            (0, 1): x_0,
            (1, 0): 0.5*x_1
        }
        srm = SmoothReservoirModel(
            [x_0, x_1],
            t,
            inputs,
            outputs,
            internal_fluxes
        )
        nr_bins = 20
        nr_bins_fine = 80
        t_max = 2*np.pi
        times = np.linspace(0, t_max, nr_bins + 1)
        times_fine = np.linspace(0, t_max, nr_bins_fine + 1)
        x0 = np.float(10)
        start_values = np.array([x0, x0])
        parameter_dict = {
            k: 0.012,
            u: 10.7}

        smr = SmoothModelRun(srm, parameter_dict, start_values, times)
        smr_fine = SmoothModelRun(
            srm,
            parameter_dict,
            start_values,
            times_fine
        )

        xs, net_Us, net_Fs, net_Rs = smr.fake_net_discretized_output(times)
        xs, gross_Us, gross_Fs, gross_Rs \
            = smr.fake_gross_discretized_output(times)
        xs_fine, gross_Us_fine, gross_Fs_fine, gross_Rs_fine \
            = smr_fine.fake_gross_discretized_output(times_fine)

        dmr_from_pwc = DMR.from_SmoothModelRun(smr, nr_bins)
        dmr_from_fake_net_data = DMR.from_fluxes_and_solution(
            times,
            xs,
            net_Fs,
            net_Rs
        )
        dmr_from_fake_gross_data_ffas \
            = DMR.from_fluxes_and_solution(
                times,
                xs,
                gross_Fs,
                gross_Rs
            )
        dmr_from_fake_gross_data_ff = DMR.from_fluxes(
            start_values,
            times,
            gross_Us,
            gross_Fs,
            gross_Rs
        )
        dmr_from_fake_gross_data_ff_fine = DMR.from_fluxes(
            start_values,
            times_fine,
            gross_Us_fine,
            gross_Fs_fine,
            gross_Rs_fine
        )

        self.assertTrue(
            np.allclose(
                smr.solve(),
                dmr_from_pwc.solve()
            )
        )

        self.assertTrue(
            np.allclose(
                smr.solve(),
                dmr_from_fake_net_data.solve()
            )
        )

        # very unexpectedly the solution
        # can be reconstructed from the right start_value
        # the WRONG inputs WRONG internal fluxes and
        # WRONG outputs
        self.assertTrue(
            np.allclose(
                smr.solve(),
                dmr_from_fake_gross_data_ff.solve(),
                rtol=1e-3
            )
        )

        # Here we also expect different results.
        # We again abuse the DiscreteModelRun class
        # but this time we give it the right solution
        # which will be reproduced
        self.assertTrue(
            np.allclose(
                smr.solve(),
                dmr_from_fake_gross_data_ffas.solve()
            )
        )
        # but the net influxes will be wrong
        self.assertFalse(
            np.allclose(
                smr.acc_net_external_input_vector(),
                dmr_from_fake_gross_data_ffas.net_Us
            )
        )
#        plot_attributes(
#            [
#                smr,
#                dmr_from_fake_net_data,
#                dmr_from_fake_gross_data_ff,
#                dmr_from_fake_gross_data_ffas
#            ],
#            'plot.pdf'
#        )
#        plot_stocks_and_fluxes(
#            [
#                smr,
#                # dmr_from_fake_net_data,
#                # dmr_from_pwc,
#                dmr_from_fake_gross_data_ff,
#                dmr_from_fake_gross_data_ff_fine
#            ],
#            'stocks_and_fluxes.pdf'
#        )
#        plot_stocks_and_gross_fluxes(
#            [
#                smr,
#                dmr_from_fake_net_data,
#                dmr_from_fake_gross_data_ff,
#                dmr_from_fake_gross_data_ffas
#            ],
#            'stocks_and_gross_fluxes.pdf'
#        )

#    @unittest.skip
    def test_start_value_format(self):
        # create ReservoirModel
        C_1, C_2, C_3 = symbols('C_1 C_2 C_3')
        state_vector = Matrix(3, 1, [C_1, C_2, C_3])
        t = symbols('t')
        B = Matrix([[-2, 0, 1], [2, -2, 0], [0, 2, -2]])
        u = Matrix(3, 1, [1, 0, 0])

        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        # create ModelRun
        ss = (-B**(-1)*u)
#        start_values = np.array(ss).astype(np.float64).reshape((3,))
        start_values = np.array(ss).astype(np.float64)
        times = np.linspace(1919, 2009, 901)
        parameter_dict = {}
        smr = SmoothModelRun(srm, parameter_dict, start_values, times)
        smr.initialize_state_transition_operator_cache(
            lru_maxsize=None
        )

        nt = 10
        DMR.from_SmoothModelRun(smr, nt)

    @unittest.skip(
        '''the reference solution is not implemented 
        in smooth_model_run yet'''
    )
    def test_pool_age_densities_1(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = Matrix([C_0, C_1])
        time_symbol = Symbol('t')
        #fixme: both input anoutput should be 1, 2, C_0, C_1
        #input_fluxes = {0: 1, 1: 2}
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(
                state_vector,
                time_symbol,input_fluxes,
                output_fluxes,
                internal_fluxes
        )

        start_values = np.array([5, 3])
        dt = 1e-9
        max_bin_nr = 100

        # The times array is 1 longer than the array of bin indices
        # At the moment it is supposed to be equidistant
        times=np.linspace(0, (max_bin_nr)*dt, max_bin_nr+1)
        print('times', times)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        dmr = DMR.from_SmoothModelRun(smr,max_bin_nr)


        def start_age_distributions_of_bins(age_bin_index):
            # Note (mm 12-14 2020):
            # - in the continuous case this the value is the
            #   integral \int_{ia*dt}^{(ia+1)*dt start_age_density
            #   (for each pool)
            # - more generally it can be defined directly by the
            #   user as the difference of the
            #   distribution function between F(dt*ai)-F(dt*(ai-1))
            #   as it the case here where the whole initial mass
            #   is in the first age bin (in python ia ==0)
            return start_values * (1 if a ==0 else 0)

        p_dmr = dmr.pool_age_densities_func(start_age_distributions_of_bins)

        # To test this localized distribution against the smooth model run
        # we would have to generalize the smoth model run class to accept
        # distributions or distribution differences since no density
        # exists for these cases.
        start_age_distributions = lambda a: (1 if a >= 0 else 0 ) * start_values
        # the next function is not implemented yet but would be the appropriate
        # tool to test a
        p_smr2 = smr.pool_age_densities_func_from_start_age_distributions(start_age_distributions)

        # we can choose arbitrary age bin indices
        # which are the arguments for the discrete computations
        age_bin_indices = np.array([1,5,10,20])

        # The ages fed to the reference smooth_model_run
        # are therefore integral multiples of dt
        ages = age_bin_indices * dt
        # since the grid for the smooth model run has one more index then the
        # discrete one we discard the last result here (for the comparison)
        ts = slice(0,-1)
        pool_age_densities_smr = p_smr(ages)[:, ts]

        pool_age_densities_dmr = p_dmr(age_bin_indices)
        ## dims = len(age_bin_indices),len(times)-1,npool

        self.assertTrue(
            np.allclose(
                pool_age_densities_dmr,
                pool_age_densities_smr,
                atol=1e-06
            )
        )

    def test_pool_age_densities_2(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = Matrix([C_0, C_1])
        time_symbol = Symbol('t')
        #fixme: both input anoutput should be 1, 2, C_0, C_1
        #input_fluxes = {0: 1, 1: 2}
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(
                state_vector,
                time_symbol,input_fluxes,
                output_fluxes,
                internal_fluxes
        )

        start_values = np.array([5, 3])
        dt = 1e-09
        max_bin_nr = 10

        # The times array is 1 longer than the array of bin indices
        # At the moment it is supposed to be equidistant
        times=np.linspace(0, (max_bin_nr)*dt, max_bin_nr+1)
        print('times', times)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
        dmr = DMR.from_SmoothModelRun(smr,max_bin_nr)


        # we can choose arbitrary age bin indices
        # which are the arguments for the discrete computations
        age_bin_indices = np.array([0,1,5,10,20])

        # The ages fed to the reference smooth_model_run
        # are therefore integral multiples of dt
        ages = age_bin_indices * dt


        #p_dmr = dmr.pool_age_densities_func(wrapper_maker(start_age_densities_bin))


        start_age_densities = lambda a: (np.exp(-a)  if a>=0 else 0)* start_values
        p_smr = smr.pool_age_densities_func(start_age_densities)

        # now create a function that yields the mass/da in every pool
        # as a function of the age bin index
        #def start_age_distributions_of_bin(ai: np.int64)->np.ndarray:
        #    da  = dt
        #    return np.array(
        #        [
        #            quad(
        #                lambda a:start_age_densities(a)[i],
        #                ai*da,
        #                (ai+1)*da
        #            )[0] / da
        #            for i in range(dmr.nr_pools)
        #        ]
        #    )

        start_age_densities_of_bin_index = \
        hr.pool_wise_bin_densities_from_smooth_densities_and_index(
            start_age_densities,
            dmr.nr_pools,
            dt
        )
        p_dmr = dmr.pool_age_densities_func(start_age_densities_of_bin_index)

        #def start_age_densities(a):
        #    return start_values*(1 if a == 0 else 0.0)



        # since the grid for the smooth model run has one more index then the
        # discrete one we discard the last result here (for the comparison)
        ts = slice(0,-1) 
        pool_age_densities_smr = p_smr(ages)[:, ts]

        pool_age_densities_dmr = p_dmr(age_bin_indices)
        ## dims = len(age_bin_indices),len(times)-1,npool
        print(
            pool_age_densities_smr[0,:,0 ], '\n',
            pool_age_densities_dmr[0,:,0 ], '\n',
            pool_age_densities_smr[0,:,0 ] -
            pool_age_densities_dmr[0,:,0 ]
        )
        print(times.shape)
        print(age_bin_indices.shape)
        print(pool_age_densities_dmr.shape)

        self.assertTrue(
            np.allclose(
                pool_age_densities_dmr,
                pool_age_densities_smr,
                atol=1e-06
            )
        )
        # A third way would be to compute the whole field of
        # all dt*dt bins from the beginning

        # initialize the initial age for all bins from 
        # 0 to ia_max and move forward by the application of B
        # and addition of the influxes
        #a0 = np.array(
        #   [ start_age_densities_bin(ia) for ia in range(max(age_bin_indices))]   
        #)
        #for it in range(nt):
        #    for ia in range(na-it):
        #        #p1
        #        a_next=

    #@unittest.skip
    def test_backward_transit_time_moment(self):
        # <outflux_vector, age_moment_vector> / sum(outflux_vector)

        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        dt = 1e-4
        max_bin_nr = 9

        # The times array is 1 longer than the array of bin indices
        # At the moment it is supposed to be equidistant
        times=np.linspace(0, (max_bin_nr)*dt, max_bin_nr+1)
        start_values = np.array([1,1])

        smr = SmoothModelRun(srm, {}, start_values, times=times)
        dmr = DMR.from_SmoothModelRun(smr,max_bin_nr)
        #smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        # poolwise vector of age densities as function of age: dm_i(a)/da(a)  
        start_age_densities = lambda a: (np.exp(-a)  if a>=0 else 0)* start_values

        order = 1
        start_age_moments = smr.moments_from_densities(order, start_age_densities)

        mbtt_smr = smr.backward_transit_time_moment(order, start_age_moments)
        mbtt_dmr = dmr.backward_transit_time_moment(order, start_age_moments)
        ts = slice(0,max_bin_nr)
        #print(
        #       mbtt_smr[ts], '\n',
        #       mbtt_dmr[ts], '\n',
        #       mbtt_smr[ts] - \
        #       mbtt_dmr[ts]
        #)
        self.assertTrue(
            np.allclose(
                mbtt_smr[ts], 
                mbtt_dmr,
                atol=1e-06
            )
        )

    def test_age_moment_vector(self):
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1, 0],
                    [ 0,-2]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([1,1])
        dt = 1e-09
        max_bin_nr = 1000
        times=np.linspace(0, (max_bin_nr)*dt, max_bin_nr+1)
        smr = SmoothModelRun(srm, {}, start_values, times=times)
#        print('smr.times',smr.times)
        start_age_densities = lambda a: np.exp(-a) * start_values
        max_order = 1
        start_age_moments = smr.moments_from_densities(max_order, start_age_densities)

        res_smr= smr.age_moment_vector(1, start_age_moments)
        start_values = np.array([1,1])

        smr = SmoothModelRun(srm, {}, start_values, times=times)
        dmr = DMR.from_SmoothModelRun(smr,max_bin_nr)
        # build startage_moment_vector samv
        res_dmr = dmr.age_moment_vector(max_order,start_age_moments)

#        print(
#               res_smr[...], '\n',
#               res_dmr[...], '\n',
#               #res_smr[...] - \
#               #res_dmr[...]
#        )
        self.assertTrue(
            np.allclose(
                res_dmr,
                res_smr,
                atol=1e-06
            )
        )

    def test_pool_age_distributions_quantiles(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 0, 1: 1}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(
            state_vector,
            time_symbol,
            input_fluxes,
            output_fluxes,
            internal_fluxes
        )

        start_values = np.array([1, 0])
        nr_bins = 10
        t_max = 1.0
        dt = t_max/nr_bins
        times = np.linspace(0,t_max,nr_bins+1)
        smr = SmoothModelRun(srm, {}, start_values, times)
        dmr = DMR.from_SmoothModelRun(smr,nr_bins)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)


        start_age_densities = lambda a: np.exp(-a)*start_values
        start_age_densities_of_bins = \
        hr.pool_wise_bin_densities_from_smooth_densities_and_index(
            start_age_densities,
            dmr.nr_pools,
            dt
        )

        # compute the median with different numerical methods
        start_age_moments = smr.moments_from_densities(1, start_age_densities)
        start_values_q = smr.age_moment_vector(1, start_age_moments)
        a_star_newton = smr.pool_age_distributions_quantiles(
            0.5,
            start_values=start_values_q,
            start_age_densities=start_age_densities,
            method='newton'
        )
        print(a_star_newton.shape)
        #a_star_dmr = dmr.pool_age_distributions_quantiles(
        #    0.5,
        #    start_age_densities_of_bins
        #    #start_values=start_values_q,
        #    #start_age_densities=start_age_densities,
        #    #method='newton'
        #)
        #self.assertTrue(
        #    np.allclose(
        #        a_star_dmr[:,0],
        #        np.log(2)+times,
        #        rtol=1e-3
        #    )
        #)

        #a, t = symbols('a t')
        #ref_sym = solve(Eq(1/2*(1-exp(-t)), 1 - exp(-a)), a)[0]
        #ref = np.array(
        #    [ref_sym.subs({t: time}) for time in times],
        #    dtype=np.float
        #)
        #ref[0] = np.nan

        #self.assertTrue(
        #    np.allclose(a_star_dmr[:,1],
        #        ref,
        #        equal_nan=True,
        #        rtol=1e-03
        #    )
        #)


    def test_age_quantile_bin_at_time_bin(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        input_fluxes = {0: 0, 1: 1}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(
            state_vector,
            time_symbol,
            input_fluxes,
            output_fluxes,
            internal_fluxes
        )

        start_values = np.array([1, 0])
        nr_bins = 10
        t_max = 1.0
        dt = t_max/nr_bins
        times = np.linspace(0,t_max,nr_bins+1)
        smr = SmoothModelRun(srm, {}, start_values, times)
        dmr = DMR.from_SmoothModelRun(smr,nr_bins)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        start_age_densities = lambda a: np.exp(-a)*start_values
        start_age_densities_of_bin = hr.pool_wise_bin_densities_from_smooth_densities_and_index(
                start_age_densities,
                dmr.nr_pools,
                dt
        )
        dmr = DMR.from_SmoothModelRun(smr,nr_bins)
        # just test that we can call the function properly
        res = dmr.age_quantile_bin_at_time_bin(
            q=0.5,
            it=0,
            pools=[0],
            start_age_densities_of_bin=start_age_densities_of_bin
        )
        print(res)

    def test_age_quantile_at_time(self):
        # two-dimensional
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        time_symbol = Symbol('t')
        #input_fluxes = {0: 0, 1: 1}
        #output_fluxes = {0: C_0, 1: C_1}
        input_fluxes = {0: 1, 1: 1}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(
            state_vector,
            time_symbol,
            input_fluxes,
            output_fluxes,
            internal_fluxes
        )

        #start_values = np.array([1, 0])
        start_values = np.array([1, 1])
        nr_bins = 2
        print(sys.getrecursionlimit())
        #t_max = 1.0
        #dt = t_max/nr_bins
        dt = 1e-3
        sys.setrecursionlimit(int(10.0/dt))
        times = np.linspace(0, nr_bins*dt, nr_bins+1)
        ts = slice(0,nr_bins)
        smr = SmoothModelRun(srm, {}, start_values, times)
        smr.initialize_state_transition_operator_cache(lru_maxsize=None)

        dmr = DMR.from_SmoothModelRun(smr,nr_bins)

        start_age_densities = lambda a: np.exp(-a)*start_values
        start_age_densities_of_bin = hr.pool_wise_bin_densities_from_smooth_densities_and_index(
                start_age_densities,
                dmr.nr_pools,
                dt
        )
        s=dmr.solve()
        print(s)
        results = np.array(
            [
                dmr.age_quantile_at_time(
                    q=0.5,
                    t=time,
                    pools=[0,1],
                    start_age_densities_of_bin=start_age_densities_of_bin
                )
                for time in times[ts]+0.5*dt 
            ]
        )
        print('results',results)
        #a, t = symbols('a t')
        #ref_sym = solve(Eq(1/2*(1-exp(-t)), 1 - exp(-a)), a)[0]
        #ref = np.array(
        #    [ref_sym.subs({t: time}) for time in times[ts]],
        #    dtype=np.float
        #)

        ref = np.array(
            [-log(0.5) for time in times[ts]],
            dtype=np.float
        )
        print('ref',ref)
        #ref[0] = np.nan

        self.assertTrue(
            np.allclose(
                results,
                ref,
                equal_nan=True,
                rtol=1e-02
            )
        )

    @unittest.skip
    def test_backward_transit_time_quantile_from_density(self):
        raise

    @unittest.skip("stub")
    def test_age_densities_vs_mean_age_vector(self):
        # this is some piece of code that can be reused to test the age densities
        # versus the mean age system

        def p2_sv(ai, ti):
            Phi = dmr._state_transition_operator
        
            if (ai < 0) or (ti <= ai):
                return np.zeros((dmr.nr_pools,))

            U = dmr.net_Us[ti-ai-1]
            res = Phi(ti, ti-ai, U)  # age 0 just arrived
    
            return res # dt=1

        def p1_sv(ai, ti):
            Phi = dmr._state_transition_operator
    
            if (ai < 0) or (ai < ti):
                return np.zeros((dmr.nr_pools,))

            return Phi(ti, 0, p0(ai-ti))
    
        p_sv = lambda ai, ti: p1_sv(ai, ti) + p2_sv(ai, ti)
    
        mean_system_age_sv = lambda ti: sum([k * p_sv(k, ti).sum() for k in range(ti+1)]) / dmr.solve()[ti].sum()

        print(mean_system_age)
        print([mean_system_age_sv(ti) for ti in dmr.times])

        dmr_p1_sv = dmr.age_densities_1_single_value_func(p0)
        dmr_p2_sv = dmr.age_densities_2_single_value_func()
        dmr_p_sv = dmr.age_densities_single_value_func(p0)

        print(p_sv(3, 7) - dmr_p_sv(3, 7))

    @unittest.skip("stub")
    def test_CS(self):
        self.assertEqual(dmr.Cs(0, 0), 0)
        self.assertEqual(dmr.Cs(0, 1), dmr.net_Us[0].sum())
        self.assertEqual(
            dmr.Cs(0, 2), 
            (self.Bs[0] @ dmr.net_Us[0] + dmr.net_Us[1]).sum()
        )




