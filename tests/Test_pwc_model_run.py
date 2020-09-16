import unittest

import numpy as np
from scipy.special import factorial
from sympy import Function, Matrix, symbols

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.pwc_model_run import PWCModelRun


class TestPWCModelRun(unittest.TestCase):

    def setUp(self):
        x, y, t, k = symbols("x y t k")
        u_1 = Function('u_1')(x, t)
        state_vector = Matrix([x, y])
        B = Matrix([[-1,  1.5],
                    [k/(t+1), -2]])
        u = Matrix(2, 1, [u_1, 1])
        self.srm = SmoothReservoirModel.from_B_u(
            state_vector,
            t,
            B,
            u
        )

        start_values = np.array([10, 40])
        t_0 = 0
        t_max = 10
        times = np.linspace(t_0, t_max, 11)
        disc_times = [5]

        parameter_dicts = [{k: 1}, {k: 0.5*t}]
        func_dicts = [{u_1: lambda x, t: 9}, {u_1: lambda x, t: 3*t}]

        self.pwc_mr = PWCModelRun(
            self.srm,
            parameter_dicts,
            start_values,
            times,
            disc_times,
            func_dicts
        )

        timess = [
            np.linspace(t_0, disc_times[0], 6),
            np.linspace(disc_times[0], t_max, 6)
        ]

        smrs = []
        tmp_start_values = start_values
        for i in range(len(disc_times)+1):
            smrs.append(
                SmoothModelRun(
                    self.srm,
                    parameter_dict=parameter_dicts[i],
                    start_values=tmp_start_values,
                    times=timess[i],
                    func_set=func_dicts[i]
                )
            )
            tmp_start_values = smrs[i].solve()[-1]
        self.smrs = smrs

    def test_nr_intervals(self):
        self.assertEqual(len(self.smrs), self.pwc_mr.nr_intervals)

    def test_boundaries(self):
        smrs = self.smrs
        self.assertTrue(
            np.all(
                [smrs[0].times[0]]
                + self.pwc_mr.disc_times
                + [smrs[-1].times[-1]]
                == self.pwc_mr.boundaries
            )
        )

    def test_nr_pools(self):
        self.assertEqual(self.srm.nr_pools, self.pwc_mr.nr_pools)

    def test_dts(self):
        dts_smrs = [smr.dts for smr in self.smrs]
        dts_ref = np.concatenate(dts_smrs)
        self.assertTrue(np.all(dts_ref == self.pwc_mr.dts))

    def test_init(self):
        # automatically tested by setUp
        pass

    def test_solve(self):
        soln_smrs = [smr.solve() for smr in self.smrs]
        L = [soln[:-1] for soln in soln_smrs[:-1]] + [soln_smrs[-1]]
        soln_ref = np.concatenate(L, axis=0)
        self.assertTrue(np.allclose(soln_ref, self.pwc_mr.solve()))

    def test_B_func(self):
        B_func_smrs = [smr.B_func() for smr in self.smrs]
        for k, B_func in enumerate(B_func_smrs):
            with self.subTest():
                t = self.smrs[k].times[0]
                self.assertTrue(
                    np.all(B_func(t) == self.pwc_mr.B_func()(t))
                )

    def test_external_input_vector_func(self):
        u_func_smrs = [smr.external_input_vector_func() for smr in self.smrs]

        u_func_pwc = self.pwc_mr.external_input_vector_func()
        for k, u_func_k in enumerate(u_func_smrs):
            with self.subTest():
                t = self.smrs[k].times[0]
                self.assertTrue(np.all(u_func_k(t) == u_func_pwc(t)))

    def test_acc_gross_external_input_vector(self):
        ageiv_smrs = [smr.acc_gross_external_input_vector()
                      for smr in self.smrs]
        ageiv_ref = np.concatenate(ageiv_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                ageiv_ref,
                self.pwc_mr.acc_gross_external_input_vector()
            )
        )

    def test_acc_gross_internal_flux_matrix(self):
        agifm_smrs = [smr.acc_gross_internal_flux_matrix()
                      for smr in self.smrs]
        agifm_ref = np.concatenate(agifm_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                agifm_ref,
                self.pwc_mr.acc_gross_internal_flux_matrix()
            )
        )

    def test_acc_gross_external_output_vector(self):
        ageov_smrs = [smr.acc_gross_external_output_vector()
                      for smr in self.smrs]
        ageov_ref = np.concatenate(ageov_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                ageov_ref,
                self.pwc_mr.acc_gross_external_output_vector()
            )
        )

    def test_acc_net_external_input_vector(self):
        aneiv_smrs = [smr.acc_net_external_input_vector()
                      for smr in self.smrs]
        aneiv_ref = np.concatenate(aneiv_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                aneiv_ref,
                self.pwc_mr.acc_net_external_input_vector()
            )
        )

    def test_acc_net_internal_flux_matrix(self):
        anifm_smrs = [smr.acc_net_internal_flux_matrix()
                      for smr in self.smrs]
        anifm_ref = np.concatenate(anifm_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                anifm_ref,
                self.pwc_mr.acc_net_internal_flux_matrix()
            )
        )

    def test_acc_net_external_output_vector(self):
        aneov_smrs = [smr.acc_net_external_output_vector()
                      for smr in self.smrs]
        aneov_ref = np.concatenate(aneov_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                aneov_ref,
                self.pwc_mr.acc_net_external_output_vector()
            )
        )

#    def test_moments_from_densities(self):
#        # two_dimensional
#        start_values = np.array([1,2])
#        def start_age_densities(a):
#            p1 = np.exp(-a) * start_values[0]
#            p2 = 2*np.exp(-2*a) * start_values[1]
#        
#            return np.array([p1, p2])
#
#        max_order = 5
#        moments = PWCModelRun.moments_from_densities(max_order, start_age_densities)
#
#        ref1 = np.array([factorial(n)/1**n for n in range(1, max_order+1)])
#        ref2 = np.array([factorial(n)/2**n for n in range(1, max_order+1)]) 
#        ref = np.array([ref1, ref2]).transpose()
#
#        self.assertTrue(np.allclose(moments, ref,rtol=1e-3))
#
#        # test empty pool
#        start_values = np.array([0,2])
#        def start_age_densities(a):
#            p1 = np.exp(-a) * start_values[0]
#            p2 = 2*np.exp(-2*a) * start_values[1]
#        
#            return np.array([p1, p2])
#
#        max_order = 1
#        moments = PWCModelRun.moments_from_densities(max_order, start_age_densities)
#        self.assertTrue(np.isnan(moments[0,0]))
#
#    def test_age_moment_vector_semi_explicit(self):
#        start_values = self.smrs[0].start_values
#
#        def start_age_densities(a):
#            p1 = np.exp(-a) * start_values[0]
#            p2 = 2*np.exp(-2*a) * start_values[1]
#        
#            return np.array([p1, p2])
#
#        start_age_moments = PWCModelRun.moments_from_densities(1, start_age_densities)
#
#        amvse_smrs = []
#        amvse_smrs.append(start_age_moments)
#        for smr in self.smrs:
#            smr.initialize_state_transition_operator_cache(lru_maxsize=None)
#            start_age_moments = smr.age_moment_vector_semi_explicit(
#                1,
#                start_age_moments
#            )[-1]
#            amvse_smrs.append(start_age_moments)
#
#        amvse_ref = np.concatenate(amvse_smrs, axis=0)
#        self.assertTrue(
#            np.allclose(
#                aneov_ref,
#                self.pwc_mr.age_moment_vector_semi_explicit(1, amvse_smrs[0])
#            )
#        )
#
##        x, y, t = symbols("x y t")
##        X = Matrix([x,y])
##        u = Matrix(2, 1, [1, 2])
##        srm = SmoothReservoirModel.from_B_u(X, t, Matrix([[-1,0],[0,-1]]), u)
##        
##        start_values = np.array([1, 1])
##        times = np.linspace(0, 1, 10)
##        smr = SmoothModelRun(srm, {}, start_values, times)
##        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
##        n = smr.nr_pools
##
##
##        ma_ref = smr.age_moment_vector(1, start_age_moments)
##        ma_semi_explicit = smr.age_moment_vector_semi_explicit(1, start_age_moments)
##        self.assertTrue(np.allclose(ma_semi_explicit, ma_ref,rtol=1e-3))
##
##        # test empty start_ages
##        ma_ref = smr.age_moment_vector(1)
##        ma_semi_explicit = smr.age_moment_vector_semi_explicit(1)
##        self.assertTrue(np.allclose(ma_semi_explicit, ma_ref,rtol=1e-3))  
##
##        # test that nothing fails for second moment
##        start_age_moments = smr.moments_from_densities(2, start_age_densities)
##        smr.age_moment_vector_semi_explicit(2, start_age_moments)
##        smr.age_moment_vector_semi_explicit(2)
##
##        # test empty second pool at beginning
##        x, y, t = symbols("x y t")
##        X = Matrix([x,y])
##        u = Matrix(2, 1, [0, 1])
##        srm = SmoothReservoirModel.from_B_u(X, t, Matrix([[-1,0],[0,-1]]), u)
##        
##        start_values = np.array([1,0])
##        times = np.linspace(0, 1, 11)
##        smr = SmoothModelRun(srm, {}, start_values, times)
##        smr.initialize_state_transition_operator_cache(lru_maxsize=None)
##        n = smr.nr_pools
##
##        def start_age_densities(a):
##            p1 = np.exp(-a) * start_values[0]
##            p2 = np.exp(-a) * start_values[1]
##        
##            return np.array([p1, p2])
##
##        start_age_moments = smr.moments_from_densities(2, start_age_densities)
##
##        ma_ref = smr.age_moment_vector(2, start_age_moments)
##        ma_semi_explicit = smr.age_moment_vector_semi_explicit(2, start_age_moments)
##        self.assertTrue(np.allclose(ma_semi_explicit, ma_ref, equal_nan=True,rtol=1e-3))

    def test_age_moment_vector(self):
        start_values = self.smrs[0].start_values

        def start_age_densities(a):
            p1 = np.exp(-a) * start_values[0]
            p2 = 2*np.exp(-2*a) * start_values[1]
        
            return np.array([p1, p2])

        start_age_moments = PWCModelRun.moments_from_densities(1, start_age_densities)

        start_age_moments_tmp = start_age_moments.copy()
        amv_smrs = []
        amv_smrs.append(start_age_moments)
        for smr in self.smrs:
            age_moments = smr.age_moment_vector(
                1,
                start_age_moments_tmp
            )
            amv_smrs.append(age_moments[1:])
            start_age_moments_tmp = age_moments[-1].reshape(1, -1)

        amv_ref = np.concatenate(amv_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                amv_ref,
                self.pwc_mr.age_moment_vector(1, start_age_moments)
            )
        )

    def test_system_age_moment(self):
        start_values = self.smrs[0].start_values

        def start_age_densities(a):
            p1 = np.exp(-a) * start_values[0]
            p2 = 2*np.exp(-2*a) * start_values[1]
        
            return np.array([p1, p2])

        start_age_moments = PWCModelRun.moments_from_densities(1, start_age_densities)

        start_age_moments_tmp = start_age_moments.copy()
        sam_smrs = []
        for nr, smr in enumerate(self.smrs):
            age_moments = smr.age_moment_vector(
                1,
                start_age_moments_tmp
            )

            sam = smr.system_age_moment(1, start_age_moments_tmp)
            if nr == 0:
                sam_smrs.append(sam[0])
            sam_smrs.extend(sam[1:].flatten())
            
            start_age_moments_tmp = age_moments[-1].reshape(1, -1)

        self.assertTrue(
            np.allclose(
                sam_smrs,
                self.pwc_mr.system_age_moment(1, start_age_moments)
            )
        )

    ###########################################################################

    def test_fake_discretized_Bs(self):
        Bs_smrs = [smr.fake_discretized_Bs()
                   for smr in self.smrs]
        Bs_ref = np.concatenate(Bs_smrs, axis=0)
        self.assertTrue(
            np.allclose(
                Bs_ref,
                self.pwc_mr.fake_discretized_Bs()
            )
        )

        # test with cache
        self.pwc_mr.initialize_state_transition_operator_cache(
            100,
            True,
            4
        )
        self.assertTrue(
            np.allclose(
                Bs_ref,
                self.pwc_mr.fake_discretized_Bs(),
                rtol=1e-03
            )
        )

###############################################################################


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(".", pattern=__file__)
    unittest.main()
