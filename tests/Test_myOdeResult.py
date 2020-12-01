import unittest
from testinfrastructure.InDirTest import InDirTest

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sympy import Symbol, Piecewise
from scipy.interpolate import interp1d

from CompartmentalSystems.myOdeResult import get_sub_t_spans, solve_ivp_pwc


class TestmyOdeResult(InDirTest):
    def test_solve_ivp_pwc(self):
        t = Symbol('t')
        ms = [1, -1, 1]
        disc_times = [410, 820]
        t_start = 0
        t_end = 1000
        ref_times = np.arange(t_start, t_end, 10)
        times = np.array([0, 300, 600, 900])
        t_span = (t_start, t_end)

        # first build a single rhs
        m = Piecewise(
            (ms[0], t < disc_times[0]),
            (ms[1], t < disc_times[1]),
            (ms[2], True)
        )

        def rhs(t, x): return m.subs({'t': t})
        x0 = np.asarray([0])
        # To see the differencte between a many piece and
        # a one piece solotion
        # we deliberately chose a combination
        # of method and first step where the
        # solver will get lost if it does not restart
        # at the disc times.
        sol_obj = solve_ivp_pwc(
            rhs,
            t_span,
            x0,
            t_eval=times,
            method='RK45',
            first_step=None
        )

        def funcmaker(m):
            return lambda t, x: m

        rhss = [funcmaker(m) for m in ms]

        ref_func = interp1d(
            [t_start] + list(disc_times) + [t_end],
            [0.0, 410.0, 0.0, 180.0]
        )

        self.assertFalse(
            np.allclose(
                ref_func(times),
                sol_obj.y[0, :],
                atol=400
            )
        )

        sol_obj_pw = solve_ivp_pwc(
            rhss,
            t_span,
            x0,
            method='RK45',
            first_step=None,
            disc_times=disc_times
        )

        self.assertTrue(
            np.allclose(
                ref_func(sol_obj_pw.t),
                sol_obj_pw.y[0, :]
            )
        )

        self.assertTrue(
            np.allclose(
                ref_func(ref_times),
                sol_obj_pw.sol(ref_times)[0, :]
            )
        )

        sol_obj_pw_t_eval = solve_ivp_pwc(
            rhss,
            t_span,
            x0,
            t_eval=times,
            method='RK45',
            first_step=None,
            disc_times=disc_times
        )

        self.assertTrue(
            np.allclose(
                times,
                sol_obj_pw_t_eval.t
            )
        )

        self.assertTrue(
            np.allclose(
                ref_func(times),
                sol_obj_pw_t_eval.y[0, :]
            )
        )

        fig, ax = plt.subplots(
                nrows=1,
                ncols=1
        )
        ax.plot(
            ref_times,
            ref_func(ref_times),
            color='blue',
            label='ref'
        )
        ax.plot(
            times,
            ref_func(times),
            '*',
            label='ref points',
        )
        ax.plot(
            sol_obj.t,
            sol_obj.y[0, :],
            'o',
            label='pure solve_ivp'
        )

        ax.plot(
            sol_obj_pw.t,
            sol_obj_pw.y[0, :],
            '+',
            label='solve_ivp_pwc',
            ls='--'
        )
        ax.legend()
        fig.savefig('inaccuracies.pdf')#, tight_layout=True)

    def test_sub_t_spans(self):
        disc_times = np.array([2, 3, 4])
        t_spans = [
            (0, 1), (0, 2), (0, 2.5), (0, 5),
            (2, 2), (2, 2.5), (2, 3.2), (2, 5),
            (3.2, 4), (3.5, 4.5),
            (4, 5), (5, 7)
        ]

        refs = [
            [(0, 1), (), (), ()],
            [(0, 2), (2, 2), (), ()],
            [(0, 2), (2, 2.5), (), ()],
            [(0, 2), (2, 3), (3, 4), (4, 5)],

            [(2, 2), (2, 2), (), ()],
            [(2, 2), (2, 2.5), (), ()],
            [(2, 2), (2, 3), (3, 3.2), ()],
            [(2, 2), (2, 3), (3, 4), (4, 5)],

            [(), (), (3.2, 4), (4, 4)],
            [(), (), (3.5, 4), (4, 4.5)],

            [(), (), (4, 4), (4, 5)],
            [(), (), (), (5, 7)]
        ]

        for t_span, ref in zip(t_spans, refs):
            with self.subTest():
                sub_t_spans = get_sub_t_spans(t_span, disc_times)
                self.assertEqual(sub_t_spans, ref)


###############################################################################


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(".", pattern=__file__)

#    # Run same tests across 16 processes
#    concurrent_suite = ConcurrentTestSuite(suite, fork_for_tests(1))
#    runner = unittest.TextTestRunner()
#    res=runner.run(concurrent_suite)
#    # to let the buildbot fail we set the exit value !=0
#    # if either a failure or error occurs
#    if (len(res.errors)+len(res.failures))>0:
#        sys.exit(1)

    unittest.main()
