from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
import numpy as np
from collections.abc import Iterable


def get_sub_t_spans(t_span, disc_times):
    t_0, t_max = t_span
    disc_times = [-np.inf] + list(disc_times) + [np.inf]
    intervls = zip(disc_times[:-1], disc_times[1:])

    sub_t_spans = []
    for k, span in enumerate(intervls):
        left = max(t_0, span[0])
        right = min(t_max, span[1])
        if left > right:
            sub_t_span = ()
        else:
            sub_t_span = (left, right)

        sub_t_spans.append(sub_t_span)

    return sub_t_spans


def solve_ivp_pwc(rhss, t_span, y0, disc_times=(), **kwargs):
#    print("myOdeResult.py 27", disc_times, flush=True)
    
    if not isinstance(rhss, Iterable):
        rhss = (rhss,)

    assert(len(rhss) == len(disc_times) + 1)

    kwargs['dense_output'] = True

    if 'method' not in kwargs.keys():
        kwargs['method'] = 'Radau'

    if 't_eval' in kwargs.keys():
        t_eval = kwargs['t_eval']
        del kwargs['t_eval']
    else:
        t_eval = None

    def sub_solve_ivp(sub_fun, sub_t_span, sub_y0, **kwargs):
        # prevent the solver from overreaching (scipy bug)
        if 'first_step' not in kwargs.keys():
            t_min, t_max = sub_t_span
            kwargs['first_step'] = (t_max-t_min)/2 if t_max != t_min else None

        sol_obj = solve_ivp(
            fun=sub_fun,
            t_span=sub_t_span,
            y0=sub_y0,
            **kwargs
        )

        if not sol_obj.success:
            msg = "ODE solver '{}' failed with ".format(kwargs['method'])
            msg += "status {} and ".format(sol_obj.status)
            msg += "message '{}'".format(sol_obj.message)
            raise(ValueError(msg))

        return sol_obj

    if len(disc_times) == 0:
        return sub_solve_ivp(
            rhss[0],
            t_span,
            y0,
            t_eval=t_eval,
            **kwargs
        )

    else:
        solns = dict()
        soln_times = dict()
        sol_funcs = dict()
        y0_i = y0.copy()

        sub_t_spans = get_sub_t_spans(t_span, disc_times)
        for i, sub_t_span in enumerate(sub_t_spans):
            if len(sub_t_span) > 0:
                sol_obj_i = sub_solve_ivp(
                    rhss[i],
                    sub_t_span,
                    y0_i,
                    **kwargs
                )
                ys_i = sol_obj_i.y
                ts_i = sol_obj_i.t
                y0_i = ys_i[:, -1]

                solns[i] = ys_i
                soln_times[i] = ts_i

                sol_funcs[i] = sol_obj_i.sol

        # build return object

        boundaries = np.unique(
            np.array(
                [-np.inf] + list(disc_times) + [np.inf]
            )
        )

        def sol(times):
            def index(t):
                i = np.where(boundaries <= t)[0][-1]
                return min(i, len(boundaries)-2)

            def sol_func(t):
                i = index(t)
                return sol_funcs[i](t)

            sol_func_v = np.vectorize(sol_func, signature='()->(n)')
            return sol_func_v(times).transpose()

        if t_eval is not None:
            t = t_eval
        else:
            # inds = range(len(soln_times.keys()))
            # L=[soln_times[i][:-1] for i in inds[:-1]]+[soln_times[inds[-1]]]

            L = [val for key, val in soln_times.items()]
            t = np.unique(np.concatenate(L))

        return myOdeResult(sol(t), t, sol)


###############################################################################


class myOdeResult(OdeResult):
    def __init__(self, y, t, sol):
        self.y = y
        self.t = t
        self.sol = sol
