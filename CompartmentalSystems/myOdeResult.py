from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
import numpy as np
from functools import lru_cache
from collections.abc import Iterable

def solve_ivp_pwc(rhss, t_span, y0, disc_times=(), **kwargs):
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


    def sub_solve_ivp(sub_fun, t_span, y0_sub, **kwargs):   
        # prevent the solver from overreaching (scipy bug)
        if 'first_step' not in kwargs.keys():
            t_min, t_max = t_span
            kwargs['first_step'] = (t_max-t_min)/2 if t_max != t_min else None

        sol_obj = solve_ivp(
             fun=sub_fun
            ,t_span=t_span
            ,y0=y0_sub
            ,**kwargs
        )

        if not sol_obj.success:
            msg = "ODE solver '{}' failed with ".format(kwargs['method'])
            msg += "status {} and ".format(sol_obj.status)
            msg += "message '{}'".format(sol_obj.message)
            raise(ValueError(msg))

        return sol_obj

    if len(disc_times) == 0 :
        return sub_solve_ivp(
            rhss[0],
            t_span,
            y0,
            t_eval = t_eval,
            **kwargs
        )

    else:
        boundaries = np.unique(np.array([t_span[0]]+list(disc_times)+[t_span[1]]))

        solns = dict()
        soln_times = dict()
        sol_funcs = dict()
        y0_i = y0.copy()
        for i, sub_t_span in enumerate(zip(boundaries[:-1], boundaries[1:])):
            sol_obj_i = sub_solve_ivp(
                rhss[i],
                sub_t_span,
                y0_i,
                **kwargs
            )
            ys_i = sol_obj_i.y
            ts_i = sol_obj_i.t
            y0_i = ys_i[:,-1]

            solns[i] = ys_i
            soln_times[i] = ts_i

            sol_funcs[i] = sol_obj_i.sol

        # build return object

        def sol(times):
            def index(t):
                i = np.where((boundaries<=t) & (t<=boundaries[-1]))[0][-1]
                return min(i, len(boundaries)-2)
        
            def sol_func(t):
                i = index(t)
                return sol_funcs[i](t)
        
            sol_func_v = np.vectorize(sol_func, signature='()->(n)')
            return sol_func_v(times).transpose()

        if t_eval is not None:
            t = t_eval
        else:
            inds = range(len(soln_times.keys()))
            l = [soln_times[i][:-1] for i in inds[:-1]] + [soln_times[inds[-1]]]
            t = np.concatenate(l)
        
        return myOdeResult(sol(t), t, sol)

        
################################################################################

        
class myOdeResult(OdeResult):
    def __init__(self, y, t, sol):
        self.y = y
        self.t = t
        self.sol = sol





