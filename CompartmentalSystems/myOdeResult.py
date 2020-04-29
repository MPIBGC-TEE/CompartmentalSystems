from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
import numpy as np
from functools import lru_cache

def custom_solve_ivp(deriv_l, t_span, y0, **kwargs):
    ## disc_times should only be used when there are
    ## real discontinuities
    ## it seems not to be working too accurately

    if 'dense_output' not in kwargs.keys():
        kwargs['dense_output'] = False

    dense_output = kwargs['dense_output']

    if 'method' not in kwargs.keys():
        kwargs['method'] = 'Radau'
    
    if 'disc_times' in kwargs.keys():
        disc_times = kwargs['disc_times']
        del kwargs['disc_times']

        assert 'deriv_r' in kwargs
        deriv_r = kwargs['deriv_r']
        del kwargs['deriv_r']
    else:
        disc_times = None


    def sub_solve_ivp(sub_fun, t_span, y0_sub, **kwargs):   
        print(t_span)
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


    if disc_times is None:
        return sub_solve_ivp(deriv_l, t_span, y0, **kwargs)

    else:
        #order = np.argsort(t_eval)
        boundaries = np.unique(np.array([t_span[0]]+list(disc_times)+[t_span[1]]))

        #soln = np.nan * np.zeros(y0.shape + (len(disc_times),))
        #soln[:,0] = y0

        solns = dict()
        soln_times = dict()
        if dense_output:
            sol_funcs = dict()
        else:
            sol_funcs = None
        #for i, t_span in enumerate(zip(disc_times[:-1], disc_times[1:])):
        y0_i_l = y0.copy()
        for i, sub_t_span in enumerate(zip(boundaries[:-1], boundaries[1:])):
            # eventuell t_evals zurecht schneiden

            t_span_l = (sub_t_span[0], sub_t_span[0]+(sub_t_span[1]-sub_t_span[0])/2)
            t_span_r = (t_span_l[1], sub_t_span[1])

            #def fun(t, x):
            #    if t == t_span[0]:
            #        return mrs[i](x)

            sol_obj_i_l = sub_solve_ivp(deriv_r, t_span_l, y0_i_l, **kwargs)
            ys_i_l = sol_obj_i_l.y 
            ts_i_l = sol_obj_i_l.t

            y0_i_r = ys_i_l[:,-1]
            sol_obj_i_r = sub_solve_ivp(deriv_l, t_span_r, y0_i_r, **kwargs)
            ys_i_r = sol_obj_i_r.y 
            ts_i_r = sol_obj_i_r.t
            
            ys_i = np.concatenate(
                (ys_i_l[:,:-1], ys_i_r),
                axis = -1
            )
            y0_i_l = ys_i[:,-1]
            ts_i = np.concatenate((ts_i_l[:-1], ts_i_r))

            solns[i] = ys_i
            soln_times[i] = ts_i

            ############# make a wrapper maker ###########
            raise()
            if dense_output:
                def wrapper(t):
                    if t <= t_span_l[1]:
                        return sol_obj_i_l.sol(t)
                    else:
                        return sol_obj_i_r.sol(t)
            
                sol_funcs[i] = wrapper

        return myOdeResult(solns, soln_times, sol_funcs, disc_times)

        
class myOdeResult(OdeResult):
    def __init__(self, solns, soln_times, sol_funcs, disc_times):
        inds = range(len(soln_times.keys()))
        l = [soln_times[i][:-1] for i in inds[:-1]] + [soln_times[inds[-1]]]
        print('WEG')
        l_weg = [soln_times[i][-1] for i in inds[:-1]]
        print(l_weg)
        self.t = np.concatenate(l)

        l_weg = [solns[i][:,-1] for i in inds[:-1]]
        print(l_weg)
        l = [solns[i][:,:-1] for i in inds[:-1]] + [solns[inds[-1]]]
        self.y = np.concatenate(l,axis=-1) 

        self.sol = sol_funcs

        if sol_funcs is not None:
            def sol(self):
                def index(t):
                    i = np.where((disc_times<=t) & (t<=disc_times[-1]))[0][-1]
                    return min(i, len(disc_times)-2)
        
                def sol_func(t):
                    i = index(t)
                    return sol_funcs[i](t)
        
                sol_func_v = np.vectorize(sol_func, signature='()->(n)')
        
                sol = lambda t: sol_func_v(t).transpose()
            self.sol = sol


