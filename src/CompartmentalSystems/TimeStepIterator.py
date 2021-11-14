
import numpy as np
import scipy
from copy import copy
##############################################################################

class TimeStep:
    """
    Just a collection of values per time step which are necessarry
    to compute the next one.
    """
    def __init__(
        self,
        B,
        u,
        x,
        t
    ):
        self.B = B
        self.u = u
        self.x = x
        self.t = t

    def __repr__(self):
        return str(self.B) + str(self.u) + str(self.x) + str(self.t)

class TimeStepIterator:
    """iterator for looping over the results of a difference equation"""

    def __init__(
        self,
        initial_ts,
        B_func,
        u_func,
        number_of_steps,
        delta_t
    ):
        self.initial_ts = initial_ts
        self.B_func = B_func
        self.u_func = u_func
        self.number_of_steps = number_of_steps
        self.delta_t = delta_t
        self.reset()

    def reset(self):
        self.i = 0
        self.ts = self.initial_ts

    def __iter__(self):
        self.reset()
        return(self)

    def __next__(self):
        if self.i == self.number_of_steps:
            raise StopIteration
        ts = copy(self.ts)
        # fixme mm 7-20-2021
        # possibly B and u one index lower than x ...
        B =ts.B
        u =ts.u
        x =ts.x
        t = ts.t

        it = self.i
        B_func = self.B_func
        u_func = self.u_func
        delta_t = self.delta_t
        # compute x_i+1
        B_new = B_func(it,ts.x)
        u_new = u_func(it,ts.x)
        x_new = x + u + np.matmul(B,x)
        t_new = t + delta_t
        self.ts = TimeStep(B_new, u_new, x_new, t_new)
        self.i += 1
        return ts

class ImplicitTimeStepIterator:
    """iterator for looping over the results of a difference equation"""

    def __init__(
        self,
        initial_ts,
        B_func, # normal B wihtout 1 
        u_func,
        number_of_steps,
        delta_t
    ):
        self.initial_ts = initial_ts
        self.B_func = B_func
        self.u_func = u_func
        self.number_of_steps = number_of_steps
        self.delta_t = delta_t
        self.reset()

    def reset(self):
        self.i = 0
        self.ts = self.initial_ts

    def __iter__(self):
        self.reset()
        return(self)

    def __next__(self):
        if self.i == self.number_of_steps:
            raise StopIteration
        ts = copy(self.ts)
        # fixme mm 7-20-2021
        # possibly B and u one index lower than x ...
        B =ts.B
        u =ts.u
        x =ts.x.reshape(u.shape)
        t = ts.t

        it = self.i
        B_func = self.B_func
        u_func = self.u_func
        delta_t = self.delta_t
        # build a function for the righthandside
        # of the ode
        def rhs(i,X): 
            return (np.matmul(B_func(i,X),X) + u_func(i,X))
        
        # compute x_i+1 satisfying 
        # x_{i+1} = x_{i} + delta_t * rhs(x_{i+1},i+1)
        # 0 = x_{i} + delta_t * rhs(x_{i+1},i+1)-x_{i+1}

        def mini(x_new,i,x):
            return x + delta_t * rhs(i+1,x_new)-x_new
        
        x_start= u + np.matmul(B,x) #use euler forward to make a guess for the new x
        # use euler backwards 
        x_new  = scipy.optimize.newton(
            func=mini,
            x0=x_start,
            fprime=None,
            args=(it,x),
            tol=1.48e-08,
            maxiter=100,
            fprime2=None,
            x1=None,
            rtol=0.0,
            full_output=False,
            disp=True
        )
        #from IPython import embed;embed()
        B_new = B_func(it,ts.x)
        u_new = u_func(it,ts.x)
        # 
        t_new = t + delta_t
        self.ts = TimeStep(B_new, u_new, x_new, t_new)
        self.i += 1
        return ts

class TimeStepIterator2():
    """iterator for looping forward over the results of a difference equation
    X_{i+1}=f(X_{i},i)"""

    def __init__(
        self,
        initial_values, # a tupel of values that will be
        f, # the function to compute the next ts
        max_it=False
    ):
        self.initial_values = initial_values
        self.f= f
        self.reset()
        self.max_it = max_it

    def reset(self):
        self.i = 0
        self.ts = self.initial_values

    def __iter__(self):
        self.reset()
        return(self)

    def __next__(self):
        if self.max_it:
            if self.i == self.max_it:
                raise StopIteration

        ts = copy(self.ts)
        ts_new = self.f(self.i, ts)
        self.ts = ts_new
        self.i += 1
        return ts

