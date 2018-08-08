# vim:set ff=unix expandtab ts=4 sw=4:
from __future__ import division

import numpy as np 
from scipy.integrate import odeint
from scipy.interpolate import lagrange
from scipy.optimize import brentq
from scipy.stats import norm
from string import Template
from sympy import flatten, gcd, lambdify, DiracDelta, solve
from sympy.polys.polyerrors import PolynomialError

def pp(strng,env,comment=""):
    print(comment+"\n"+strng+"=:")
    print(env[strng])

def flux_dict_string(d,indent=0):
    s=""
    for k,val in d.items():
        s+=' '*indent+str(k)+": "+str(val)+"\n"

    return s


#fixme: test
def has_pw(expr):
    if expr.is_Matrix:
        for c in list(expr):
            if has_pw(c):
                return True
        return False

    if expr.is_Piecewise:
        return True

    for a in expr.args:
        if has_pw(a):
            return True
    return False


def is_DiracDelta(expr):
    """Check if expr is a Dirac delta function."""
    if len(expr.args) != 1: 
        return False

    arg = expr.args[0]
    return DiracDelta(arg) == expr


def parse_input_function(u_i, time_symbol):
    """Return an ordered list of jumps in the input function u.

    Args:
        u (SymPy expression): input function in :math:`\\dot{x} = B\\,x + u`

    Returns:
        ascending list of jumps in u
    """
    impulse_times = []
    pieces = []

    def rek(expr, imp_t, p):
        if hasattr(expr, 'args'):
            for arg in expr.args:
                if is_DiracDelta(arg):
                    dirac_arg = arg.args[0]
                    zeros = solve(dirac_arg)
                    imp_t += zeros
    
                if arg.is_Piecewise:
                    for pw_arg in arg.args:
                        cond = pw_arg[1]
                        if cond != True:
                            atoms = cond.args
                            zeros = solve(atoms[0] - atoms[1])
                            p += zeros
                
                rek(arg, imp_t, p)

    rek(u_i, impulse_times, pieces)

    impulses = []
    impulse_times = sorted(impulse_times)
    for impulse_time in impulse_times:
        intensity = u_i.coeff(DiracDelta(impulse_time-time_symbol))
        impulses.append({'time': impulse_time, 'intensity': intensity})

    jump_times = sorted(pieces + impulse_times)
    return (impulses, jump_times)


def factor_out_from_matrix(M):
    if has_pw(M):
        return(1)

    try:
        return gcd(list(M))
    except(PolynomialError):
        #print('no factoring out possible')
        #fixme: does not work if a function of X, t is in the expressios,
        # we could make it work...if we really wanted to
        return 1


def numerical_rhs(state_vector, time_symbol, rhs, 
        parameter_set, func_set, times):
    rhs_par = rhs.subs(parameter_set)

    # first check if the rhs is defined piecewise since lambdify does not work
    if not has_pw(rhs):
    #if False:
        #https://www.python.org/dev/peps/pep-0008/ 
        # we have an expression for the derivative
        # but the ode solver wants a function 
        # operating on lists.
        # We proceed in steps to construct this function:
        # 1.)  Create a Matrix valued function from the Matrix valued expression
        #  Fpar
        #      which we can do  automatically with sympys lambdify function
        #      a) assemble tuple for lambdify 
        tup = tuple(state_vector) + (time_symbol,)
        #     b) use lambdify

        # cut off the parentheses from the keys, because lamdify wants it
        #print('fs', func_set)
        cut_func_set = {key[:key.index('(')]: val 
                            for key, val in func_set.items()}
        #print('cfs', cut_func_set)
        #print('rhs_par', [(a, type(a)) for a in rhs_par.atoms()])
        #print('rhs_par', rhs_par)
        #FL = lambdify(tup, rhs_par, modules=[cut_func_set,"numpy"])
        
        #FL = lambdify(tup, rhs_par, modules=[cut_func_set, TRANSLATIONS])
        FL = lambdify(tup, rhs_par, modules=[cut_func_set, 'numpy'])
        
        # 2.) Write a wrapper that transformes Matrices to lists 
        # (or numpy.ndarrays)
        # 
        def num_rhs(X,t):
            # the ode solver delivers X as numpy.ndarray 
            # however, our FL requires a tuple of arguments
            Xt = tuple(X) + (t,)
            #print('Xt', Xt)
            #cut_func_set
            #print('num_rhs', tup, Xt)
            Fval = FL(*Xt)
            #print(Fval)
            #pp("Fval",locals())
            return flatten(Fval.tolist())
    
    else:
        def funcmaker(expr, state_vector, time_symbol):
            # parse out the signatures of involved functions
            name_tup = (tuple([sv.name for sv in state_vector]) + 
                            (time_symbol.name,))
            signature_indices = {}
            for key, func in func_set.items():
                # find the signature of func
                pars = [s.strip() 
                    for s in key[key.index('(')+1:key.index(')')].split(',')]
                signature_indices[key] = np.array([s in pars for s in name_tup])

            #print('expr', expr)
            #print('fs', func_set)
            #print(signature_indices)

            def f(X,t):
                Xt = np.array(tuple(X) + (t,))
                #create the dictionary for substitute
                edict = {sv: X[i] for i, sv in enumerate(state_vector)}
                edict.update({time_symbol: t})

                # evaluate the functions in func_set at X,t
                func_vals = {}
                for key, func in func_set.items():
                    Y = Xt[signature_indices[key]]
                    #print(key, Xt, Y, func)
                    ft = func(*Y)
                    func_vals[key] = np.float(ft) 
               
                #substitute
                eval_expr = expr.subs(func_vals)
                eval_expr = eval_expr.subs(edict)
                #transform the resulting matrix to a list
                #return(list(eval_expr))
                return np.array(list(eval_expr), dtype='float64')

            return f
        
        num_rhs = funcmaker(rhs_par, state_vector, time_symbol) 

    def bounded_num_rhs(X,t):
        # fixme 1:
        # maybe odeint (or another integrator) 
        # can be told >>not<< to look outside
        # the interval 

        # fixme 2:
        # actually the times vector is not the smallest
        # possible allowed set but the intersection of
        # all the intervals where the 
        # time dependent functions are defined
        # this should be tested in init
        t_max = times[-1]

        #fixme: we should die hard here, because now we think we can compute the
        # state transition operator till any time in the future,
        # but it is actually biased by the fact, that we use the last value over
        # and over again
        # and hence assume some "constant" future
        if t > t_max:
            res = num_rhs(X, t_max)
        else:
            res = num_rhs(X, t)

        #print('brhs', 't', t, 'X', X, 'res', res)
        #print('t', t)
        return res

    return bounded_num_rhs


def numsol_symbolic_system(
        state_vector, 
        time_symbol, 
        rhs, 
        parameter_set, 
        func_set, 
        start_values, 
        times
    ):

    nr_pools = len(state_vector)
    
    if times[0] == times[-1]: return start_values.reshape((1, nr_pools))

    num_rhs = numerical_rhs(
        state_vector,
        time_symbol,
        rhs, 
        parameter_set,
        func_set,
        times
    )

    return odeint(num_rhs, start_values, times, mxstep=10000)


def arrange_subplots(n):
    if n <=3:
        rows = 1
        cols = n
    if n == 4 :
        rows = 2
        cols = 2
    if n >= 5:
        rows = n // 3
        if n % 3 != 0:
            rows += 1
        cols = 3

    return (rows, cols)


def melt(ndarr, identifiers = None):
    shape = ndarr.shape

    if identifiers == None:
        identifiers =  [range(shape[dim]) for dim in range(len(shape))]

    def rek(struct, ids, melted_list, dim):
        if type(struct) != np.ndarray:
            melted_list.append(ids + [struct])
        else:
            shape = struct.shape
            for k in range(shape[0]):
                rek(struct[k], ids + [identifiers[dim][k]], melted_list, dim+1)

    melted_list = []
    rek(ndarr, [], melted_list, 0)
    rows = len(melted_list)
    cols = len(melted_list[0])
    melted = np.array(melted_list).reshape((rows, cols))
    
    return melted


#fixme: test
# compute inverse of CDF at u for quantiles or generation of random variables
def generalized_inverse_CDF(CDF, u, start_dist = 1e-4, tol = 1e-8):
    #print('u', u)
    #f = lambda a: u - CDF(a)
    def f(a):
        res = u-CDF(a)
        #print('gi', a, res)
        return res

    x1 = start_dist
 
    # go so far to the right such that CDF(x1) > u, the bisect in 
    # interval [0, x1]
    y1 = f(x1)
    while y1 >= 0:
        x1 = x1*2 + 0.1
        y1 = f(x1)
    
    if np.isnan(y1):
        res = np.nan
    else:
        #print('calling brentq on [0,', x1, ']')
        res =  brentq(f, 0, x1, xtol=tol)
    #if f(res) > tol: res = np.nan
    #print('gi_res', res)
    #print('finished', method_f.__name__, 'on [0,', x1, ']')
    
    return res


# draw a random variable with given CDF
def draw_rv(CDF):
    return generalized_inverse_CDF(CDF, np.random.uniform())


# return function g, such that g(normally distributed sv) is distributed 
# according to CDF
def stochastic_collocation_transform(M, CDF):
    # collocation points for normal distribution, 
    # taken from Table 10 in Appendix 3 of Grzelak2015SSRN
    cc_data = { 2: [1],
                3: [0.0, 1.7321],
                4: [0.7420, 2.3344],
                5: [0.0, 1.3556, 2.8570],
                6: [0.6167, 1.8892, 3.3243],
                7: [0.0, 1.1544, 2.3668, 3.7504],
                8: [0.5391, 1.6365, 2.8025, 4.1445],
                9: [0.0, 1.0233, 2.0768, 3.2054, 4.5127],
               10: [0.4849, 1.4660, 2.8463, 3.5818, 4.8595],
               11: [0.0, 0.9289, 1.8760, 2.8651, 3.9362, 5.1880]}


    if not M in cc_data.keys(): return None
    cc_points = [-x for x in reversed(cc_data[M]) if x != 0.0] + cc_data[M]
    cc_points = np.array(cc_points)
    #print('start computing collocation transform')
    ys = np.array([generalized_inverse_CDF(CDF, norm.cdf(x)) 
                    for x in cc_points])
    #print('ys', ys)
    #print('finished computing collocation transform')

    return lagrange(cc_points, ys)


# Metropolis-Hastings sampling for PDFs with nonnegative support
# no thinning, no burn-in period
def MH_sampling(N, PDF, start = 1.0):
    xvec = np.ndarray((N,))
    x = start
    PDF_x = PDF(x)
    norm_cdf_x = norm.cdf(x)
   
    for i in range(N):
        xs = -1.0
        while xs <= 0:
            xs = x + np.random.normal()

        PDF_xs = PDF(xs)
        A1 = PDF_xs/PDF_x
        norm_cdf_xs = norm.cdf(xs)
        A2 = norm_cdf_x/norm_cdf_xs
        A = A1 * A2

        if np.random.uniform() < A: 
            x = xs
            PDF_x = PDF_xs
            norm_cdf_x = norm_cdf_xs
    
        xvec[i] = x
  
    return xvec


def save_csv(filename, melted, header):
    np.savetxt(filename, melted, header = header, 
                    delimiter=',', fmt="%10.8f", comments='')


def load_csv(filename):
    #return np.loadtxt(filename, skiprows=1, delimiter=',', comments='')
    return np.loadtxt(filename, skiprows=1, delimiter=',')
    

def tup2str(tup):
    # uses for stoichiometric models
    string=Template("${f}_${s}").substitute(f=tup[0], s=tup[1])
    return(string)


# use only every (k_1,k_2,...,k_n)th element of the n-dimensional numpy array 
# data,
# strides is a list of k_j of length n
# always inlcude first and last elements
def stride(data, strides):
    if isinstance(strides, int):
        strides = [strides]

    index_list = []
    for dim in range(data.ndim):
        n = data.shape[dim]
        stride = strides[dim]
        ind = np.arange(0, n, stride).tolist()
        if (n-1) % stride != 0:
            ind.append(n-1)

        index_list.append(ind)

    return data[np.ix_(*index_list)]

def is_compartmental(M):    
    gen=range(M.shape[0])
    return all([M.is_square,all([M[i,i]<=0 for i in gen]), all([sum(M[i,:])<=0 for i in gen])])    
    


def start_age_moments_from_empty_spin_up(srm,parameter_set,func_set,a_max,max_order):
    # to do:
    # run a spin up and observe the age distribution at the end
    # then compute the moments numerically
    raise(Exception("Not implemented yet")

def start_age_moments_from_steady_state(srm,parameter_set,func_set,max_order):
    """
    The age distribution at the start :math`t_0` is NOT 
    defined by the constituents of objects of class SmoothModelRun. 
    In fact EVERY age distribution can be chosen. 
    The implemented algorithms will correcly project it 
    to any time :math`t`.
    The distribution eaisiest to imagine is the one with all pools empty.
    However often one is interested in the CHANGE of an age distribution that has been ESTABLISHED over a long period of time.        
    There are several ways to generate such an established
    distribution.

    1 ) Finite age spin up from empty pools ( see function start_age_moments_from_empty_spin_up)
    Start the computation with all pools empty at time 
    :math `t_0` and run it till time :math `t = a_{max}` where :math `a_max` is the oldest age you care about about in your interpretation of the results. 
    The resulting distribution will be non zero only in the interval :math `[0,a_max]`. This has to be considere regarding the questions that can be asked. 
    It would for instance not make sense to mistake the fact that the percentage of material older than :math `a_max` will increase over time as a property of the system, where it is actually a property of the (spun up) start distribution.  
    
    2 ) find a steady state  of the autonumuous system
    If you take the term ESTABLISHED to the limit of infinity you can look for a related system that has persisted unchanged for all  times :math t<t_0 and start with the age distribution created by this system.
      1.) Transform the general nonlinear non-autonomous system into a nonlinear autonomous system by freezing it at time :math `t=t_0`: 
    Compute :math `u_0(x)=u(t_0,x_0)` and :math `B_0(x)=B(t_0,x_0)` 
    Numerically look for an equilibrium :math `x*` of the nonlinear system :math `0=B_0(x*)+u_0(x*).
    :math `x*` is the compatible startvalue for the following simulation. 
    A special case occures for linear systems where B(x,t)=B(t)
    and u(x,t) = u(t)
    We can compute the :math `x*` in one step: `x*=B_0**-1 u_0`.
    

    Args:
        srm (SmoothReservoirModel) : The (symbolic) model
        par_set : The parameter set that transforms the symbolic model into a numeric one. 
        func_set : The set of (numeric) fucntion that replace the symbolic function in srm. 
        max_order (int): The highest order up to which moments are
            to be computed.

    Returns:
        numpy.ndarray: moments x pools, containing the moments of the
            pool ages in equilibrium.
    """
    
    #check for linearity 
    if srm.is_linear:
        # fixme mm 8.8.2018
        # this linear case should actually be handled by LAPM and just call a function from there.
        times = self.times
        B0 = self.B(times[0])
        # old: x0 = self.solve()[0]
        # fixed mm 8.8.2018 x0 is not free to choose
        # see helptext above: 
        u0=self.u(times[0])
        x0=inv(B0)*u0
        X0 = x0 * np.identity(len(x0))
        start_age_moments = []
        for n in range(1, max_order+1):
            start_age_moment = (-1)**n * factorial(n) \
                            * inv(X0) @ matrix_power(inv(B0), n) @ x0
                            # fixed mm 8.8.2018: 
                            #* pinv(X0) @ matrix_power(pinv(B0), n) @ x0
                            # I think that pinv (pseudo inverse) is not justified here since 
                            # the computation relies on the same assumptions as LAPMs steady state
                            # formulas.
                            # If a singular matrix B_0 occurs as B(x*,t0)
                            # consequence of the solution of the steady state x*            
                            # Then the system would have traps. A steady state could then only occour
                            # if the components of u0=u(x*,t0) would be zero for the pools connected
                            # to the trap. In this (unlikely) event the startage distribution would
                            # be ambigous because x* is not uniqe (The content of the trap is a free 
                            # parameter and so is its age.)
                            # In this case we should at least issue  a warning 
                            
            start_age_moments.append(start_age_moment)

        return np.array(start_age_moments)
    else:
        raise Exception("""
        At the moment the algorithm assumes an equilibrium
        at t_0 with B_0=B(t_0) but starts at arbitrarz startvalues.
        Actually it could start at the equilibrium point of the 
        nonlinear autonomous system. 
        Until this is actually implemented  the tests should fail 
        at least for all nonlinear models.
         
        """
        # to do:
        # try to find an equilibrium
        # and implement 2a) or fail
