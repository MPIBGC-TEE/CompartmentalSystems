# vim:set ff=unix expandtab ts=4 sw=4:
from typing import Callable, Tuple, Sequence
from functools import lru_cache, _CacheInfo, _lru_cache_wrapper
import numpy as np
import matplotlib.pyplot as plt
import inspect
from collections import namedtuple
from numbers import Number
from scipy.integrate import odeint, quad
from scipy.interpolate import lagrange
from scipy.optimize import brentq
from scipy.stats import norm
from string import Template
from sympy import gcd, lambdify, DiracDelta, solve, Matrix, diff
from sympy.polys.polyerrors import PolynomialError
from sympy.core.function import UndefinedFunction, Function, sympify
from sympy import Symbol
from .BlockOde import BlockOde
from .myOdeResult import solve_ivp_pwc


ALPHA_14C = 1.18e-12
DECAY_RATE_14C_DAILY = 0.0001209681


def warning(txt):
    print('############################################')
    calling_frame = inspect.getouterframes(inspect.currentframe(), 2)
    func_name = calling_frame[1][3]
    print("Warning in function {0}:".format(func_name))
    print(txt)


def deprecation_warning(txt):
    print('############################################')
    calling_frame = inspect.getouterframes(inspect.currentframe(), 2)
    func_name = calling_frame[1][3]
    print("The function {0} is deprecated".format(func_name))
    print(txt)


def flux_dict_string(d, indent=0):
    s = ""
    for k, val in d.items():
        s += ' '*indent+str(k)+": "+str(val)+"\n"

    return s


def func_subs(t, Func_expr, func, t0):
    """
    returns the function part_func
    where part_func(_,_,...) =func(_,t=t0,_..) (func partially applied to t0)
    The position of argument t in the argument list is found
    by examining the Func_expression argument.
    Args:
        t (sympy.symbol): the symbol to be replaced by t0
        t0 (value)      : the value to which the function will be applied
        func (function) : a python function
        Func_exprs (sympy.Function) : An expression for an undefined Function

    """
    assert(isinstance(type(Func_expr), UndefinedFunction))
    pos = Func_expr.args.index(t)

    def frozen(*args):
        # tuples are immutable
        L = list(args)
        L.insert(pos, t0)
        new_args = tuple(L)
        return func(*new_args)
    return frozen


def jacobian(vec, state_vec):
    dim1 = vec.rows
    dim2 = state_vec.rows
    return Matrix(dim1, dim2, lambda i, j: diff(vec[i], state_vec[j]))


# fixme: test
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
                        # 'if not cond' led to strange behavior
                        if cond != True:  # noqa: E712
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
        # print('no factoring out possible')
        # fixme: does not work if a function of X, t is in the expressios,
        # we could make it work...if we really wanted to
        return 1


def numerical_function_from_expression(expr, tup, parameter_dict, func_set):
    # the function returns a function that given numeric arguments
    # returns a numeric result.
    # This is a more specific requirement than a function returned by lambdify
    # which can still return symbolic
    # results if the tuple argument to lambdify does not contain all free
    # symbols of the lambdified expression.
    # To avoid this case here we check this.
    expr_par = expr.subs(parameter_dict)
    ss_expr = expr_par.free_symbols
    ss_tup = set([s for s in tup])

    if not(ss_expr.issubset(ss_tup)):
        raise Exception(
            """The free symbols of the expression: ${0} are not a subset of
               the symbols in the tuple argument:${1}
            """.format(ss_expr, ss_tup))

    cut_func_set = make_cut_func_set(func_set)
#    expr_par=expr.subs(parameter_dict)
    expr_func = lambdify(tup, expr_par, modules=[cut_func_set, 'numpy'])
    def expr_func_safe_0_over_0(*val):
        with np.errstate(invalid='raise'):
            try:
                res = expr_func(*val)
            except FloatingPointError as e:
                if e.args[0] == 'invalid value encountered in double_scalars':
                    with np.errstate(invalid='ignore'):
                        res = expr_func(*val)
                        res = np.nan_to_num(res, copy=False)
        return res

    return expr_func_safe_0_over_0


def numerical_rhs(
    state_vector,
    time_symbol,
    rhs,
    parameter_dict,
    func_dict
):

    FL = numerical_function_from_expression(
        rhs,
        (time_symbol,)+tuple(state_vector),
        parameter_dict,
        func_dict
    )

    # 2.) Write a wrapper that transformes Matrices to numpy.ndarrays and
    # accepts array instead of the separate arguments for the states)
    def num_rhs(t, X):
        # we need the arguments to be numpy arrays to be able to catch 0/0
        # Holger: I made 0/0 being caught by a subfunction in 
        # numerical_function_from_expression
#        Y = (np.array([x]) for x in X) # Markus' version, lead to 
#       deprecation warnings, so I rewrote it to (Holger):
        Y = np.array([x for x in X]) # 

        Fval = FL(t, *Y)
        return Fval.reshape(X.shape,)

    return num_rhs


def numerical_rhs_old(
    state_vector,
    time_symbol,
    rhs,
    parameter_dict,
    func_set,
    times
):

    FL = numerical_function_from_expression(
        rhs,
        tuple(state_vector) + (time_symbol,),
        parameter_dict,
        func_set
    )

    # 2.) Write a wrapper that transformes Matrices numpy.ndarrays and accepts
    # array instead of the separate arguments for the states)
    def num_rhs(X, t):
        Fval = FL(*X, t)
        return Fval.reshape(X.shape,)

    def bounded_num_rhs(X, t):
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

        # fixme: we should die hard here, because now we think we can compute
        # the state transition operator till any time in the future,
        # but it is actually biased by the fact, that we use the last value
        # over and over again
        # and hence assume some "constant" future
        if t > t_max:
            res = num_rhs(X, t_max)
        else:
            res = num_rhs(X, t)

#        print('brhs', 't', t, 'X', X, 'res', res)
#        print('t', t)
        return res

    return bounded_num_rhs


def numsol_symbolic_system_old(
    state_vector,
    time_symbol,
    rhs,
    parameter_dict,
    func_set,
    start_values,
    times
):

    nr_pools = len(state_vector)

    if times[0] == times[-1]:
        return start_values.reshape((1, nr_pools))

    num_rhs = numerical_rhs_old(
        state_vector,
        time_symbol,
        rhs,
        parameter_dict,
        func_set,
        times
    )
    return odeint(num_rhs, start_values, times, mxstep=10000)


def numsol_symbolical_system(
    state_vector,
    time_symbol,
    rhs,
    parameter_dicts,
    func_dicts,
    start_values,
    times,
    disc_times=()
):
    nr_pools = len(state_vector)
    t_min = times[0]
    t_max = times[-1]

    if times[0] == times[-1]:
        return start_values.reshape((1, nr_pools))

    num_rhss = tuple(
        numerical_rhs(
            state_vector,
            time_symbol,
            rhs,
            parameter_dict,
            func_dict
        )
        for parameter_dict, func_dict in zip(parameter_dicts, func_dicts)
    )

    res = solve_ivp_pwc(
        rhss=num_rhss,
        t_span=(t_min, t_max),
        y0=start_values,
        t_eval=tuple(times),
        disc_times=disc_times
    )

    # adapt to the old ode_int interface
    # since our code at the moment expects it
    values = np.rollaxis(res.y, -1, 0)
    return (values, res.sol)


def arrange_subplots(n):
    if n <= 3:
        rows = 1
        cols = n
    if n == 4:
        rows = 2
        cols = 2
    if n >= 5:
        rows = n // 3
        if n % 3 != 0:
            rows += 1
        cols = 3

    return (rows, cols)


def melt(ndarr, identifiers=None):
    shape = ndarr.shape

    if identifiers is None:
        identifiers = [range(shape[dim]) for dim in range(len(shape))]

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


# fixme: test
# compute inverse of CDF at u for quantiles or generation of random variables
def generalized_inverse_CDF(CDF, u, start_dist=1e-4, tol=1e-8):
    def f(a):
        return u-CDF(a)

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
        res = brentq(f, 0, x1, xtol=tol)
#    if f(res) > tol: res = np.nan
#    print('gi_res', res)
#    print('finished', method_f.__name__, 'on [0,', x1, ']')
    return res


# draw a random variable with given CDF
def draw_rv(CDF):
    return generalized_inverse_CDF(CDF, np.random.uniform())


# return function g, such that g(normally distributed sv) is distributed
# according to CDF
def stochastic_collocation_transform(M, CDF):
    # collocation points for normal distribution,
    # taken from Table 10 in Appendix 3 of Grzelak2015SSRN
    cc_data = {
         2: [1],
         3: [0.0, 1.7321],
         4: [0.7420, 2.3344],
         5: [0.0, 1.3556, 2.8570],
         6: [0.6167, 1.8892, 3.3243],
         7: [0.0, 1.1544, 2.3668, 3.7504],
         8: [0.5391, 1.6365, 2.8025, 4.1445],
         9: [0.0, 1.0233, 2.0768, 3.2054, 4.5127],
        10: [0.4849, 1.4660, 2.8463, 3.5818, 4.8595],  # noqa: E131
        11: [0.0, 0.9289, 1.8760, 2.8651, 3.9362, 5.1880]  # noqa: E131
    }

    if M not in cc_data.keys():
        return None
    cc_points = [-x for x in reversed(cc_data[M]) if x != 0.0] + cc_data[M]
    cc_points = np.array(cc_points)
#    print('start computing collocation transform')
    ys = np.array(
        [generalized_inverse_CDF(CDF, norm.cdf(x)) for x in cc_points]
    )
#    print('ys', ys)
#    print('finished computing collocation transform')

    return lagrange(cc_points, ys)


# Metropolis-Hastings sampling for PDFs with nonnegative support
# no thinning, no burn-in period
def MH_sampling(N, PDF, start=1.0):
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
    np.savetxt(
        filename,
        melted,
        header=header,
        delimiter=',',
        fmt="%10.8f",
        comments=''
    )


def load_csv(filename):
    return np.loadtxt(filename, skiprows=1, delimiter=',')


def tup2str(tup):
    # uses for stoichiometric models
    string = Template("${f}_${s}").substitute(f=tup[0], s=tup[1])
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
    gen = range(M.shape[0])
    return all(
        [
            M.is_square,
            all([M[j, j] <= 0 for j in gen]),
            all([sum(M[:, j]) <= 0 for j in gen])
        ]
    )


def make_cut_func_set(func_set):
    def unify_index(expr):
        # for the case Function('f'):f_numeric
        if isinstance(expr, UndefinedFunction):
            res = str(expr)
        # for the case {f(x, y): f_numeric} f(x, y)
        elif isinstance(expr, Symbol):
            res = str(expr)
        elif isinstance(expr, Function):
            res = str(type(expr))
        elif isinstance(expr, str):
            expr = sympify(expr)
            res = unify_index(expr)
        else:
            print(type(expr))
            raise(TypeError(
                """
                funcset indices should be indexed by instances of
                sympy.core.functions.UndefinedFunction
                """
            ))
        return res

    cut_func_set = {unify_index(key): val for key, val in func_set.items()}
    return cut_func_set


def f_of_t_maker(sol_funcs, ol):
    def ot(t):
        sv = [sol_funcs[i](t) for i in range(len(sol_funcs))]
        tup = tuple(sv) + (t,)
        res = ol(*tup)
        return res
    return ot


def const_of_t_maker(const):
    def const_arr_fun(possible_vec_arg):
        if isinstance(possible_vec_arg, Number):
            return const  # also a number
        else:
            return const*np.ones_like(possible_vec_arg)
    return const_arr_fun


def x_phi_ode(
    srm,
    parameter_dicts,
    func_dicts,
    x_block_name='x',
    phi_block_name='phi',
    disc_times=()
):
    nr_pools = srm.nr_pools

    sol_rhss = []
    for pd, fd in zip(parameter_dicts, func_dicts):
        sol_rhs = numerical_rhs(
            srm.state_vector,
            srm.time_symbol,
            srm.F,
            pd,
            fd
        )
        sol_rhss.append(sol_rhs)

    B_sym = srm.compartmental_matrix
    tup = (srm.time_symbol,) + tuple(srm.state_vector)

    B_funcs_non_lin = []
    for pd, fd in zip(parameter_dicts, func_dicts):
        B_func_non_lin = numerical_function_from_expression(
            B_sym,
            tup,
            pd,
            fd
        )
        B_funcs_non_lin.append(B_func_non_lin)

    def Phi_rhs_maker(B_func_non_lin):
        def Phi_rhs(t, x, Phi_2d):
            return np.matmul(B_func_non_lin(t, *x), Phi_2d)
        return Phi_rhs

    Phi_rhss = []
    for B_func_non_lin in B_funcs_non_lin:
        Phi_rhss.append(Phi_rhs_maker(B_func_non_lin))

    functionss = []
    for sol_rhs, Phi_rhs in zip(sol_rhss, Phi_rhss):
        functions = [
            (sol_rhs, [srm.time_symbol.name, x_block_name]),
            (Phi_rhs, [srm.time_symbol.name, x_block_name, phi_block_name])
        ]
        functionss.append(functions)

    return BlockOde(
        time_str=srm.time_symbol.name,
        block_names_and_shapes=[
            (x_block_name, (nr_pools,)),
            (phi_block_name, (nr_pools, nr_pools,))
        ],
        functionss=functionss,
        disc_times=disc_times
    )


def integrate_array_func_for_nested_boundaries(
    f: Callable[[float], np.ndarray],
    integrator: Callable[
        [
            Callable[[float], np.ndarray],
            float,
            float
        ],
        np.ndarray
    ],  # e.g. array_quad_result
    tuples: Sequence[Tuple[float, float]]
) -> Sequence[float]:
    # we assume that the first (a,b) tuple contains the second,
    # the second the third and so on from outside to inside
    def compute(f, tuples, results: Sequence[float]):
        (a_out, b_out), *tail = tuples
        if len(tail) == 0:
            # number=quad(f, a_out, b_out)[0]
            arr = integrator(f, a_out, b_out)
        else:
            (a_in, b_in) = tail[0]
            results = compute(f, tail, results)
            arr = (
                integrator(f, a_out, a_in)
                + results[0]
                + integrator(f, b_in, b_out)
            )

        results = [arr] + results
        return results

    return compute(f, tuples, [])


def array_quad_result(
    f: Callable[[float], np.ndarray],
    a: float,
    b: float,
    epsrel=1e-3,  # epsabs would be a dangerous default
    *args,
    **kwargs
) -> np.ndarray:
    # integrates a vectorvalued function of a single argument
    # we transform the result array of the function into a one dimensional
    # vector compute the result for every component
    # and reshape the result to the form of the integrand
    test = f(a)
    n = len(test.flatten())
    vec = np.array(
        [quad(
            lambda t:f(t).reshape(n,)[i],
            a,
            b,
            *args,
            epsrel=epsrel,
            **kwargs
        )[0] for i in range(n)]
    )
    return vec.reshape(test.shape)


def array_integration_by_ode(
    f: Callable[[float], np.ndarray],
    a: float,
    b: float,
    *args,
    **kwargs
) -> np.ndarray:
    # another integrator like array_quad_result
    test = f(a)
    n = len(test.flatten())

    def rhs(tau, X):
        # although we do not need X we have to provide a
        # righthandside s uitable for solve_ivp

        # avoid overshooting if the solver
        # tries to look where the integrand might not be defined
        if tau < a or tau > b:
            return 0
        else:
            return f(tau).flatten()

    ys = solve_ivp_pwc(
        rhss=(rhs,),
        y0=np.zeros(n),
        t_span=(a, b)
    ).y
    val = ys[:, -1].reshape(test.shape)
    return val


def array_integration_by_values(
    f: Callable[[float], np.ndarray],
    taus: np.ndarray,
    *args,
    **kwargs,
) -> np.ndarray:
    # only allow taus as vector
    assert(len(taus.shape) == 1)
    assert(len(taus) > 0)
    test = f(taus[0])
    # create a big 2 dimensional array suitable for trapz
    integrand_vals = np.stack([f(tau).flatten() for tau in taus], 1)
    vec = np.trapz(y=integrand_vals, x=taus)
    return vec.reshape(test.shape)


def x_phi_tmax(s, t_max, block_ode, x_s, x_block_name, phi_block_name):
    x_s = np.array(x_s)
    nr_pools = len(x_s)

    start_Phi_2d = np.identity(nr_pools)
    start_blocks = [
        (x_block_name, x_s),
        (phi_block_name, start_Phi_2d)
    ]
    blivp = block_ode.blockIvp(start_blocks)
    return blivp


def phi_tmax(s, t_max, block_ode, x_s, x_block_name, phi_block_name):
    blivp = x_phi_tmax(s, t_max, block_ode, x_s, x_block_name, phi_block_name)
    phi_func = blivp.block_solve_functions(t_span=(s, t_max))[phi_block_name]

    return phi_func


@lru_cache()
def x_tmax(s, t_max, block_ode, x_s, x_block_name, phi_block_name):
    blivp = x_phi_tmax(s, t_max, block_ode, x_s, x_block_name, phi_block_name)
    x_func = blivp.block_solve_functions(t_span=(s, t_max))[x_block_name]

    return x_func


_CacheStats = namedtuple(
    'CacheStats',
    ['hitss', 'missess', 'currsizes', 'hitting_ratios']
)


def custom_lru_cache_wrapper(maxsize=None, typed=False, stats=False):
    if stats:
        hitss = []
        missess = []
        currsizes = []
        hitting_ratios = []

    def decorating_function(user_function):
        func = _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo)

        def wrapper(*args, **kwargs):
            nonlocal stats, hitss, missess, currsizes, hitting_ratios

            result = func(*args, **kwargs)
            if stats:
                hitss.append(func.cache_info().hits)
                missess.append(func.cache_info().misses)
                currsizes.append(func.cache_info().currsize)
                hitting_ratios.append(
                    round(hitss[-1]/(hitss[-1]+missess[-1])*100.0, 2)
                )
            return result

        wrapper.cache_info = func.cache_info
        if stats:
            def cache_stats():
                nonlocal hitss, missess, currsizes
                return _CacheStats(hitss, missess, currsizes, hitting_ratios)

            wrapper.cache_stats = cache_stats

            def plot_hitss():
                nonlocal hitss
                plt.plot(range(len(hitss)), hitss)
                plt.title('Hitss')
                plt.show()

            wrapper.plot_hitss = plot_hitss

            def plot_hit_history():
                nonlocal hitss
                plt.scatter(
                    range(len(hitss)-1),
                    np.diff(hitss),
                    s=1,
                    alpha=0.2
                )
                plt.title('Hit history')
                plt.show()

            wrapper.plot_hit_history = plot_hit_history

            def plot_hitting_ratios():
                nonlocal hitss, hitting_ratios
                plt.plot(
                    range(len(hitss)),
                    hitting_ratios
                )
                plt.title('Hitting ratios')
                plt.show()

            wrapper.plot_hitting_ratios = plot_hitting_ratios

            def plot_currsizes():
                nonlocal currsizes
                plt.plot(
                    range(len(currsizes)),
                    currsizes
                )
                plt.title('Currsizes')
                plt.show()

            wrapper.plot_currsizes = plot_currsizes

            def plot_hitting_ratios_over_currsizes():
                nonlocal hitting_ratios, currsizes
                plt.plot(
                    range(len(hitting_ratios)),
                    [hitting_ratios[i]/currsizes[i]
                     for i in range(len(hitting_ratios))]
                )
                plt.title('Hitting ratios over currsizes')
                plt.show()

            wrapper.plot_hitting_ratios_over_currsizes =\
                plot_hitting_ratios_over_currsizes

            def plot_hitting_ratios_vs_currsizes():
                nonlocal hitting_ratios, currsizes
                plt.plot(
                    currsizes,
                    hitting_ratios
                )
                plt.title('Hitting ratios vs currsizes')
                plt.show()

            wrapper.plot_hitting_ratios_vs_currsizes =\
                plot_hitting_ratios_vs_currsizes

        def cache_clear():
            nonlocal hitss, missess, currsizes
            hitss = []
            missess = []
            currsizes = []
            func.cache_clear()

        wrapper.cache_clear = cache_clear
        return wrapper

    return decorating_function


def print_quantile_error_statisctics(qs_ode, qs_pi):
    print('ODE          :', ['{: 7.2f}'.format(v) for v in qs_ode])
    print('Expl.        :', ['{: 7.2f}'.format(v) for v in qs_pi])
    abs_err = np.abs(qs_ode-qs_pi)
    print('abs. err.    :', ['{: 7.2f}'.format(v) for v in abs_err])
    rel_err = np.abs(qs_ode-qs_pi)/qs_pi * 100
    print('rel. err. (%):', ['{: 7.2f}'.format(v) for v in rel_err])
    print()
    print('mean abs. err    :', '{: 6.2f}'.format(abs_err.mean()))
    print('mean rel. err (%):', '{: 6.2f}'.format(rel_err.mean()))
    print('max. abs. err    :', '{: 6.2f}'.format(np.max(abs_err)))
    print('max. rel. err (%):', '{: 6.2f}'.format(np.max(rel_err)))
    print()


def net_Fs_from_discrete_Bs_and_xs(Bs, xs):
    nr_pools = xs.shape[1]
    nt = len(Bs)

    net_Fs = np.zeros((nt, nr_pools, nr_pools))
    for k in range(nt):
        for j in range(nr_pools):
            for i in range(nr_pools):
                if i != j:
                    net_Fs[k, i, j] = Bs[k, i, j] * xs[k, j]

    return net_Fs


def net_Rs_from_discrete_Bs_and_xs(Bs, xs, decay_corr=None):
    nr_pools = xs.shape[1]
    nt = len(Bs)

    if decay_corr is None:
        decay_corr = np.ones((nt,))

    net_Rs = np.zeros((nt, nr_pools))
    for k in range(nt):
        for j in range(nr_pools):
            net_Rs[k, j] = (1-sum(Bs[k, :, j])*decay_corr[k]) * xs[k, j]

    return net_Rs


def net_Us_from_discrete_Bs_and_xs(Bs, xs):
    nr_pools = xs.shape[1]
    nt = len(Bs)

    net_Us = np.zeros((nt, nr_pools))
    for k in range(nt):
        net_Us[k] = xs[k+1] - Bs[k] @ xs[k]

    return net_Us


def check_parameter_dict_complete(model, parameter_dict, func_set):
    """Check if the parameter set  the function set are complete
       to enable a model run.

    Args:
        model (:class:`~.smooth_reservoir_model.SmoothReservoirModel`):
            The reservoir model on which the model run bases.
        parameter_dict (dict): ``{x: y}`` with ``x`` being a SymPy symbol
            and ``y`` being a numerical value.
        func_set (dict): ``{f: func}`` with ``f`` being a SymPy symbol and
            ``func`` being a Python function. Defaults to ``dict()``.
    Returns:
        free_symbols (set): set of free symbols, parameter_dict is complete if
                            ``free_symbols`` is the empty set
    """
    free_symbols = model.F.subs(parameter_dict).free_symbols
#    print('fs', free_symbols)
    free_symbols -= {model.time_symbol}
#    print(free_symbols)
    free_symbols -= set(model.state_vector)
#    print(free_symbols)

    # remove function names, are given as strings
    free_names = set([symbol.name for symbol in free_symbols])
    func_names = set([key for key in func_set.keys()])
    free_names = free_names - func_names

    return free_names


def F_Delta_14C(C12, C14, alpha=None):
    if alpha is None:
        alpha = ALPHA_14C

    C12[C12 == 0] = np.nan
    return (C14/C12/alpha - 1) * 1000
