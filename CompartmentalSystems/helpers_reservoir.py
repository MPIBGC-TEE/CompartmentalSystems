# vim:set ff=unix expandtab ts=4 sw=4:
from __future__ import division
from typing import Callable,Iterable,Union,Optional,List,Tuple 

import numpy as np 
import inspect
from numbers import Number
from scipy.integrate import odeint
from scipy.interpolate import lagrange
from scipy.optimize import brentq
from scipy.stats import norm
from string import Template
from sympy import flatten, gcd, lambdify, DiracDelta, solve, Matrix,diff
from sympy.polys.polyerrors import PolynomialError
from sympy.core.function import UndefinedFunction, Function, sympify
from sympy import Symbol
#from testinfrastructure.helpers import pe

def warning(txt):
    print('############################################')
    calling_frame=inspect.getouterframes(inspect.currentframe(),2)
    func_name=calling_frame[1][3]
    print("Warning in function {0}:".format(func_name))
    print(txt)

def deprecation_warning(txt):
    print('############################################')
    calling_frame=inspect.getouterframes(inspect.currentframe(),2)
    func_name=calling_frame[1][3]
    print("The function {0} is deprecated".format(func_name))
    print(txt)

def flux_dict_string(d,indent=0):
    s=""
    for k,val in d.items():
        s+=' '*indent+str(k)+": "+str(val)+"\n"

    return s

def func_subs(t,Func_expr,func,t0):
    """
    returns the function part_func 
    where part_func(_,_,...) =func(_,t=t0,_..) (func partially applied to t0)
    The position of argument t in the argument list is found 
    by examining the Func_expression argument.
    Args: 
        t (sympy.symbol): the symbol to be replaced by t0
        t0 (value)      : the value the function will be applied to
        func (function) : a python function 
        Func_exprs (sympy.Function) : An expression for an undefined Function

    """
    assert(isinstance(type(Func_expr),UndefinedFunction))
    pos=Func_expr.args.index(t)
    def frozen(*args):
        #tuples are immutable
        l=list(args)
        l.insert(pos,t0)
        new_args=tuple(l)
        return func(*new_args)
    return frozen

def jacobian(vec,state_vec):
    dim1 = vec.rows
    dim2 = state_vec.rows
    return(Matrix(dim1,dim2,lambda i,j: diff(vec[i],state_vec[j])))

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

def numerical_function_from_expression(expr,tup,parameter_dict:dict,func_set):
    # the function returns a function that given numeric arguments
    # returns a numeric result.
    # This is more specific requirement than a function returned lambdify 
    # which can still return symbolic
    # results if the tuple argument to lambdify does not contain all free
    # symbols of the lambdified expression.
    # To avoid this case here we check this.
    expr_par=expr.subs(parameter_dict)
    ss_expr=expr_par.free_symbols
    ss_tup=set([s for s in tup])
    if not(ss_expr.issubset(ss_tup)):
        raise Exception("The free symbols of the expression: ${0} are not a subset of the symbols in the tuple argument:${1}".format(ss_expr,ss_tup))


    
    cut_func_set=make_cut_func_set(func_set)
   

    expr_par=expr.subs(parameter_dict)
    expr_func = lambdify(tup, expr_par, modules=[cut_func_set, 'numpy'])
    return expr_func

def numerical_rhs2(state_vector, time_symbol, rhs, 
        parameter_dict, func_set):

    FL=numerical_function_from_expression(
            rhs
            ,(time_symbol,)+tuple(state_vector)
            ,parameter_dict
            ,func_set
    )
    
    # 2.) Write a wrapper that transformes Matrices numpy.ndarrays and accepts array instead of the separate arguments for the states)
    # 
    def num_rhs(t,X):
        Fval = FL(t,*X)
        return Fval.reshape(X.shape,)

    return num_rhs    

def numerical_rhs(state_vector, time_symbol, rhs, 
        parameter_dict, func_set, times):

    
    FL=numerical_function_from_expression(
            rhs
            ,tuple(state_vector) + (time_symbol,)
            ,parameter_dict
            ,func_set
    )
    # 2.) Write a wrapper that transformes Matrices numpy.ndarrays and accepts array instead of the separate arguments for the states)
    # 
    def num_rhs(X,t):
        Fval = FL(*X,t)
        return Fval.reshape(X.shape,)

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
        parameter_dict, 
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
        parameter_dict,
        func_set,
        times
    )

    #return odeint(num_rhs, start_values, times, mxstep=10000)
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
    
def make_cut_func_set(func_set):
    def unify_index(expr):
        # for the case Function('f'):f_numeric
        if isinstance(expr,UndefinedFunction):
            res=str(expr)
        # for the case {f(x,y):f_numeric} f(x,y) 
        elif isinstance(expr,Symbol):
            res=str(expr)
        elif isinstance(expr,Function):
            res=str(type(expr))
        elif isinstance(expr,str):
            expr=sympify(expr)
            res=unify_index(expr)
        else:
            print(type(expr))
            raise(TypeError("funcset indices should be indexed by instances of sympy.core.functions.UndefinedFunction"))
        return res

    cut_func_set={unify_index(key):val for key,val in func_set.items()}
    return cut_func_set


def f_of_t_maker(sol_funcs,ol):
    def ot(t):
        sv = [sol_funcs[i](t) for i in range(len(sol_funcs))]
        tup = tuple(sv)+(t,)
        res = ol(*tup)
        return(res)
    return(ot)
def const_of_t_maker(const):
    
    def const_arr_fun(possible_vec_arg):
        if isinstance(possible_vec_arg,Number):
            return const #also a number
        else:
            return(const*np.ones_like(possible_vec_arg))
    return const_arr_fun

def block_rhs(
         time_str  : str
        ,X_blocks  : List[ Tuple[str,int] ]
        ,functions : List[ Tuple[Callable,List[str]]]
    )->Callable[[np.double,np.ndarray],np.ndarray]:
    """
    The function returns a function dot_X=f(t,X) suitable as the righthandside 
    for the ode solver scipy.solve_ivp from a collection of vector valued
    functions that compute blocks of dot_X from time and blocks of X 
    rather than from single equations.

    A special application is the creation of block triangular systems, to 
    integrate variables whose time derivative depend on the solution
    of an original system instantaniously along with it.
    
    Assume that
    X_1(t) is the solution of the initial value problem (ivp)
    
    ivp_1:
    dot_X_1=f_1(t,X) ,X_1(t_0) 

    and X_2(t) the solution of another ivp 

    ivp_2:
    dot_X_2=f_2(t,X_1,X_2) ,X_2(t_0) whose righthand side depends on x_1
    
    Then we can obtain the solution of both ivps simultaniously by
    combining the them into one.
    
    (dot_X_1, dox_X_2)^t = (f_1(t,X_1),f_2(t,X_1,X_2))^t

    For n instead of 2 variables one has:  
    (dot_X_1, dox_X_2,...,dot_X_n)^t = (f_1(t,X_1),f_2(t,X_1,X_2),...,f_n(t,X_1,...X_n))^t
    
    For a full lower triangular system  the block derivative dot_X_i depend on t,
    and ALL the blocks X_1,...,X_i but often they will only depend on 
    SOME of the previous blocks so that f_m has a considerably smaller argument list.

    This function therefore allows to specify WHICH blocks the f_i depend on.
    Consider the following 7 dimensional block diagonal example:

    b_s=block_rhs(
         time_str='t'
        ,X_blocks=[('X1',5),('X2',2)]
        ,functions=[
             ((lambda x   : x*2 ),  ['X1']    )
            ,((lambda t,x : t*x ),  ['t' ,'X2'])
         ])   
    

    The first argument 'time_str' denotes the alias for the t argument to be used
    later in the signature of the blockfunctions.
    The second argument 'X_blocks' describes the decomposition of X into blocks
    by a list of tuples of the form ('Name',size)
    The third argument 'functions' is a list of tuples of the function itself
    and the list of the names of its block arguments as specified in the
    'X_blocks' argument. 
    Order is important for the 'X_blocks' and the 'functions'
    It is assumed that the i-th function computes the derivative of the i-th block.
    The names of the blocks itself are arbitrary and have no meaning apart from
    their correspondence in the X_blocks and functions argument.
    """
    block_names=[t[0] for t in X_blocks]
    dims=[t[1] for t in X_blocks]
    nb=len(dims)
    strArgLists=[f[1] for f in functions]
    # make sure that all argument lists are really lists
    assert(all([isinstance(l,list) for l in strArgLists])) 
    # make sure that the function argument lists do not contain block names
    # that are not mentioned in the Xblocks argument
    flatArgList=[arg for argList in strArgLists for arg in argList]
    assert(set(flatArgList).issubset(block_names+[time_str]))
    

    # first compute the indices of block boundaries in X by summing the dimensions 
    # of the blocks
    indices=[0]+[ sum(dims[:(i+1)]) for i in range(nb)]
    def rhs(t,X):
        blockDict={block_names[i]: X[indices[i]:indices[i+1]] for i in range(nb)}
        #pe('blockDict',locals())
        blockDict[time_str]=t
        arg_lists=[ [blockDict[k] for k in f[1]] for f in functions]
        blockResults=[ functions[i][0]( *arg_lists[i] )for i in range(nb)]
        #pe('blockResults',locals())
        return np.concatenate(blockResults).reshape(X.shape)
    
    return rhs
