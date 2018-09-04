"""Module for computing age distributions or moments thereofto be used 
as start distributions in subsequent simulations. 

The age distribution at the start :math:`t_0` is NOT 
defined by the reservoir model and the initial values. 
In fact EVERY age distribution can be chosen. 
The implemented algorithms will correcly project it 
to any time :math:`t`.
This module provides several ways to generate such a distribution.


The functions containing the word 'distributions' usually return a vector
valued function of age representing pool wise age distributions that 
are NOT normalized. 
(Integrating of a vector component over all ages 
yields the mass of the corresponding pool.)
The functions containing the word 'moments' usually return an
array: moments x pools, containing the moments of the
pool ages .
representing the initial values needed for the moment systems e.g. the
mean of a startdistribution to be used as initial value for the
mean age system. The length of the list is determined by the maximum order of
the moment system to be solved. 



Zero start age distributions
----------------------------
The distributions eaisiest to imagine are those that start with zero age: 

#.  The one with all pools empty provided by: 
    :py:meth:`start_age_distributions_from_zero_initial_content` or 
#.  The one where all initial mass has age zero, provided by: 
    :py:meth:`start_age_distributions_from_zero_age_initial_content` 
    
Established distributions
----------------------------
However for many applications one is interested in the CHANGE of an age
distribution that has been established over a (possibly infinitely) long period of time.        

#.  Spinup

    If we start the computation with all pools empty at time :math:`0` 
    and run it till time :math:`t = a_{max}`, 
    the resulting distribution will be non zero only in the interval
    :math:`[0,a_{max}]`. 
    Such a distribution is provided by:
    :py:meth:`start_age_distributions_from_empty_spinup`                                       
    
    Note that the finiteness of the spin up time has to be considered in the
    choice of questions that can be asked. 
    For instance do not mistake the fact that the percentage  
    of material older than :math:`a_{max}` will increase over 
    time for a property of the system, where it is actually a 
    property of the start distribution resulting from the finiteness 
    of the spin up time.  

    
#.  Distributions induced by steady states of the autonumuous system, if those
    exist.

    If the term 'established' is taken to the limit of infinity one can look 
    for a related system that has persisted unchanged for all  times :math:`t<t_0` and 
    start with the age distribution created by such a system. Such a
    distribution can only occure if the system has been in a steady state.
    For a general non-autonomous system this is very unlikely that such a
    steady state exist at all.
    However we can consider a related autonomous system resulting from 
    'freezing' the general non-autonomous system at a time :math:`t_0`. 
    Even for such an autonomous system it is uncertain if and where equilibria
    exist. This has to be checked before an equilibrium age distribution can be
    computed. 
    Actually the following steps have to be performed:

    a.  Transform the general nonlinear non-autonomous system 
        into a nonlinear autonomous system by freezing it 
        at time :math:`t=t_0`: 

    b.  Compute :math:`u_0(x)=u(t_0,x_0)` and :math:`B_0(x)=B(t_0,x_0)` 

    c.  Look for an equilibrium :math:`x_{fix}` of the frozen system 
        such that :math:`0=B_0(x_{fix})+u_0(x_{fix})`.  
        If the frozen system is linear the we can compute 
        the fixed point explicitly : :math:`x_{fix}=B_0^{-1}u_0`.
        In general the frozen system will be nonlinear and we will have to 
        look for the fixed point numerically.
    d.  Compute the age distribution of the system at equilibrium :math:`x_{fix}`.
        This can be done using the formulas for linear 
        autonomous pool models. (At the fixed point the nonlinear system 
        is identical to a linear one. This is a special case of the general
        idea of linearizing a nonlinear model with respect to a trajectory,
        which in case of a fixed point is constant.)

    All these steps are performed by  
    :py:meth:`start_age_distributions_from_steady_state`.                                      
    Note that :math:`x_{fix}` is the compatible startvalue that has to be used 
    along with this start distributions for the following computation.
"""
from sympy import Matrix,Function
import numpy as np 
from scipy.linalg import inv, LinAlgError
from scipy.special import factorial
from scipy.optimize import root,fsolve
from CompartmentalSystems.helpers_reservoir import \
    jacobian, func_subs,\
    numerical_function_from_expression,pe,\
    warning,deprecation_warning
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel

def start_age_distributions_from_empty_spinup(srm,t0,parameter_set,func_set):
    """
    Finite age spin up from empty pools 
    """
    a_dist_at_start_of_spinup= start_age_distributions_from_zero_initial_content(srm)
    # unfortunately the numer of timesteps has some influence on accuracy
    # although the ode solver guarantees a minimum it gets better if you force it to make smaller steps...
    #times=[0,t0]
    times=np.linspace(0,t0,101) 
    spin_up_mr = SmoothModelRun(
            srm, 
            parameter_set=parameter_set, 
            start_values=np.zeros(srm.nr_pools), 
            times=times,
            func_set=func_set)
    
    # p_sv(a,t) returns the mass of age a at time t
    p_sv = spin_up_mr.pool_age_densities_single_value(a_dist_at_start_of_spinup)
    sol=spin_up_mr.solve()[-1,:]

    def a_dist_at_end_of_spinup(age):
        return p_sv(age,t0)

    return a_dist_at_end_of_spinup,sol

def start_age_distributions_from_steady_state(srm,t0,parameter_set,func_set,x0=None):
    """
    Compute the age distribution of the system at equilibrium :math:`x_{fix}`
    , by means of a linear autonomous pool model with identical age
    distributions.
    The fixed point and the linear model are provided by: 
    :py:meth:`lapm_for_steady_state` 
    

    Args:
        srm (SmoothReservoirModel) : The (symbolic) model
        par_set (dict) : 
            The parameter set that transforms the symbolic model into a numeric one. 
            The keys are the sympy symbols, the values are the values used for the simulation.
        func_set (dict): 
            The keys are the symbolic 
            sympy expressions for external functions 
            the values are the numeric functions to be used in the simulation  
        t0 (float):
            The time where the non-autonomous system is frozen. 

        x0 (numpy.ndarray): 
            An initial guess to start the fixed point iteration.
            If the frozen model is linear it will be ignored and 
            can therefore be omitted. 
            
    Returns: 
        (a_dist_function, x_fix)  (tuple):

        a_dist_function is a vector valued function of age. 
        a_dist_function(a)[i] reflects the mass of age :math:`a` 
        in pool i.
        :math:`x_{fix}` is a one dimensional vector representing the equilibrium.
        This is returned since it is very likely needed as start vector in the
        simulation for which the start distributions has been computed.
        (The computed distribution assumes the system to be in this state.) 
    """
    lapm,x_fix= lapm_for_steady_state(srm,t0,parameter_set,func_set,x0=None)
    def a_dist_function(age): 
        mat_func=lapm.a_density 
        # Lapm returns a function that returns a sympy.Matrix
        # of NORMALIZED functions per pool
        # we have to transform it to a numpy array and 
        # then multiply it with the start values (x_fix)  
        mat=mat_func(age) 
        arr=np.array(mat).astype(np.float).reshape(srm.nr_pools)
        return x_fix*arr

    return a_dist_function,x_fix

def start_age_distributions_from_zero_age_initial_content(srm,start_values):
    """
    The function returns a vector valued function f(a) of the age a 
    that returns the startvector for a=0 and a zero vector of the same size everywhere else.
    This represents the age distribution of mass a compartmental system where the age of all
    pool contents is zero.
    This means that the distribution is NOT normalized to 1.
    """
    svs=start_values.shape
    n=srm.nr_pools
    # first check that the start vector has the correct size
    assert svs==(n,) ,"The initial_values had shape {0} while The reservoir model had s {1} pools".format(svs,n)
    def dist(a):
        if a == 0:
            return start_values
        else:
            return np.zeros(n)

    return dist

def start_age_distributions_from_zero_initial_content(srm):
    """
    The function returns a vector valued function f(a) of the age a 
    that is zero everywhere f(a)=0 for all a.
    This represents the age distribution of a compartmental system with all
    pools empty.
    """
    return start_age_distributions_from_zero_age_initial_content(srm,np.zeros(srm.nr_pools))

def compute_fixedpoint_numerically(srm,t0,x0,parameter_set,func_set):
    B_sym = srm.compartmental_matrix
    u_sym=srm.external_inputs

    t=srm.time_symbol
     
    tup = tuple(srm.state_vector) + (t,)
    u_func=numerical_function_from_expression(u_sym,tup,parameter_set,func_set)
    B_func=numerical_function_from_expression(B_sym,tup,parameter_set,func_set)
   
    # get functions of x1,...,xn by partly applying to t0 
    B0_func=func_subs(t,Function("B")(*tup),B_func,t0)
    u0_func=func_subs(t,Function("u")(*tup),u_func,t0)

    # build the kind of function that scipy.optimize.root expects 
    # it has to have a single vector like argument
    def ex_func(x):
        tup=tuple(x)
        return B0_func(*tup)@x+u0_func(*tup).reshape(x.shape)
    
    res = root(fun=ex_func,jac=False,x0=x0)
    assert(res.success)
    # chose a method that does not use the jacobian
    #res = root(fun=ex_func,method='krylov',x0=x0,tol=1e-5)
    #pe('res',locals())
    return res.x

def start_age_moments_from_empty_spin_up(srm,parameter_set,func_set,a_max,max_order):
    # to do:
    # run a spin up and observe the age distribution at the end
    # then compute the moments numerically
    raise Exception("Not implemented yet")

def start_age_moments_from_zero_initial_content(srm,max_order):
    return [ np.zeros(srm.nr_pools,1) for n in range(1, max_order+1)]


def lapm_for_steady_state(srm,t0,parameter_set,func_set,x0=None):
    """
    The function performs the following steps:

    a.  Transform the general nonlinear non-autonomous system 
        into a nonlinear autonomous system by freezing it 
        at time :math:`t=t_0`: 

    b.  Compute :math:`u_0(x)=u(t_0,x_0)` and :math:`B_0(x)=B(t_0,x_0)` 

    c.  Look for an equilibrium :math:`x_{fix}` of the frozen system 
        such that :math:`0=B_0(x_{fix})+u_0(x_{fix})`.  
        If the frozen system is linear the we can compute 
        the fixed point explicitly : :math:`x_{fix}=B_0^{-1}u_0`.
        In general the frozen system will be nonlinear and we will have to 
        look for the fixed point numerically.
    d.  Create a linear autonomous pool model. 
        that can be investigated with py:mo:`LAPM`
        This is a special case of the general
        linearization of a nonlinear model along a trajectory,
        which in case of a fixed point is constant.
        At the fixed point the age distribution and solution of the 
        nonlinear system are identical to those of a linear one. 
    
    Args:
        srm (SmoothReservoirModel) : The (symbolic) model
        par_set (dict) : 
            The parameter set that transforms the symbolic model into a numeric one. 
            The keys are the sympy symbols, the values are the values used for the simulation.
        func_set (dict): 
            The keys are the symbolic 
            sympy expressions for external functions 
            the values are the numeric functions to be used in the simulation  
        t0 (float):
            The time where the non-autonomous system is frozen. 

        x0 (numpy.ndarray): 
            An initial guess to start the fixed point iteration.
            If the frozen model is linear it will be ignored and 
            can therefore be omitted. 
            
    Returns: 
        (lapm, x_fix)  (tuple):
        lapm (:py:class:`LinearAutonomousPoolModel` ) The linearization with
        respect to the (constant) fixed point trajectory. It yields the 
        same age distribution as the frozen possibly nonlinear system at 
        the fixed point.
        :math:`x_{fix}` is a one dimensional vector representing the equilibrium.
        This is returned since it is very likely needed as start vector in the
        simulation for which the start distributions has been computed.
        (The computed distribution assumes the system to be in this state.) 
    """
    B_sym = srm.compartmental_matrix
    u_sym=srm.external_inputs
    if srm.is_linear:
        if srm.is_state_dependent(u_sym):
            # in this case we can in principle transform to a linear Model with constant
            # imput and new B
            # compute the jacobian of u
            sv=Matrix(srm.state_vector)
            M=jacobian(u_sym,sv)
            u_sym=u_sym-M*sv
            B_sym=B_sym+M
        
        t=srm.time_symbol
        tup = (t,)
        u_func=numerical_function_from_expression(u_sym,tup,parameter_set,func_set)
        B_func=numerical_function_from_expression(B_sym,tup,parameter_set,func_set)
        B0=B_func(t0)
        u0=u_func(t0)
        try:
            x_fix=(-inv(B0)@u0).reshape(srm.nr_pools)
            pe('x_fix',locals())
        except LinAlgError as e:
            print("""
            B_0=B(t_0) is not invertable
            If a singular matrix B_0 occurs, then the system would have traps. 
            A steady state could then only occour
            if the components of u0=u(t0) would be zero for the pools connected
            to the trap. 
            In this (unlikely) event the startage distribution would
            be ambigous because the fixedpoint x_{fix} is not uniqe (The content of the trap is a free 
            parameter and so is its age.)
            """)
            raise e

    else:
        if x0 is None:
            x0=np.ones(srm.nr_pools)
            warning("""
            No initial guess for the fix point iteration given.
            For nonlinear models equilibria can in general only be found
            numerically by an iterative process starting from an initial guess x0, 
            which has not been provided.
            Since a nonlinear model can have several equilibria the initial
            guess determines which one will be found by the iteration.
            A good guess might also increase the likelihood to find an
            equilibrium at all.
            In absence of an initial guess we will start with
            numpy.ones(smr.nr_pools) which might not be a good choice.
            We strongly advise to specify the x0 argument."""
            )
            

        x_fix= compute_fixedpoint_numerically(srm,t0,x0,parameter_set,func_set)
        pe('x_fix',locals())

        t=srm.time_symbol
        tup = tuple(srm.state_vector)+(t,)
        u_func=numerical_function_from_expression(u_sym,tup,parameter_set,func_set)
        B_func=numerical_function_from_expression(B_sym,tup,parameter_set,func_set)
        B0=B_func(*x_fix,t0)
        u0=u_func(*x_fix,t0)

    lapm=LinearAutonomousPoolModel(Matrix(u0),Matrix(B0))
    return lapm,x_fix

def start_age_moments_from_steady_state(srm,t0,parameter_set,func_set,max_order):
    """
    Args:
        srm (SmoothReservoirModel) : The (symbolic) model
        par_set : The parameter set that transforms the symbolic model into a numeric one. 
        max_order (int): The highest order up to which moments are
        to be computed..

    Returns:
        numpy.ndarray: moments x pools, containing the moments of the
        pool ages in equilibrium.
    """
    
    lapm,x_fix= lapm_for_steady_state(srm,t0,parameter_set,func_set,x0=None)
    start_age_moments = []
    for n in range(1, max_order+1):
        start_age_moment_sym=lapm.a_nth_moment(n)
        start_age_moment=np.array(start_age_moment_sym).astype(np.float).reshape(srm.nr_pools)
                        
        start_age_moments.append(start_age_moment)
    ret=np.array(start_age_moments)
    
    return ret
