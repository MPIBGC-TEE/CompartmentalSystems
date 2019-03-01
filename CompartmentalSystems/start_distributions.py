"""Module for computing age distributions or moments thereof to be used 
as start distributions in subsequent simulations. 

The age distribution at the start :math:`t_0` is NOT 
defined by the reservoir model and the initial values. 
In fact EVERY age distribution can be chosen. 
The implemented algorithms will correctly project it 
to any time :math:`t`.
This module provides several ways to generate such a distribution.


The functions containing the word 'distributions' usually return a vector
valued function of age representing pool wise age distributions that 
are NOT normalized (Integrating of a vector component over all ages 
yields the mass of the corresponding pool.)
, and in some cases a start vector that should be used in the subsequent
simulation for which the start age distribution is computed. 

The functions containing the word 'moments' usually return an
array: moments x pools, containing the moments of the
pool ages .
representing the initial values needed for the moment systems e.g. the
mean of a start distribution to be used as initial value for the
mean age system. The length of the list is determined by the maximum order of
the moment system to be solved. 
In some cases a consistent start vector is also provided.


Zero start age distributions
----------------------------
The distributions eaisiest to imagine are those that start with zero age: 

#.  The one with all pools empty provided by: 
    :py:meth:`start_age_distributions_from_zero_initial_content` 
    and the respective moments by:  
    :py:meth:`start_age_moments_from_zero_initial_content`     

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
    and the moments by: 
    :py:meth:`start_age_moments_from_empty_spinup`                                       
    
    Note that the finiteness of the spin up time has to be considered in the
    choice of questions that can be asked. 
    For instance do not mistake the fact that the percentage  
    of material older than :math:`a_{max}` will increase over 
    time for a property of the system, where it is actually a 
    property of the start distribution resulting from the finiteness 
    of the spin up time.  

    
#.  Distributions induced by steady states of the autonomous system, if those
    exist.

    If the term 'established' is taken to the limit of infinity one can look 
    for a related system that has persisted unchanged for all  times :math:`t<t_0` and 
    start with the age distribution created by such a system. Such a
    distribution can only occur if the system has been in a steady state.
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
    :py:meth:`start_age_distributions_from_steady_state`
    and 
    :py:meth:`start_age_moments_from_steady_state`.

    Note that :math:`x_{fix}` is the compatible start vector that has to be used 
    along with this start distributions for the following computation.
"""
from sympy import Matrix,Function
import numpy as np 
from scipy.linalg import inv, LinAlgError
from scipy.special import factorial
from scipy.optimize import root,fsolve
from testinfrastructure.helpers import pe
from CompartmentalSystems.helpers_reservoir import \
    jacobian, func_subs,\
    numerical_function_from_expression,\
    warning,deprecation_warning
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel
from testinfrastructure.helpers import pe

def start_age_distributions_from_empty_spinup(srm,t_max,parameter_dict,func_set):
    """
    Finite age spin up from empty pools 
 
    Creates a SmoothModelRun object with empty pools at :math:`t=0`, 
    runs it until :math:`t=t_{max}` an returns the age distribution at
    :math:`t_{max}`

    Args:
        srm (SmoothReservoirModel) : The (symbolic) model
        t_max (float):
            The end of the spinup (which starts at t=0 and runs until t=t_max)

        parameter_dict (dict) : 
            The parameter set that transforms the symbolic model into a numeric one. 
            The keys are the sympy symbols, the values are the values used for the simulation.
        func_set (dict): 
            The keys are the symbolic 
            sympy expressions for external functions 
            the values are the numeric functions to be used in the simulation  
    Returns: 
        (a_dist_at_end_of_spinup, sol_t_max)  (tuple):

        a_dist_at_end_of_spinup is a vector valued function of age. 
        a_dist_at_end_of_spinup(a)[i] reflects the mass of age :math:`a` 
        in pool i.

        sol_t_max is a one dimensional vector 
        representing the pool contents at the end of the spinup.
        This is returned since it is very likely needed as start vector in the
        simulation for which the start distributions has been computed.
    """
    a_dist_at_start_of_spinup= start_age_distributions_from_zero_initial_content(srm)
    # unfortunately the number of time steps has some influence on accuracy
    # although the ode solver guarantees a minimum it gets better if you force it to make smaller steps...
    #times=[0,t_max]
    times=np.linspace(0,t_max,101) 
    spin_up_mr = SmoothModelRun(
            srm, 
            parameter_dict=parameter_dict, 
            start_values=np.zeros(srm.nr_pools), 
            times=times,
            func_set=func_set)
    
    # p_sv(a,t) returns the mass of age a at time t
    p_sv = spin_up_mr.pool_age_densities_single_value(a_dist_at_start_of_spinup)
    sol_t_max=spin_up_mr.solve()[-1,:]

    def a_dist_at_end_of_spinup(age):
        return p_sv(age,t_max)

    return a_dist_at_end_of_spinup,sol_t_max



def start_age_distributions_from_steady_state(srm,t0,parameter_dict,func_set,x0=None):
    """
    Compute the age distribution of the system at equilibrium :math:`x_{fix}`
    , by means of a linear autonomous pool model with identical age
    distributions.
    The fixed point and the linear model are provided by: 
    :py:meth:`lapm_for_steady_state` 
    

    Args:
        srm (SmoothReservoirModel) : The (symbolic) model
        parameter_dict (dict) : 
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
    lapm,x_fix= lapm_for_steady_state(srm,t0,parameter_dict,func_set,x0=None)
    def a_dist_function(age): 
        mat_func=lapm.a_density 
        # LAPM returns a function that returns a sympy.Matrix
        # of NORMALIZED functions per pool
        # we have to transform it to a numpy array and 
        # then multiply it with the start values (x_fix)  
        mat=mat_func(age) 
        arr=np.array(mat).astype(np.float).reshape(srm.nr_pools)
        return x_fix*arr

    return a_dist_function,x_fix

def start_age_distributions_from_zero_age_initial_content(srm,x0):
    """
    Returns the age distribution (function) for a system into 
    which all initial mass is injected instantaneous at :math:`t=0`.

    The function returns a vector valued function dist(a) of the age a 
    that returns the start vector for a=0 and a
    zero vector of the same size everywhere else.
    This represents the age distribution of mass in a 
    compartmental system where the age of all pool contents is zero.
    This means that the distribution is NOT normalized to 1.

    Args:
        srm (SmoothReservoirModel) : The (symbolic) model
        x0 (numpy.ndarray) : The contents of the pools at :math:`t=0`
    Returns: 
        dist (callable):a vector valued function of age. 
        dist(a)[i] reflects the mass of age :math:`a` 
        in pool i.
    """
    svs=x0.shape
    n=srm.nr_pools
    # first check that the start vector has the correct size
    assert svs==(n,) ,"The initial_values had shape {0} while The reservoir model had s {1} pools".format(svs,n)
    def dist(a):
        if a == 0:
            return x0 
        else:
            return np.zeros(n)

    return dist

def start_age_distributions_from_zero_initial_content(srm):
    """
    Returns the age distribution (function) for an empty system. 

    The function returns a vector valued function dist(a) of the age a 
    that is zero everywhere dist(a)=0 for all a.
    This represents the age distribution of a compartmental system with all
    pools empty.

    Args:
        srm (SmoothReservoirModel) : The (symbolic) model

    Returns: 
        callable: dist, a vector valued function of age. 
        dist(a)[i] reflects the mass of age :math:`a` 
        in pool i.
    """
    return start_age_distributions_from_zero_age_initial_content(srm,np.zeros(srm.nr_pools))

def compute_fixedpoint_numerically(srm,t0,x0,parameter_dict,func_set):
    B_sym = srm.compartmental_matrix
    u_sym=srm.external_inputs

    t=srm.time_symbol
     
    tup = tuple(srm.state_vector) + (t,)
    u_func=numerical_function_from_expression(u_sym,tup,parameter_dict,func_set)
    B_func=numerical_function_from_expression(B_sym,tup,parameter_dict,func_set)
   
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

def start_age_moments_from_empty_spinup(srm,t_max,parameter_dict,func_set,max_order):
    times=np.linspace(0,t_max,101) 
    spin_up_mr = SmoothModelRun(
            srm, 
            parameter_dict=parameter_dict, 
            start_values=np.zeros(srm.nr_pools), 
            times=times,
            func_set=func_set)
    m0=start_age_moments_from_zero_initial_content(srm,max_order)
    
    
    moment_vector_list=[spin_up_mr.age_moment_vector(order,m0[0:order,:])[-1,:] for order in
            range(1,max_order+1)]
    #stack the moment vectors line wise
    #first line first moment (columns are pools)
    #second line second moment (columns are pools)
    moment_arr=np.stack(moment_vector_list,0) 
    
    sol_t_max=spin_up_mr.solve()[-1,:]
    return moment_arr, sol_t_max

def start_age_moments_from_zero_initial_content(srm,max_order):
    """
    The function returns an array of shape (max_order, srm.nr_pools)
   
    The values are set to numpy.nan to be consistent with other parts of the
    code. 
    For instance the mean age (first moment) of an empty pool is undefined )
    Args:
        srm (SmoothReservoirModel) : The (symbolic) model
        max_order (int): 
            The highest order up to which moments are to be computed.

    """
    start_age_moments = np.empty((max_order, srm.nr_pools))
    start_age_moments[:,:]=np.nan

    return  start_age_moments 


def lapm_for_steady_state(srm,t0,parameter_dict,func_set,x0=None):
    """
    If a fixedpoint of the frozen system can be found, create a linear
    autonomous model as an equivalent for the frozen (generally nonlinear)
    system there.

    The function performs the following steps:

    #.  Substitute symbols and symbolic functions with the parameters and 
        numeric functions.
    
    #.  Transform the general nonlinear non-autonomous system 
        into a nonlinear autonomous system by freezing it 
        at time :math:`t=t_0`: 

    #.  Compute :math:`u_0(x)=u(t_0,x_0)` and :math:`B_0(x)=B(t_0,x_0)` 

    #.  Look for an equilibrium :math:`x_{fix}` of the frozen system 
        such that :math:`0=B_0(x_{fix})+u_0(x_{fix})`.  
        If the frozen system is linear the we can compute 
        the fixed point explicitly : :math:`x_{fix}=B_0^{-1}u_0`.
        In general the frozen system will be nonlinear and we will have to 
        look for the fixed point numerically.
    
    #.  Create a linear autonomous pool model. 
        that can be investigated with the 
        package `LAPM <https://github.com/MPIBGC-TEE/LAPM>`_ 

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
        lapm is an instance of
        (:py:class:`LAPM.linear_autonomous_pool_model.LinearAutonomousPoolModel`
        ) representing the linearization with
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
            # in this case we can in principle transform to 
            # a linear Model with constant
            # input and new B
            # compute the jacobian of u
            sv=Matrix(srm.state_vector)
            M=jacobian(u_sym,sv)
            u_sym=u_sym-M*sv
            B_sym=B_sym+M
        
        t=srm.time_symbol
        tup = (t,)
        u_func=numerical_function_from_expression(u_sym,tup,parameter_dict,func_set)
        B_func=numerical_function_from_expression(B_sym,tup,parameter_dict,func_set)
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
            

        x_fix= compute_fixedpoint_numerically(srm,t0,x0,parameter_dict,func_set)
        pe('x_fix',locals())

        t=srm.time_symbol
        tup = tuple(srm.state_vector)+(t,)
        u_func=numerical_function_from_expression(u_sym,tup,parameter_dict,func_set)
        B_func=numerical_function_from_expression(B_sym,tup,parameter_dict,func_set)
        B0=B_func(*x_fix,t0)
        u0=u_func(*x_fix,t0)


    B0_m=Matrix(B0) 
    u0_m=Matrix(u0) 
    pe('B0',locals())
    pe('u0',locals())
    lapm=LinearAutonomousPoolModel(u0_m,B0_m,force_numerical=True)
    return lapm,x_fix

def start_age_moments_from_steady_state(srm,t0,parameter_dict,func_set,max_order):
    """
    Compute the age moments of the system at equilibrium :math:`x_{fix}`
    , by means of a linear autonomous pool model with identical age
    distributions.
    The fixed point and the linear model are provided by: 
    :py:meth:`lapm_for_steady_state` 
    
    Args:
        srm (SmoothReservoirModel) : The (symbolic) model
        parameter_dict (dict) : 
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
        max_order (int): 
            The highest order up to which moments are to be computed.

    Returns:
        numpy.ndarray: moments x pools, containing the moments of the
        pool ages in equilibrium.
    """
    
    lapm,x_fix= lapm_for_steady_state(srm,t0,parameter_dict,func_set,x0=None)
    start_age_moments = []
    for n in range(1, max_order+1):
        start_age_moment_sym=lapm.a_nth_moment(n)
        start_age_moment=np.array(start_age_moment_sym).astype(np.float).reshape(srm.nr_pools)
                        
        start_age_moments.append(start_age_moment)
    ret=np.array(start_age_moments)
    
    return ret
