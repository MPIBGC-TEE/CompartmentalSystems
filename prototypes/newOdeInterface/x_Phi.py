import numpy as np
from sympy import var,Matrix
from scipy.integrate import solve_ivp
from testinfrastructure.helpers import pe
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.helpers_reservoir import x_phi_ivp,numerical_rhs2,numerical_function_from_expression
# The state transition operator Phi is defined for linear systems only
# To compute it we have to create a linear system first by substituting
# the solution into the righthandside
# This could be done in different ways:
# 1.)   By solving the ODE with the actual start vector first and then
#       substituting the interpolation into the righthandside used to compute the state transition operator
# 2.)   Creation of a skewproductsystem whose solution yields
#       the solution for the initial value problem and the state transition operator for Phi(t,t_0) simultaniously.
k_0_val=1
k_1_val=2
x0_0=np.float(2)
x0_1=np.float(1)
delta_t=np.float(1./4.)
# 
var(["x_0","x_1","k_0","k_1","t","u"])
#
inputs={
     0:u
    ,1:u*t
}
outputs={
     0:k_0*x_0**2
    ,1:k_1*x_1
}
internal_fluxes={}
svec=Matrix([x_0,x_1])
srm=SmoothReservoirModel(
         state_vector       =svec
        ,time_symbol        =t
        ,input_fluxes       =inputs
        ,output_fluxes      =outputs
        ,internal_fluxes    =internal_fluxes
)
t_max=4
times = np.linspace(0, t_max, 11)
parameter_dict = {
     k_0: k_0_val
    ,k_1: k_1_val
    ,u:1}
func_dict={}
start_x= np.array([x0_0,x0_1]) #make it a column vector for later use
my_x_phi_ivp=x_phi_ivp(srm,parameter_dict,func_dict,start_x)


# we now express the solution as integral expression of the state transition operator
# x_t=Phi(t,t0)*x_0+int_t0^t Phi(t,tau)*u(tau) dtau
# and check that we get the original solution back
t_0=0
t_span=(t_0,t_max)
times = np.linspace(t_0, t_max, 11)
ts   =my_x_phi_ivp.get_values("t",t_span=t_span,max_step=.2)
xs   =my_x_phi_ivp.get_values("sol",t_span=t_span)
phis =my_x_phi_ivp.get_values("Phi_1d",t_span=t_span)

sol_rhs=numerical_rhs(
     srm.state_vector
    ,srm.time_symbol
    ,srm.F
    ,parameter_dict
    ,func_dict
)
x_func=solve_ivp(sol_rhs,y0=start_x,t_span=t_span,dense_output=True).sol
def check_phi_args(t,s):
    # first check 
    if t < t_0:
        raise(Error("Evaluation of Phi(t,s) with t before t0 is not possible"))
    if s < t_0 :
        raise(Error("Evaluation of Phi(t,s) wiht s before t0 is not possible"))
    
    if s == t:
        return np.identity(srm.nr_pools)

def Phi(t,s):
    # this whole function would make sense if 
    # one could guarantee that the state transition operator will
    # be required only for a fixed grid 
    # (ideally to be determined by the functions unsing it before this function is called)
    # so that no interpolation takes place (as it is now)
    # but only cached values would be required
    # The the cost of computing the inverses for every time step would be
    # justified
    """
    For t_0 <s <t we have
    
    Phi(t,t_0)=Phi(t,s)*Phi(s,t_0)
    
    If we know $Phi(s,t_0) \forall s \in [t,t_0] $
    
    We can reconstruct Phi(t,s)=Phi(t,t_0)* Phi(s,t_0)^-1
    This is what this function does.
    """
    check_phi_args(t,s)
    # the next call will cost nothing if s and t are smaller than t_max
    # since the ivp caches the solutions up to t_0 after the first call.
    t_span=(t_0,max(s,t,t_max))
    Phi_t0 =my_x_phi_ivp.get_function("Phi_1d",t_span=t_span)
    def Phi_t0_mat(t):
        return Phi_t0(t).reshape(srm.nr_pools,srm.nr_pools)

    if s>t:
        # we could call the function recoursively with exchanged arguments but
        #
        # res=np.linalg.inv(Phi(s,t))
        # 
        # But this would involve 2 inversions. 
        # To save one inversion we use (A*B)^-1=B^-1*A^-1
        return np.matmul(np.linalg.inv(Phi_t0_mat(s)),Phi_t0_mat(t))
        

    
    if s == t_0:
        return Phi_t0_mat(t)
    # fixme: mm 3/9/2019
    # the following inversion is very expensive and should be cached or avoided
    # to save space in a linear succession state Transition operators from step to step
    # then everything can be reached by a 
    return np.matmul(Phi_t0_mat(t),np.linalg.inv(Phi_t0_mat(s)))

tup=(srm.time_symbol,)+tuple(srm.state_vector)
B_func=numerical_function_from_expression(srm.compartmental_matrix,tup,parameter_dict,func_dict)
def Phi_direct(t,s):
    """
    For t_0 <s <t we have
    
    compute Phi(t,s) directly
    by integrating 
    d Phi/dt= B(x(tau))  from s to t with the startvalue Phi(s)=UnitMatrix
    It assumes the existence of a solution x 
    """

    check_phi_args(t,s)
    # the next call will cost nothing if s and t are smaller than t_max
    # since the ivp caches the solutions up to t_0 after the first call.
    t_span=(t_0,max(s,t,t_max))
    
    def Phi_rhs(tau,Phi_1d):
        B=B_func(tau,*x_func(tau))
        #Phi_cols=[Phi_1d[i*nr_pools:(i+1)*nr_pools] for i in range(nr_pools)]
        #Phi_ress=[np.matmul(B,pc) for pc in Phi_cols]
        #return np.stack([np.matmul(B,pc) for pc in Phi_cols]).flatten()
        return np.matmul(B,Phi_1d.reshape(nr_pools,nr_pools)).flatten()
     
    Phi_values=solve_ivp(Phi_rhs,y0=np.identity(srm.nr_pools).flatten(),t_span=(s,t)).y
    val=Phi_values[:,-1].reshape(srm.nr_pools,srm.nr_pools)
    
    if s>t:
        # we could probably express this by a backward integration
        return np.linalg.inv(val)
    else: 
        return val




    return np.matmul(Phi_t0_mat(t),np.linalg.inv(Phi_t0_mat(s)))

nr_pools=srm.nr_pools
rs=(nr_pools,len(ts))

u_sym=srm.external_inputs
u_num=numerical_function_from_expression(u_sym,(t,),parameter_dict,{})
#
def trapez_integral(i):
    # We compute the integral 
    # NOT as the solution of an ivp but with an integration rule that 
    # works with arrays instead of functions
    t=ts[i]
    if i==0:
        return np.zeros((nr_pools,1))
    # the integrals boundaries grow with time
    # so the vector for the trapezrule becomes longer and longer
    taus=ts[0:i]
    sh=(nr_pools,len(taus)) 
    #t=taus[-1]
    phi_vals=np.array([Phi(t,tau) for tau in taus])
    #print(phi_vals)
    integrand_vals=np.stack([np.matmul(Phi(t,tau),u_num(tau)).flatten() for tau in taus],1)
    #pe('integrand_vals',locals())
    val=np.trapz(y=integrand_vals,x=taus).reshape(nr_pools,1)
    #pe('val',locals())
    return val

def continiuous_integral(t,Phi_func):
    # We compute the integral of the continious function
    # With an integration rule that 
    #print(phi_vals)
    def rhs(tau,X):
        # although we do not need X we have to provide a 
        # righthandside suitable for solve_ivp
        return np.matmul(Phi_func(t,tau),u_num(tau)).flatten()

    #pe('integrand_vals',locals())
    ys= solve_ivp(rhs,y0=np.zeros(nr_pools),t_span=(t_0,t)).y
    pe('ys.shape',locals())
    val=ys[:,-1].reshape(nr_pools,1)
    #pe('val',locals())
    return val

## reconstruct the solution with Phi and the integrand
# x_t=Phi(t,t0)*x_0+int_t0^t Phi(tau,t0)*u(tau) dtau
# x_t=a(t)+b(t)
a_list=[np.matmul(Phi(t,t_0),start_x.reshape(nr_pools,1)) for t in ts]
b_list=[ trapez_integral(i) for i,t in enumerate(ts)]
b_cont_list=[continiuous_integral(t,Phi) for t in ts]
b_cont2_list=[continiuous_integral(t,Phi_direct) for t in ts]
x2_list=[a_list[i]+b_list[i] for i,t in enumerate(ts)]
x3_list=[a_list[i]+b_cont_list[i] for i,t in enumerate(ts)]
x4_list=[a_list[i]+b_cont2_list[i] for i,t in enumerate(ts)]
def vectorlist2array(l):
    return np.stack( [vec.flatten() for vec in l],1)

a,b,x2,x3,x4=map(vectorlist2array,[a_list,b_list,x2_list,x3_list,x4_list])
fig=plt.figure(figsize=(10,17))
ax1=fig.add_subplot(2,1,1)
ax1.plot(ts,xs[0,:],'o',color='blue' ,label="sol[0]")
ax1.plot(ts,xs[1,:],'o',color='blue' ,label="sol[1]")

ax1.plot(ts,x2[0,:],'+',color='red' ,label="x2[0]")
ax1.plot(ts,x2[1,:],'+',color='red' ,label="x2[0]")

ax1.plot(ts,x3[0,:],'x',color='green' ,label="x3[0]")
ax1.plot(ts,x3[1,:],'x',color='green' ,label="x3[1]")

ax1.plot(ts,x3[0,:],'+',color='orange' ,label="x4[0]")
ax1.plot(ts,x3[1,:],'+',color='orange' ,label="x4[1]")
#ax1.plot(ts,xs2[1,:],'x',color='red' ,label="sol2[1]")

#ax1.plot(ts,a[0,:],'o',color='orange' ,label="a[0]")
#ax1.plot(ts,a[1,:],'x',color='orange' ,label="a[1]")

#ax1.plot(ts,b[0,:],'o',color='green' ,label="b[0]")
#ax1.plot(ts,b[1,:],'x',color='green' ,label="b[1]")
ax1.legend()

#ax2=fig.add_subplot(2,1,2)
#ax2.plot(ts,integrand_vals[0,:],'x',color='green' ,label="integrand")
#ax2.plot(ts,phi_int_vals1[0,:],'x',color='red' ,label="phi 1")
#ax2.plot(ts,phi_int_vals2[0,:],'x',color='red' ,label="phi 2")
#ax2.legend()

fig.savefig("solutions.pdf")
