from sympy import atan ,pi,lambdify, symbols ,solve,Piecewise
import numpy as np
import matplotlib.pyplot as plt
def half_saturation(z,eps):
    return z/(eps+z)

def atan_ymax(z,y_max,sfz):
    sfy=2/pi*y_max
    res=sfy*atan(sfz/sfy*z)
    return res

def cubic(z,z_max):
    """ yields a piece wise expression with a cubic interpolation f(z)= a3*z**3+a2*z**2+a1*z+a0 with  for z<z_max and  1.0 for z>=z_max. 
    The a_i are adjusted to guarantee f(0)=0,f'(0)= 1/z_max f(z<z_max)<=1, f'(z>z_max)=0"""
    a0,a1,a2,a3=symbols('a0,a1,a2,a3')
    f=a3*z**3+a2*z**2+a1*z+a0
    #compute coefficients
    par_dict=solve([
        f.subs({z:0}), #f(0)=0
        (f.diff(z)-1/z_max).subs({z:0}),#f'(0)=1/z_max
        (f-1).subs({z:z_max}), #f(z_max)=1
        f.diff(z).subs({z:z_max}), #f'(z_max)=0
        ],
        a0,a1,a2,a3)
    f_ai=f.subs(par_dict)
    f_piece=Piecewise((f_ai,z<z_max),(1,True))
    return f_piece

def deceleration(z,z_max,alpha):
    f=(1-((z_max-z)/z_max)**alpha)
    f_piece=Piecewise((f,z<z_max),(1,True))
    return f_piece

z,z_max=symbols('z,z_max')
f_piece=cubic(z,z_max)
fig=plt.figure(figsize=(9,15))
ax1=fig.add_subplot(5,1,1)
ax2=fig.add_subplot(5,1,2)
ax3=fig.add_subplot(5,1,3)
ax4=fig.add_subplot(5,1,4)
ax5=fig.add_subplot(5,1,5)
plt.subplots_adjust(hspace=0.4)

zms=[10,20,30]
z_val=1.5*np.linspace(0,max(zms),101)
ax1.set_title("cubic with slope=1/z_max")

for z_max_val in zms: 
    f_par=f_piece.subs({z_max:z_max_val})
    f_num=lambdify(z,f_par,modules='numpy')
    ax1.plot(z_val,f_num(z_val))


z,z_max,alpha =symbols('z,z_max,alpha')
f_piece=deceleration(z,z_max,alpha)
ax2.set_title("decelaration with constant alpha and varying z_max")
for z_max_val in zms: 
    pd1={z_max:z_max_val,alpha:2}
    f_par=f_piece.subs(pd1)
    f_num=lambdify(z,f_par)
    ax2.plot(z_val,f_num(z_val))

ax3.set_title("decelaration with constant z_max and varying alpha  ")
z_max_val=max(zms)
for alpha_val in [1.5,2.5,4,8]: 
    pd1={z_max:z_max_val,alpha:alpha_val}
    f_par=f_piece.subs(pd1)
    f_num=lambdify(z,f_par)
    ax3.plot(z_val,f_num(z_val))

eps,z,z_max=symbols('eps,z,z_max')
f=half_saturation(z,eps)
ax4.set_title("half saturation with different epsilons")
z_max_val=max(zms)
for eps_val in [1,100,1000]: 
    pd1={eps:eps_val}
    f_par=f.subs(pd1)
    f_num=lambdify(z,f_par)
    ax4.plot(z_val,f_num(z_val))


z,sfz,y_max=symbols('z,sfz,y_max')
f=atan_ymax(z,y_max,sfz)
ax5.set_title("atan with different sfz")
z_max_val=max(zms)
for sfz_val in [.1,.5,1,2]: 
    pd1={y_max:1,sfz:sfz_val}
    f_par=f.subs(pd1)
    f_num=lambdify(z,f_par,modules=['numpy'])
    ax5.plot(z_val,f_num(z_val))


fig.savefig('limiters.pdf')

