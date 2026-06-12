from sympy import atan ,pi,lambdify, symbols ,solve,Piecewise
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
