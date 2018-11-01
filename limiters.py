from sympy import lambdify
from sympy import lambdify, symbols
from sympy import solve
import numpy as np
y,y_max,z,z0,alpha =symbols('y,y_max,z,z0,alpha')
a0,a1,a2,a3=symbols('a0,a1,a2,a3')
f=a3*z**3+a2*z**2+a1*z+a0
a0_val=solve(f.subs({z:0}),a0)[0]#=0
a1_val=solve((f.diff(z)-1).subs({z:0,a0:a0_val}),a1)[0]#=1

par_dict=solve([
    f.subs({z:0}), #f(0)=0
    (f.diff(z)-1).subs({z:0}),#f'(0)=1
    (f-1).subs({z:z0}), #f(z0)=1
    f.diff(z).subs({z:z0}), #f'(z0)=0
    ],
    a0,a1,a2,a3)
f_ai=f.subs(par_dict)
f_par=f_ai.subs({z0:50})
f_num=lambdify(z,f_par,modules='numpy')
print(f_num(np.linspace(1,50,100)))
