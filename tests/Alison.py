#from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel
from sympy.printing import pprint
from sympy import var 
var("lambda_1 lambda_2 t")
M=Matrix([[lambda_1,0,0],[1,lambda_1,0],[0,0,lambda_2]])
pprint(M.eigenvects())
var("t")
pprint(M.exp())
M.jordan_form()
pprint((M*t).exp())
T=Matrix([[1,2,0],[2,1,0],[1,1,1]])
T**-1
A=T*M*T**-1

