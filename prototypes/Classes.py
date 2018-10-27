from sympy import Matrix
from CompartmentalSystems.helpers_reservoir import  numsol_symbolic_system ,numerical_function_from_expression
from copy import copy,deepcopy
from testinfrastructure.helpers  import pe
class BastinModel():
    # Bastonification of a reservoir model
    def __init__(self,srm,u_expr,z_sym):
        self.z_sym=z_sym
        self.u_expr=u_expr
        crm=deepcopy(srm)
        cof= crm.output_fluxes
        cif= crm.input_fluxes
        # up to now we can only built 
        # single input single output models
        assert(len(cof)==1)
        assert(len(cif)==1)
        F_SD=list(cof.values())[0]
        cif= crm.input_fluxes
        #index of the single input receiving pool
        ii=list(cif.items())[0][0]
        d=cif[ii]
        cif[ii] = u_expr*d
        crm.input_fluxes=cif
    
        self.state_vector=Matrix(list(srm.state_vector)+[z_sym])
        z_dot=F_SD-u_expr*d
        #rhs
        self.F=Matrix(list(crm.F)+[z_dot])
        self.time_symbol=srm.time_symbol



class BastinModelRun():
    def __init__(self,bm,par_dict, start_values, times, func_dict):
        self.bm=bm
        self.par_dict=par_dict
        self.start_values=start_values
        self.times=times
        self.func_dict=func_dict

    def solve(self):
        bm=self.bm
        soln = numsol_symbolic_system(
            bm.state_vector,
            bm.time_symbol,
            bm.F,
            self.par_dict,
            self.func_dict,
            self.start_values, 
            self.times
        )
        return soln
    def phi_num(self,tup):
        bm=self.bm
        u_num= numerical_function_from_expression(bm.u_expr,tup,self.par_dict,self.func_dict)
        return u_num
