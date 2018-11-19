from sympy import Matrix
from numbers import Number
from CompartmentalSystems.helpers_reservoir import  numsol_symbolic_system ,numerical_function_from_expression ,make_cut_func_set,f_of_t_maker
from copy import copy,deepcopy
from testinfrastructure.helpers  import pe
from scipy.interpolate import interp1d, UnivariateSpline
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
        self.srm=crm



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
    def _flux_funcs(self, expr_dict):
        bm = self.bm
        srm=bm.srm
        sol_funcs = self.sol_funcs()
        flux_funcs = {}
        tup = tuple(bm.state_vector) + (bm.time_symbol,)

        for key, expr in expr_dict.items():
            if isinstance(expr,Number):
                # if expr is a number like 5.1 lambdify does not create a vectorized function 
                # so the output is always a number and not an array with identical exprs which is a problem in plots
                def expr_func(arg_arr):
                    return expr*np.ones_like(arg_arr)
                flux_funcs[key]=expr_func

            else:
                ol=numerical_function_from_expression(expr,tup,self.par_dict,self.func_dict)
                flux_funcs[key] = f_of_t_maker(sol_funcs, ol)

        return flux_funcs
        
        
    def sol_funcs(self):
        """Return linearly interpolated solution functions.

        Returns:
            Python function ``f``: ``f(times)`` returns a numpy.array containing the
            pool contents at times ``times``.
        """
        sol = self.solve()
        times = self.times
        sol_funcs = []
        for i in range(sol.shape[1]):
            sol_inter = interp1d(times, sol[:,i])
            sol_funcs.append(sol_inter)

        return sol_funcs
    
    def external_input_flux_funcs(self):
        """Return a dictionary of the external input fluxes.
        
        The resulting functions base on sol_funcs and are linear interpolations.

        Returns:
            dict: ``{key: func}`` with ``key`` representing the pool which 
            receives the input and ``func`` a function of time that returns 
            a ``float``.
        """
        return self._flux_funcs(self.bm.srm.input_fluxes)

    def internal_flux_funcs(self):
        """Return a dictionary of the internal fluxes.
        
        The resulting functions base on sol_funcs and are linear interpolations.

        Returns:
            dict: ``{key: func}`` with ``key=(pool_from, pool_to)`` representing
            the pools involved and ``func`` a function of time that returns 
            a ``float``.
        """
        return self._flux_funcs(self.bm.srm.internal_fluxes)

    def output_flux_funcs(self):
        """Return a dictionary of the external output fluxes.
        
        The resulting functions base on sol_funcs and are linear interpolations.

        Returns:
            dict: ``{key: func}`` with ``key`` representing the pool from which
            the output comes and ``func`` a function of time that returns a 
            ``float``.
        """
        return self._flux_funcs(self.bm.srm.output_fluxes)
    

    def phi_num(self,tup):
        bm=self.bm
        u_num= numerical_function_from_expression(bm.u_expr,tup,self.par_dict,self.func_dict)
        return u_num
