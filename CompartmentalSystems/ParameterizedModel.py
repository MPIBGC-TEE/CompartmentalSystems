import .SmoothReservoirModel
from BlockIvp import BlockIvp
class ParameterizedSmootReservoirModel:
    # think of the instances as a SymbolicModel applied to a parameter set
    # with this combination some things can allready be computed numerically
    # (e.g. steady states)
    # but others can not (like for instance the solution vector which additionally
    # needs a start_vector and a time grid)
    # to do:
    # Check if similar class could be a build for a discrete Model and 
    # a bin model
    # and organize the similarities in a possible interface or abstract 
    # Base class
    def __init__(self,srm:SmoothReservoirModel,parameter_dict,funct_dict)
        self.srm=srm
        self.parameter_dict=parameter_dict

    def x_phi_ivp(self,start_x,x_bloc_name='x',phi_block_name='phi'):
        parameter_dict  = self.parameter_dict
        funct_dict      = self.func_dict
        #B=srm.compartmental_matrix
        nr_pools=srm.nr_pools
        nq=nr_pools*nr_pools
        #tup=(t,)+tuple(srm.state_vector)
        sol_rhs=numerical_rhs2(
             srm.state_vector
            ,srm.time_symbol
            ,srm.F
            ,parameter_dict
            ,func_dict
        )
        # for comparison solve the original system 
        
        B_sym=srm.compartmental_matrix
        ## now use the interpolation function to compute B in an alternative way.
        #symbolic_sol_funcs = {sv: Function(sv.name + '_sol')                        for sv in svec}
        #sol_func_exprs =     {sv: symbolic_sol_funcs[sv](srm.time_symbol)           for sv in svec}# To F add the (t) 
        #def func_maker(pool):
        #    def func(t):
        #        return sol.sol(t)[pool]
        #
        #    return(func)
        #
        #sol_dict = {symbolic_sol_funcs[svec[pool]]:func_maker(pool) for pool in range(srm.nr_pools)}
        #lin_func_dict=copy(func_dict)
        #lin_func_dict.update(sol_dict)
        #
        #linearized_B = B_sym.subs(sol_func_exprs)
        #
        tup=(t,)+tuple(srm.state_vector)
        B_func=numerical_function_from_expression(B_sym,tup,parameter_dict,func_dict)
        x_i_start=0
        x_i_end=nr_pools
        Phi_1d_i_start=x_i_end
        Phi_1d_i_end=(nr_pools+1)*nr_pools
        #
        def Phi_rhs(t,x,Phi_1d):
            B=B_func(t,*x)
            #Phi_cols=[Phi_1d[i*nr_pools:(i+1)*nr_pools] for i in range(nr_pools)]
            #Phi_ress=[np.matmul(B,pc) for pc in Phi_cols]
            #return np.stack([np.matmul(B,pc) for pc in Phi_cols]).flatten()
            return np.matmul(B,Phi_1d.reshape(nr_pools,nr_pools)).flatten()

        #create the additional startvecot for the components of Phi
        start_Phi_1d=np.identity(nr_pools).flatten()

        block_ivp=BlockIvp(
            time_str='t'
            ,start_blocks  = [('sol',start_x),('Phi_1d',start_Phi_1d)]
            ,functions = [
                 (sol_rhs,['t','sol'])
                ,(Phi_rhs,['t','sol','Phi_1d'])
             ]
        )
        return block_ivp

    def x_ivp(self,start_x):    
        parameter_dict  = self.parameter_dict
        funct_dict      = self.func_dict
