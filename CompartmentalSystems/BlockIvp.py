from typing import Callable,Iterable,Union,Optional,List,Tuple 
from scipy.integrate import solve_ivp
import numpy as np

def custom_solve_ivp(fun, t_span, y0 , **kwargs):
    dense_output = kwargs.get('dense_output', False)
    method = kwargs.get('method', 'Radau')
    t_min, t_max = t_span
    
    def f(method):
        return solve_ivp(
             fun=fun
            ,t_span=t_span
            ,y0=y0
            ,method=method
            #
            # prevent the solver from overreaching (scipy bug)
            ,first_step=(t_max-t_min)/2 if t_max != t_min else None
            #,max_step=1
            ,**kwargs
        )
    sol_obj = f(method=method)
    #sol_obj = f(method='RK45')
    #sol_obj = f(method='RK23')
    #sol_obj = f(method='Radau')
    #sol_obj = f(method='BDF')
    #sol_obj = f(method='LSODA')

    #print('status', sol_obj.status)
    #print('message', sol_obj.message)
    #print('success', sol_obj.success)
    return sol_obj

class BlockIvp:
    """Helper class to build initial value systems from functions that operate on blocks of the state_variables"""
    @classmethod
    def build_rhs(
             cls
            ,time_str  : str
            #,X_blocks  : List[ Tuple[str,int] ]
            ,start_blocks : List[ Tuple[str,np.ndarray] ]
            ,functions : List[ Tuple[Callable,List[str]]]
        )->Callable[[np.double,np.ndarray],np.ndarray]:
        """
        The function returns a function dot_X=f(t,X) suitable as the right-hand side 
        for the ode solver scipy.solve_ivp from a collection of array valued
        functions that compute blocks of dot_X from time and blocks of X 
        rather than from single equations.
    
        A special application is the creation of block triangular systems, to 
        integrate variables whose time derivative depends on the solution
        of an original system instantaneously along with it.
        
        Assume that
        X_1(t) is the solution of the initial value problem (ivp)
        
        ivp_1:
        dot_X_1=f_1(t,X), X_1(t_0) 
    
        and X_2(t) the solution of another ivp 
    
        ivp_2:
        dot_X_2=f_2(t,X_1,X_2), X_2(t_0) whose right-hand side depends on x_1.
        
        Then we can obtain the solution of both ivps simultaneously by
        combining them into one.
        
        (dot_X_1, dox_X_2)^t = (f_1(t,X_1),f_2(t,X_1,X_2))^t
    
        For n instead of 2 variables one has:  
        (dot_X_1, dot_X_2,..., dot_X_n)^t = (f_1(t,X_1), f_2(t,X_1,X_2),..., f_n(t,X_1,...X_n))^t
        
        For a full lower triangular system the block derivative dot_X_i depends on t and ALL the blocks X_1,...,X_i  
        but often it will only depend on 
        SOME of the previous blocks so that f_i has a considerably smaller argument list.
    
        This function therefore allows to specify WHICH blocks the f_i depend on.
        Consider the following 5+2*2 = 9 -dimensional block diagonal example:
    
        b_s=block_rhs(
             time_str='t'
            ,start_blocks=[('X1',np.ones((5,1)),('X2',np.ones((2,2)))]
            ,functions=[
                 ((lambda x   : x*2 ),  ['X1']    )
                ,((lambda t,x : t*x ),  ['t' ,'X2'])
             ])   
        
    
        The first argument 'time_str' denotes the alias for the t argument to be used
        later in the signature of the block functions.
        The second argument 'start_blocks' describes the decomposition of X into blocks
        by a list of tuples of the form ('Name',array).
        The third argument 'functions' is a list of tuples of the function itself
        and the list of the names of its block arguments as specified in the
        'start_blocks' argument. 
        Order is important for the 'start_blocks' and the 'functions'.
        It is assumed that the i-th function computes the derivative of the i-th block.
        The names of the blocks itself are arbitrary and have no meaning apart from
        their correspondence in the start_blocks and functions argument.
        """

        start_block_dict={ t[0]:t[1] for t in start_blocks }
        block_names=[t[0] for t in start_blocks]
        sizes=[t[1].size for t in start_blocks]
        nb=len(sizes)
        strArgLists=[f[1] for f in functions]
        # make sure that all argument lists are really lists
        assert(all([isinstance(l,list) for l in strArgLists])) 
        # make sure that the function argument lists do not contain block names
        # that are not mentioned in the Xblocks argument
        flatArgList=[arg for argList in strArgLists for arg in argList]
        assert(set(flatArgList).issubset(block_names+[time_str]))
        
    
        # first compute the indices of block boundaries in X by summing the dimensions 
        # of the blocks
        indices=np.array([0]+[ sum(sizes[:(i+1)]) for i in range(nb)])
        #vecBlockDict={block_names[i]: X[indices[i]:indices[i+1]] for i in range(nb)}
        #blockDict={name:vecBlock.reshape(start_block_dict[name].shape) for name,vecBlock in vecBlockDict.items()}
        #blockDict[time_str]=t
        #arg_lists=[ [blockDict[name] for name in f[1]] for f in functions]
        #def rhs(t,X):
        #    Y=tuple(t,X)
        #    for blnr in range(nb):
        #        x=X[indices[blnr]:indices[blnr+1]]
        #        f=functions[blnr][0]
        #        f_arglist=[Y[block_start,block_end] for block_start,block_endin arg_indeces[blnr]]
        #        res=f(arglist) 


        #    vecResults=[ functions[i][0]( *arg_lists[i] ).flatten()  for i in range(nb)]
        #    return np.concatenate(vecResults)#.reshape(X.shape)
        #
        #return rhs
        indices=[0]+[ sum(sizes[:(i+1)]) for i in range(nb)]
        def rhs(t,X):
            vecBlockDict={block_names[i]: X[indices[i]:indices[i+1]] for i in range(nb)}
            blockDict={name:vecBlock.reshape(start_block_dict[name].shape) for name,vecBlock in vecBlockDict.items()}
            blockDict[time_str]=t

            arg_lists=[ [blockDict[name] for name in f[1]] for f in functions]
            vecResults=[ functions[i][0]( *arg_lists[i] ).flatten()  for i in range(nb)]
            return np.concatenate(vecResults)#.reshape(X.shape)
        
        return rhs

    def __init__(
             self
            ,time_str       : str
            ,start_blocks   : List[ Tuple[str,np.ndarray] ]
            ,functions      : List[ Tuple[Callable,List[str]]]
        ):
        
        self.array_dict={tup[0]:tup[1] for tup in start_blocks}
        self.time_str=time_str
        names           =[sb[0]   for sb in start_blocks]
        start_arrays    =[sb[1]   for sb in start_blocks]
        #
        sizes=[a.size for a in start_arrays]
        nb=len(sizes)
        r=range(nb)
        #X_blocks=[(names[i],sizes[i]) for i in r]
        indices=[0]+[ sum(sizes[:(i+1)]) for i in r]
        self.index_dict={names[i]:(indices[i],indices[i+1]) for i in r}
        self.rhs=self.build_rhs(
             time_str  = time_str
            ,start_blocks= start_blocks
            ,functions = functions
        )
        self.start_vec=np.concatenate([ a.flatten() for a in start_arrays])
        self._cache=dict()
        

    def check_block_exists(self,block_name):
        if not(block_name in set(self.index_dict.keys()).union(self.time_str)):
            raise Exception("There is no block with this name")

        
    def block_solve(self, t_span, first_step=None, **kwargs):
        # fixme:
        # This will be the new solve 
        # returns a dict of the block solutions (either values or functions depending on the "dense_output" keyword 
        # of solve_ivp
        sol_obj = custom_solve_ivp(
                 fun          = self.rhs
                ,t_span       = t_span
                ,y0           = self.start_vec
                ,dense_output = False
                ,**kwargs
            )

        def block_sol(block_name):
            start_array = self.array_dict[block_name]
            lower, upper = self.index_dict[block_name] 
            time_dim_size = sol_obj.y.shape[-1]
            tmp = sol_obj.y[lower:upper,:].reshape(
                    start_array.shape+(time_dim_size,)
            )        
            # solve_ivp returns an array that has time as the LAST dimension
            # but our code usually expects it as FIRST dimension 
            # Therefore we move the last axis to the first position
            return np.moveaxis(tmp, -1, 0)

        block_names = self.index_dict.keys()
        block_sols = {block_name: block_sol(block_name) for block_name in block_names}
        #print(block_sols)
        return block_sols

    def block_solve_functions(self, t_span, first_step=None, **kwargs):
        sol_obj = custom_solve_ivp(
                 fun          = self.rhs
                ,t_span       = t_span
                ,y0           = self.start_vec
                ,dense_output = True
                ,**kwargs
            )
        def block_sol(block_name):
            start_array = self.array_dict[block_name]
            lower, upper = self.index_dict[block_name] 
            def func(times):
                tmp =  sol_obj.sol(times)[lower:upper]
                if isinstance(times, np.ndarray): 
                    res = tmp.reshape(
                            (start_array.shape+(len(times),))
                    ) 
                    return np.moveaxis(res, -1, 0)
                else:
                    return tmp.reshape(start_array.shape)
            
            # solve_ivp returns an array that has time as the LAST dimension
            # but our code usually expects it as FIRST dimension 
            # Therefore we move the last axis to the first position
            return func

        block_names = self.index_dict.keys()
        block_sols = {block_name: block_sol(block_name) for block_name in block_names}
        #print(block_sols)
        return block_sols


