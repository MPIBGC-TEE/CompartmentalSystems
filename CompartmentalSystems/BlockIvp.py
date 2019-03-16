from typing import Callable,Iterable,Union,Optional,List,Tuple 
from scipy.integrate import solve_ivp
import numpy as np
class BlockIvp:
    """Helper class to build initial value systems from functions that operate on blocks of the state_variables"""
    @classmethod
    def build_rhs(
             cls
            ,time_str  : str
            ,X_blocks  : List[ Tuple[str,int] ]
            ,functions : List[ Tuple[Callable,List[str]]]
        )->Callable[[np.double,np.ndarray],np.ndarray]:
        """
        The function returns a function dot_X=f(t,X) suitable as the righthandside 
        for the ode solver scipy.solve_ivp from a collection of vector valued
        functions that compute blocks of dot_X from time and blocks of X 
        rather than from single equations.
    
        A special application is the creation of block triangular systems, to 
        integrate variables whose time derivative depend on the solution
        of an original system instantaniously along with it.
        
        Assume that
        X_1(t) is the solution of the initial value problem (ivp)
        
        ivp_1:
        dot_X_1=f_1(t,X) ,X_1(t_0) 
    
        and X_2(t) the solution of another ivp 
    
        ivp_2:
        dot_X_2=f_2(t,X_1,X_2) ,X_2(t_0) whose righthand side depends on x_1
        
        Then we can obtain the solution of both ivps simultaniously by
        combining the them into one.
        
        (dot_X_1, dox_X_2)^t = (f_1(t,X_1),f_2(t,X_1,X_2))^t
    
        For n instead of 2 variables one has:  
        (dot_X_1, dox_X_2,...,dot_X_n)^t = (f_1(t,X_1),f_2(t,X_1,X_2),...,f_n(t,X_1,...X_n))^t
        
        For a full lower triangular system  the block derivative dot_X_i depend on t,
        and ALL the blocks X_1,...,X_i but often they will only depend on 
        SOME of the previous blocks so that f_m has a considerably smaller argument list.
    
        This function therefore allows to specify WHICH blocks the f_i depend on.
        Consider the following 7 dimensional block diagonal example:
    
        b_s=block_rhs(
             time_str='t'
            ,X_blocks=[('X1',5),('X2',2)]
            ,functions=[
                 ((lambda x   : x*2 ),  ['X1']    )
                ,((lambda t,x : t*x ),  ['t' ,'X2'])
             ])   
        
    
        The first argument 'time_str' denotes the alias for the t argument to be used
        later in the signature of the blockfunctions.
        The second argument 'X_blocks' describes the decomposition of X into blocks
        by a list of tuples of the form ('Name',size)
        The third argument 'functions' is a list of tuples of the function itself
        and the list of the names of its block arguments as specified in the
        'X_blocks' argument. 
        Order is important for the 'X_blocks' and the 'functions'
        It is assumed that the i-th function computes the derivative of the i-th block.
        The names of the blocks itself are arbitrary and have no meaning apart from
        their correspondence in the X_blocks and functions argument.
        """
        block_names=[t[0] for t in X_blocks]
        dims=[t[1] for t in X_blocks]
        nb=len(dims)
        strArgLists=[f[1] for f in functions]
        # make sure that all argument lists are really lists
        assert(all([isinstance(l,list) for l in strArgLists])) 
        # make sure that the function argument lists do not contain block names
        # that are not mentioned in the Xblocks argument
        flatArgList=[arg for argList in strArgLists for arg in argList]
        assert(set(flatArgList).issubset(block_names+[time_str]))
        
    
        # first compute the indices of block boundaries in X by summing the dimensions 
        # of the blocks
        indices=[0]+[ sum(dims[:(i+1)]) for i in range(nb)]
        def rhs(t,X):
            blockDict={block_names[i]: X[indices[i]:indices[i+1]] for i in range(nb)}
            #pe('blockDict',locals())
            blockDict[time_str]=t
            arg_lists=[ [blockDict[k] for k in f[1]] for f in functions]
            blockResults=[ functions[i][0]( *arg_lists[i] )for i in range(nb)]
            #pe('blockResults',locals())
            return np.concatenate(blockResults).reshape(X.shape)
        
        return rhs

    def __init__(
         self
        ,time_str       : str
        ,start_blocks   : List[ Tuple[str,np.ndarray] ]
        ,functions      : List[ Tuple[Callable,List[str]]]):
        
        self.time_str=time_str
        names           =[sb[0]   for sb in start_blocks]
        start_arrays    =[sb[1]   for sb in start_blocks]
        shapes          =[a.shape  for a in start_arrays]
        #assert that we have only vectors or n,1 arrays as startvalues
        assert( all([len(s)==1 or (len(s)==2 and s[1]==1) for s in shapes]))
        #
        dims=[s[0] for s in shapes]
        nb=len(dims)
        r=range(nb)
        X_blocks=[(names[i],dims[i]) for i in r]
        indices=[0]+[ sum(dims[:(i+1)]) for i in r]
        self.index_dict={names[i]:(indices[i],indices[i+1]) for i in r}
        self.rhs=self.build_rhs(
             time_str  = time_str
            ,X_blocks  = X_blocks
            ,functions = functions
        )
        self.start_vec=np.concatenate(start_arrays)
        self._cache=dict()
        print(self.rhs(0,self.start_vec))
        
    def solve(self,t_span,dense_output=False,**kwargs):
        # this is just a caching proxy for scypy.solve_ivp
        # remember the times for the solution
        if not(isinstance(t_span,tuple)):
            raise Exception('''
            scipy.solve_ivp actually allows a list for t_span, but we insist 
            that it should be an (immutable) tuple, since we want to cache the solution 
            and want to use t_span as a hash.''')
        cache_key=(t_span,dense_output)
        if cache_key in self._cache.keys():
            # caching can be made much more sophisticated by
            # starting at the end of previos solution with
            # the same start times and a smaller t_end
            return self._cache[cache_key]
        #
        if 'vectorized' in kwargs.keys():
            del(kwargs['vectorized'])
            print('''The vectorized flag is forbidden for $c 
            since we rely on decomposing the argument vector'''.format(s=self.__class__))
        sol=solve_ivp(
             fun=self.rhs
            ,y0=self.start_vec
            ,t_span=t_span
            ,dense_output=dense_output
            ,**kwargs
        )
        self._cache[cache_key]=sol
        return sol
    def check_block_exists(self,block_name):
        if not(block_name in set(self.index_dict.keys()).union(self.time_str)):
            raise Exception("There is no block with this name")

    def get_values(self,block_name,t_span,**kwargs):
        self.check_block_exists(block_name)
        sol=self.solve(t_span=t_span,**kwargs)
        if block_name==self.time_str:
            return sol.t
        
        lower,upper=self.index_dict[block_name] 
        return sol.y[lower:upper,:]
        
    def get_function(self,block_name,t_span,**kwargs):
        self.check_block_exists(block_name)
        if block_name==self.time_str:
            print("""
            warning:
            $s interpolated with respect to s$ is probably an accident...
            """.format(s=self.time_str))
            # this is silly since it means somebody asked for the interpolation of t with respect to t
            # so we give back the identiy
            return lambda t:t

        lower,upper=self.index_dict[block_name] 
        complete_sol_func=self.solve(t_span=t_span,dense_output=True).sol
        def block(t):
            return complete_sol_func(t)[lower:upper]
        
        return block

