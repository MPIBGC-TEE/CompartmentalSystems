from typing import Callable,Iterable,Union,Optional,List,Tuple 
from .helpers_reservoir import block_rhs
import numpy as np
class BlockIvp:
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
        self.rhs=block_rhs(
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

