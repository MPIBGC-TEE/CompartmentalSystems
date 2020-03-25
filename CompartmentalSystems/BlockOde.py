from typing import Callable,Iterable,Union,Optional,List,Tuple 
#from scipy.integrate import solve_ivp
import numpy as np
from .BlockIvp import BlockIvp
class BlockOde:
    """
    Helper class to build a system from functions that operate on blocks of the state_variables.
    """

    def __init__(
         self
        ,time_str               : str
        ,block_names_and_shapes : List[Tuple[str,Tuple[int]]] 
        ,functions              : List[ Tuple[Callable,List[str]]]):
        
        self.time_str       = time_str
        self.block_names_and_shapes    = block_names_and_shapes
        self.functions    = functions

    def check_block_exists(self,block_name):
        if not(block_name in set(self.block_names).union(self.time_str)):
            raise Exception("There is no block with this name")

    def blockIvp(
            self,
            start_blocks   : List[ Tuple[str,np.ndarray] ]
        )->BlockIvp:
        """
        Extends the system to an initial value problem by adding startvalues for the blocks 
        It checks that the names of the names of the blocks coincide.
        """
        for ind,tup in enumerate(self.block_names_and_shapes):
            name, shape=tup
            assert( name==start_blocks[ind][0])
            assert( shape==start_blocks[ind][1].shape)
        return BlockIvp(
            self.time_str, 
            start_blocks,   
            self.functions      
        ) 
                
