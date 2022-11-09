from typing import Callable, List, Tuple, Dict, TypeVar #, Self
import numpy as np
from copy import copy
from functools import reduce
from itertools import islice
#from collections import  OrderedDict
import inspect
from .InfiniteIterator import InfiniteIterator


class BlockDictIterator(InfiniteIterator):
    def __init__(
            self, #: Self,
            iteration_str: str,
            start_seed_dict: Dict,
            present_step_funcs: Dict[str,Callable],
            next_step_funcs: Dict[str,Callable]
    ):#-> Self:
        func_dict = {**present_step_funcs, **next_step_funcs}
        arglists  = {
            target_var: inspect.getfullargspec(fun).args
            for target_var,fun in func_dict.items()
        }    
        def apply(acc: Dict, k: str) -> Dict:
            arg_values = [acc[cl] for cl in arglists[k]]
            res = copy(acc)
            res.update({k: func_dict[k](*arg_values)})
            return res
        
        def complete(seed_dict):
            return reduce(apply, present_step_funcs.keys(), seed_dict) 
        
        # produce the first value for the general iterator
        # by calling update we preserve the dict subclass 
        seed_value_dict_0 = copy(start_seed_dict)
        seed_value_dict_0.update({iteration_str: 0, })
            
        
        # create the function for the general iterator
        def f(i,present_val_dict):
            # make the new iteration number available for the functions
            # that compute the new seed
            present_val_dict[iteration_str]=i
            new_dict = reduce(apply, next_step_funcs.keys(), present_val_dict) 
            # compute the extended values from the new seed 
            return complete(new_dict)

        super().__init__(
            start_value=complete(seed_value_dict_0),
            func=f
        )
        
