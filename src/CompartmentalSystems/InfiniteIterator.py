from typing import Callable, List, Tuple, Dict, TypeVar #, Self
import numpy as np
from copy import copy
from functools import reduce
from itertools import islice
import inspect
T=TypeVar('T')
class InfiniteIterator:
    """the equivalent to an InitialValueProblem for a discrete dynamical system
    If max_iter is not set the Iterator will never stop, which is useful to 
    represent an autonomous dynamical system.
    In case that some of the fucntions rely on data that are only
    available for a limited number of iterations  (typically a timeline)
    max_iter can be set to avoid indexing errors.
    """
    def __init__(
        self,
        start_value,
        func: Callable[[int,T],T],
        max_iter=None 
    ):  # ,n):
        self.start_value = start_value
        self.func = func

        self.cur = start_value
        self.pos = 0
        self.max_iter = max_iter

    def reset(self):
        self.cur = self.start_value
        self.pos = 0

    def __iter__(self):
        # return a fresh instance that starts from the first step)
        # This is important for the __getitem__ to work
        # as expected and not have side effects
        # for
        # res1=itr[0:10]
        # res2=itr[0:10]

        #c = self.__class__(self.start_value, self.func)
        #return c
        c = copy(self)
        c.reset()
        return c

    def __next__(self):
        # print(self.pos, self.cur)
        if self.max_iter is None or self.pos < self.max_iter: 
            val = self.cur 
            self.pos += 1
            self.cur = self.func(self.pos, val) 
            return val
        else:
            raise StopIteration()



