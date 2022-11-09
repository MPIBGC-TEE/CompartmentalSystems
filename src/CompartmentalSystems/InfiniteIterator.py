from typing import Callable, List, Tuple, Dict, TypeVar #, Self
import numpy as np
from copy import copy
from functools import reduce
from itertools import islice
import inspect
T=TypeVar('T')
class InfiniteIterator:
    """the equivalent to an InitialValueProblem for a discrete dynamical system
    """
    def __init__(
        self,
        start_value,
        func: Callable[[int,T],T]
    ):  # ,n):
        self.start_value = start_value
        self.func = func

        self.cur = start_value
        self.pos = 0

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
        val = self.cur 
        self.cur = self.func(self.pos, val) 
        self.pos += 1
        return val
        # raise StopIteration()

    # @lru_cache
    def value_at(self, it_max):
        I = self.__iter__()

        def f_i(acc, i):
            return I.__next__()

        return reduce(f_i, range(it_max), I.start_value)

    def __getitem__(self, arg):
        # this functions implements the python index notation itr[start:stop:step]
        # fixme mm 4-26-2022
        # we could use the cache for value_at if we dont use isslice
        # it behaves like the [] on a tuple 
        # (1,2,3)[0:0] = ()
        # (1,2,3)[0:1] = (1)
        # (1,2,3)[1:3] = (2,3)
        # 
        # but it will not allow negative indices since not max_iter is given
        if isinstance(arg, slice):
            start = arg.start
            stop = arg.stop
            step = arg.step
            return tuple(islice(self, start, stop, step))

        elif isinstance(arg, int):
            #return (self.value_at(it_max=arg),)
            return (self.value_at(it_max=arg))
        else:
            raise IndexError(
                """arguments to __getitem__ have to be either
                indeces or slices."""
            )
