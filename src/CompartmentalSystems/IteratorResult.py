from typing import Iterator, Dict
from functools import reduce
from .ArrayDict import ArrayDict
from .helpers_reservoir import average_iterator_from_partitions
from itertools import islice, tee
import numpy as np


class IteratorResult():
    # provides iterators 
    # with a  __getitem__ method 
    def __init__(
        self, 
        iterator: Iterator,
        ):
        self.iterator = iterator


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
        # we copy the iterator to avoid consuming the original we use
        # itertools.tee because it also works for generators (for which
        # copy.copy or copy.deepcopy dont)
        iterator,_ = tee(self.iterator)
        if isinstance(arg, slice):
            start = arg.start
            stop = arg.stop
            if stop is None:  
                raise IndexError(
                    """infinite slices like [:] [:] ,[0:] , [::1] and so on
                    are not supported because the iterator could be
                    infinite. Provide a slice with finite stop like [0:5]."""
                )
            step = arg.step
            return tuple(islice(iterator, start, stop, step))

        elif isinstance(arg, int):
            return tuple(islice(iterator, arg, arg+1, 1))[0]
        else:
            raise IndexError(
                """arguments to __getitem__ have to be either
                indeces or slices."""
            )
