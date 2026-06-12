from typing import Iterator
from functools import reduce
from .ArrayDict import ArrayDict
from .DictResult import DictResult
from .helpers_reservoir import average_iterator_from_partitions
from itertools import islice, tee
import numpy as np


class ArrayDictResult(DictResult):
    # provides iterators that have ArrayDict values
    # with a  __getitem__ method that yields arrays with the iteration
    # as the first dimension
    def __init__(self, iterator: Iterator[ArrayDict]):
        self.iterator = iterator

    def __getitem__(self, arg):
        # we copy the iterator to avoid consuming the original we use
        # itertools.tee because it also works for generators (for which
        # copy.copy or copy.deepcopy dont)
        iterator,_=tee(self.iterator)
        if isinstance(arg, int):
            return super().__getitem__(arg)
            #return tuple(islice(iterator, arg, arg+1, 1))[0]
        elif isinstance(arg,slice):
            #start = arg.start
            #stop = arg.stop
            #step = arg.step
            #res = tuple(islice(iterator, start, stop, step))
            res = super().__getitem__(arg)
            if len(res) == 0:
                return res
            else:    
                #from IPython import embed; embed()
                cls=res.__class__
                return cls(
                    {
                        name:
                        np.stack(tup)
                        for name , tup in res.items()
                    }
                )
        else:
            raise IndexError(
                """arguments to __getitem__ have to be either
                indeces or slices."""
            )
    
    def averaged_values(self, partitions):
        try:
            max_ind = len(partitions)
        except TypeError:
            raise TypeError("only finite partitions are allowed.")
        
        iterator,_ = tee(self.iterator)
        av_it = average_iterator_from_partitions(iterator, partitions) 
        return self.__class__(av_it)[:len(partitions)]

