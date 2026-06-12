from typing import Iterator, Dict
from itertools import islice, tee
from .IteratorResult import IteratorResult


class DictResult(IteratorResult):
    # provides iterators that have Dict values
    # with a  __getitem__ method that yields dictionaries of tuples
    # instead of a tuple of dictionaries
    # as the first dimension
    def __init__(self, iterator: Iterator[Dict]):
        self.iterator = iterator

    def __getitem__(self, arg):
        # we copy the iterator to avoid consuming the original we use
        # itertools.tee because it also works for generators (for which
        # copy.copy or copy.deepcopy dont)
        if isinstance(arg, int):
            return super().__getitem__(arg)
        
        elif isinstance(arg,slice):
            res = super().__getitem__(arg) # a tuple of dicts
            if len(res) == 0:
                return res
            else:    
                start_tup = res[0]
                cls=start_tup.__class__
                return cls(
                    {
                        name:
                        tuple([tup[name] for tup in res])
                        for name in start_tup.keys()
                    }
                )
        else:
            raise IndexError(
                """arguments to __getitem__ have to be either
                indeces or slices."""
            )
