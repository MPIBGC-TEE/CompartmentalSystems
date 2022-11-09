import numpy as np
from collections import OrderedDict
from .BlockDictIterator import BlockDictIterator
from .ArrayDict import ArrayDict 

class BlockArrayIterator(BlockDictIterator):
    # overload methods specific to blocks of arrays
    def __getitem__(self, arg):
        # fixme mm 11-07-2022
        # at the moment we call the [] method of the superclass
        # which returns a tuple of dict like objects with arrays inside
        # We could speed this up by allocating the result arrays 
        # once (since we know the slice) and filling them in a timestep loop
        # using the islice function directly
        # this would also probably save half of the ram (since the final array and ONE iterator dict)
        # coexist at the same time
        res = super().__getitem__(arg)
        if isinstance(arg, int):
            return res
        elif isinstance(arg,slice):
            start_tup = res[0]
            cls=start_tup.__class__
            return cls(
                {
                    name:
                    np.stack([tup[name] for tup in res])
                    for name in start_tup.keys()
                }
        )
        else:
            raise IndexError(
                """arguments to __getitem__ have to be either
                indeces or slices."""
            )

    
