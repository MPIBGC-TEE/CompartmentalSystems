import numpy as np
from collections import OrderedDict

def averaged_1d_array(arr, partitions):
    """this function also works for multidimensional arrays
    It assumes that the first dimension is time/iteration
    """
    def f(p):
        start, stop = p
        sub_arr = arr[start : stop]
        #from IPython import embed; embed()
        return sub_arr.sum(axis=0)/(stop - start)

    return np.array([f(p) for p in partitions])


class ArrayDict(OrderedDict):

    def __getattribute__(self, name):
        # make the dictionary content available via the . operator
        try:
            return self[name]
        except KeyError:
            return super().__getattribute__(name)

    @property
    def _fields(self):
        return self.keys()

    def averaged_values(self, partitions):
        return self.__class__(
            {
                name:
                averaged_1d_array(self.__getattribute__(name), partitions)
                for name in self._fields
            }
        )

    def __add__(self, other):
        """overload + which is useful for averaging"""
        return self.__class__(
            {
                name:
                self.__getattribute__(name) + other.__getattribute__(name)
                for name in self._fields
            }
        )

    def __truediv__(self, number):
        """overload / for scalars  which is useful for averaging"""

        return self.__class__(
            { 
                name: 
                self.__getattribute__(name) / number 
                for name in self._fields
            }
        )

    def __eq__(self, other):
        """overload == which is useful for tests"""
        return np.all([
            self.__getattribute__(name) == other.__getattribute__(name)
            for name in self._fields
        ])
