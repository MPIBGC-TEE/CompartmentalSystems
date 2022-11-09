import numpy as np
from collections import OrderedDict


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

    def averages(self, partitions):
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
        return np.all(
            self.__getattribute__(name) == other.__getattribute__(name)
            for name in self._fields
        )
