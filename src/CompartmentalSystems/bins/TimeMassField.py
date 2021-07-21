# vim: set ff=unix expandtab ts=4 sw=4:
import numpy as np
from .TimeField import TimeField


class TimeMassField(TimeField):
    def __add__(self, other):
        if not (isinstance(other, self.__class__)):
            raise (
                Exception(
                    "The two operands must be both children of " + str(self__class__)
                )
            )
        arr = self.arr + other.arr
        obj = self.__new__(self.__class__)
        obj.__init__(arr, self.tss)
        return obj
