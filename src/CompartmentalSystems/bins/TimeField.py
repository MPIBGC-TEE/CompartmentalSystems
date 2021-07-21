# vim: set ff=unix expandtab ts=4 sw=4:
import numpy as np
from .gv import default_data_type


class TimeField:
    def __init__(self, arr, tss):
        self.arr = arr
        self.tss = tss

    def __getitem__(self, *args, **kwargs):
        # delegate to numpy
        return self.arr.__getitem__(*args, **kwargs)

    def __setitem__(self, key, val):
        # delegate to numpy
        self.arr.__setitem__(key, val)

    @property
    def number_of_Ts_entries(self):
        return self.arr.shape[0]

    @property
    def shape(self):
        # delegate to numpy
        return self.arr.shape

    @property
    def max_Ts(self):
        return self.number_of_Ts_entries * self.tss
