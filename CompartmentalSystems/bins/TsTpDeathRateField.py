# vim: set ff=unix expandtab ts=4 sw=4:
from .TsTpField import TsTpField
class TsTpDeathRateField(TsTpField):
    def __init__(self,arr,tss):
        if arr.max() <=1 and arr.min()>=0:
            super().__init__(arr,tss)
        else:
            raise(Exception("Death rates have to have values in [0,1]"))

