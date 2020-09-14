# vim: set ff=unix expandtab ts=4 sw=4:
import numpy as np

class FieldsPerTimeStep(list):
    def __init__(self,listOfTimeFields,start):
        super().__init__(listOfTimeFields)
        self.start=start

    
    @property
    def tss(self):
        return(self[0].tss)
    
    @property
    def times(self):
        return(np.arange(len(self))*self.tss+self.start)
    
    @property
    def t_min(self):
        return(min(self.times))

    @property
    def t_max(self):
        return(max(self.times))
