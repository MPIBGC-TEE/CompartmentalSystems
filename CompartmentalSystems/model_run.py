
import numpy as np
from abc import ABCMeta, abstractmethod


class ModelRun(metaclass=ABCMeta):
    # abstractmehtods HAVE to be overloaded in the subclasses
    # the decorator should only be used inside a class definition
    @abstractmethod
    def solve(self,alternative_start_values:np.ndarray=None): 
        pass
    
 
