
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty


class ModelRun(metaclass=ABCMeta):
    # abstractmehtods HAVE to be overloaded in the subclasses
    # the decorator should only be used inside a class definition
    @abstractmethod
    def solve(self,alternative_start_values:np.ndarray=None): 
        pass
    
    @abstractproperty
    def external_input_vector(self) :
        pass

    @abstractproperty
    def internal_flux_matrix(self):
        pass
    
    @abstractproperty
    def external_output_vector(self) :
        pass

