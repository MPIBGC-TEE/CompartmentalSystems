
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty


class ModelRun(metaclass=ABCMeta):
    # abstractmehtods HAVE to be overloaded in the subclasses
    # the decorator should only be used inside a class definition
    @abstractmethod
    def solve(self,alternative_start_values:np.ndarray=None): 
        pass
    
    @abstractmethod
    def acc_external_input_vector(self):
        """
        Accumulated fluxes (flux u integrated over the time step)
        """
        pass

    @abstractmethod
    def acc_internal_flux_matrix(self):
        pass
    
    @abstractmethod
    def acc_external_output_vector(self) :
        pass

