# vim: set ff=unix expandtab ts=4 sw=4:
import numpy as np
from .TsTpField import TsTpField
from .TimeMassField import TimeMassField
from .TsMassField import TsMassField
from .TsTpDeathRateField import TsTpDeathRateField
from .gv import default_data_type

class TsTpMassField(TimeMassField,TsTpField):
    #The class represents a Ts Tp rectangle 
    def loss(self,eta_dist):
        if not(isinstance(eta_dist,TsTpDeathRateField)):
            raise(Exception("argument must me a deathrate"))

        if eta_dist.tss!=self.tss:
            raise(Exception("time step size not compatible"))

        arr=self.arr*eta_dist.arr   
        obj=self.__new__(self.__class__)# create a new instance
        obj.__init__(arr,self.tss)
        return(obj)
  
  #def __sub__(self,other):
    #    if not(isinstance(other,self.__class__)):
    #        raise(Exception("The two operands must be both children of "+str(self__class__)))    
    #    arr=self.arr-other.arr
    #    obj=self.__new__(self.__class__)
    #    obj.__init__(arr,self.tss)
    #    return(obj)
    
    def remove_loss(self,loss):
        if not(isinstance(loss,self.__class__)):
            raise(Exception("The two operands must be both children of "+str(self__class__)))   
        self.arr+=-loss.arr

    def sum_over_all_pool_ages(self):        
        #(first index=SystemAge,second_index PoolAge)
        return(TsMassField(np.sum(self.arr,1),self.tss))
    
    def receive(self,age_shifted_gain):
        if not(isinstance(age_shifted_gain,TsMassField)):
            raise(Exception("argument must be a TsField"))
        self.arr[:,0]+=age_shifted_gain.arr
        
    def receive_external(self,brand_new_stuff):
        # we could check for positive numbers here...
        n=self.arr
        n[0,0]=brand_new_stuff
        
    #def receive_shifting(self,stuff_from_previos_time_step):
    #    # this method avoids shifting the gains before reception
    #    # thereby avoiding one array creation
    #    if not(isinstance(stuff_from_previous_time_step,TsField)):
    #        raise(Exception("argument must be a TsField"))
    #    n=self.arr
    #    n[1:,0]+=new_stuff.arr
    
    def resize(self,max_index):
        x,y=self.arr.shape
        x_new=max_index
        new_arr=np.zeros((x_new,y))
        new_arr[:x,:]=self.arr
        self.arr=new_arr
        
    def shift(self):
        #pool age and system age increase by one tss
        x,y=self.arr.shape
        ns=(x+1,)+(y+1,)
        new_arr=np.ndarray(ns,dtype=default_data_type())
        new_arr[0,:]=np.zeros(y+1)
        new_arr[:,0]=np.zeros(x+1)
        new_arr[1:,1:]=self.arr
        self.arr=new_arr

    @property
    def total_content(self):
        # complete mass regardless of either pool or system age
        return(self.arr.sum())
        
