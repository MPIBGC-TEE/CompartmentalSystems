# vim: set ff=unix expandtab ts=4 sw=4:
from string import Template
import numpy as np
from functools import reduce
from .TsTpMassField import TsTpMassField
from .TsMassField import TsMassField

class TimeMassFields(dict):
    def shift(self):
        #shift everything forward one tss step
        # this has to be done for all the pools even if they neither loose nor gain in this time step
        for el in self.values():
            el.shift()
# the  classes are mutually dependent and have to be part of the same module to avoid 
# circular imports

class TsMassFieldsPerPool(TimeMassFields):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for key,value in self.items():
            if not(isinstance(key,int)):
                raise(Exception("only integers (describing the pool number) are allowed as indices here"))
                atom_type=TsMassField
            if not(isinstance(value,atom_type)):
                raise(Exception(Template("only "+atome_type +"elements are allowed as values, got an element of type ${vt}").substitute(vt=type(value))))

class TsTpMassFieldsPerPool(TimeMassFields):
    # Although derived from dict the class implements 
    # an array like structure 
    # of TsTpFields indexed by pool number 
    # It is in the current implementation derived from 
    # dictionary to avoid the neccessaty to 
    # set all item from 0..n-1 before setting item n
    # So we can have incomplete lists used in the losses method
    

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for key,value in self.items():
            if not(isinstance(key,int)):
                raise(Exception("only integers (describing the pool number) are allowed as indices here"))
            if not(isinstance(value,TsTpMassField)):
                raise(Exception(Template("only TsTpMassFields are allowed as values, got an element of type ${vt}").substitute(vt=type(value))))

    def shift(self):
        #shift everything forward one tss step
        # this has to be done for all the pools even if they neither loose nor gain in this time step
        for el in self.values():
            el.shift()

        
    def internal_losses(self,internal_death_rate_fields):
        losses=TsTpMassFieldsPerPipe()
        for key,death_rate in internal_death_rate_fields.items():
            sending_pool=key[0]
            r=self[sending_pool]
            losses[key]=r.loss(death_rate)
        return(losses)
    
    def receive_external(self,external_inputs):
        for receiving_pool,stuff in external_inputs.items():
            self[receiving_pool].receive_external(stuff)
            
        
    def receive(self,gains):
        for receiving_pool,gain in gains.items():
            self[receiving_pool].receive(gain)
        
#    def receive_shifting(self,gains_from_previous_time_step):
#        # calling this method avoids the shifting operation
#        # on the gains before incorporation
#        # this avoids one array creation 
#        for receiving_pool,gain in gains_from_previous_time_step.items():
#            rfield=self[receiving_pool]
#            rfield.receive_shifting(gain)
        
    
    def remove(self,l):
        if isinstance(l,TsTpMassFieldsPerPool):
            for sending_pool,loss in l.items():
                sfield=self[sending_pool]
                sfield.remove_loss(loss) 
            #note that self[sending_pool] will be changed (reference vs. copy)
        if isinstance(l,TsTpMassFieldsPerPipe):
            for pipe_key,loss in l.items():
                sending_pool=pipe_key[0]
                sfield=self[sending_pool]
                sfield.remove_loss(loss) 
            
      
        
    def external_losses(self,external_death_rate_fields):
        losses=TsTpMassFieldsPerPool([]) #call constructor of this class (while self might be member of a subclass)
        for sending_pool,death_rate in external_death_rate_fields.items():
            r=self[sending_pool]
            losses[sending_pool]=r.loss(death_rate)
        return(losses)
        
##############################################################################
##############################################################################
class TsTpMassFieldsPerPipe(TimeMassFields):
    # The class implements a list of TsTpFields indexed by a tuple=(sender,receiver) 

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for key,value in self.items():
            if not(isinstance(key,tuple)and len(key)==2 and key[0]!=key[1]):
                raise(Exception("only tuples (describing the sending and receiving pool ) are allowed as keys here"))
            if not(isinstance(value,TsTpMassField)):
                raise(Exception("only TsTpMassFields are allowed as values"))
    @property
    def gains(self):
        # gains always have pool age 0
        # and a minimal system age of 0+tss
        # gains are represented by a pool-number indexed dict
        # of one-dimensional fields which for Ts values which are 
        # longer by one compared to the loss fields they are computed from
        # This represents the aging (shift) that occurs for them 
        # only with respect to Ts
        res=TsMassFieldsPerPool()
        for pipe_key,mass_field in self.items():
            receiving_pool=pipe_key[1]
            pipe_gain=mass_field.sum_over_all_pool_ages().shifted()
            if receiving_pool in res.keys():
                res[receiving_pool]+=pipe_gain
            else:
                res[receiving_pool]=pipe_gain
        return(res)


