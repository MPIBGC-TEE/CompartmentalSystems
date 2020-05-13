# vim:set ff=unix expandtab ts=4 sw=4
from .TimeStep import TimeStep
import numpy as np
from scipy.integrate import quad 
from scipy.interpolate import interp1d

from .TsTpMassFields import TsTpMassFieldsPerPool,TsTpMassFieldsPerPipe
from .CompatibleTsTpMassFieldsPerPool import CompatibleTsTpMassFieldsPerPool
from .TsTpMassField import TsTpMassField 
from .TsTpDeathRateField import TsTpDeathRateField
from .TsTpMassFieldsPerPoolPerTimeStep import TsTpMassFieldsPerPoolPerTimeStep

def external_death_rate_maker(sender,func,solfs):
    def wrapper(field,t):
        tss=field.tss
        loss=quad(func,t,t+tss)[0]
        stock=solfs[sender](t)
        if stock != 0:
            relative_loss=loss/stock
        else:
            relative_loss = 0
        dr=TsTpDeathRateField(relative_loss*np.ones(field.shape),tss)
        return(dr)
    return(wrapper)

def internal_death_rate_maker(key,func,solfs):
    def wrapper(field,t):
        sender=key[0]
        tss=field.tss
        loss=quad(func,t,t+tss)[0]
        stock=solfs[sender](t)
        
        if stock != 0:
            relative_loss=loss/stock
        else:
            relative_loss = 0

        dr=TsTpDeathRateField(relative_loss*np.ones(field.shape),tss)
        return(dr)
    return(wrapper)

def external_input_maker(tss,receiver,func):
    def wrapper(t):
        return(quad(func,t,t+tss)[0])
    return(wrapper)
        
#########################################################################
#########################################################################
#########################################################################
#########################################################################
class TimeStepIterator:
    """iterator for looping over the results of a difference equation"""
    def __init__(
        self,
        initial_plains,
        external_input_funcs,
        internal_death_rate_funcs,
        external_death_rate_funcs,
        t0,
        number_of_steps
        ):
        self.t0=t0
        self.initial_plains=initial_plains
        self.number_of_steps=number_of_steps

        self.external_input_funcs=external_input_funcs
        self.internal_death_rate_funcs=internal_death_rate_funcs
        self.external_death_rate_funcs=external_death_rate_funcs
        self.reset()

    ######################################################################
    @classmethod
    def from_ode_reservoir_model_run(cls,mr,initial_plains=None):
        obj=cls.__new__(cls)
        number_of_pools=mr.nr_pools
        start_values=mr.start_values
        # to avoid excess of numerical cost we limit to 100 time steps here
        obj.number_of_steps=100
        # and adapt the time step size accordingly
        #holger: change to //4+1 and find out what goes wrong
        # with bare fallow in ICBM
        times=mr.times[:len(mr.times)//4]
#        times=mr.times[:obj.number_of_steps]
        #print(times)
        tss=(times[-1]-times[0])/obj.number_of_steps
#        tss=(times[1]-times[0])
#        print(times)
#        print(tss)
        #fixme: find right times
        
        if not(initial_plains):
            obj.initial_plains=CompatibleTsTpMassFieldsPerPool(
                [
                    TsTpMassField(start_values[i]*np.ones((1,1)),tss) 
                    for i in range(number_of_pools)
                ]
            )

            #holger: added initial distr
#            init_list = []
#            for i in range(number_of_pools):
#                k=20
#                pool_field = np.zeros((k,1))  
#                pool_field[:k,0]=[start_values[i]/k for j in range(k)]
##                pool_field[:50,0] = [0.028*(1-4/5*tss)**Ts for Ts in range(50)]
#                print(sum(pool_field))
#                init_list.append(pool_field)
#            
#            obj.initial_plains=CompatibleTsTpMassFieldsPerPool(
#                [
#                    TsTpMassField(init_list[i],tss) 
#                    for i in range(number_of_pools)
#                ]
#            )
            

        else: #adjust tss of the plains
            for plane in initial_planes:
                plane.tss=tss
        
        ## we now build the deathrate functions
        ## note that the factories depend
        ## on the solution funtions 

        # produce the output deathrate functions

        obj.external_death_rate_funcs=dict()
        solfs=mr.sol_funcs()
        for sender,func in mr.external_output_flux_funcs().items():
            obj.external_death_rate_funcs[sender]=external_death_rate_maker(sender,func,solfs)
            
        ## produce the internal deathrate functions
        obj.internal_death_rate_funcs=dict()
        for key,func in mr.internal_flux_funcs().items():
            obj.internal_death_rate_funcs[key]=internal_death_rate_maker(key,func,solfs)


        # produce the external inputs
        obj.external_input_funcs=dict()
        for receiver,func in mr.external_input_flux_funcs().items():
            obj.external_input_funcs[receiver]=external_input_maker(tss,receiver,func)
            
        obj.t0=times[0]
        obj.reset()
        return(obj)


    ######################################################################
    @property
    def tss(self):
        return(self.initial_plains[0].tss)

    def reset(self):
        self.i=0
        self.time=self.t0
        self.rectangles=self.initial_plains

    def __iter__(self):
        self.reset()
        return(self)

    def __next__(self):
        number_of_steps=self.number_of_steps
        if self.i == number_of_steps:
            raise StopIteration
        # compute deathrate fields
        t=self.t0+self.i*self.tss
        internal_death_rate_fields={pipe_key:f(self.rectangles[pipe_key[0]],t) for pipe_key,f in self.internal_death_rate_funcs.items()}
            
        external_death_rate_fields={pool_key:f(self.rectangles[pool_key],t) for pool_key,f in self.external_death_rate_funcs.items()}
        # compute external inputs
        external_input_numbers={key:f(t) for key,f in self.external_input_funcs.items()}

        ts=TimeStep(
            t,
            self.rectangles,
            internal_death_rate_fields,
            external_death_rate_fields,
            external_input_numbers
            )
        self.rectangles=ts.updated_content
        #print(t, "%0.9f" % self.rectangles[0].total_content)
        #holger: external losses were not removed,
        # they still seem to be at least a little wrong
        #print(self.rectangles[0].total_content)
        self.i+=1
        return(ts)
        
