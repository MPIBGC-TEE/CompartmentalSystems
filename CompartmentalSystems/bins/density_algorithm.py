# vim: set ff=unix expandtab ts=4 sw=4:
from ..helpers_reservoir import pp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D


##def loss(age_dist_plain,eta_plain ):
##    return(age_dist_plain*eta_plain)    

def losses(old_rectangles,death_rates,time):
    internal_death_rates= {k:v for k,v in death_rates.items() if isinstance(k,tuple)}
    losses=dict()
    for key,death_rate in internal_death_rates.items():
    #for pool_number,r in enumerate(old_rectangles):
        sending_pool=key[0]
        receiving_pool=key[1]
        r=old_rectangles[sending_pool]
        losses[key]=r*death_rate(r,time)
    return(losses)

def gains(pool_number,internal_losses):
    #sum over all pool ages
    #(first index=SystemAge,second_index PoolAge)
    #find pipelines leading to me
    
    this_pool_gains={k:v for k,v in internal_losses.items() if k[1]==pool_number}
    return(sum([np.sum(v,1) for v in this_pool_gains.values()]))

#def age_distributions(initial_plains,external_inputs,external_death_rates,internal_death_rates,start,end):
#    tss=initial_plains[0].tss
#    # infer the number of pools from the number of start distributions
#    number_of_pools=len(initial_plains)
#
#    # start the list with an element for the first time step 
#    mpp=[initial_plains]
#   
#    old_rectangles=initial_plains
#
#    times= np.arange(start+tss,end+tss,tss)
#    for t in times:
#        new_rectangles=advance_rectangles(old_rectangles,t-tss,external_inputs,death_rates)
#        mpp.append(new_rectangles)
#        old_rectangles=new_rectangles
#    return(mpp)

#def advance_rectangles(old_rectangles,time,external_inputs,death_rates):
#    new_rectangles=[]
#    #extract the deathrates out of the system
#    internal_death_rates= {k:v for k,v in death_rates.items() if isinstance(k,tuple)}
#    internal_losses=losses(old_rectangles,death_rates,time)
#    outward_death_rates= {k:v for k,v in death_rates.items() if not(isinstance(k,tuple))}
#
#    for pool_number,r in enumerate(old_rectangles):
#        x,y=r.shape
#        #first remove the outflow for all external deathrates defined for this pool
#        this_pool_outward_death_rates={k:v for k,v in outward_death_rates.items() if k==pool_number}
#        
#        internal_input_to_first_pool=np.zeros(x) #fixme: faked since we know that we have no contributions from other pools
#        #accumulate losses through different output channels
#        l=np.zeros(r.shape)
#        for dr in this_pool_outward_death_rates.values():
#            #create the eta field of appropriate size
#            eta_rect=dr(r,time)
#            l+=loss(r,eta_rect)
#        
#        x,y=r.shape
#        n=np.ndarray((x+1,y+1),dtype=default_data_type())
#        n[1:,1:]=r-l
#        #print("n[0,0]",n[0,0])
#        #print("n=",n[0,:])
#        #print("inp=",external_inputs[pool_number](y+1,time))
#        n[0,:]=external_inputs[pool_number](y+1,time)
#        n[1:,0]=internal_input_to_first_pool
#
#        new_rectangles.append(n)
#    return(new_rectangles)
