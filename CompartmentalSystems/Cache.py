from . import picklegzip
import numpy as np

# Fixme: mm 03-30-2020 
# At the moment this class is not used since we do all
# the caching in the lru cache at runtime.  We might reuse it if we decide
# to persist the lru cache in the future
class Cache:
    
    def __init__(self,keys,values,smr_hash):
        self.keys=keys
        self.values=values
        self.myhash=smr_hash

    @classmethod
    def from_file(cls,filename):
        return picklegzip.load(filename)
        
    def save(self,filename):
        picklegzip.dump(self,filename)
        
    def __eq__(self,other):
        return all(
            (
                self.myhash==other.myhash,
                np.array_equal(self.keys,other.keys),
                np.array_equal(self.values,other.values)
            )
        )

    def phi_ind(self,tau):
        cache_times=self.keys
        """
        Helper function to compute the index of the cached state transition operator values. 
        E.g. two matrices require 3 times (0 , 2 ,4 )
        Where Phi[0]=Phi(t=2,s=0),Phi[1]= Phi(t=4,s=2)
        """
        # intervals before tau
        m=cache_times[-1] 
        if tau==m:
            return len(cache_times)-1-1
        else:
            time_ind=cache_times.searchsorted(tau,side='right')
            return time_ind-1


    def end_time_from_phi_ind(self,ind):
        cache_times=self.keys
        if len(cache_times<2):
            return cache_times[-1]
        else:
            return cache_times[ind+1]
    

    def start_time_from_phi_ind(sefl,ind):
        cache_times=self.keys
        return cache_times[ind]
