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

