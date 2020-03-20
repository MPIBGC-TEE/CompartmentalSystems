from . import picklegzip
import numpy as np


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

