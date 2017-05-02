# vim: set ff=unix expandtab ts=4 sw=4:
import unittest
import numpy as np
from CompartmentalSystems.bins.TsTpMassField import TsTpMassField 
from CompartmentalSystems.bins.TsTpDeathRateField import TsTpDeathRateField 
from CompartmentalSystems.bins.TsTpMassFields import TsTpMassFieldsPerPool,TsTpMassFieldsPerPipe


class TestTsTpMassFieldsPerPool(unittest.TestCase):
    def setUp(self):
        self.tss=0.1
        x,y=3,3
        s=(x,y)
        arr=np.zeros(s)
        arr[2,2]=2
        arr2=2*arr
        #copy for references in the tests
        self.arr=arr
        self.arr2=arr2
        self.age_dist_0=TsTpMassField(arr,self.tss)
        self.age_dist_1=TsTpMassField(arr2,self.tss)
        size_diff=5
        age_dist_2=TsTpMassField(np.zeros((x+size_diff,y+size_diff)),self.tss)
        self.fields=TsTpMassFieldsPerPool({0:self.age_dist_0,1:self.age_dist_1})
        self.time=10
        self.loss_factor=0.3
        
        def constant_well_mixed_death_rate(age_dist,t):
            # these functions must be able to define a TsTpDeathRateField eta
            # of the same size as the age distribution it gets 
            # for all the system ages and pool ages present in age_dist it must
            # be able to compute the deathrate
            return(TsTpDeathRateField(self.loss_factor*np.ones(age_dist.arr.shape),age_dist.tss))
        self.func=constant_well_mixed_death_rate
        external_death_rate_funcs=dict()
        external_death_rate_funcs[0]=self.func
        external_death_rate_funcs[1]=self.func
        fields=self.fields
        t=self.time
        external_death_rate_fields={pool_key:f(fields[pool_key],t) for pool_key,f in external_death_rate_funcs.items()}
        self.external_death_rate_fields=external_death_rate_fields
        
        
    def test_init_Exceptions(self):
        with self.assertRaises(Exception) as cm:
            TsTpMassFieldsPerPool({(1,2):self.age_dist_0}) #only integers allowed as keys not tuples

        with self.assertRaises(Exception) as cm:
            TsTpMassFieldsPerPool({1:2}) #only TsTpMassFields allowed as values

    # fixme:
    @unittest.skip("reimplementing")
    def test_setitem(self):
        # should contain the checks of init 
        # and be called by it
        raise(Exception("not implemented yet"))

        
    def test_internal_losses(self):
        internal_death_rate_funcs=dict()
        internal_death_rate_funcs[(0,1)]=self.func
        internal_death_rate_funcs[(1,0)]=self.func
        fields=self.fields
        t=self.time
        internal_death_rate_fields={pipe_key:f(fields[pipe_key[0]],t) for pipe_key,f in internal_death_rate_funcs.items()}
        #print(fields)
        computed_losses=fields.internal_losses(internal_death_rate_fields)
        ref=TsTpMassFieldsPerPipe()
        ref[(0,1)]=TsTpMassField(self.loss_factor*self.age_dist_0.arr ,self.tss)
        ref[(1,0)]=TsTpMassField(self.loss_factor*self.age_dist_1.arr ,self.tss)
        for key,v in ref.items():
            f=computed_losses[key]
            self.assertTrue((v.arr==f.arr).all())

    def test_external_losses(self):
        computed_losses=self.fields.external_losses(self.external_death_rate_fields)
        ref=TsTpMassFieldsPerPool()
        ref[0]=TsTpMassField(self.loss_factor*self.age_dist_0.arr,self.tss)
        ref[1]=TsTpMassField(self.loss_factor*self.age_dist_1.arr,self.tss)
        for k,v in ref.items():
            f=computed_losses[k]
            self.assertTrue((v.arr==f.arr).all())

    def test_shift(self):
        fields=self.fields
        fields.shift()
        ref0=np.zeros((4,4))
        ref0[3,3]=2
        self.assertTrue((fields[0].arr==ref0).all())
        ref1=np.zeros((4,4))
        ref1[3,3]=4
        self.assertTrue((fields[1].arr==ref1).all())
        
if __name__=="__main__":
    unittest.main()
