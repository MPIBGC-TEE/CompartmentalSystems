# vim: set ff=unix expandtab ts=4 sw=4:
import unittest
import numpy as np
from bgc_md.TsTpMassField import TsTpMassField 
from bgc_md.TsTpDeathRateField import TsTpDeathRateField 
from bgc_md.TsTpMassFields import TsTpMassFieldsPerPool,TsTpMassFieldsPerPipe
from bgc_md.CompatibleTsTpMassFieldsPerPool import CompatibleTsTpMassFieldsPerPool

class TestCompatibleTsTpMassFieldsPerPool(unittest.TestCase):
    def test_init(self):
        tss=0.1
        with self.assertRaises(Exception) as cm:
            #empty list
            initial_fields=CompatibleTsTpMassFieldsPerPool([])
            
        # check a one pool example
        age_dist_0=TsTpMassField(np.zeros((4,3)),tss)
        age_dist_0[2,2]=100
        initial_fields=CompatibleTsTpMassFieldsPerPool([age_dist_0])
        
        # check a one 3 pool example
        initial_fields=CompatibleTsTpMassFieldsPerPool([
            TsTpMassField(np.zeros((3,3)),tss),
            TsTpMassField(np.zeros((2,1)),tss),
            TsTpMassField(np.zeros((4,2)),tss)])
        self.assertEqual(len(initial_fields),3) 
        
        # check that all pools can now receive sytem ages up to 4*tss althoug they were smaller at the beginnign
        tss=0.1
        initial_fields=CompatibleTsTpMassFieldsPerPool([
            TsTpMassField(np.zeros((3,3)),tss),
            TsTpMassField(np.zeros((2,1)),tss),
            TsTpMassField(np.zeros((4,2)),tss)])
        for field in initial_fields.values():
            self.assertTrue(field.number_of_Ts_entries==4)    

class TestCompatibleTsTpMassFieldsPerPoolSetUp(unittest.TestCase):
    def setUp(self):
        self.tss=0.1
        arr0=np.zeros((3,3))
        ref0=arr0
        arr1=np.zeros((3,3))
        ref1=arr1
        f0=TsTpMassField(arr0,self.tss)
        f1=TsTpMassField(arr1,self.tss)
        self.fields=CompatibleTsTpMassFieldsPerPool([f0,f1])
    
        
if __name__=="__main__":
    unittest.main()
