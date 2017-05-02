# vim: set ff=unix expandtab ts=4 sw=4:
import unittest
import numpy as np
from CompartmentalSystems.bins.TsMassField import TsMassField 
from CompartmentalSystems.bins.TsTpMassField import TsTpMassField 
from CompartmentalSystems.bins.TsTpDeathRateField import TsTpDeathRateField 
from CompartmentalSystems.bins.TsTpMassFields import TsTpMassFieldsPerPool,TsTpMassFieldsPerPipe


class TestTsTpMassFieldsPerPipe(unittest.TestCase):
    def test_init(self):
        f_0=TsTpMassField(np.zeros((3,3)),0.1)
        with self.assertRaises(Exception) as cm:
            TsTpMassFieldsPerPipe({1:f_0}) #only tuples of integer are allowed as indices

        with self.assertRaises(Exception) as cm:
            TsTpMassFieldsPerPipe({1:2}) #only TsTpMassFields allowed as values

        with self.assertRaises(Exception) as cm:
            TsTpMassFieldsPerPipe({(1,1):2}) # sender and receiver have to be different

    def test_gains(self):
        tss=0.1
        x,y=3,3
        s=(x,y)
        val=1
        arr=np.zeros(s)
        arr[2,2]=val
        f_0=TsTpMassField(arr,tss)
        mult=2
        f_1=TsTpMassField(mult*arr,tss)
        # only pipes can contribute to a pool
        pipe_contents=TsTpMassFieldsPerPipe({(0,1):f_0,(2,1):f_1})
        computed_gains=pipe_contents.gains
        
        # only pool one gains 
        receivers=computed_gains.keys()
        self.assertEqual(list(computed_gains.keys()),[1])
        
        # the result is one dimensional
        self.assertTrue(isinstance(computed_gains[1],TsMassField))

        # the length of the result has increased by one 
        # representing our policy to pinpoint the moment when 
        # material becomes older excatly between the moments of 
        # leaving one pool and reaching the next 
        # material  aging in the pipeline
        self.assertTrue(computed_gains[1].shape,f_0.shape+(1,))
        
        ref=np.zeros(x+1)
        # the contributions are added up correctly
        # and appear one step shifted in Ts
        ref[3]=val+mult*val
        self.assertTrue((computed_gains[1].arr==ref).all())


if __name__=="__main__":
    unittest.main()
