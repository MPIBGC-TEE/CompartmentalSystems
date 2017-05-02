# vim: set ff=unix expandtab ts=4 sw=4:
import unittest
import numpy as np
from CompartmentalSystems.bins.TsMassField import TsMassField
from CompartmentalSystems.bins.TsTpMassField import TsTpMassField 
from CompartmentalSystems.bins.TsTpDeathRateField import TsTpDeathRateField


class TestTsTpMassField(unittest.TestCase):
    def setUp(self):
        self.x,self.y=6,6
        self.s=(self.x,self.y)
        self.arr=np.zeros(self.s)
        self.arr[5,5]=10
        self.arr[5,4]=20
        self.tss=0.1## the time step size 
        self.spad=TsTpMassField(self.arr,self.tss) 

    def test_loss(self):
        eta_dist=TsTpDeathRateField(np.ones(self.s)*0.5,self.tss)
        l=self.spad.loss(eta_dist)
        
        self.assertEqual(l[5,5],5)
        self.assertEqual(l[5,4],10)

    def test_sum_over_all_pool_ages(self):
        res=self.spad.sum_over_all_pool_ages()
        self.assertTrue(isinstance(res,TsMassField))
        ref=np.zeros(self.x)
        ref[5]=30 #10+20
        self.assertTrue((res.arr==ref).all())

    def test_shift(self):
        spad=self.spad
        spad.shift()
        ref=np.zeros((self.x+1,self.y+1))
        ref[6,6]=10
        ref[6,5]=20
        print("\n##########shift",spad.arr)
        self.assertTrue((spad.arr==ref).all())
    
    def test_resize(self):
        spad=self.spad
        spad.resize(10)
        ref=np.zeros((10,self.y))
        ref[5,5]=10
        ref[5,4]=20
        self.assertTrue((spad.arr==ref).all())

    def test_receive_external(self):
        spad=self.spad
        spad.receive_external(5)
        ref=np.zeros((self.x,self.y))
        ref[5,5]=10
        ref[5,4]=20
        ref[0,0]=5
        self.assertTrue((spad.arr==ref).all())
        
    def test_receive(self):
        x,y=5,2
        valf=1
        field=TsTpMassField(np.zeros((x,y)),self.tss)
        # although not relevant for this test the shifted 
        # field would not have entries in [0,:],and [:,0] therefor we choose
        field[1,1]=valf
        # note that the gains method  has already shifted the TsMassField
        # by tss so that the gains are the same size as the (also shifted) 
        # receiver in Ts direction
        gain=TsMassField(5*np.zeros(x),self.tss)
        # Since the gains have also been collected from other pools
        # there system age is at least tss
        # size in Ts direction so gains[0] must stay 0
        valg=2
        gain[1]=valg
        field.receive(gain)
        ref=np.zeros((x,y))
        ref[1,1]=valf #as before 
        ref[1,0]=valg #gains incorporated with pool age  0
        self.assertTrue((field.arr==ref).all())
        

if __name__ == "__main__":
    unittest.main()
