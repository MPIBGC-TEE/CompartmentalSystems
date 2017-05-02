# vim: set ff=unix expandtab ts=4 sw=4:
from unittest import TestCase,main
import numpy as np
from CompartmentalSystems.bins.TimeField import TimeField

class TestTimeField(TestCase):
    def setUp(self):
        self.ar=np.zeros(3)
        self.ar[2]=2
        self.arr=np.zeros((3,2))
        self.arr[2,1]=2
        
    def test_number_of_Ts_entries(self):
        tf=TimeField(self.arr,0.1)
        self.assertEqual(tf.number_of_Ts_entries,3)

if __name__=="__main__":
    main()
