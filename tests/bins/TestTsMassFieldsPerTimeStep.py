# vim: set ff=unix expandtab ts=4 sw=4:
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from CompartmentalSystems.bins.TsMassField import TsMassField 
from CompartmentalSystems.bins.TsMassFieldsPerTimeStep import TsMassFieldsPerTimeStep
from testinfrastructure.InDirTest import InDirTest

class TestTsMassFieldsPerTimeStep(InDirTest):
    def test_plot_bins(self):
        x=6
        s=(x)
        arr=np.zeros(s)
        val=10
        arr[1]=val
        arr[2]=val/2.
        arr[3]=val/3.
        arr[4]=val/3.
        tss=0.1

        sapts=TsMassFieldsPerTimeStep([
            TsMassField(arr,tss), 
            TsMassField(0.7*arr,tss), 
            TsMassField(0.7**2*arr,tss)],
            5)
        
        #res=sad.default_plot_args(t)
        #self.assertEqual(res,(6*tss,t,t,10.0))
        #
       # res=sad.default_plot_args()
       # self.assertEqual(res,(max_shape,10.0))
        #
        #res=sad.default_plot_args(max_shape,20)
        #self.assertEqual(res,(max_shape,20.0))
        #print(res)

        fig = plt.figure()
        ax1=fig.add_subplot(1,1,1,projection="3d")
        sapts.plot_bins(ax1)
        fig.savefig("plot.svg")
        
if __name__ == "__main__":
    unittest.main()
        
