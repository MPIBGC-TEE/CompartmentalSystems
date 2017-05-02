# vim: set ff=unix expandtab ts=4 sw=4:
import unittest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from CompartmentalSystems.bins.TsMassField import TsMassField 
from testinfrastructure.InDirTest import InDirTest
class TestTsMassField(InDirTest):
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
        sad=TsMassField(arr,tss) 
        t=4.0
        res=sad.default_plot_args(t)
        #self.assertEqual(res,(6*tss,t,t,10.0))
        #
       # res=sad.default_plot_args()
       # self.assertEqual(res,(max_shape,10.0))
        #
        #res=sad.default_plot_args(max_shape,20)
        #self.assertEqual(res,(max_shape,20.0))
        #print(res)

        fig = plt.figure()
        ax1=fig.add_subplot(2,2,1,projection="3d")
        ax2=fig.add_subplot(2,2,2,projection="3d")
        ax3=fig.add_subplot(2,2,3,projection="3d")
        ax4=fig.add_subplot(2,2,4,projection="3d")
        sad.plot_bins(ax1,t)
        sad.plot_bins(ax2,t,Ts_max=10*tss,t_min=t-tss)
        sad.plot_bins(ax3,t,Ts_max=10*tss,t_min=t-tss,t_max=t+3*tss)
        sad.plot_bins(ax4,t,Ts_max=10*tss,t_min=t-tss,t_max=t+3*tss,z_max=15)
        fig.savefig("plot.svg")
        
if __name__ == "__main__":
    unittest.main()
        
