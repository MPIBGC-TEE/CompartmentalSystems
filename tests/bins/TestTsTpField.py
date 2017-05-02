# vim: set ff=unix expandtab ts=4 sw=4:
import unittest
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from CompartmentalSystems.bins.TsTpField import TsTpField 
from testinfrastructure.InDirTest import InDirTest
class TestTsTpField(InDirTest):
    
    def test_init(self):
        x,y=10,10
        s=(x,y)
        arr=np.zeros(s)
        arr[5,5]=10
        tss=0.1## the time step size 
        ## This defines 3 things:
        ## 1. The time between snapshots 
        ## 2. The size of SystemTime bins 
        ## 3. The size of PoolTime  bins 
        ## 
        ## 1  is the most obvious but irrelavent here
        ## 2&3 are need    ed to interpret the arr as a Ts,Tp plane
        spad=TsTpField(    arr,tss) 
                           
        # fixme:
        # we should check that nobody populates the triangular part of the array where
        # Tp>Ts 
        # It is however not totally clear how to do this numerically efficiently 
        # with the current implementation that uses the full rectangle.
        # the best option also in terms of memory usage would
        # be not to store the Tp>Ts part at all.
        # At the moment this looks as if we would have to craft this in fortran or C...
        # An intermediate workaround might be to provide a function to check the field
        # on demand but is not called by TsTpFields __init__ automatically
        
        # we at least chek that it is not possible to initialize an array that has room for maxTp>maxTs
        with self.assertRaises(Exception) as cm:
            f=TsTpMassField(np.zeros(3,4),0.1)

    def test_plot_surface(self):
        x,y=6,3
        max_shape=(20,10)
        s=(x,y)
        arr=np.zeros(s)
        val=10
        arr[1,1]=val
        tss=0.1
        spad=TsTpField(arr,tss) 

        fig = plt.figure()
        ax1=fig.add_subplot(1,3,1,projection="3d")
        ax2=fig.add_subplot(1,3,2,projection="3d")
        ax3=fig.add_subplot(1,3,3,projection="3d")
        spad.plot_surface(ax1)
        spad.plot_surface(ax2,max_shape)
        spad.plot_surface(ax3,max_shape,20)
        fig.savefig("plot.pdf")

    def test_plot_bins(self):
        x,y=6,3
        max_shape=(20,10)
        s=(x,y)
        arr=np.zeros(s)
        val=10
        arr[1,1]=val
        arr[2,2]=val/2.
        arr[2,1]=val/3.
        arr[2,1]=val/3.
        tss=0.1
        spad=TsTpField(arr,tss) 

        res=spad.default_plot_args()
        self.assertEqual(res,((6,3),10.0))
        
        res=spad.default_plot_args(max_shape)
        self.assertEqual(res,(max_shape,10.0))
        
        res=spad.default_plot_args(max_shape,20)
        self.assertEqual(res,(max_shape,20.0))
        print(res)

        fig = plt.figure()
        fig.clf()
        ax1=fig.add_subplot(1,3,1,projection="3d")
        ax2=fig.add_subplot(1,3,2,projection="3d")
        ax3=fig.add_subplot(1,3,3,projection="3d")
        spad.plot_bins(ax1)
        spad.plot_bins(ax2,max_shape)
        spad.plot_bins(ax3,max_shape,20)
        fig.savefig("plot.pdf")
        
    def test_getitem(self):
        x,y=10,10
        s=(x,y)
        arr=np.zeros(s)
        val=10
        arr[5,5]=val
        tss=0.1
        spad=TsTpField(arr,tss) 
        
        # indexing is possible directly 
        self.assertEqual(spad[5,5],val) 

if __name__ == "__main__":
    unittest.main()
