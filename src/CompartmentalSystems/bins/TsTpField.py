# vim: set ff=unix expandtab ts=4 sw=4:
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter

from .TimeField import TimeField
from .TsTpBin import TsTpBin

class TsTpField(TimeField):
    # instances respresent a distribution field
    # with a tss (time step size) spacing 
    def __init__(self,arr,tss):
        s=arr.shape
        if len(s)!=2:
            raise(Exception("arr has to be 2 dimensional"))
        if s[0]<s[1]:
            raise(Exception("""Pool age can not exceed System age by definition. 
            Therefore at least arr.shape[0]>=arr.shape[1] should hold!"""))
        super().__init__(arr,tss)

    @property
    def number_of_Tp_entries(self):     
        return(self.arr.shape[1])

    @property
    def max_Tp(self):     
        return(self.number_of_Tp_entries*self.tss)

    def default_plot_args(self,max_shape=None,z_max=None):
        if not(max_shape):
            max_shape=self.shape
        if not(z_max):
            z_max=self.arr.max()
        return((max_shape,z_max))    
        
    def plot_surface(self,ax,max_shape=None,z_max=None):
        max_shape,z_max=self.default_plot_args(max_shape,z_max)
        rect=self.arr
        tss=self.tss
        systemAges  =np.arange(self.number_of_Ts_entries)*tss
        poolAges    =np.arange(self.number_of_Tp_entries)*tss
        X,Y=np.meshgrid(
            systemAges,
            poolAges,
            indexing="ij" # see help of meshgrid
        )
        
        ax.plot_surface(X, Y, rect, rstride=1, cstride=1, linewidth=1)
        #ax.plot_wireframe(X, Y, rect, rstride=1, cstride=1, linewidth=1)
        #ax.plot_surface(X, Y, Z,linewidth=0)
        self.set_limits(ax,max_shape,z_max)
        self.set_ticks_and_labels(max_shape,ax)
    
    def set_ticks_and_labels(self,max_shape,ax):
        tss=self.tss
        systemAges  =np.arange(max_shape[0])*tss
        poolAges    =np.arange(max_shape[1])*tss
        ax.set_xticks(systemAges)
        ax.set_yticks(poolAges)
        
        ax.set_xlabel("system age")
        ax.set_ylabel("pool age")

        ax.invert_xaxis()

    def plot_bins(self,ax,max_shape=None,z_max=None,facecolors=None,offset_field=None):
        if not(isinstance(offset_field,np.ndarray)):
            offset_field=np.zeros(self.shape)
        max_shape,z_max=self.default_plot_args(max_shape,z_max)
        arr=self.arr
        tss=self.tss
        
        ax.set_zlim3d((0,z_max))
        for Ts in range(self.number_of_Ts_entries):
            for Tp in range(self.number_of_Tp_entries):
                offset=offset_field[Ts,Tp]
                val=arr[Ts,Tp]
                if val!=0:
                    b=TsTpBin(tss,Ts*tss,Tp*tss,arr[Ts,Tp],facecolors=facecolors,offset=offset)
                    b.plot(ax)
        self.set_limits(ax,max_shape,z_max)
        self.set_ticks_and_labels(max_shape,ax)

        

    def set_limits(self,ax,max_shape,z_max):
        nTs,nTp=max_shape
        max_system_age=nTs*self.tss
        max_pool_age=nTp*self.tss
        ax.set_xlim(max_system_age,0) #the order (big,small) avoids the axis inversion
        ax.set_ylim(max_pool_age,0)
        ax.set_zlim(0,z_max)
