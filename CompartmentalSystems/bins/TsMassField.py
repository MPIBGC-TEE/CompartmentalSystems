# vim: set ff=unix expandtab ts=4 sw=4:
import mpl_toolkits.mplot3d as a3
import numpy as np
from matplotlib.collections import PolyCollection
from .TimeMassField import TimeMassField
from .TstBin import TstBin
from .plot_helpers import cc

class TsMassField(TimeMassField):
    # could replace TimeMassField as parent 
    # but this would confuse the naming scheme a bit..
    
    def plot_bins(self,ax,time,Ts_max=None,t_min=None,t_max=None,z_max=None,facecolors=None,offset_field=None):
        if not(isinstance(offset_field,np.ndarray)):
            offset_field=np.zeros(self.shape)
        Ts_max,t_min,t_max,z_max=self.default_plot_args(time,Ts_max,t_min,t_max,z_max)
        ax.invert_xaxis()
        ax.set_xlim((0,Ts_max)) 
        #ax.set_ylim((t_max,0))
        ax.set_ylim((t_max,t_min))
        ax.set_zlim((0,z_max))
        tss=self.tss
        arr=self.arr
        verts=[]
        cols=[]
        if time >= t_min and time <= t_max:
            lim1 = int(Ts_max/tss)+1
            lim2 = self.number_of_Ts_entries
            #holger: cut off bins with Ts>Ts_max
            #for Ts in range(self.number_of_Ts_entries):
            for Ts in range(min(lim1,lim2)):
                offset=offset_field[Ts]
                val=arr[Ts]+offset
                if val-offset!=0:
                    b=TstBin(tss,Ts*tss,time,arr[Ts],facecolors=facecolors,offset=offset)
                    b.plot(ax)
                    #print(b.__dict__)
                    #verts+=b.verts()
                    #cols=cols+[cc("r"),cc("b"),cc("y")]
            #p1 = a3.art3d.Poly3DCollection(verts)
            #p1.set_facecolors(cols)
            #p1.set_linewidth(0.1)
            #p1.set_edgecolor(cc("b"))
            #ax.add_collection3d(p1 )

    def default_plot_args(self,time,Ts_max=None,t_min=None,t_max=None,z_max=None):
        tss=self.tss
        #print("tss=",tss)
        arr=self.arr
        if not(Ts_max):
            Ts_max=self.number_of_Ts_entries*tss
        if not(t_min):
            t_min=time
        if not(t_max):
            t_max=time+tss
        if not(z_max):
            z_max=self.arr.max()
        return(Ts_max,t_min,t_max,z_max)    

    def shifted(self):
        # move all existent mass in Ts direction by one time step
        y=self.shape[0]
        new_arr=np.ndarray(y+1)
        new_arr[1:]=self.arr
        # no mass in the age 0 bin
        new_arr[0]=0
        return(self.__class__(new_arr,self.tss))
