# vim: set ff=unix expandtab ts=4 sw=4:
import numpy as np
from sympy import latex
from .FieldsPerTimeStep import FieldsPerTimeStep
from matplotlib import cm
import matplotlib.pyplot as plt
class TsMassFieldsPerTimeStep(FieldsPerTimeStep):

    @property
    def max_number_of_Ts_entries(self):
        return(max([v.number_of_Ts_entries for v in self]))

    @property
    def max_Ts(self):
        return(self.tss*self.max_number_of_Ts_entries)

        
    #fixme: treatment of units
    def plot_bins(self,ax,mr=None,pool=None):
        tss=self.tss
        times=self.times
        z_max=max([vec.arr.max() for vec in self])
        #print(max([vec.arr.max() for vec in self]))
        #print(min([vec.arr.min() for vec in self]))

        for i,vec in enumerate(self):
            vec.plot_bins(ax,self.times[i])
        
        ax.set_ylim(self.t_min,self.t_max*1.05)
        #ax.set_ylim(self.t_min,(self.t_max+tss)*1.05)
        #ax.invert_yaxis()
        
        ax.set_xlim(0,self.max_Ts*1.05)
        ax.set_zlim(0,z_max)
        ax.invert_xaxis()
        self.set_ticks_and_labels(ax,mr,pool)
        
    #fixme: treatment of units
    def set_ticks_and_labels(self,ax,mr=None,pool=None, fontsize=20):
        #fixme:
        # no ticksetting yet

        if mr and mr.model.time_unit:
            ax.set_xlabel("System age ($" + latex(mr.model.time_unit) + "$)", fontsize=fontsize)
            ax.set_ylabel("time ($" + latex(mr.model.time_unit) + "$)", fontsize=fontsize)
        else:
            ax.set_xlabel("system age")
            ax.set_ylabel("time")

        if mr and (pool != None) and mr.model.units and mr.model.units[pool]:
            pool_unit = mr.model.units[pool]
           
            if pool_unit:
                ax.set_zlabel("content ($" + latex(pool_unit) + "$)", fontsize=fontsize)
            else:
                ax.set_zlabel("content")

        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        ax.zaxis.labelpad = 15

    def plot_surface(self,ax,mr=None,pool=None):
        times=self.times
        Ts_max_index=max([vec.shape[0] for vec in self]) 
        z_max=max([vec.arr.max() for vec in self])
        tss=self.tss
        systemAges  =np.arange(Ts_max_index)*tss
        X,Y=np.meshgrid(
            systemAges,
            times,
            indexing="ij" # see help of meshgrid
        )
        Z=np.ndarray((Ts_max_index,len(times)))*np.NaN
        for i,vec in enumerate(self):
            l=vec.shape[0]
            Z[:l,i]=vec.arr
        ax.plot_surface(
            X, Y, Z, 
            rstride=1, 
            cstride=1,
            #color="y", 
            linewidth=0.0,
            cmap=cm.coolwarm,
            norm=plt.Normalize(0,z_max),
            antialiased=False
        )
        #ax.plot_wireframe(X, Y, Z,cmap=cm.coolwarm,norm=plt.Normalize(0,z_max),linewidth=0.3) 
        #ax.plot_surface(X, Y, Z,cmap=cm.coolwarm,linewidth=0.1,antialiased=False)


        ax.set_xlim(0,self.max_Ts*1.05)
        ax.set_ylim(self.t_min,(self.t_max+tss)*1.05)
        ax.invert_yaxis()
        self.set_ticks_and_labels(ax,mr,pool)
        #print(ax.get_zlim())
    

