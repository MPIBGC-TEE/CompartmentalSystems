# vim: set ff=unix expandtab ts=4 sw=4:
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
from .plot_helpers import cc
class TimeBin():
    def __init__(self,tss,smin,pmin,content,facecolors=None,offset=0):
        self.tss=tss
        self.smin=smin
        self.pmin=pmin
        self.content=content
        self.offset=offset
        self.facecolors=facecolors
    
    def verts(self):
        ts=     self.tss
        smin=   self.smin
        pmin=   self.pmin
        content=self.content
        offset=self.offset
        verts=[]
        #polygon for Tp,content plane small Ts
        xs=[smin,smin   ,smin   ,smin   ]
        ys=[pmin,pmin+ts,pmin+ts,pmin   ]
        zs=[offset   ,   offset   ,content+offset,content+offset]
        verts.append(list(zip(xs,ys,zs)))

        #polygon for Tp,content plane bigger Ts
        xs=[smin+ts,smin+ts,smin+ts,smin+ts]
        ys=[pmin   ,pmin+ts,pmin+ts,pmin   ]
        zs=[offset      ,offset      ,content+offset,content+offset]
        verts.append(list(zip(xs,ys,zs)))

        #polygon for Ts,content plane big Ts
        xs=[smin   ,smin+ts,smin+ts,smin   ]
        ys=[pmin+ts,pmin+ts,pmin+ts,pmin+ts]
        zs=[offset      ,   offset   ,content+offset,content+offset]
        verts.append(list(zip(xs,ys,zs)))

        #polygon for Ts,content plane big Ts
        xs=[smin   ,smin+ts,smin+ts,smin   ]
        ys=[pmin   ,pmin   ,pmin   ,pmin   ]
        zs=[offset      ,   offset   ,content+offset,content+offset]
        verts.append(list(zip(xs,ys,zs)))

        #polygon for cap
        xs=[smin,smin+ts,smin+ts,smin   ]
        ys=[pmin,pmin,   pmin+ts,pmin+ts  ]
        zs=[content+offset,content+offset,content+offset,content+offset]
        verts.append(list(zip(xs,ys,zs)))
        return(verts)

    def plot(self,ax):
        
        p1 = a3.art3d.Poly3DCollection(self.verts())
        #p1.set_color(cc("r"))
        if not self.facecolors:
            self.facecolors=[
            cc("b"),
            cc("b"),
            cc("b"),
            cc("b"),
            cc("b"),
            cc("y")
            ]
        p1.set_facecolors(self.facecolors)
        p1.set_edgecolors([
            cc("b"), 
            cc("b"), 
            cc("y"), 
            cc("y"), 
            cc("y")
        ])
        p1.set_linewidth(0.5)
        ax.add_collection3d(p1)
        ax.set_zlim3d(0,self.content+self.offset)


        
        
