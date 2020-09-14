# vim: set ff=unix expandtab ts=4 sw=4:
from .TimeBin import TimeBin

class TstBin(TimeBin):
    def plot(self,ax):
        super().plot(ax)
        ax.set_xlabel("system age")
        ax.set_ylabel("time")


        
        
