# vim: set ff=unix expandtab ts=4 sw=4:
import unittest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from CompartmentalSystems.bins.TsTpBin import TsTpBin
from tests.test_infrastructure import InDirTest
class TestAgeBin(InDirTest):
    def test_plot(self):
        tss=0.1
        smin=1
        pmin=1
        content=0.5
        b=TsTpBin(tss,smin,pmin,content)
        fig = plt.figure()
        ax=fig.add_subplot(1,1,1,projection="3d")
        b.plot(ax)
        fig.savefig("bin.pdf")



if __name__ == "__main__":
    unittest.main()
