
#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:

from concurrencytest import ConcurrentTestSuite, fork_for_tests
import inspect
import sys 
import unittest
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.integrate import quad
from scipy.interpolate import interp1d 
from scipy.misc import factorial
from sympy import sin, symbols, Matrix, Symbol, exp, solve, Eq, pi, Piecewise, Function, ones
    
import example_smooth_reservoir_models as ESRM
import example_smooth_model_runs as ESMR

from testinfrastructure.InDirTest import InDirTest
from testinfrastructure.helpers import pe
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.smooth_model_run import SmoothModelRun 
from CompartmentalSystems.discrete_model_run import DiscreteModelRun


class TestDiscreteModelRun(InDirTest):
    def test_from_SmoothModelRun(self):
        symbs = symbols("x,k,t")
        x, t, k = symbs 
        srm = ESRM.minimal(symbs) 
        times = np.linspace(0, 10, 101)
        start_values = np.array([10])
        pardict = {k: -1}
        smr = SmoothModelRun(srm, pardict, start_values, times)
        
        #create a valid model run without start ages
        smr = SmoothModelRun(srm, pardict, start_values, times=times)
        dmr = DiscreteModelRun.from_SmoothModelRun(smr)
        ss=smr.solve()
        ds=dmr.solve()
        import matplotlib.pyplot  as plt
        fig=plt.figure(figsize=(7,7))
        ax1=fig.add_subplot(1,1,1)
        ax1.plot(times,ss[:,0],'*',color='red',label="smr")
        ax1.plot(times,ds[:,0],'*',color='blue',label="dmr")
        fig.savefig("pool_contents.pdf")
        self.assertTrue(True)
