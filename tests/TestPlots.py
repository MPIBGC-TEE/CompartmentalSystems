import sys
import unittest
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from sympy import (
    Symbol,
    Matrix,
    diag,
    symbols,
    sin,
    cos,
    var,
    Piecewise,
    DiracDelta,
    Function,
    simplify,
    zeros,
    var
)
import CompartmentalSystems.helpers_reservoir as hr 
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from testinfrastructure.InDirTest import InDirTest

class TestPlots(InDirTest):
    def test_matplotlib_part_plot(self):
        
        var("A")
        var("A_long_name")
        var("B")
        var("C")
        var("t")
        # InfluxesBySymbol
        IFBS={
            A: sin(t)*A,
            A_long_name: sin(t)*A_long_name,
            C: 1
        }
        # OutfluxesBySymbol
        OFBS={
            B: cos(t)*B,
            C: 0.5*C
        }
        # InternalFluxesBySymbol
        IntFB = {
            (A,A_long_name): 2*A,
            (A,B): 2*A,
            (B,A): 3*B,
            (C,B): C**2,
            (B,C): 2*C,
        }
        part_dict = {
            frozenset([A,B]): "green",
            frozenset([C]): "brown",
        }

        fig = plt.figure(figsize=(8, 12))
        ax = fig.add_subplot(1, 1, 1)
        hr.matplotlib_part_plot(
            (A, B, C, A_long_name),
            IFBS,
            IntFB,
            OFBS,
            part_dict,
            ax
        )
        fig.savefig("plot.pdf")

    def test_igraph_part_plot(self):
        
        var("A")
        var("A_long_name")
        var("B")
        var("C")
        var("t")
        # InfluxesBySymbol
        IFBS={
            A: sin(t)*A,
            A_long_name: sin(t)*A_long_name,
            C: 1
        }
        # OutfluxesBySymbol
        OFBS={
            B: cos(t)*B,
            C: 0.5*C
        }
        # InternalFluxesBySymbol
        IntFB = {
            (A,A_long_name): 2*A,
            (A,B): 2*A,
            (B,A): 3*B,
            (C,B): C**2,
            (B,C): 2*C,
        }
        part_dict = {
            frozenset([A,B]): "green",
            frozenset([C]): "brown",
        }

        hr.igraph_part_plot(
            (A, B, C, A_long_name),
            IFBS,
            IntFB,
            OFBS,
            part_dict,
            "plot.pdf"
        )
        


