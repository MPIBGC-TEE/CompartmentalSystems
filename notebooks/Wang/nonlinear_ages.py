# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Computation of ages and transit times for a two-pool nonlinear system

# This document presents the computation of age and transit time distributions for a nonlinear autonomous model with two compartments. 
# The model describes the dynamics of carbon in soils according to a microbial biomass pool and a soil carbon substrate compartment. 
# The model is described in Wang et al. (2014, Biogeosciences 11: 1817-1831). 
#
# To run the code, you need to install the required libraries below.

# +
# file system operations
import os

# all array-like data structures are numpy.array
import numpy as np

import pandas as pd

# for 2d plots we use Matplotlib
import matplotlib.pyplot as plt

# for 3d plots we use plotly
# load plotly's jupyter notebook functions
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=False)

# symbolic treatment of reservoir models as basis of model runs
from sympy import Matrix, symbols, Symbol, Function, latex
from scipy.integrate import quad

# load the compartmental model packages
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun

## technical part for notebook ##

# Hide unnecessary warnings in integration coming from 'quad'
import warnings

# enable latex in plotly labels
from IPython.display import display, HTML, Markdown
display(HTML(
    '<script>'
        'var waitForPlotly = setInterval( function() {'
            'if( typeof(window.Plotly) !== "undefined" ){'
                'MathJax.Hub.Config({ SVG: { font: "STIX-Web" }, displayAlign: "center" });'
                'MathJax.Hub.Queue(["setRenderer", MathJax.Hub, "SVG"]);'
                'clearInterval(waitForPlotly);'
            '}}, 250 );'
    '</script>'
))
# -

# # Model definition
# We will first create the model following the symbols from the original publication by using the syntax of the sympy package, so we can manipulate the equations and print the output.

# +
########## symbol definitions ##########

# time symbol
time_symbol = symbols('t')

# State variables
C_s, C_b = symbols('C_s C_b')

# Inputs, NPP flux
F_NPP = symbols('F_NPP')

# Parameters
epsilon, V_s, K_s, mu_b = symbols('epsilon V_s K_s mu_b')
# -

x = Matrix([C_s, C_b]) # State variables vector
u = Matrix([F_NPP, 0]) # Input vector
lamda = (C_b*V_s)/(C_s+K_s) # Decomposition rates
B=Matrix([[ -lamda, mu_b], # Compartmental matrix
          [epsilon*lamda, -mu_b]])

# Compartmental system
u+B*x

# # Create a numerical instance of the model
# We now use the set of parameter values described in the original publication, and define the initial conditions for the state variables, so we can run a forward simulation. 
# We use here the class SmoothReservoirModel from the CompartmentalSystem package to load all the ingridients for the model and then be able to compute the solution.

# Replace symbolic model by numerical values
par_dict = {mu_b: 4.38, epsilon:0.39, K_s: 53954.83, V_s: 59.13, F_NPP: 345.0}
start_values = np.array([11000, 50])
times = np.linspace(0, 100, 1000)

# Create smooth reservoir model (a class instance)
srm = SmoothReservoirModel.from_B_u(x, time_symbol, B, u)

# create the nonlinear model run (also a class instance, now from SmoothModelRun)
smrs = []
smr = SmoothModelRun(srm, par_dict, start_values, times)

soln = smr.solve()

# +
# plot the solution
plt.figure(figsize = (10, 7))
plt.title('Total carbon')

plt.plot(times,   soln[:,0], color =   'blue', label = 'C_s: carbon substrate')
plt.plot(times,   soln[:,1], color =  'green', label = 'C_b: microbial biomass')
#plt.plot(times, soln.sum(1), color =    'black', label = 'Total')

plt.legend(loc = 2)
plt.xlabel('Time (yr)')
plt.ylabel('Mass (gC/m2)')
plt.show()
# -

plt.figure(figsize = (10, 7))
plt.plot(soln[:,0], soln[:,1])
plt.xlabel('C_s: carbon substrate')
plt.ylabel('C_b: microbial biomass')


# # Computation of the state transition operator
# For this example, we will compute CBS for carbon entering at year $t_0 = 10$, so we need to obtain the fate of this carbon $M_s(t)$ according to equation (25) from the manuscript. For this purpose, we need to obtain the state transition operator $\Phi(t, t_0)$, which can be computed automatically by the Phi.func from the CompartmentalSystems package.

# We obtain the state transition operator from the smooth model run
Phi = smr.Phi_func()
