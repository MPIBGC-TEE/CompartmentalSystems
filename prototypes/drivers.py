
import numpy as np
from scipy.interpolate import interp1d

def u_A_step(t_val):
    t_step=2100
    lower,higher=10,30
    res = lower if t_val<t_step else higher
    return res

# fossil fuel and land use change data
ff_and_lu_data = np.loadtxt('emissions.csv', usecols = (0,1,2), skiprows = 38)
# column 0: time, column 1: fossil fuels
ff_data = ff_and_lu_data[:,[0,1]]
# linear interpolation of the (nonnegative) data points
u_A_interp = interp1d(ff_data[:,0], np.maximum(ff_data[:,1], 0))

def u_A_func(t_val):
    # here we could do whatever we want to compute the input function
    # we return only the linear interpolation from above
    return u_A_interp(t_val)

# column 0: time, column 2: land use effects
lu_data = ff_and_lu_data[:,[0,2]]
f_TA_func = interp1d(lu_data[:,0], lu_data[:,1])
