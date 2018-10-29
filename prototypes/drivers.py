
import numpy as np
from numpy import ndarray
from scipy.interpolate import interp1d

def u_A_step_single(time):
    #vectorized step function
    step_time=2100
    l_val,r_val=10,30
    if time< step_time:
        return l_val
    else:
        return r_val

# fossil fuel and land use change data
# column 0: time, column 2: land use effects
ff_and_lu_data = np.loadtxt('emissions.csv', usecols = (0,1,2), skiprows = 38)

def func_maker(interpol_times,interpol_data):
    inter_func=interp1d(interpol_times, interpol_data)
    first_time=interpol_times[0]
    first_data=interpol_data[0]
    def scalar_func(time):
        # here we could do whatever we want to compute the input function
        # we return only the linear interpolation from above extendet on the lower 
        # end by constant values equal to the first value of the timeline
        # The resulting function has to be vectorized to be used on arrays
        
        if time < first_time: 
            res = first_data
        else:
            res = inter_func(time)
        return(res)
    
    vec_func=np.vectorize(scalar_func)    
    return vec_func

u_A_func=func_maker(ff_and_lu_data[:,0],ff_and_lu_data[:,1])
f_TA_func =func_maker(ff_and_lu_data[:,0],ff_and_lu_data[:,2])
