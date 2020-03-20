#from dask.distributed import Client
import xarray as xr
import numpy as np
import pandas as pd

import importlib
import ELMlib
importlib.reload(ELMlib)

#client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')
#client

#ds = xr.open_dataset('../Data/14C_spinup_holger_fire.2x2_small.nc')
from netCDF4 import Dataset
ds = Dataset('../Data/14C_spinup_holger_fire.2x2_small.nc')


#lat, lon = ds.coords['lat'], ds.coords['lon']
lat, lon = ds['lat'][:], ds['lon'][:]
lat_indices, lon_indices = np.meshgrid(
    range(len(lat)),
    range(len(lon)),
    indexing='ij'
)

lats, lons = np.meshgrid(lat, lon, indexing='ij')
df_pd = pd.DataFrame(
    {
        'cell_nr': range(len(lat)*len(lon)),
        'lat_index': lat_indices.flatten(),
        'lon_index': lon_indices.flatten(),
        'lat': lats.flatten(),
        'lon': lons.flatten()
    }
)

import dask.array as da

import dask.dataframe as dask_df

df_dask = dask_df.from_pandas(df_pd, npartitions=4)
df_dask

parameter_set = ELMlib.load_parameter_set(
    ds_filename = '../Data/14C_spinup_holger_fire.2x2_small.nc',
    time_shift  = -198*365,
    nstep       = 10
)

def func(line):
    location_dict = {
        'cell_nr':   int(line.cell_nr),
        'lat_index': int(line.lat_index),
        'lon_index': int(line.lon_index)    
    }
    
    cell_nr, log, xs_12C_data, us_12C_data, rs_12C_data= ELMlib.load_model_12C_data(parameter_set, location_dict)
    return cell_nr, log, xs_12C_data, us_12C_data, rs_12C_data

df_dask_2 = df_dask.apply(func, axis=1, meta=('A', 'object'))

df_dask_2.compute()
type(df_dask_2)

df_dask_2

list(df_dask_2)

pd.DataFrame(list(df_dask_2), columns=('cell_nr', 'log', 'xs_12C_data', 'us_12C_data', 'rs_12C_data'))


