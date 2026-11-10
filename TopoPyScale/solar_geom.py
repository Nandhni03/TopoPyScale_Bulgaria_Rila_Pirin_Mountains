"""
Function to compute Solar angles
S. Filhol, Oct 2021

"""

import pvlib
import pandas as pd
import xarray as xr
import numpy as np
from pyproj import Transformer
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mproc
from TopoPyScale import topo_export as te
from pathlib import Path


def cos_illumination(mu0, zenith, azimuth, slope, aspect):
    """
    Cosine of the angle between the sun and the normal of a tilted surface (0 when the sun is behind the slope).

    Angle conventions (Dozier / IPW, used consistently in TopoPyScale):
      - azimuth: solar azimuth [rad], 0 towards SOUTH, positive towards EAST, negative towards west
        (solar_geom.get_solar_geom stores 180 - pvlib azimuth).
      - aspect: slope aspect [rad], same reference: 0 = south-facing, pi/2 = east-facing, pi = north-facing,
        3*pi/2 (= -pi/2) = west-facing. This is what topocalc.gradient.gradient_d8(aspect_rad=False) returns
        (in degrees) for a north-up DEM, i.e. with dy < 0 as passed by topo_param.compute_dem_param.
      - slope [rad], zenith [rad], mu0 = cos(zenith).

    Args:
        mu0, zenith, azimuth: solar geometry (scalars, numpy arrays or DataArrays)
        slope, aspect: surface geometry [rad]
    Returns:
        cos of the illumination angle, clipped at 0
    """
    c = mu0 * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(azimuth - aspect)
    return c * (c > 0)


def aspect_to_bearing_deg(aspect):
    """Convert a TopoPyScale aspect [rad, 0 = south, positive towards east] to a compass bearing
    [deg clockwise from north, 0 = north-facing, 90 = east-facing]."""
    return (180.0 - np.rad2deg(aspect)) % 360.0


def get_solar_geom(df_position,
                   start_date,
                   end_date,
                   tstep,
                   sr_epsg="4326",
                   num_threads=None,
                   fname='ds_solar.nc',
                   project_ouput=Path('./'),
                   method_solar='nrel_numpy'):
    """
    Function to compute solar position for each location given in the dataframe
    azimuth is define with 0 towards South, negative in W-dir, and posiive towards E-dir

    Args:
        df_position (dataframe): point_name as index, latitude, longitude
        start_date (str): start date  "2014-05-10"
        end_date (str: end date   "2015-05-10"
        tstep (str): time step, ex: '6h'
        sr_epsg (str): source EPSG code for the input coordinate
        num_threads (int): number of threads to parallelize computation on. default is number of core -2
        fname (str): name of netcdf file to store solar geometry
        project_ouput (str): path to project root directory

    Returns: 
        dataset: solar angles in degrees

    """
    print('\n---> Computing solar geometry')
    if (int(sr_epsg) != "4326") or ('longitude' not in df_position.columns):
        trans = Transformer.from_crs("epsg:" + sr_epsg, "epsg:4326", always_xy=True)
        df_position['longitude'], df_position['latitude'] = trans.transform(df_position.x.values, df_position.y.values)
    # tstep_dict = {'1h': 1, '3h': 3, '6h': 6}

    times = pd.date_range(start_date, pd.to_datetime(end_date)+pd.to_timedelta('1D'), freq=tstep, tz='UTC', inclusive='left')


    df_pool = pd.DataFrame()
    df_pool[['latitude', 'longitude', 'elevation']] = df_position[['latitude', 'longitude', 'elevation']]
    df_pool['times'] = df_pool.latitude.apply(lambda x: times)

    def compute_solar_geom(time_step, latitude, longitude, elevation):
        arr = pvlib.solarposition.get_solarposition(time_step, latitude, longitude, elevation, method=method_solar)[['zenith', 'azimuth', 'elevation']].values.T
        return arr

    if num_threads is None:
        pool = ThreadPool(mproc.cpu_count() - 2)
    else:
        pool = ThreadPool(num_threads)

    arr = pool.starmap(compute_solar_geom, zip(list(df_pool.times),
                                                   list(df_pool.latitude),
                                                   list(df_pool.longitude),
                                                   list(df_pool.elevation)))
    pool.close()
    pool.join()

    arr_val = np.empty((df_position.shape[0], 3, times.shape[0]))
    for i, a in enumerate(arr):
        arr_val[i,:,:]=a

    ds = xr.Dataset(
        {
            "zenith": (["point_name", "time"], np.deg2rad(arr_val[:, 0, :])),
            "azimuth": (["point_name", "time"], np.deg2rad(180 - arr_val[:,1,:])),
            "elevation": (["point_name", "time"], np.deg2rad(arr_val[:, 2, :])),
        },
        coords={
            "point_name":(("point_name"), df_position.point_name),
            "time": pd.date_range(start_date, pd.to_datetime(end_date)+pd.to_timedelta('1D'), freq=tstep, inclusive='left'),
            "reference_time": pd.Timestamp(start_date),
        },
    )

    ds['mu0'] = np.cos(ds.zenith) * (np.cos(ds.zenith)>0)
    S0 = 1370 # Solar constat (total TOA solar irradiance) [Wm^-2] used in ECMWF's IFS
    ds['SWtoa'] = S0 * ds.mu0
    ds['sunset'] = ds.mu0 < np.cos(89 * np.pi/180)
    ds.SWtoa.attrs = {'units': 'W/m**2', 'standard_name': 'Shortwave radiations downward top of the atmosphere'}
    ds.sunset.attrs = {'units': 'bool', 'standard_name': 'Sunset'}

    te.save_dataset(ds, Path(project_ouput) / fname)   # 'xxx.zarr' -> Zarr store, else NetCDF

    return ds

