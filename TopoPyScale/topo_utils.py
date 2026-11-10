import pandas as pd
import pyproj
import rasterio
import linecache
from matplotlib import pyplot as plt
import numpy as np
import glob
#from TopoPyScale import topo_da as da
import git
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mproc
import warnings
import xarray as xr


def vertical_interp_plev(plev_interp, elevation, variables=('t', 'u', 'v', 'q')):
    """
    Vertical interpolation of pressure-level fields to a point elevation, decided PER TIME STEP.

    At each time step: linear interpolation (in height) between the nearest level below and the nearest level above
    the point. At time steps when the point lies below every level (below the lowest pressure level's geopotential
    height), the values of the nearest level above are used (no vertical interpolation, no extrapolation).

    Before 10 Nov 2026 this choice was made once for the whole series: a single time step below the lowest level sent
    every time step to the no-interpolation branch, so results depended on the length of the period downscaled.
    The interpolated time steps are computed with exactly the same arithmetic as before (bit-identical).

    Args:
        plev_interp (dataset): pressure-level fields horizontally interpolated to the point, dims (time, level),
            geopotential height z in m
        elevation (float): point elevation [m]
        variables (tuple): variables to interpolate
    Returns:
        dataset (time): the interpolated variables, top_level [hPa], top_z [m], top_t [K] of the nearest level above
        (used for the pressure), and fallback (bool: point below all levels at that time step)
    """
    z = plev_interp.z.transpose('time', 'level').values
    e = float(elevation)
    above, below = z > e, z < e
    if not above.any(axis=1).all():
        raise ValueError(f'ERROR: Upper pressure level {plev_interp.level.min().values} hPa geopotential is lower than '
                         f'cluster mean elevation {elevation} at some time step')
    it = np.arange(z.shape[0])
    i_top = np.where(above, z - e, np.inf).argmin(axis=1)
    i_bot = np.where(below, z - e, -np.inf).argmax(axis=1)
    fallback = ~below.any(axis=1)
    d_bot = np.abs(z[it, i_bot] - e)
    d_top = np.abs(z[it, i_top] - e)
    with np.errstate(invalid='ignore', divide='ignore'):
        dist = np.array([d_bot, d_top])
        w = dist / np.sum(dist, axis=0)            # same weights as the former code: bot * w[1] + top * w[0]
    out = xr.Dataset(coords={'time': plev_interp.time})
    for v in variables:
        a = plev_interp[v].transpose('time', 'level').values
        top, bot = a[it, i_top], a[it, i_bot]
        out[v] = ('time', np.where(fallback, top, bot * w[1] + top * w[0]))
    out['top_level'] = ('time', plev_interp.level.values[i_top])
    out['top_z'] = ('time', z[it, i_top])
    out['top_t'] = ('time', plev_interp.t.transpose('time', 'level').values[it, i_top])
    out['fallback'] = ('time', fallback)
    return out


def read_downscaled(path='outputs/down_pt*.nc'):
    """
    Function to read downscaled points saved into netcdf files into one single dataset using Dask

    Args:
        path (str): path and pattern to use into xr.open_mfdataset

    Returns:
        dataset: merged dataset readily to use and loaded in chuncks via Dask
    """

    down_pts = xr.open_mfdataset(path, concat_dim='point_name', combine='nested')
    return down_pts


def ds_to_indexed_dataframe(ds, dim_order=None):
    """
    Function to convert an Xarray dataset with multi-dimensions to indexed dataframe (and not a multilevel indexed dataframe).
    WARNING: this only works if the variable of the dataset have all the same dimensions!

    By default the ds.to_dataframe() returns a multi-index dataframe. Here the coordinates are transfered as columns in the dataframe

    Args:
        ds (dataset): xarray dataset with all variable of same number of dimensions

    Returns:
        pandas dataframe:
    """
    # dim_order (e.g. ['y', 'x']) fixes the row order explicitly; by default it follows ds.dims,
    # which can differ between datasets on the same grid
    df = ds.to_dataframe(dim_order=dim_order)
    n_levels = df.index.names.__len__()
    return df.reset_index(level=list(range(0, n_levels)))

def multicore_pooling(fun, fun_param, n_cores):
    '''
    Function to perform multiprocessing on n_cores
    Args:
        fun (obj): function to distribute
        fun_param zip(list): zip list of function arguments
        n_core (int): number of cores
    '''
    if n_cores is None:
        n_cores = mproc.cpu_count() - 2
        warnings.warn(f'WARNING: number of cores to use not provided. By default {n_cores} cores will be used')
    elif n_cores > mproc.cpu_count():
        n_cores = mproc.cpu_count() - 2
        warnings.warn(f'WARNING: Only {mproc.cpu_count()} cores available on this machine, reducing n_cores to {n_cores} ')

    # make sure it will run on one core at least
    if n_cores == 0:
        n_cores = 1

    pool = Pool(n_cores)
    pool.starmap(fun, fun_param)
    pool.close()
    pool.join()
    pool = None


def multithread_pooling(fun, fun_param, n_threads):
    '''
    Function to perform multiprocessing on n_threads
    Args:
        fun (obj): function to distribute
        fun_param zip(list): zip list of functino arguments
        n_core (int): number of threads
    '''
    tpool = ThreadPool(n_threads)
    tpool.starmap(fun, fun_param)
    tpool.close()
    tpool.join()
    tpool = None

def get_versionning():
    import TopoPyScale as tps
    from importlib import metadata
    try:
        repo = git.Repo(path=tps.__file__, search_parent_directories=True)
        hash = repo.head.object.hexsha
    except:
        hash = 'Not available'
    try:
        version = metadata.version('TopoPyScale')
    except (metadata.PackageNotFoundError, ImportError):
        # Fallback for development/source installations
        version = "dev"
    ver_dic = {'git_commit': hash,
               'package_version': version}
    return ver_dic



def FsmMetParser(file, freq="1h", resample=False):
    '''
    Parses FSM forcing files from toposcale sims
    '''

    df = pd.read_csv(file, delim_whitespace=True,
                     header=None, index_col='datetime',
                     parse_dates={'datetime': [0, 1, 2, 3]},
                     date_format=('%Y %m %d %H'))

    df.columns = ['ISWR', 'ILWR', 'Sf', 'Rf', 'TA', 'RH', 'VW', 'P']

    if resample == "TRUE":
        raise ValueError('ERROR: line 24 to be fixed!')
        #df = df.resample(freq).apply(resample_func)

    return (df)


def FsmSnowParser(file, freq="1h", resample=False):
    '''
    parses FSM output fuiles
    '''

    df = pd.read_csv(file, delim_whitespace=True, parse_dates=[[0, 1, 2]], header=None)
    df.set_index(df.iloc[:, 0], inplace=True)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.columns = ['albedo', 'Rof', 'HS', 'SWE', 'TS10', 'TS50']

    if resample == "TRUE":
        raise ValueError('ERROR: line 39 to be fixed!')
        #df = df.resample(freq).apply(resample_func)

    return (df)


#def FsmPlot(df):
#    df.plot(subplots=True)
#    plt.title("Sample=" + str(sampleN) + " Elev=" + str(np.round(lp.elevation[sampleN])))
#    plt.show()

def FsmPlot_ensemble(wdir, simvar, sampleN):
    """
    Function to plot ensemble results for a given variable and point.

    Args:
        wdir: full path to sim directory
        simvar: variable to plot (str)
        sampleN: sample number to plot
    Returns:
        sampleN: sample number

    Example:
        FsmPlot_ensemble('/home/joel/sim/topoPyscale_davos/', "HS", 0)

    """
    sampleNpad = "{:0>2}".format(sampleN)
    myensemble = glob.glob(wdir + "/fsm_sims/sim_ENS"+"*"+"_FSM_pt_"+str(sampleNpad)+".txt")
    lp = pd.read_csv(wdir+"/listpoints.csv")


    ax = plt.subplot(1, 1, 1)
    for ensembleN in myensemble:
        df = FsmSnowParser(ensembleN)
        plt.plot(df[simvar], alpha=0.3)

    plt.title("Sample="+str(sampleN) + " Elev="+str(np.round(lp.elevation[sampleN])))
    plt.show()

def SmetParser(file, doresample=True, freq='1h', NA_val=-999):
    '''
    val_file = full path to a smet
    resample = "TRUE" or "FALSE"
    freq = '1D' '1h' etc

    '''
    resample_func = "mean"
    # find statrt of data
    lookup = '[DATA]'
    with open(file) as myFile:
        for num, line in enumerate(myFile, 1):
            if lookup in line:
                skip = num

    df = pd.read_csv(file, delim_whitespace=True, index_col='datetime',
                     parse_dates={'datetime': [0]}, header=None, skiprows=skip, na_values=NA_val)

    # get col names
    lookup = 'fields'
    with open(file) as myFile:
        for num, line in enumerate(myFile, 1):
            if lookup in line:
                fields = num


    myfields = linecache.getline(file, fields)
    colnames = myfields.split("= ")[1]
    cnames = colnames.split(" ")

    df.columns = cnames[1:]

    if doresample:
        df = df.resample(freq).apply(resample_func)

    return (df)


def getCoordinatePixel(map_path, x_coord, y_coord, epsg_in, epsg_out):
    """
    Function to find which sample a point exists in. Can accept any combination of projections of the map and points.

    Args:
        map_path: full path to map
        x_coord: x coord of point
        y_coord: y coord of point
        epsg_in: input proj epsg code (lonlat: 4326)
        epsg_out: ouput proj epsg code

    Returns:
        sampleN: sample number

    Example:
        # get epsg of map
        from TopoPyScale import topo_da as da
        epsg_out, bbxox = da.projFromLandform(map_path)

        getCoordinatePixel("/home/joel/sim/topoPyscale_davos/landform.tif", 9.80933, 46.82945, 4326, 32632)

    """
    # define proj
    proj = pyproj.Transformer.from_crs(int(epsg_in), int(epsg_out), always_xy=True)
    # project
    x2, y2 = proj.transform(x_coord, y_coord)
    # open map
    dataset = rasterio.open(map_path)
    # get pixel x+y of the coordinate
    py, px = dataset.index(x2, y2)
    # create 1x1px window of the pixel
    window = rasterio.windows.Window(px - 1 // 2, py - 1 // 2, 1, 1)
    # read rgb values of the window
    clip = dataset.read(window=window)
    return (int(clip))

'''
Commented as it relies on topo_da.py which is currently having dependencies from gdal and osgeo. To Be Changed to Rasterio
def val_plots():
    # script from topo_utils (validation stuff)
    map_path = "/home/joel/sim/topoPyscale_davos/landform.tif"
    epsg_out, bbxox = da.projFromLandform(map_path)
    sample = getCoordinatePixel(map_path, 9.80933, 46.82945, 4326, epsg_out)
    filename = "/home/joel/sim/topoPyscale_davos/outputs/FSM_pt_" + str(sample-1)+ ".txt"
    df = FsmMetParser(filename, freq="1h", resample=False)
    wfj = SmetParser("/home/joel/data/wfj_optimal/WFJ_optimaldataset_v8.smet")

    start_date = df.index[0]
    end_date = df.index[-1]

    #greater than the start date and smaller than the end date
    mask = (wfj.index >= start_date) & (wfj.index <= end_date)
    wfj = wfj.loc[mask]


    def plot_xyline(ax):
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    ax = plt.subplot(2, 2, 1)
    plt.scatter(df.TA, wfj.TA, alpha=0.3)
    #r, bias, rmse = mystats(df.TA, wfj.TA)
    #plt.title("uncorrected rmse=" + str(round(rmse, 2)))
    plt.xlabel("Modelled")
    plt.ylabel("Measured")
    plot_xyline(ax)


    ax = plt.subplot(2, 2, 2)
    plt.scatter(df.ISWR, wfj.ISWR, alpha=0.3)
    #r, bias, rmse = mystats(df.TA, wfj.TA)
    #plt.title("uncorrected rmse=" + str(round(rmse, 2)))
    plt.xlabel("Modelled")
    plt.ylabel("Measured")
    plot_xyline(ax)


    ax = plt.subplot(2, 2, 3)
    plt.scatter(df.RH, wfj.RH*100, alpha=0.3)
    #r, bias, rmse = mystats(df.TA, wfj.TA)
    #plt.title("uncorrected rmse=" + str(round(rmse, 2)))
    plt.xlabel("Modelled")
    plt.ylabel("Measured")
    plot_xyline(ax)

    ax = plt.subplot(2, 2, 4)
    plt.scatter(df.ILWR, wfj.ILWR, alpha=0.3)
    #r, bias, rmse = mystats(df.TA, wfj.TA)
    #plt.title("uncorrected rmse=" + str(round(rmse, 2)))
    plt.xlabel("Modelled")
    plt.ylabel("Measured")
    plot_xyline(ax)
    plt.show()

    myfile = "/home/joel/sim/topoPyscale_davos/outputs/FSM_pt_00.txt"
    df = FsmMetParser(myfile)
    #FsmPlot(df)

    myfile = "/home/joel/sim/topoPyscale_davos/fsm_sims/sim_ENS1_FSM_pt_00.txt"
    df = FsmSnowParser(myfile)
    #FsmPlot(df)
'''

def time_vecs(start_date, end_date):
    """Function to build timestamp vectors of year starts and year ends for arbitrary time range.
  
    Args:
        start_date (str): _description_
        end_date (str): _description_

    Returns:
        start_year: vector of start of years (datetime64)
        end_year: vector of end of year (datetime64)
    """
    
    current_start = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    start_year = []
    end_year = []

    while current_start <= end_date:
        # Append the current start date
        start_year.append(current_start)

        # Calculate the end date for the current cycle
        next_start = current_start + pd.DateOffset(years=1)
        current_end = next_start - pd.DateOffset(days=1)

        # If the cycle's end date goes beyond the overall end date, break and add the final segment
        if current_end > end_date:
            current_end = end_date
            end_year.append(current_end)
            break

        # Append the current end date
        end_year.append(current_end)

        # Update current start date to next year start
        current_start = next_start

    start_year = pd.to_datetime(start_year)
    end_year = pd.to_datetime(end_year)
    
    return start_year, end_year



# ---------------------------------------------------------------------------------------------
#   Cached intermediate files (ds_param, ds_solar, da_horizon) as NetCDF *or* Zarr
#   The format follows the file name given in the config: 'xxx.zarr' -> Zarr store, else NetCDF.
#   A Zarr store is a directory, so "does the cache exist" cannot use Path.is_file().
# ---------------------------------------------------------------------------------------------

def is_zarr(path):
    """True if `path` names a Zarr store (by its '.zarr' suffix)."""
    return str(path).rstrip('/').endswith('.zarr')


def dataset_exists(path):
    """True if a cached NetCDF file or a complete Zarr store exists at `path`."""
    from pathlib import Path
    path = Path(path)
    if is_zarr(path):
        return path.is_dir() and ((path / 'zarr.json').is_file() or (path / '.zgroup').is_file())
    return path.is_file()


def open_dataset(path, **kwargs):
    """Open a cached NetCDF file or Zarr store. kwargs (e.g. engine, chunks) apply to NetCDF only,
    except `chunks`, which is passed to both."""
    if is_zarr(path):
        return xr.open_zarr(str(path), chunks=kwargs.get('chunks', {}))
    return xr.open_dataset(path, **kwargs)


def open_dataarray(path, **kwargs):
    """Open a single-variable cached NetCDF file or Zarr store as a DataArray (like xr.open_dataarray)."""
    if is_zarr(path):
        ds = xr.open_zarr(str(path), chunks=kwargs.get('chunks', {}))
        if len(ds.data_vars) != 1:
            raise ValueError(f'{path} holds {len(ds.data_vars)} variables, expected one')
        return ds[list(ds.data_vars)[0]]
    return xr.open_dataarray(path, **kwargs)
