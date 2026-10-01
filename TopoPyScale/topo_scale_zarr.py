"""
S. Filhol, 2025. Reworked 2026 (Rila-Pirin project, see PROGRESS.md).

TopoScale downscaling reading ERA5 from ONE local Zarr store, points processed in parallel.

Input store (built by fetch_era5_edh.fetch_era5_edh, or fetch_era5.convert_netcdf_stack_to_zarr):
    PLEV  z, t, u, v, q, r   (time, level ascending [hPa], latitude, longitude)
    SURF  z_surf, d2m, sp, strd, ssrd, tp, t2m   (time, latitude, longitude)

Physics: identical to the classic path `topo_scale.pt_downscale_interp` +
`topo_scale.pt_downscale_radiations` (checked by tests/test_topo_scale_zarr_parity.py).
Pressure-level and surface fields are kept in separate datasets, as in the classic path,
because the meteo_util helpers write to fixed variable names (q, w, vp) and would overwrite
each other in a merged dataset. Accumulations follow commit 91ab64c: ERA5 values are 1-hour
accumulations, so radiation / 3600 s and tp x timestep.

What changed compared to the 2025 version, and why:
    - q from pressure levels was overwritten by the surface q (merged dataset) -> separate datasets
    - precip lapse-rate line did not run (`... / {} (1 - ...)`: calling a dict) -> classic formula
    - radiation divided by timestep seconds, tp divided by timestep -> classic (91ab64c) convention
    - float32 arithmetic -> float64, and outputs rounded to 5 decimals, as in the classic path
    - errors were swallowed ("Error processing subset") -> the real exception propagates
    - output store recreated with mode='w' on every run -> resumable: existing store reused,
      finished points recorded in <store>.progress/ and skipped on rerun
    - horizon angles: read from the `hori_azi_*` columns that Topoclass.compute_horizon()
      already sampled into df_centroids (same nearest-neighbour lookup as before, without
      re-reading the multi-GB horizon file for every point)

Parallelism
    multicore: a multiprocessing Pool; each worker opens the input store once, then handles
               one point at a time (reads the 3x3 ERA5 cells around it for the whole period,
               downscales in memory, writes its own slice of the output).
    dask:      same work function submitted to a dask.distributed LocalCluster.
    Points are independent and each writes to its own output chunk, so no locking is needed.
"""
import multiprocessing as mproc
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

try:
    from zarr.codecs import BloscCodec
except ImportError:
    try:
        from zarr.codecs._blosc import BloscCodec
    except ImportError:
        from zarr import Blosc as BloscCodec

from TopoPyScale import meteo_util as mu
from TopoPyScale import topo_utils as tu

g = 9.81      # Acceleration of gravity [ms^-1]
R = 287.05    # Gas constant for dry air [JK^-1kg^-1]
SECONDS_PER_ACCUMULATION = 3600   # ERA5 accumulations cover the previous 1 hour (commit 91ab64c)

PLEV_VARS = ['z', 't', 'u', 'v', 'q', 'r']
SURF_VARS = ['z_surf', 'd2m', 'sp', 'strd', 'ssrd', 'tp', 't2m']

varout_default = ['t', 'u', 'v', 'q', 'p', 'tp', 'wd', 'ws', 'w', 'LW', 'SW_diffuse', 'SW_direct', 'SW']


# =====================================================================================
#   Physics for one point (ported line by line from topo_scale.py)
# =====================================================================================

def horizon_for_point(row, solar_azimuth_rad, da_horizon=None):
    """
    Horizon angle [deg] along the sun's azimuth for each timestep.

    Same result as `da_horizon.sel(x=row.x, y=row.y, azimuth=np.rad2deg(solar_azimuth), method='nearest')`
    (the classic code), but from the `hori_azi_<az>` columns Topoclass.compute_horizon() stored in
    df_centroids with that same nearest-neighbour sampling. Falls back to da_horizon if absent.
    """
    cols = [c for c in vars(row) if c.startswith('hori_azi_')] if isinstance(row, SimpleNamespace) else \
        [c for c in row.index if c.startswith('hori_azi_')]
    az_target = np.rad2deg(np.asarray(solar_azimuth_rad))
    if cols:
        az_bins = np.array([float(c[len('hori_azi_'):]) for c in cols])
        order = np.argsort(az_bins)
        az_bins = az_bins[order]
        values = np.array([getattr(row, c) if isinstance(row, SimpleNamespace) else row[c] for c in cols])[order]
        idx = pd.Index(az_bins).get_indexer(az_target, method='nearest')   # what xarray .sel(method='nearest') does
        return values[idx]
    if da_horizon is None:
        raise ValueError('No hori_azi_* columns in df_centroids and no da_horizon given')
    return da_horizon.sel(x=row.x, y=row.y, azimuth=az_target, method='nearest').values


def downscale_point(row, plev_pt, surf_pt, solar_pt, horizon_values, meta):
    """
    TopoScale for one point.

    Args:
        row: centroid (x, y, elevation, slope, aspect, svf, point_name, point_ind, ...)
        plev_pt: pressure levels on the 3x3 cells around the point (time, level, latitude, longitude), z in m2/s2
        surf_pt: surface fields on the same cells (time, latitude, longitude), z_surf in m2/s2
        solar_pt: ds_solar for this point (time): sunset, mu0, SWtoa, zenith, azimuth, elevation
        horizon_values: horizon angle along the solar azimuth for each timestep [deg]
        meta: interp_method, lw_terrain_flag, tstep (hours), target_epsg, precip_lapse_rate_flag, transformer

    Returns:
        xr.Dataset (time) with the downscaled variables, rounded to 5 decimals like topo_scale.py
    """
    pt_id = row.point_name

    # ---- inputs as in the classic path: z converted to m (in the stored dtype), then float64
    surf_pt = surf_pt.rename({'z_surf': 'z'})
    plev_pt['z'] = plev_pt.z / g
    surf_pt['z'] = surf_pt.z / g
    plev_pt = plev_pt.astype('float64')
    surf_pt = surf_pt.astype('float64')

    # ====== Horizontal interpolation ====================
    interp_method = meta.get('interp_method')
    lons, lats = np.meshgrid(plev_pt.longitude.values, plev_pt.latitude.values)
    trans = meta.get('transformer') or Transformer.from_crs("epsg:4326", f"epsg:{meta.get('target_epsg')}", always_xy=True)
    Xs, Ys = trans.transform(lons.flatten(), lats.flatten())
    Xs, Ys = Xs.reshape(lons.shape), Ys.reshape(lons.shape)

    dist = np.sqrt((row.x - Xs) ** 2 + (row.y - Ys) ** 2)
    if interp_method == 'idw':
        idw = 1 / (dist ** 2)
        weights = idw / np.sum(idw)
    elif interp_method == 'linear':
        weights = dist / np.sum(dist)
    else:
        raise ValueError(f'interpolation method {interp_method} not available')
    da_idw = xr.DataArray(data=weights, dims=["latitude", "longitude"],
                          coords={"latitude": plev_pt.latitude.values, "longitude": plev_pt.longitude.values}
                          ).astype('float64')
    plev_interp = plev_pt.weighted(da_idw).sum(['longitude', 'latitude'], keep_attrs=True)
    surf_interp = surf_pt.weighted(da_idw).sum(['longitude', 'latitude'], keep_attrs=True)

    # ============ Specific humidity at surface, dew point on levels ============
    surf_interp = mu.dewT_2_q_magnus(surf_interp, mu.var_era_surf)
    plev_interp = mu.t_rh_2_dewT(plev_interp, mu.var_era_plevel)

    down_pt = xr.Dataset(coords={'time': plev_interp.time, 'point_name': pt_id})

    if (row.elevation < plev_interp.z.isel(level=-1)).sum():
        pt_elev_diff = np.round(np.min(row.elevation - plev_interp.z.isel(level=-1).values), 0)
        print(f"---> WARNING: Point {pt_id} is {pt_elev_diff} m lower than the {plev_interp.isel(level=-1).level.data} hPa geopotential\n=> "
              "Values sampled from Psurf and lowest Plevel. No vertical interpolation", flush=True)
        vertical_fallback = True
        ind_z_top = (plev_interp.where(plev_interp.z > row.elevation).z - row.elevation).argmin('level')
        top = plev_interp.isel(level=ind_z_top)
        down_pt['t'] = top['t']
        down_pt['u'] = top.u
        down_pt['v'] = top.v
        down_pt['q'] = top.q
        down_pt['p'] = top.level * (10 ** 2) * np.exp(-(row.elevation - top.z) / (0.5 * (top.t + down_pt.t) * R / g))
    else:
        vertical_fallback = False
        # ========== Vertical interpolation at the DEM surface z  ===============
        ind_z_bot = (plev_interp.where(plev_interp.z < row.elevation).z - row.elevation).argmax('level')
        try:
            ind_z_top = (plev_interp.where(plev_interp.z > row.elevation).z - row.elevation).argmin('level')
        except Exception as e:
            raise ValueError(f'ERROR: Upper pressure level {plev_interp.level.min().values} hPa geopotential is '
                             f'lower than cluster mean elevation {row.elevation}') from e
        top = plev_interp.isel(level=ind_z_top)
        bot = plev_interp.isel(level=ind_z_bot)
        dist = np.array([np.abs(bot.z - row.elevation).values, np.abs(top.z - row.elevation).values])
        weights = dist / np.sum(dist, axis=0)
        down_pt['t'] = bot.t * weights[1] + top.t * weights[0]
        down_pt['u'] = bot.u * weights[1] + top.u * weights[0]
        down_pt['v'] = bot.v * weights[1] + top.v * weights[0]
        down_pt['q'] = bot.q * weights[1] + top.q * weights[0]
        down_pt['p'] = top.level * (10 ** 2) * np.exp(-(row.elevation - top.z) / (0.5 * (top.t + down_pt.t) * R / g))

    # ============ Precipitation, wind ============
    down_pt['month'] = ('time', down_pt.time.dt.month.data)
    if meta.get('precip_lapse_rate_flag'):
        monthly_coeffs = xr.Dataset({'coef': (['month'], [0.35, 0.35, 0.35, 0.3, 0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35])},
                                    coords={'month': np.arange(1, 13)})
        coef = monthly_coeffs.coef.sel(month=down_pt.month.values).data
        elev_diff = (row.elevation - surf_interp.z) * 1e-3
        down_pt['precip_lapse_rate'] = (1 + coef * elev_diff) / (1 - coef * elev_diff)
    else:
        down_pt['precip_lapse_rate'] = down_pt.t * 0 + 1

    down_pt['tp'] = down_pt.precip_lapse_rate * surf_interp.tp * 1 * meta.get('tstep') * 10 ** 3
    down_pt['theta'] = np.arctan2(-down_pt.u, -down_pt.v)
    down_pt['theta_neg'] = (down_pt.theta < 0) * (down_pt.theta + 2 * np.pi)
    down_pt['theta_pos'] = (down_pt.theta >= 0) * down_pt.theta
    down_pt = down_pt.drop_vars('theta')
    down_pt['wd'] = (down_pt.theta_pos + down_pt.theta_neg)
    down_pt['ws'] = np.sqrt(down_pt.u ** 2 + down_pt.v ** 2)
    down_pt = down_pt.drop_vars(['theta_pos', 'theta_neg', 'month'])

    # ============ Longwave ============
    x1, x2 = 0.43, 5.7
    sbc = 5.67e-8
    down_pt = mu.mixing_ratio(down_pt, mu.var_era_plevel)
    down_pt = mu.vapor_pressure(down_pt, mu.var_era_plevel)
    surf_interp = mu.mixing_ratio(surf_interp, mu.var_era_surf)
    surf_interp = mu.vapor_pressure(surf_interp, mu.var_era_surf)
    down_pt['cse'] = 0.23 + x1 * (down_pt.vp / down_pt.t) ** (1 / x2)
    surf_interp['cse'] = 0.23 + x1 * (surf_interp.vp / surf_interp.t2m) ** (1 / x2)
    tstep_seconds = SECONDS_PER_ACCUMULATION
    surf_interp['cle'] = (surf_interp.strd / tstep_seconds) / (sbc * surf_interp.t2m ** 4) - surf_interp['cse']
    surf_interp['aef'] = down_pt['cse'] + surf_interp['cle']
    if meta.get('lw_terrain_flag'):
        down_pt['LW'] = row.svf * surf_interp['aef'] * sbc * down_pt.t ** 4 + \
                        0.5 * (1 + np.cos(row.slope)) * (1 - row.svf) * 0.99 * 5.67e-8 * (273.15 ** 4)
    else:
        down_pt['LW'] = row.svf * surf_interp['aef'] * sbc * down_pt.t ** 4

    # ============ Shortwave ============
    kt = surf_interp.ssrd * 0
    sunset = solar_pt.sunset.astype(bool)
    mu0 = solar_pt.mu0
    SWtoa = solar_pt.SWtoa
    kt[~sunset] = (surf_interp.ssrd[~sunset] / tstep_seconds) / SWtoa[~sunset]
    kt[kt < 0] = 0
    kt[kt > 1] = 1
    kd = 0.952 - 1.041 * np.exp(-1 * np.exp(2.3 - 4.702 * kt))

    surf_interp['SW'] = surf_interp.ssrd / tstep_seconds
    surf_interp['SW'][surf_interp['SW'] < 0] = 0
    surf_interp['SW_diffuse'] = kd * surf_interp.SW
    down_pt['SW_diffuse'] = row.svf * surf_interp.SW_diffuse
    surf_interp['SW_direct'] = surf_interp.SW - surf_interp.SW_diffuse
    ka = surf_interp.ssrd * 0
    ka[~sunset] = (g * mu0[~sunset] / down_pt.p) * np.log(SWtoa[~sunset] / surf_interp.SW_direct[~sunset])
    down_pt['cos_illumination_tmp'] = mu0 * np.cos(row.slope) + np.sin(solar_pt.zenith) * \
                                      np.sin(row.slope) * np.cos(solar_pt.azimuth - row.aspect)
    down_pt['cos_illumination'] = down_pt.cos_illumination_tmp * (down_pt.cos_illumination_tmp > 0)
    down_pt = down_pt.drop_vars(['cos_illumination_tmp'])
    down_pt['cos_illumination'][down_pt['cos_illumination'] < 0] = 0

    shade = xr.DataArray(horizon_values, dims='time', coords={'time': solar_pt.time}) > solar_pt.elevation
    down_pt['SW_direct_tmp'] = down_pt.t * 0
    down_pt['SW_direct_tmp'][~sunset] = SWtoa[~sunset] * np.exp(-ka[~sunset] * down_pt.p[~sunset] / (g * mu0[~sunset]))
    down_pt['SW_direct'] = down_pt.t * 0
    down_pt['SW_direct'][~sunset] = down_pt.SW_direct_tmp[~sunset] * (down_pt.cos_illumination[~sunset] / mu0[~sunset]) * (1 - shade)
    down_pt['SW'] = down_pt.SW_diffuse + down_pt.SW_direct
    down_pt = down_pt.drop_vars(['level', 'SW_direct_tmp']).round(5)
    down_pt.attrs['vertical_fallback'] = int(vertical_fallback)   # 1: point below z(lowest level) at some hour

    down_pt.t.attrs = {'units': 'K', 'long_name': 'Temperature', 'standard_name': 'air_temperature'}
    down_pt.q.attrs = {'units': 'kg kg**-1', 'long_name': 'Specific humidity', 'standard_name': 'specific_humidity'}
    down_pt.u.attrs = {'units': 'm s**-1', 'long_name': 'U component of wind', 'standard_name': 'eastward_wind'}
    down_pt.v.attrs = {'units': 'm s**-1', 'long_name': 'V component of wind', 'standard_name': 'northward wind'}
    down_pt.p.attrs = {'units': 'bar', 'long_name': 'Pression atmospheric', 'standard_name': 'pression_atmospheric'}
    down_pt.ws.attrs = {'units': 'm s**-1', 'long_name': 'Wind speed', 'standard_name': 'wind_speed'}
    down_pt.wd.attrs = {'units': 'deg', 'long_name': 'Wind direction', 'standard_name': 'wind_direction'}
    down_pt.tp.attrs = {'units': 'mm hr**-1', 'long_name': 'Precipitation', 'standard_name': 'precipitation'}
    down_pt.LW.attrs = {'units': 'W m**-2', 'long_name': 'Surface longwave radiation downwards',
                        'standard_name': 'longwave_radiation_downward'}
    down_pt.cse.attrs = {'units': 'xxx', 'standard_name': 'Clear sky emissivity'}
    down_pt.SW.attrs = {'units': 'W m**-2', 'long_name': 'Surface solar radiation downwards',
                        'standard_name': 'shortwave_radiation_downward'}
    down_pt.SW_diffuse.attrs = {'units': 'W m**-2', 'long_name': 'Surface solar diffuse radiation downwards',
                                'standard_name': 'shortwave_diffuse_radiation_downward'}
    return down_pt


# =====================================================================================
#   Per-process worker state (multicore and dask share this)
# =====================================================================================

_WORKER = {}


def _init_worker(ctx):
    """Open the inputs once per worker process."""
    _WORKER.clear()
    _WORKER.update(ctx)
    _WORKER['ERA'] = xr.open_zarr(str(ctx['era5_zarr_path']))
    _WORKER['meta'] = dict(ctx['meta'], transformer=Transformer.from_crs(
        "epsg:4326", f"epsg:{ctx['meta']['target_epsg']}", always_xy=True))


def _run_point(i):
    """Downscale centroid number i (row position in df_centroids) and store it. Returns seconds."""
    t0 = time.time()
    w = _WORKER
    row = w['rows'][i]
    ERA = w['ERA']
    ilat = np.abs(ERA.latitude.values - row.lat).argmin()
    ilon = np.abs(ERA.longitude.values - row.lon).argmin()
    cells = dict(latitude=[ilat - 1, ilat, ilat + 1], longitude=[ilon - 1, ilon, ilon + 1])
    if min(ilat, ilon) < 1 or ilat + 1 >= ERA.latitude.size or ilon + 1 >= ERA.longitude.size:
        raise ValueError(f'Point {row.point_name} is at the edge of the ERA5 domain; fetch a larger bbox')
    sub = ERA.sel(time=w['tvec']).isel(**cells)
    plev_pt = sub[PLEV_VARS].load()
    surf_pt = sub[SURF_VARS].load()
    solar_pt = w['ds_solar'].sel(point_name=row.point_name).sel(time=w['tvec']).load()
    hori = horizon_for_point(row, solar_pt.azimuth.values, w.get('da_horizon'))
    res = downscale_point(row, plev_pt, surf_pt, solar_pt, hori, w['meta'])
    fallback = bool(res.attrs.get('vertical_fallback', 0))
    _store_point(res, i, row)
    return time.time() - t0, fallback


def _store_point(res, i, row):
    w = _WORKER
    if w['output_format'] == 'zarr':
        out = res[w['varout']].drop_vars(['point_name'], errors='ignore').expand_dims(point_ind=[row.point_ind])
        for v in out.variables:
            out[v].encoding = {}
        # region writes take only variables along the region dims; coordinates are already in the store
        out = out.drop_vars([v for v in out.variables if v in ('time', 'point_ind')
                             or not set(out[v].dims) & {'point_ind', 'time'}])
        out.to_zarr(w['store'], region={'point_ind': slice(i, i + 1), 'time': slice(0, len(w['tvec']))},
                    mode='r+', zarr_format=3, consolidated=False)
        (w['progress_dir'] / f'{i}.done').touch()
    else:
        ver = tu.get_versionning()
        res.attrs = {'title': 'Downscaled timeseries with TopoPyScale',
                     'created with': 'TopoPyScale (topo_scale_zarr), see https://topopyscale.readthedocs.io',
                     'package_version': ver.get('package_version'), 'git_commit': ver.get('git_commit'),
                     'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        f = w['output_path'] / w['file_pattern'].replace('*', str(row.point_name))
        tmp = f.with_name(f.name + '.tmp')
        comp = dict(zlib=True, complevel=5)
        res.to_netcdf(tmp, engine='h5netcdf', encoding={v: comp for v in res.data_vars})
        os.replace(tmp, f)


def _run_point_safe(i):
    try:
        secs, fallback = _run_point(i)
        return i, secs, None, fallback
    except Exception:  # report which point failed, with the real traceback
        return i, None, traceback.format_exc(), False


# =====================================================================================
#   Driver
# =====================================================================================

class ClimateDownscaler:
    def __init__(self,
                 era5_zarr_path,
                 output_path,
                 df_centroids,
                 da_horizon,
                 ds_solar,
                 target_EPSG,
                 start_date,
                 end_date,
                 tstep,
                 varout=varout_default,
                 interp_method='idw',
                 lw_terrain_flag=True,
                 precip_lapse_rate_flag=False,
                 file_pattern='down_pt*.nc',
                 store_name=None,
                 overwrite=False):
        """
        Args:
            era5_zarr_path: input ERA5 Zarr store (see module docstring)
            output_path: directory for outputs (netcdf files, or the Zarr store `store_name`)
            df_centroids (DataFrame): points (from Topoclass), incl. hori_azi_* columns if available
            da_horizon (DataArray): horizon angles; only used if df_centroids lacks hori_azi_* columns
            ds_solar (Dataset): solar geometry (point_name, time)
            target_EPSG: EPSG code of the DEM
            start_date, end_date: period (end day included)
            tstep (str): '1h', '3h' or '6h'
            varout (list): variables written to the Zarr store
            file_pattern (str): per-point netcdf output 'xxx_*.nc' (used when store_name is None)
            store_name (str): 'xxx.zarr' -> one Zarr store (point_ind, time) instead of netcdf files
            overwrite (bool): False (default) resumes: finished points are skipped. True starts over.
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.era5_zarr_path = Path(era5_zarr_path)
        self.df_centroids = df_centroids
        self.ds_solar = ds_solar
        self.da_horizon = da_horizon
        self.varout = list(varout)
        self.overwrite = overwrite
        self.tvec = pd.date_range(start_date, pd.to_datetime(end_date) + pd.to_timedelta('1D'), freq=tstep, inclusive='left')
        tstep_dict = {'1h': 1, '3h': 3, '6h': 6}
        if tstep not in tstep_dict:
            raise ValueError(f'tstep must be one of {list(tstep_dict)}')
        self.n_digits = len(str(self.df_centroids.index.max()))
        self.meta = {'interp_method': interp_method, 'lw_terrain_flag': lw_terrain_flag, 'tstep': tstep_dict[tstep],
                     'n_digits': self.n_digits, 'file_pattern': file_pattern, 'target_epsg': target_EPSG,
                     'precip_lapse_rate_flag': precip_lapse_rate_flag, 'output_directory': self.output_path}

        if (store_name is None) == (file_pattern is None):
            raise ValueError('Give exactly one of store_name (Zarr output) or file_pattern (netcdf output)')
        if store_name is not None:
            if not str(store_name).endswith('.zarr'):
                raise ValueError("store_name must be xxx.zarr")
            self.output_format = 'zarr'
            self.store_name = store_name
            self.store = self.output_path / store_name
            self.progress_dir = self.output_path / f'{store_name}.progress'
        else:
            if not str(file_pattern).endswith('*.nc'):
                raise ValueError("file_pattern must finish with *.nc")
            self.output_format = 'netcdf'
            self.file_pattern = file_pattern
        self._check_inputs()

    # ---------------------------------------------------------------- checks / setup
    def _check_inputs(self):
        ERA = xr.open_zarr(str(self.era5_zarr_path))
        missing = [v for v in PLEV_VARS + SURF_VARS if v not in ERA]
        if missing:
            raise KeyError(f'{self.era5_zarr_path} lacks variables {missing}')
        if not np.all(np.diff(ERA.level.values) > 0):
            raise ValueError('ERA5 store levels must be ascending in hPa (level[-1] = lowest level)')
        have = pd.DatetimeIndex(ERA.time.values)
        lacking = self.tvec.difference(have)
        if len(lacking):
            raise ValueError(f'ERA5 store lacks {len(lacking)} requested timesteps, first {lacking[:3].tolist()}')
        missing_t = self.tvec.difference(pd.DatetimeIndex(self.ds_solar.time.values))
        if len(missing_t):
            raise ValueError(f'ds_solar lacks {len(missing_t)} timesteps of the period')

    def setup_output_store(self):
        """Create the output Zarr store, or reuse a compatible existing one (resume)."""
        points = self.df_centroids.point_ind.values
        if self.store.exists() and self.overwrite:
            shutil.rmtree(self.store)
            shutil.rmtree(self.progress_dir, ignore_errors=True)
        if self.store.exists():
            old = xr.open_zarr(self.store)
            same = (old.sizes.get('time') == len(self.tvec) and np.array_equal(old.point_ind.values, points)
                    and pd.Timestamp(old.time.values[0]) == self.tvec[0] and set(self.varout) <= set(old.data_vars))
            if not same:
                raise ValueError(f'{self.store} exists but was made for other points/period/variables. '
                                 f'Delete it or use overwrite=True.')
            print(f'---> Reusing output store {self.store.name} (resume)')
            return
        shutil.rmtree(self.progress_dir, ignore_errors=True)
        shape = (len(points), len(self.tvec))
        chunks = (1, len(self.tvec))
        ds = xr.Dataset(
            {v: (('point_ind', 'time'), da.full(shape, np.nan, chunks=chunks, dtype='float64')) for v in self.varout},
            coords={'time': self.tvec, 'point_ind': points,
                    'point_name': ('point_ind', np.array(self.df_centroids.point_name.astype(str).tolist(), dtype='U'))})
        comp = BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle', blocksize=0)
        ds.attrs = {'title': 'Downscaled timeseries with TopoPyScale (topo_scale_zarr)',
                    'input_era5': str(self.era5_zarr_path), 'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ds.to_zarr(self.store, mode='w', compute=False, zarr_format=3, consolidated=True,
                   encoding={v: {'compressors': comp} for v in self.varout})
        print(f'---> Output store {self.store.name} created ({shape[0]} points x {shape[1]} timesteps)')

    def _todo(self):
        n = len(self.df_centroids)
        if self.output_format == 'zarr':
            self.progress_dir.mkdir(parents=True, exist_ok=True)
            done = {int(f.stem) for f in self.progress_dir.glob('*.done')}
        else:
            names = self.df_centroids.point_name.astype(str).values
            done = {i for i in range(n) if (self.output_path / self.file_pattern.replace('*', names[i])).exists()}
        return [i for i in range(n) if i not in done], len(done)

    def _context(self):
        cols = self.df_centroids.columns.tolist()
        rows = [SimpleNamespace(**dict(zip(cols, r))) for r in self.df_centroids.itertuples(index=False, name=None)]
        has_hori = any(c.startswith('hori_azi_') for c in cols)
        ctx = {'era5_zarr_path': str(self.era5_zarr_path), 'rows': rows, 'tvec': self.tvec,
               'ds_solar': self.ds_solar, 'da_horizon': None if has_hori else self.da_horizon,
               'meta': self.meta, 'varout': self.varout, 'output_format': self.output_format,
               'output_path': self.output_path}
        if self.output_format == 'zarr':
            ctx.update(store=str(self.store), progress_dir=self.progress_dir)
        else:
            ctx.update(file_pattern=self.file_pattern)
        return ctx

    def _report(self, results, n_done_before, t0):
        failed = [(i, err) for i, _, err, _ in results if err]
        secs = [s for _, s, err, _ in results if not err]
        fallback = sorted(str(self.df_centroids.point_name.iloc[i]) for i, _, err, fb in results if fb and not err)
        el = time.time() - t0
        print(f'---> Downscaling: {len(secs)} points done this run ({n_done_before} earlier), {len(failed)} failed, '
              f'{el:.0f}s wall, {np.mean(secs) if secs else float("nan"):.1f}s per point per worker')
        if fallback:
            print(f'---> {len(fallback)} point(s) lie below the lowest pressure level at some hour and use the '
                  f'no-vertical-interpolation fallback: {fallback}')
        self.vertical_fallback_points = fallback
        if failed:
            i, err = failed[0]
            raise RuntimeError(f'{len(failed)} point(s) failed, e.g. point #{i} '
                               f'({self.df_centroids.point_name.iloc[i]}):\n{err}')

    # ---------------------------------------------------------------- parallel runs
    def multicore_parallel_process_multiple_subsets(self, n_core=4):
        t0 = time.time()
        if self.output_format == 'zarr':
            self.setup_output_store()
        todo, n_done = self._todo()
        print(f'---> {len(todo)} points to downscale on {n_core} cores ({n_done} already done), '
              f'{len(self.tvec)} timesteps')
        if not todo:
            return
        n_core = max(1, min(n_core, mproc.cpu_count(), len(todo)))
        results = []
        # 'forkserver', not 'fork': zarr v3 runs a background I/O thread in this process, and forking a
        # process with running threads can deadlock the children. forkserver starts clean workers.
        # Preload only this module in the fork server. The default ('__main__') re-imports the calling
        # script, which re-runs a whole pipeline if the script lacks an `if __name__ == '__main__':` guard.
        ctx = mproc.get_context('forkserver')
        ctx.set_forkserver_preload(['TopoPyScale.topo_scale_zarr'])
        with ctx.Pool(n_core, initializer=_init_worker, initargs=(self._context(),)) as pool:
            for k, r in enumerate(pool.imap_unordered(_run_point_safe, todo), 1):
                results.append(r)
                if k % max(1, len(todo) // 20) == 0 or k == len(todo):
                    el = time.time() - t0
                    print(f'---> {k}/{len(todo)} points, {el:.0f}s elapsed, ETA {el / k * (len(todo) - k):.0f}s', flush=True)
        self._report(results, n_done, t0)

    def dask_parallel_process_multiple_subsets(self, dask_worker=None):
        from dask.distributed import LocalCluster, Client
        dask_worker = dask_worker or {'n_workers': 4, 'threads_per_worker': 1, 'memory_target_fraction': 0.95,
                                      'memory_limit': '1.5GB'}
        t0 = time.time()
        if self.output_format == 'zarr':
            self.setup_output_store()
        todo, n_done = self._todo()
        if not todo:
            return
        ctx = self._context()
        with LocalCluster(processes=True, **dask_worker) as cluster, Client(cluster) as client:
            print(f"Dask client started with {len(client.scheduler_info()['workers'])} workers")
            client.run(_init_worker, ctx)
            results = client.gather(client.map(_run_point_safe, todo, pure=False))
        self._report(results, n_done, t0)

    def downscale_parallel(self, parallel_method='multicore', n_core=4, dask_worker=None):
        if parallel_method == 'multicore':
            self.multicore_parallel_process_multiple_subsets(n_core)
        elif parallel_method == 'dask':
            self.dask_parallel_process_multiple_subsets(dask_worker)
        else:
            raise ValueError('Method not available')


def open_downscaled_store(store):
    """Open a topo_scale_zarr output store with `point_name` as dimension (what Topoclass exports expect)."""
    ds = xr.open_zarr(store)
    return ds.swap_dims({'point_ind': 'point_name'})
