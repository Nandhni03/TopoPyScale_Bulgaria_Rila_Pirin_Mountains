"""
Parity test: topo_scale_zarr.ClimateDownscaler must give the same numbers as the classic,
reference implementation topo_scale.downscale_climate on identical inputs.

A synthetic but physically plausible ERA5 sample (2 days, hourly, 0.25 deg around Rila-Pirin) is
written twice: as yearly PLEV/SURF NetCDF files (classic input) and as one Zarr store (zarr input).
Three points cover both vertical branches: a normal mountain point, a high point, and a point
below the 1000 hPa geopotential (surface fallback, no vertical interpolation).

Run:  python -m pytest tests/test_topo_scale_zarr_parity.py -v
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Transformer

from TopoPyScale import topo_scale as ta
from TopoPyScale import topo_scale_zarr as tz
from TopoPyScale import solar_geom as sg

EPSG = 32634
START, END = '2025-01-10', '2025-01-11'
LEVELS = np.array([600., 700., 850., 925., 1000.])
LATS = np.arange(42.75, 40.99, -0.25)
LONS = np.arange(22.5, 24.51, 0.25)


def _era(seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range(START, pd.Timestamp(END) + pd.Timedelta('23h'), freq='h')
    nt, nl, ny, nx = len(time), len(LEVELS), len(LATS), len(LONS)
    hour = time.hour.values[:, None, None]
    # geopotential heights of the levels (m), slight spatial/temporal variation
    zh = np.array([4200., 3000., 1450., 760., 110.])[None, :, None, None] + rng.normal(0, 15, (nt, nl, ny, nx))
    t = (288. - 6.5e-3 * zh) + 3 * np.sin(2 * np.pi * (time.hour.values - 9) / 24)[:, None, None, None]
    t[:, 3:, 4:6, 3:5] += 4.0          # a low-level inversion patch
    plev = xr.Dataset(
        {'z': (('time', 'level', 'latitude', 'longitude'), (zh * 9.81).astype('float32')),
         't': (('time', 'level', 'latitude', 'longitude'), t.astype('float32')),
         'u': (('time', 'level', 'latitude', 'longitude'), rng.normal(3, 4, (nt, nl, ny, nx)).astype('float32')),
         'v': (('time', 'level', 'latitude', 'longitude'), rng.normal(-1, 4, (nt, nl, ny, nx)).astype('float32')),
         'q': (('time', 'level', 'latitude', 'longitude'), rng.uniform(5e-4, 4e-3, (nt, nl, ny, nx)).astype('float32')),
         'r': (('time', 'level', 'latitude', 'longitude'), rng.uniform(30, 95, (nt, nl, ny, nx)).astype('float32'))},
        coords={'time': time, 'level': LEVELS, 'latitude': LATS, 'longitude': LONS})
    t2m = 272. + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1, (nt, ny, nx))
    day = np.clip(np.sin(np.pi * (hour - 6) / 10), 0, None)
    surf = xr.Dataset(
        {'z': (('time', 'latitude', 'longitude'), (rng.uniform(400, 1600, (1, ny, nx)) * 9.81 + np.zeros((nt, ny, nx))).astype('float32')),
         'd2m': (('time', 'latitude', 'longitude'), (t2m - rng.uniform(1, 6, (nt, ny, nx))).astype('float32')),
         'sp': (('time', 'latitude', 'longitude'), rng.uniform(84000, 96000, (nt, ny, nx)).astype('float32')),
         'strd': (('time', 'latitude', 'longitude'), (rng.uniform(230, 320, (nt, ny, nx)) * 3600).astype('float32')),
         'ssrd': (('time', 'latitude', 'longitude'), (day * rng.uniform(100, 450, (nt, ny, nx)) * 3600).astype('float32')),
         'tp': (('time', 'latitude', 'longitude'), rng.exponential(2e-4, (nt, ny, nx)).astype('float32')),
         't2m': (('time', 'latitude', 'longitude'), t2m.astype('float32'))},
        coords={'time': time, 'latitude': LATS, 'longitude': LONS})
    return plev, surf


def _points():
    pts = pd.DataFrame({'lat': [42.18, 41.77, 41.95], 'lon': [23.58, 23.40, 23.10],
                        'elevation': [1800., 2850., 60.],             # last one is below z(1000 hPa)
                        'slope': [0.35, 0.8, 0.02], 'aspect': [2.9, 0.4, 5.5], 'svf': [0.93, 0.8, 0.99]})
    tr = Transformer.from_crs('epsg:4326', f'epsg:{EPSG}', always_xy=True)
    pts['x'], pts['y'] = tr.transform(pts.lon.values, pts.lat.values)
    pts['aspect_cos'], pts['aspect_sin'] = np.cos(pts.aspect), np.sin(pts.aspect)
    pts['point_ind'] = np.arange(len(pts))
    pts['point_name'] = pts.point_ind.astype(str).str.zfill(1)
    return pts


def _horizon(pts):
    az = np.arange(-180., 180., 10.)
    xs = np.sort(pts.x.values)[[0, -1]] + np.array([-200, 200])
    ys = np.sort(pts.y.values)[[0, -1]] + np.array([-200, 200])
    x = np.linspace(xs[0], xs[1], 9)
    y = np.linspace(ys[1], ys[0], 9)
    rng = np.random.default_rng(3)
    return xr.DataArray(rng.uniform(0, 25, (len(az), len(y), len(x))), dims=('azimuth', 'y', 'x'),
                        coords={'azimuth': az, 'y': y, 'x': x}, name='horizon')


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp('parity')
    plev, surf = _era()
    clim = d / 'climate'
    clim.mkdir()
    plev.to_netcdf(clim / 'PLEV_2025.nc')
    surf.to_netcdf(clim / 'SURF_2025.nc')
    xr.merge([plev, surf.rename({'z': 'z_surf'})]).to_zarr(clim / 'ERA5.zarr', zarr_format=3, mode='w')

    pts = _points()
    da_hor = _horizon(pts)
    # horizon columns sampled exactly like Topoclass.compute_horizon()
    tx, ty = xr.DataArray(pts.x.values, dims='points'), xr.DataArray(pts.y.values, dims='points')
    for az in da_hor.azimuth.values:
        pts['hori_azi_' + str(az)] = da_hor.sel(x=tx, y=ty, azimuth=az, method='nearest').values.flatten()

    out = d / 'outputs'
    for sub in ('tmp', 'downscaled'):
        (out / sub).mkdir(parents=True)
    ds_solar = sg.get_solar_geom(pts.copy(), START, END, '1h', str(EPSG), 2, 'ds_solar.nc', out)
    ds_solar = xr.open_dataset(out / 'ds_solar.nc')
    return dict(d=d, clim=clim, out=out, pts=pts, da_hor=da_hor, ds_solar=ds_solar)


def _classic(s):
    ta.downscale_climate(s['d'], s['clim'], s['out'], s['pts'].copy(), s['da_hor'], s['ds_solar'], EPSG,
                         START, END, '1h', 'idw', True, True, 'classic_*.nc', 2)
    return {pn: xr.open_dataset(s['out'] / 'downscaled' / f'classic_{pn}.nc') for pn in s['pts'].point_name}


VARS = ['t', 'u', 'v', 'q', 'p', 'tp', 'wd', 'ws', 'w', 'vp', 'cse', 'LW', 'SW_diffuse', 'SW_direct', 'SW', 'cos_illumination']


@pytest.mark.parametrize('use_hori_columns', [True, False])
def test_netcdf_output_matches_classic(setup, use_hori_columns, tmp_path):
    s = setup
    ref = _classic(s)
    pts = s['pts'] if use_hori_columns else s['pts'][[c for c in s['pts'].columns if not c.startswith('hori_azi_')]]
    cd = tz.ClimateDownscaler(s['clim'] / 'ERA5.zarr', tmp_path, pts, s['da_hor'], s['ds_solar'], EPSG,
                              START, END, '1h', interp_method='idw', lw_terrain_flag=True,
                              precip_lapse_rate_flag=True, file_pattern='zarr_*.nc')
    cd.multicore_parallel_process_multiple_subsets(n_core=2)
    for pn, r in ref.items():
        z = xr.open_dataset(tmp_path / f'zarr_{pn}.nc')
        for v in VARS:
            np.testing.assert_allclose(z[v].values, r[v].values, rtol=0, atol=1e-5, err_msg=f'point {pn} var {v}')


def test_zarr_output_and_resume(setup, tmp_path):
    s = setup
    ref = _classic(s)
    kw = dict(interp_method='idw', lw_terrain_flag=True, precip_lapse_rate_flag=True,
              file_pattern=None, store_name='down.zarr')
    args = (s['clim'] / 'ERA5.zarr', tmp_path, s['pts'], s['da_hor'], s['ds_solar'], EPSG, START, END, '1h')
    tz.ClimateDownscaler(*args, **kw).multicore_parallel_process_multiple_subsets(n_core=2)
    out = tz.open_downscaled_store(tmp_path / 'down.zarr')
    for pn, r in ref.items():
        for v in tz.varout_default:
            np.testing.assert_allclose(out[v].sel(point_name=pn).values, r[v].values, rtol=0, atol=1e-5,
                                       err_msg=f'point {pn} var {v}')
    # resume: forget one point, rerun -> only that point is recomputed, others untouched
    prog = tmp_path / 'down.zarr.progress'
    (prog / '1.done').unlink()
    before = {f: f.stat().st_mtime_ns for f in prog.glob('*.done')}
    tz.ClimateDownscaler(*args, **kw).multicore_parallel_process_multiple_subsets(n_core=2)
    assert (prog / '1.done').exists()
    assert all(f.stat().st_mtime_ns == m for f, m in before.items())
    # a store made for another period is refused
    with pytest.raises(ValueError, match='other points/period'):
        tz.ClimateDownscaler(s['clim'] / 'ERA5.zarr', tmp_path, s['pts'], s['da_hor'], s['ds_solar'], EPSG,
                             START, START, '1h', **kw).multicore_parallel_process_multiple_subsets(n_core=2)


def test_edge_point_is_refused(setup, tmp_path):
    s = setup
    pts = s['pts'].copy()
    pts.loc[0, 'lat'] = 42.75                     # on the northern edge of the ERA5 domain
    cd = tz.ClimateDownscaler(s['clim'] / 'ERA5.zarr', tmp_path, pts, s['da_hor'], s['ds_solar'], EPSG,
                              START, END, '1h', file_pattern='e_*.nc')
    with pytest.raises(RuntimeError, match='edge of the ERA5 domain'):
        cd.multicore_parallel_process_multiple_subsets(n_core=1)
