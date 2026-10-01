"""
Regression test: TopoSUB clustering must use exactly the pixels inside the clustering mask, and the
cluster map written to ds_param.nc must agree with df_centroids.

Before the fix, df_param (from ds_param, dims y,x) and the mask (read with rasterio, dims x,y) were
flattened in different orders and combined by position, so the mask hit the wrong pixels; the map
was written back with an order assumption too. This happened when ds_param.nc was reloaded from
disk (cached run): its dims are then (y, x), while a freshly computed ds_param and a rasterio-read
mask are (x, y). A tiny DEM with a deliberately asymmetric mask, clustered from the cached
ds_param, catches both.

Run:  python -m pytest tests/test_clustering_mask.py -v
"""
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from TopoPyScale import topoclass as tc

NY, NX, RES = 36, 50, 25.0
X0, Y0 = 700000.0, 4650000.0


def _write_tif(path, arr, dtype):
    with rasterio.open(path, 'w', driver='GTiff', height=NY, width=NX, count=1, dtype=dtype,
                       crs='EPSG:32634', transform=from_origin(X0, Y0, RES, RES)) as dst:
        dst.write(arr.astype(dtype), 1)


@pytest.fixture(scope='module')
def project(tmp_path_factory):
    d = tmp_path_factory.mktemp('clmask')
    (d / 'inputs' / 'dem').mkdir(parents=True)
    yy, xx = np.mgrid[0:NY, 0:NX]
    dem = 800 + 12 * xx + 5 * yy + 60 * np.sin(xx / 5.0) * np.cos(yy / 4.0)
    _write_tif(d / 'inputs' / 'dem' / 'dem.tif', dem, 'float32')
    mask = np.zeros((NY, NX))
    mask[3:20, 5:44] = 1          # a wide block in the north...
    mask[20:33, 5:15] = 1         # ...plus a leg in the south-west: not symmetric in x or y
    _write_tif(d / 'inputs' / 'dem' / 'mask.tif', mask, 'uint8')
    (d / 'config.yml').write_text(textwrap.dedent(f"""
        project:
          name: test
          directory: {d}
          start: 2025-01-01
          end: 2025-01-01
          split: {{IO: False, time: 1, space: None}}
          climate: era5
          extent:
          parallelization:
            downscaling_method: multicore
            setting: {{multicore: {{CPU_cores: 2}}}}
        climate:
          precip_lapse_rate: False
          era5: {{path: inputs/climate/, product: reanalysis, timestep: 1h, plevels: [700, 1000],
                  download_threads: 1, data_repository: cds, realtime: False, zarr_store: null}}
        dem: {{file: dem.tif, epsg: 32634, horizon_increments: 45}}
        sampling:
          method: toposub
          toposub:
            clustering_method: minibatchkmean
            n_clusters: 6
            random_seed: 2
            clustering_features: {{'x': 1, 'y': 1, 'elevation': 4, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1}}
            clustering_mask: inputs/dem/mask.tif
        toposcale: {{interpolation_method: idw, LW_terrain_contribution: True}}
        outputs:
          directory: outputs
          file: {{clean_outputs: False, clean_FSM: False, df_centroids: df_centroids.pck, ds_param: ds_param.nc,
                  ds_solar: ds_solar.nc, da_horizon: da_horizon.nc, landform: landform.tif,
                  downscaled_pt: down_pt_*.nc}}
        clean_up: {{rm_tmp_dirs: False}}
        """))
    # 1st instance computes the terrain parameters and writes ds_param.nc;
    # 2nd instance reloads it from disk (dims then y,x, unlike the freshly computed x,y order),
    # which is the cached situation where the old code mis-applied the mask.
    tc.Topoclass(str(d / 'config.yml')).compute_dem_param()
    # Rewrite ds_param.nc with (y, x) dimension order, like the Rila-Pirin terrain file (written by
    # an older TopoPyScale); a freshly computed one is (x, y) and would hide the bug.
    f = d / 'outputs' / 'ds_param.nc'
    code = (f"import xarray as xr; ds = xr.open_dataset(r'{f}').load(); ds.close(); "
            f"ds.transpose('y', 'x').to_netcdf(r'{f}', engine='h5netcdf'); "
            f"assert list(xr.open_dataset(r'{f}').dims)[:2] == ['y', 'x'], 'test setup: dims order'")
    subprocess.run([sys.executable, '-c', code], check=True)
    mp = tc.Topoclass(str(d / 'config.yml'))
    mp.compute_dem_param()
    mp.extract_topo_param()
    return d, mp, mask


def _read_saved(path):
    """Read ds_param.nc in a fresh process: xarray caches open files by path, so this process
    could still see the version Topoclass opened before rewriting it."""
    out = path.parent / 'check.nc'
    code = (f"import xarray as xr; d = xr.open_dataset(r'{path}').transpose('y', 'x')"
            f"[['point_name', 'elevation']].load(); d.to_netcdf(r'{out}')")
    subprocess.run([sys.executable, '-c', code], check=True)
    return xr.open_dataset(out, engine='h5netcdf').load()


def test_only_masked_pixels_are_clustered(project):
    d, mp, mask = project
    ds = _read_saved(d / 'outputs' / 'ds_param.nc')
    clustered = ds.point_name.values != '-9999'
    assert clustered.sum() == mask.sum()
    assert np.array_equal(clustered, mask == 1)          # same pixels, not just the same count


def test_cluster_map_agrees_with_centroids(project):
    d, mp, mask = project
    ds = _read_saved(d / 'outputs' / 'ds_param.nc')
    df = mp.toposub.df_centroids.set_index('point_name')
    s = pd.DataFrame({'lab': ds.point_name.values.ravel(), 'el': ds.elevation.values.ravel(),
                      'xx': np.broadcast_to(ds.x.values, (NY, NX)).ravel()})
    g = s[s.lab != '-9999'].groupby('lab').mean()
    j = df[['elevation', 'x']].join(g, how='inner')
    assert len(j) == 6
    # k-means centroids are (close to) the mean of their member pixels
    np.testing.assert_allclose(j.elevation, j.el, atol=15)
    np.testing.assert_allclose(j.x, j.xx, atol=3 * RES)
