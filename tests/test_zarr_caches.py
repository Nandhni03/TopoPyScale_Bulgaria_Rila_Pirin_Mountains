"""
The cached intermediate files (ds_param, ds_solar, da_horizon) can be Zarr stores instead of NetCDF:
the format follows the file name in the config ('xxx.zarr'). This test runs the first Topoclass steps
on a tiny DEM twice, once with NetCDF names and once with Zarr names, and requires identical results
(Zarr stores are packed exactly like topo_export.to_netcdf packs NetCDF). It also checks that a
second Topoclass instance finds and reuses the Zarr caches (a store is a directory, not a file).

Run:  python -m pytest tests/test_zarr_caches.py -v
"""
import textwrap

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from TopoPyScale import topo_export as te
from TopoPyScale import topo_utils as tu
from TopoPyScale import topoclass as tc

NY, NX, RES = 36, 50, 25.0
X0, Y0 = 700000.0, 4650000.0


def _write_tif(path, arr, dtype):
    with rasterio.open(path, 'w', driver='GTiff', height=NY, width=NX, count=1, dtype=dtype,
                       crs='EPSG:32634', transform=from_origin(X0, Y0, RES, RES)) as dst:
        dst.write(arr.astype(dtype), 1)


def _project(d, suffix):
    (d / 'inputs' / 'dem').mkdir(parents=True)
    yy, xx = np.mgrid[0:NY, 0:NX]
    dem = 800 + 12 * xx + 5 * yy + 60 * np.sin(xx / 5.0) * np.cos(yy / 4.0)
    _write_tif(d / 'inputs' / 'dem' / 'dem.tif', dem, 'float32')
    mask = np.zeros((NY, NX))
    mask[3:20, 5:44] = 1
    mask[20:33, 5:15] = 1
    _write_tif(d / 'inputs' / 'dem' / 'mask.tif', mask, 'uint8')
    (d / 'config.yml').write_text(textwrap.dedent(f"""
        project:
          name: test
          directory: {d}
          start: 2025-01-01
          end: 2025-01-02
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
          file: {{clean_outputs: False, clean_FSM: False, df_centroids: df_centroids.pck, ds_param: ds_param{suffix},
                  ds_solar: ds_solar{suffix}, da_horizon: da_horizon{suffix}, landform: landform.tif,
                  downscaled_pt: down_pt_*.nc}}
        clean_up: {{rm_tmp_dirs: False}}
        """))
    mp = tc.Topoclass(str(d / 'config.yml'))
    mp.compute_dem_param()
    mp.extract_topo_param()
    mp.compute_solar_geometry()
    mp.compute_horizon()
    return mp


@pytest.fixture(scope='module')
def both(tmp_path_factory):
    return {sfx: _project(tmp_path_factory.mktemp('cache' + sfx.replace('.', '_')), sfx) for sfx in ('.nc', '.zarr')}


def test_zarr_caches_are_written_as_stores(both):
    out = both['.zarr'].config.outputs.path
    for name in ('ds_param.zarr', 'ds_solar.zarr', 'da_horizon.zarr'):
        assert (out / name).is_dir() and tu.dataset_exists(out / name), name
        assert not (out / (name + '.part')).exists(), 'temporary store left behind'
    assert not list(out.glob('*.nc'))


def test_zarr_and_netcdf_caches_give_identical_results(both):
    nc, zr = both['.nc'], both['.zarr']
    pn = tu.open_dataset(nc.config.outputs.path / 'ds_param.nc').transpose('y', 'x').load()
    pz = tu.open_dataset(zr.config.outputs.path / 'ds_param.zarr').transpose('y', 'x').load()
    for v in ['elevation', 'slope', 'aspect', 'aspect_cos', 'aspect_sin', 'svf']:
        # same packed integers; decoding may differ in the last bit (~1e-13), hence the 1e-9 tolerance
        np.testing.assert_allclose(pz[v].values, pn[v].values, rtol=0, atol=1e-9, err_msg=v)
    np.testing.assert_array_equal(pz.point_name.values.astype(str), pn.point_name.values.astype(str))
    pd.testing.assert_frame_equal(zr.toposub.df_centroids, nc.toposub.df_centroids)
    sn = tu.open_dataset(nc.config.outputs.path / 'ds_solar.nc').load()
    sz = tu.open_dataset(zr.config.outputs.path / 'ds_solar.zarr').load()
    for v in sn.data_vars:
        np.testing.assert_allclose(sz[v].values, sn[v].values, rtol=0, atol=1e-9, err_msg=v)
    hn = tu.open_dataarray(nc.config.outputs.path / 'da_horizon.nc').load()
    hz = tu.open_dataarray(zr.config.outputs.path / 'da_horizon.zarr').load()
    np.testing.assert_array_equal(hz.values, hn.values)


def test_second_instance_reuses_zarr_caches(both):
    zr = both['.zarr']
    out = zr.config.outputs.path
    stamp = {n: (out / n / 'zarr.json').stat().st_mtime_ns for n in ('ds_solar.zarr', 'da_horizon.zarr')}
    mp = tc.Topoclass(str(zr.config.project.directory / 'config.yml'))
    mp.compute_dem_param()
    mp.extract_topo_param()
    mp.compute_solar_geometry()
    mp.compute_horizon()
    for n, t in stamp.items():
        assert (out / n / 'zarr.json').stat().st_mtime_ns == t, f'{n} was recomputed'
    pd.testing.assert_frame_equal(mp.toposub.df_centroids, zr.toposub.df_centroids)


def test_to_zarr_packs_like_to_netcdf(tmp_path):
    rng = np.random.default_rng(0)
    ds = xr.Dataset({'a': (('y', 'x'), rng.normal(500, 300, (7, 9))),
                     'flag': (('y', 'x'), rng.random((7, 9)) > 0.5),
                     'n': (('y', 'x'), rng.integers(0, 9, (7, 9))),
                     'point_name': (('y', 'x'), rng.integers(1, 5, (7, 9)).astype(str).astype(object))},
                    coords={'y': np.arange(7.), 'x': np.arange(9.)})
    te.save_dataset(ds, tmp_path / 'a.nc')
    te.save_dataset(ds, tmp_path / 'a.zarr')
    a, b = tu.open_dataset(tmp_path / 'a.nc').load(), tu.open_dataset(tmp_path / 'a.zarr').load()
    for v in ds.data_vars:
        np.testing.assert_array_equal(b[v].values.astype(a[v].dtype) if v == 'point_name' else b[v].values,
                                      a[v].values, err_msg=v)
    # packing is lossy (authors' choice, 10 bits): the decoded floats differ from the input
    assert np.abs(b.a.values - ds.a.values).max() < (ds.a.max() - ds.a.min()) / 2 ** 10
    # rewriting a store that is open lazily (as Topoclass does with ds_param after clustering)
    lazy = tu.open_dataset(tmp_path / 'a.zarr')
    lazy['b'] = lazy.a * 2
    te.save_dataset(lazy, tmp_path / 'a.zarr')
    c = tu.open_dataset(tmp_path / 'a.zarr').load()
    np.testing.assert_allclose(c.b.values, 2 * b.a.values, atol=float(2 * (ds.a.max() - ds.a.min()) / 2 ** 10))
