"""
Tests for the solar-geometry changes of 2026-11-15:
  - mid_interval=True: sun position at t - tstep/2, time labels kept at t
  - the cache is stored unpacked (float32): a run that reads the cache gets exactly what a run that computed it used
  - SW_direct capped at SWtoa (upstream 48c07bd): checked on the formula used in topo_scale_zarr
"""
import numpy as np
import pandas as pd
import pvlib
import xarray as xr

from TopoPyScale import solar_geom as sg
from TopoPyScale import topo_utils as tu


def _pts():
    """Two points given in EPSG:32634 (get_solar_geom always converts x/y; its EPSG test compares int with str)."""
    from pyproj import Transformer
    lat, lon = np.array([42.18, 41.55]), np.array([23.58, 23.27])
    x, y = Transformer.from_crs(4326, 32634, always_xy=True).transform(lon, lat)
    return pd.DataFrame({'point_name': ['a', 'b'], 'x': x, 'y': y, 'elevation': [2925.0, 207.0]})


def test_mid_interval_matches_pvlib_half_an_hour_earlier(tmp_path):
    ds = sg.get_solar_geom(_pts(), '2024-12-21', '2024-12-21', '1h', sr_epsg='32634', num_threads=2, fname='s.zarr',
                           project_ouput=tmp_path, mid_interval=True)
    t = pd.date_range('2024-12-21', periods=24, freq='1h')
    assert (pd.DatetimeIndex(ds.time.values) == t).all()                  # labels unchanged
    ref = pvlib.solarposition.get_solarposition((t - pd.Timedelta('30min')).tz_localize('UTC'), 42.18, 23.58, 2925.0,
                                                method='nrel_numpy')
    np.testing.assert_allclose(ds.zenith.sel(point_name='a').values, np.deg2rad(ref.zenith.values), atol=1e-6)
    np.testing.assert_allclose(ds.azimuth.sel(point_name='a').values, np.deg2rad(180 - ref.azimuth.values), atol=1e-5)


def test_mid_interval_false_is_the_former_behaviour(tmp_path):
    ds = sg.get_solar_geom(_pts(), '2024-06-21', '2024-06-21', '1h', sr_epsg='32634', num_threads=2, fname='s.zarr',
                           project_ouput=tmp_path, mid_interval=False)
    t = pd.date_range('2024-06-21', periods=24, freq='1h', tz='UTC')
    ref = pvlib.solarposition.get_solarposition(t, 41.55, 23.27, 207.0, method='nrel_numpy')
    np.testing.assert_allclose(ds.zenith.sel(point_name='b').values, np.deg2rad(ref.zenith.values), atol=1e-6)


def test_cache_reads_back_identical(tmp_path):
    ds = sg.get_solar_geom(_pts(), '2024-03-01', '2024-03-02', '1h', sr_epsg='32634', num_threads=2, fname='s.zarr',
                           project_ouput=tmp_path)
    back = tu.open_dataset(tmp_path / 's.zarr').load()
    for v in ('zenith', 'azimuth', 'elevation', 'mu0', 'SWtoa'):
        assert back[v].dtype == np.float32
        np.testing.assert_array_equal(back[v].values, ds[v].values)
    np.testing.assert_array_equal(back.sunset.values, ds.sunset.values)


def test_sw_direct_cap_removes_negative_ka():
    """Low sun: hour-mean SW_direct above the instantaneous SWtoa gives ka < 0 and a spike; the cap prevents it."""
    swtoa = xr.DataArray([5.0, 300.0, 900.0])
    sw_direct = xr.DataArray([40.0, 200.0, 950.0])
    capped = sw_direct.clip(max=swtoa)
    ka = np.log(swtoa / capped)
    assert float(ka.min()) >= 0.0
    np.testing.assert_array_equal(capped.values, [5.0, 200.0, 900.0])
