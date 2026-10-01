"""Tests for TopoPyScale.topo_daily.daily_stats.   Run: python -m pytest tests/test_topo_daily.py -v"""
import numpy as np
import pandas as pd
import xarray as xr

from TopoPyScale.topo_daily import daily_stats


def _hourly(days=3, start='2025-01-01'):
    t = pd.date_range(start, periods=24 * days, freq='h')
    h = np.arange(t.size, dtype=float)
    return xr.Dataset({
        't': (('point_name', 'time'), np.stack([270 + (h % 24), 260 + (h % 24)])),
        'tp': (('point_name', 'time'), np.ones((2, t.size)) * 0.5),                 # 0.5 mm/h
        'u': (('point_name', 'time'), np.stack([np.where(h % 2 == 0, 5., -5.), np.ones(t.size)])),
        'v': (('point_name', 'time'), np.zeros((2, t.size))),
        'ws': (('point_name', 'time'), np.stack([np.full(t.size, 5.), np.ones(t.size)])),
        'SW': (('point_name', 'time'), np.stack([np.where((h % 24 >= 6) & (h % 24 < 18), 240., 0.)] * 2)),
    }, coords={'time': t, 'point_name': ['a', 'b']})


def test_temperature_min_mean_max_and_precip_sum():
    d = daily_stats(_hourly())
    assert d.time.size == 3
    np.testing.assert_allclose(d.t_min.sel(point_name='a'), 270)
    np.testing.assert_allclose(d.t_max.sel(point_name='a'), 293)
    np.testing.assert_allclose(d.t_mean.sel(point_name='a'), 281.5)
    np.testing.assert_allclose(d.tp_sum, 12.0)                    # 24 h x 0.5 mm/h
    np.testing.assert_allclose(d.SW_mean, 120.0)                  # night zeros included
    assert d.tp_sum.attrs['units'] == 'mm day**-1'


def test_wind_mean_of_speeds_and_vector_direction():
    d = daily_stats(_hourly())
    # point a: alternating +5/-5 m/s east-west: mean speed 5, but the vectors cancel
    np.testing.assert_allclose(d.ws_mean.sel(point_name='a'), 5.0)
    np.testing.assert_allclose(d.u_mean.sel(point_name='a'), 0.0)
    # point b: steady wind blowing towards east (u>0) comes FROM the west = 3pi/2
    np.testing.assert_allclose(d.wd_vector_mean.sel(point_name='b'), 1.5 * np.pi)


def test_incomplete_days_are_nan_and_utc_offset():
    ds = _hourly().isel(time=slice(5, None))                      # first day starts at 05 UTC
    d = daily_stats(ds)
    assert np.isnan(d.t_mean.isel(time=0)).all() and np.isfinite(d.t_mean.isel(time=1)).all()
    d2 = daily_stats(_hourly(), utc_offset_hours=2)               # local days UTC+2
    assert pd.Timestamp(d2.time.values[0]) == pd.Timestamp('2025-01-01')
    assert np.isnan(d2.t_mean.isel(time=0)).all()                 # 00-02 local missing
    np.testing.assert_allclose(d2.t_min.sel(point_name='a').isel(time=1), 270)


def test_three_hourly_input():
    ds = _hourly().isel(time=slice(0, None, 3))
    d = daily_stats(ds)
    np.testing.assert_allclose(d.tp_sum, 12.0)                    # 8 steps x 0.5 mm/h x 3 h
