"""
topo_utils.vertical_interp_plev: vertical interpolation to the point elevation decided per time step.

    python -m pytest tests/test_vertical_interp.py -v
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from TopoPyScale import topo_utils as tu

LEVELS = np.array([700., 850., 925., 1000.])          # level -1 = lowest (1000 hPa), as in the ERA5 stores


def plev(z1000):
    """Synthetic pressure levels, dims (time, level); z of 1000 hPa given per time step."""
    nt = len(z1000)
    rng = np.random.default_rng(1)
    z = np.stack([np.full(nt, 3000.), np.full(nt, 1450.), np.full(nt, 760.), np.asarray(z1000, float)], 1)
    z = z + rng.normal(0, 5, z.shape) * np.array([1, 1, 1, 0])
    t = 288. - 6.5e-3 * z + rng.normal(0, 1, z.shape)
    ds = xr.Dataset({'z': (('time', 'level'), z), 't': (('time', 'level'), t),
                     'u': (('time', 'level'), rng.normal(0, 5, z.shape)),
                     'v': (('time', 'level'), rng.normal(0, 5, z.shape)),
                     'q': (('time', 'level'), rng.uniform(1e-3, 8e-3, z.shape))},
                    coords={'time': pd.date_range('2024-01-01', periods=nt, freq='h'), 'level': LEVELS})
    return ds


def former_interpolation(p, e):
    """The former (pre 10 Nov 2026) interpolation branch, verbatim, for points above the lowest level."""
    ind_z_bot = (p.where(p.z < e).z - e).argmax('level')
    ind_z_top = (p.where(p.z > e).z - e).argmin('level')
    top, bot = p.isel(level=ind_z_top), p.isel(level=ind_z_bot)
    dist = np.array([np.abs(bot.z - e).values, np.abs(top.z - e).values])
    w = dist / np.sum(dist, axis=0)
    return {v: (bot[v] * w[1] + top[v] * w[0]).values for v in ('t', 'u', 'v', 'q')}, top


def test_bit_identical_when_never_below():
    p = plev(np.full(48, 110.))
    vi = tu.vertical_interp_plev(p, 320.)
    ref, top = former_interpolation(p, 320.)
    assert not vi.fallback.any()
    for v in ('t', 'u', 'v', 'q'):
        assert np.array_equal(vi[v].values, ref[v])          # exactly the same numbers
    assert np.array_equal(vi.top_z.values, top.z.values) and np.array_equal(vi.top_level.values, top.level.values)


def test_fallback_only_at_the_hours_below():
    z1000 = np.full(48, 110.)
    z1000[10:14] = 400.                                     # strong anticyclone: 1000 hPa surface above the point
    p = plev(z1000)
    e = 320.
    vi = tu.vertical_interp_plev(p, e)
    assert vi.fallback.values.tolist() == [10 <= i < 14 for i in range(48)]
    # fallback hours: the lowest level's values (no interpolation, no extrapolation)
    for v in ('t', 'u', 'v', 'q'):
        assert np.array_equal(vi[v].values[10:14], p[v].values[10:14, -1])
    # all other hours: identical to the former interpolation computed on those hours alone
    keep = np.r_[0:10, 14:48]
    ref, _ = former_interpolation(p.isel(time=keep), e)
    for v in ('t', 'u', 'v', 'q'):
        assert np.array_equal(vi[v].values[keep], ref[v])


def test_independent_of_the_period():
    z1000 = np.full(72, 110.)
    z1000[5:8] = 400.
    p, e = plev(z1000), 320.
    full = tu.vertical_interp_plev(p, e)
    part = tu.vertical_interp_plev(p.isel(time=slice(24, 72)), e)  # a period without the anticyclone
    for v in ('t', 'u', 'v', 'q', 'top_z', 'top_t'):
        assert np.array_equal(full[v].values[24:], part[v].values)


def test_continuous_at_the_switch():
    # point slightly above / slightly below the 1000 hPa surface: values converge to the 1000 hPa level
    p = plev(np.array([319.9, 320.1]))
    vi = tu.vertical_interp_plev(p, 320.)
    assert vi.fallback.values.tolist() == [False, True]
    assert vi.t.values[0] == pytest.approx(p.t.values[0, -1], abs=1e-3)


def test_error_when_above_all_levels():
    with pytest.raises(ValueError):
        tu.vertical_interp_plev(plev(np.full(5, 110.)), 3500.)
