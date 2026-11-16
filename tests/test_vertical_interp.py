"""
Tests for topo_utils.vertical_interp_plev (2026-11-15): vertical interpolation decided per time step.

Former behaviour (upstream): a point below z(lowest level) at ANY time step of the run took, for the WHOLE period,
the values of the nearest level above (no interpolation). Now: interpolation at every step where the point lies
between two levels, lowest-level values only at the steps where it lies below every level.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from TopoPyScale import topo_utils as tu

LEVELS = [600, 700, 850, 925, 1000]


def make_plev(n=48, seed=0, z1000_mean=150.0, z1000_amp=120.0):
    """Synthetic point profile: z decreasing with pressure; z(1000) oscillates around a low point."""
    rng = np.random.default_rng(seed)
    t = pd.date_range('2024-01-01', periods=n, freq='1h')
    z1000 = z1000_mean + z1000_amp * np.sin(np.arange(n) / 5.0)
    zoff = np.array([4200.0, 3000.0, 1450.0, 750.0, 0.0])
    z = z1000[:, None] + zoff[None, :] + rng.normal(0, 5, (n, len(LEVELS)))
    ds = xr.Dataset({'z': (('time', 'level'), z)}, coords={'time': t, 'level': LEVELS})
    for v, base in (('t', 270.0), ('u', 3.0), ('v', -2.0), ('q', 0.004)):
        ds[v] = (('time', 'level'), base + rng.normal(0, 1, (n, len(LEVELS))) * (0.001 if v == 'q' else 1.0))
    return ds


def former_interpolation(plev, e):
    """The former 'else' branch (point above the lowest level at every step), copied from topo_scale_zarr."""
    ind_z_bot = (plev.where(plev.z < e).z - e).argmax('level')
    ind_z_top = (plev.where(plev.z > e).z - e).argmin('level')
    top, bot = plev.isel(level=ind_z_top), plev.isel(level=ind_z_bot)
    dist = np.array([np.abs(bot.z - e).values, np.abs(top.z - e).values])
    w = dist / np.sum(dist, axis=0)
    return {v: bot[v] * w[1] + top[v] * w[0] for v in ('t', 'u', 'v', 'q')}


def former_fallback(plev, e):
    """The former fallback branch: nearest level above, no interpolation."""
    ind_z_top = (plev.where(plev.z > e).z - e).argmin('level')
    top = plev.isel(level=ind_z_top)
    return {v: top[v] for v in ('t', 'u', 'v', 'q')}


def test_interpolated_steps_bit_identical_to_former_interpolation():
    plev = make_plev()
    e = 200.0
    _, out, below_all = tu.vertical_interp_plev(plev, e)
    assert bool(below_all.any()) and not bool(below_all.all())          # both kinds of steps occur
    ok = ~below_all.values
    ref = former_interpolation(plev.isel(time=ok), e)                    # only the bracketed steps
    for v in ref:
        np.testing.assert_array_equal(out[v].values[ok], ref[v].values)


def test_steps_below_all_levels_take_the_lowest_level():
    plev = make_plev()
    e = 200.0
    _, out, below_all = tu.vertical_interp_plev(plev, e)
    bl = below_all.values
    ref = former_fallback(plev, e)
    for v in ref:
        np.testing.assert_array_equal(out[v].values[bl], ref[v].values[bl])
        np.testing.assert_array_equal(out[v].values[bl], plev[v].sel(level=1000).values[bl])


def test_independent_of_the_period():
    plev = make_plev(n=96)
    e = 200.0
    _, full, _ = tu.vertical_interp_plev(plev, e)
    for sl in (slice(0, 24), slice(30, 60), slice(70, 96)):
        _, part, _ = tu.vertical_interp_plev(plev.isel(time=sl), e)
        for v in full:
            np.testing.assert_array_equal(part[v].values, full[v].values[sl])


def test_point_always_inside_is_unchanged():
    plev = make_plev(z1000_mean=-200.0, z1000_amp=50.0)                  # z(1000) always below the point
    e = 400.0
    _, out, below_all = tu.vertical_interp_plev(plev, e)
    assert not bool(below_all.any())
    ref = former_interpolation(plev, e)
    for v in ref:
        np.testing.assert_array_equal(out[v].values, ref[v].values)


def test_continuous_at_the_switch():
    """Where z(lowest level) crosses the point, the interpolated value tends to the lowest-level value."""
    plev = make_plev(n=2)
    e = float(plev.z.sel(level=1000).isel(time=0)) + 1e-6               # just above the lowest level
    _, out, below_all = tu.vertical_interp_plev(plev.isel(time=[0]), e)
    assert not bool(below_all.any())
    np.testing.assert_allclose(out['t'].values, plev.t.sel(level=1000).isel(time=[0]).values, atol=1e-6)


def test_error_above_all_levels():
    plev = make_plev()
    with pytest.raises(ValueError):
        tu.vertical_interp_plev(plev, 10000.0)
