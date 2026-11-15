"""
Option B (2026-10-13): one output Zarr store covering a long period (a year), filled run by run
(e.g. month by month). Filling the store in two runs, one per day, must give exactly the store that
one run over both days gives, and each run must resume on its own.

Uses the synthetic ERA5 sample and points of the parity test (2 days, 3 points).

Run:  python -m pytest tests/test_zarr_store_period.py -v
"""
import numpy as np
import pytest
import xarray as xr

from TopoPyScale import topo_scale_zarr as tz
from test_topo_scale_zarr_parity import EPSG, START, END, setup  # noqa: F401  (setup is a fixture)

DAY1, DAY2 = START, END
KW = dict(interp_method='idw', lw_terrain_flag=True, precip_lapse_rate_flag=True, file_pattern=None,
          store_name='down.zarr')


def _cd(s, out, start, end, **kw):
    return tz.ClimateDownscaler(s['clim'] / 'ERA5.zarr', out, s['pts'], s['da_hor'], s['ds_solar'], EPSG,
                                start, end, '1h', **(KW | kw))


def test_store_filled_day_by_day_equals_one_run(setup, tmp_path):
    s = setup
    one, split = tmp_path / 'one', tmp_path / 'split'
    _cd(s, one, START, END).multicore_parallel_process_multiple_subsets(n_core=2)
    # day 2 first: the order of the runs must not matter
    for d in (DAY2, DAY1):
        _cd(s, split, d, d, store_start=START, store_end=END).multicore_parallel_process_multiple_subsets(n_core=2)
    a = xr.open_zarr(one / 'down.zarr').load()
    b = xr.open_zarr(split / 'down.zarr').load()
    assert b.sizes['time'] == 48
    for v in tz.varout_default:
        assert not np.isnan(b[v].values).any(), f'{v}: part of the store not filled'
        np.testing.assert_array_equal(b[v].values, a[v].values, err_msg=v)
    # one marker folder per run period
    prog = sorted(p.name for p in (split / 'down.zarr.progress').iterdir())
    assert prog == ['2025011000_2025011023', '2025011100_2025011123']


def test_store_period_resume_and_checks(setup, tmp_path):
    s = setup
    _cd(s, tmp_path, DAY1, DAY1, store_start=START, store_end=END).multicore_parallel_process_multiple_subsets(n_core=2)
    st = xr.open_zarr(tmp_path / 'down.zarr').load()
    assert not np.isnan(st.t.isel(time=slice(0, 24)).values).any()
    assert np.isnan(st.t.isel(time=slice(24, 48)).values).all()          # day 2 not run yet
    # resume day 1: nothing to do, markers untouched
    prog = tmp_path / 'down.zarr.progress' / '2025011000_2025011023'
    before = {f: f.stat().st_mtime_ns for f in prog.glob('*.done')}
    assert len(before) == 3
    _cd(s, tmp_path, DAY1, DAY1, store_start=START, store_end=END).multicore_parallel_process_multiple_subsets(n_core=2)
    assert {f: f.stat().st_mtime_ns for f in prog.glob('*.done')} == before
    # a run outside the store period is refused
    with pytest.raises(ValueError, match='not inside the store period'):
        _cd(s, tmp_path, DAY1, DAY2, store_start=DAY2, store_end=DAY2)
    with pytest.raises(ValueError, match='both store_start and store_end'):
        _cd(s, tmp_path, DAY1, DAY1, store_start=START)
    # an existing store made for another store period is refused
    with pytest.raises(ValueError, match='other points/period'):
        _cd(s, tmp_path, DAY1, DAY1, store_start=DAY1, store_end=DAY1).multicore_parallel_process_multiple_subsets(n_core=2)
