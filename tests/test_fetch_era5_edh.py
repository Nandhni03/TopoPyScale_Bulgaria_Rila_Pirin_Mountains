"""
Tests for TopoPyScale.fetch_era5_edh.

Unit tests run against a small synthetic Zarr store with the Earth Data Hub layout
(valid_time, isobaricInhPa descending, latitude descending, longitude 0..360, one level per
chunk), so they need no network. Values encode their own coordinates, so tests can check that
exactly the right cells, levels and hours came out.

The integration test at the bottom reads 2 days from the real EDH stores and is skipped when
no EDH credentials are available.

Run:  python -m pytest tests/test_fetch_era5_edh.py -v
"""
import json
import netrc
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from TopoPyScale import fetch_era5_edh as fe

T0 = pd.Timestamp('2020-01-01T00')
NT = 24 * 8                       # 8 days hourly
PLEV_CHUNK_H, SURF_CHUNK_H = 48, 36
LEVELS = [1000, 925, 850, 700, 600, 500]      # descending, like EDH
LATS = np.arange(50.0, 29.0, -1.0)           # 1 deg grid keeps the store small
LONS = np.arange(0.0, 360.0, 1.0)


def _value(var_i, hour, lat, lon, lev=0.0):
    """Unique, decodable value per cell: lets tests assert exact selection."""
    return var_i * 1e4 + lev + lat * 10 + lon * 0.01 + hour * 1e-4


def _make_store(path, kind):
    time = pd.date_range(T0, periods=NT, freq='h')
    hour = np.arange(NT, dtype='float64')
    if kind == 'plev':
        names = fe.PLEV_VARS
        shape = (NT, len(LEVELS), LATS.size, LONS.size)
        grid = np.meshgrid(hour, np.array(LEVELS, float), LATS, LONS, indexing='ij')
        dims = ('valid_time', 'isobaricInhPa', 'latitude', 'longitude')
        coords = {'valid_time': time, 'isobaricInhPa': np.array(LEVELS, 'int32'), 'latitude': LATS, 'longitude': LONS}
        chunks = {'valid_time': PLEV_CHUNK_H, 'isobaricInhPa': 1, 'latitude': 10, 'longitude': 60}
    else:
        names = fe.SURF_VARS + ['u10']  # an extra var, like the real store
        shape = (NT, LATS.size, LONS.size)
        grid = np.meshgrid(hour, LATS, LONS, indexing='ij')
        dims = ('valid_time', 'latitude', 'longitude')
        coords = {'valid_time': time, 'latitude': LATS, 'longitude': LONS}
        chunks = {'valid_time': SURF_CHUNK_H, 'latitude': 10, 'longitude': 60}
    data = {}
    for i, n in enumerate(names):
        if kind == 'plev':
            h, lev, la, lo = grid
            arr = _value(i, h, la, lo, lev)
        else:
            h, la, lo = grid
            arr = _value(i, h, la, lo)
        data[n] = (dims, arr.astype('float64').reshape(shape),
                   {'GRIB_stepType': 'accum' if n in fe.ACCUM_VARS else 'instant'})
    ds = xr.Dataset(data, coords=coords).chunk(chunks)
    # a scalar coord that EDH/cfgrib stores carry and that must be dropped
    ds = ds.assign_coords(number=0)
    ds.to_zarr(path, mode='w', zarr_format=3, consolidated=False)
    return path


@pytest.fixture(scope='module')
def stores(tmp_path_factory):
    d = tmp_path_factory.mktemp('edh')
    return {'plev': str(_make_store(d / 'plev.zarr', 'plev')), 'surf': str(_make_store(d / 'surf.zarr', 'surf'))}


BBOX = {'latN': 42.72, 'latS': 41.03, 'lonW': 22.57, 'lonE': 24.49}   # Rila-Pirin, buffered


# ---------------------------------------------------------------- pure functions

def test_buffer_extent_matches_go_fetch():
    b = fe.buffer_extent({'latN': 42.32, 'latS': 41.43, 'lonW': 22.97, 'lonE': 24.09})
    assert b == pytest.approx({'latN': 42.72, 'latS': 41.03, 'lonW': 22.57, 'lonE': 24.49})


@pytest.mark.parametrize('requested, used, added', [
    ([600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000], [600, 700, 850, 925, 1000], []),
    ([550, 700, 1000], [500, 600, 700, 850, 925, 1000], [500]),          # 550 absent -> add 500 above
    ([620, 980], [600, 700, 850, 925, 1000], [600, 1000]),              # both ends bracketed
])
def test_resolve_plevels(requested, used, added):
    got, info = fe.resolve_plevels(requested, LEVELS)
    assert got == used
    assert sorted(info['added_to_bracket']) == sorted(added)
    assert set(info['missing_from_source']) == {float(p) for p in requested} - set(map(float, LEVELS))


def test_chunk_aligned_blocks():
    blocks = fe.chunk_aligned_blocks('2020-01-02T00', '2020-01-06T23', T0, 48)
    assert blocks == [(pd.Timestamp('2020-01-01T00'), pd.Timestamp('2020-01-02T23')),   # whole chunks
                      (pd.Timestamp('2020-01-03T00'), pd.Timestamp('2020-01-04T23')),
                      (pd.Timestamp('2020-01-05T00'), pd.Timestamp('2020-01-06T23'))]
    # every block lies inside exactly one source chunk
    for b0, b1 in blocks:
        assert (b0 - T0) // pd.Timedelta('48h') == (b1 - T0) // pd.Timedelta('48h')


def test_timestep_convention_samples_accumulations():
    t = pd.date_range(T0, periods=24, freq='h')
    ds = xr.Dataset({'tp': ('time', np.arange(24.0)), 't2m': ('time', np.arange(24.0))}, coords={'time': t})
    assert fe.apply_timestep_convention(ds, '1h') is ds
    d3 = fe.apply_timestep_convention(ds, '3h')
    assert list(d3.time.dt.hour.values) == [0, 3, 6, 9, 12, 15, 18, 21]
    # sampled, not summed: value at 03h is the 1-hour accumulation stored at 03h
    np.testing.assert_array_equal(d3.tp.values, [0, 3, 6, 9, 12, 15, 18, 21])
    with pytest.raises(ValueError):
        fe.apply_timestep_convention(ds, '2h')


def test_storage_options_local_and_token(monkeypatch):
    assert fe.get_storage_options('/some/local.zarr') is None
    monkeypatch.setenv('EDH_TOKEN', 'dummy')
    so = fe.get_storage_options('https://x')
    import base64
    assert so['client_kwargs']['headers']['Authorization'] == 'Basic ' + base64.b64encode(b'edh:dummy').decode()


# ---------------------------------------------------------------- domain selection

def test_select_domain_rila_pirin(stores):
    ds = fe.open_edh_store('surf', stores['surf'])
    sub = fe.select_domain(ds, BBOX)
    # 1 deg grid: outward snap to 41..43 N, 22..25 E
    assert sub.latitude.values.tolist() == [43.0, 42.0, 41.0]
    assert sub.longitude.values.tolist() == [22.0, 23.0, 24.0, 25.0]
    np.testing.assert_allclose(sub.t2m.isel(valid_time=0).sel(latitude=42, longitude=23).values,
                               _value(fe.SURF_VARS.index('t2m'), 0, 42, 23))


def test_select_domain_crossing_greenwich(stores):
    ds = fe.open_edh_store('surf', stores['surf'])
    sub = fe.select_domain(ds, {'latN': 45.2, 'latS': 44.1, 'lonW': -2.5, 'lonE': 1.5})
    assert sub.longitude.values.tolist() == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
    # -2 deg must hold the source value at 358 deg
    np.testing.assert_allclose(sub.sp.isel(valid_time=0).sel(latitude=45, longitude=-2).values,
                               _value(fe.SURF_VARS.index('sp'), 0, 45, 358))


# ---------------------------------------------------------------- full fetch

def _fetch(stores, out, **kw):
    args = dict(start='2020-01-02', end='2020-01-06', bbox=BBOX,
                plevels=[600, 650, 700, 750, 800, 850, 900, 925, 950, 1000],
                output_dir=out, urls=stores, max_concurrency=4, backoff_s=0)
    args.update(kw)
    return fe.fetch_era5_edh(**args)


def test_fetch_zarr_schema_matches_topo_scale_zarr(stores, tmp_path):
    rep = _fetch(stores, tmp_path, output_format='both')
    z = xr.open_zarr(rep['zarr'])
    assert set(z.dims) == {'time', 'level', 'latitude', 'longitude'}
    assert set(z.data_vars) == {'z', 't', 'u', 'v', 'q', 'r', 'z_surf', 'd2m', 'sp', 'strd', 'ssrd', 'tp', 't2m'}
    for v in fe.PLEV_VARS:
        assert z[v].dims == ('time', 'level', 'latitude', 'longitude')
    assert z.level.values.tolist() == [600.0, 700.0, 850.0, 925.0, 1000.0]   # ascending: level[-1] = lowest
    assert 'number' not in z.coords and 'valid_time' not in z.coords
    assert z.time.size == 6 * 24          # whole source chunks: 01-01 .. 01-06
    assert pd.Timestamp(z.time.values[0]) == pd.Timestamp('2020-01-01T00')
    assert pd.Timestamp(z.time.values[-1]) == pd.Timestamp('2020-01-06T23')
    # values: temperature at 850 hPa, 42N 23E, 2020-01-03T05 (hour index 2*24+5 = 53 in source)
    t = z.t.sel(time='2020-01-03T05', level=850, latitude=42, longitude=23).values
    np.testing.assert_allclose(t, _value(fe.PLEV_VARS.index('t'), 53, 42, 23, 850))
    # surface geopotential renamed, value from source 'z'
    np.testing.assert_allclose(z.z_surf.sel(time='2020-01-02T00', latitude=41, longitude=24).values,
                               _value(fe.SURF_VARS.index('z'), 24, 41, 24))
    # yearly CDS-layout NetCDF, with root symlinks for topo_scale.downscale_climate
    s = xr.open_dataset(tmp_path / 'SURF_2020.nc')
    assert 'z' in s and 'z_surf' not in s and s.time.size == 6 * 24
    assert (tmp_path / 'PLEV_2020.nc').is_symlink()
    # report
    assert rep['plev']['n_blocks'] == 3 and rep['surf']['n_blocks'] == 4


def test_fetch_is_resumable(stores, tmp_path, monkeypatch):
    _fetch(stores, tmp_path)
    blocks = sorted((tmp_path / 'edh_blocks').glob('*.nc'))
    mtimes = {f: f.stat().st_mtime_ns for f in blocks}
    victim = blocks[1]
    victim.unlink()
    calls = []
    orig = fe._load_with_retry
    monkeypatch.setattr(fe, '_load_with_retry', lambda *a, **k: calls.append(1) or orig(*a, **k))
    rep = _fetch(stores, tmp_path)
    assert len(calls) == 1                                  # only the deleted block
    assert rep['plev']['n_blocks_fetched'] + rep['surf']['n_blocks_fetched'] == 1
    for f, m in mtimes.items():
        if f != victim:
            assert f.stat().st_mtime_ns == m                # untouched


def test_manifest_refuses_changed_settings(stores, tmp_path):
    _fetch(stores, tmp_path)
    with pytest.raises(ValueError, match='different settings'):
        _fetch(stores, tmp_path, plevels=[500, 1000])


def test_fetch_3h(stores, tmp_path):
    rep = _fetch(stores, tmp_path, timestep='3h')
    z = xr.open_zarr(rep['zarr'])
    assert z.time.size == 6 * 8
    assert set(z.time.dt.hour.values) == {0, 3, 6, 9, 12, 15, 18, 21}
    # accumulated tp at 03h = the source's 1-hour value at 03h (sampled, not summed)
    np.testing.assert_allclose(z.tp.sel(time='2020-01-02T03', latitude=42, longitude=23).values,
                               _value(fe.SURF_VARS.index('tp'), 27, 42, 23))


def test_out_of_range_dates(stores, tmp_path):
    with pytest.raises(ValueError, match='covers'):
        _fetch(stores, tmp_path, end='2020-02-01')


# ---------------------------------------------------------------- retries

class _HTTPError(Exception):
    def __init__(self, status):
        super().__init__(f'HTTP {status}')
        self.status = status


def test_retry_transient_then_succeed(monkeypatch):
    ds = xr.Dataset({'a': ('x', np.arange(3.0))})
    n = {'calls': 0}
    real_load = xr.Dataset.load

    def flaky(self, **kw):
        n['calls'] += 1
        if n['calls'] < 3:
            raise _HTTPError(503)
        return real_load(self, **kw)
    monkeypatch.setattr(xr.Dataset, 'load', flaky)
    out = fe._load_with_retry(ds, 2, max_retries=5, backoff_s=0)
    assert n['calls'] == 3 and out.a.values.tolist() == [0, 1, 2]


@pytest.mark.parametrize('status, exc', [(426, RuntimeError), (401, RuntimeError), (404, _HTTPError)])
def test_no_retry_on_auth_or_not_found(monkeypatch, status, exc):
    ds = xr.Dataset({'a': ('x', np.arange(3.0))})
    n = {'calls': 0}

    def fail(self, **kw):
        n['calls'] += 1
        raise _HTTPError(status)
    monkeypatch.setattr(xr.Dataset, 'load', fail)
    with pytest.raises(exc):
        fe._load_with_retry(ds, 2, max_retries=5, backoff_s=0)
    assert n['calls'] == 1


# ---------------------------------------------------------------- integration (network)

def _have_edh_credentials():
    if os.environ.get('EDH_TOKEN'):
        return True
    try:
        return netrc.netrc().authenticators(fe.EDH_HOST) is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


@pytest.mark.skipif(not _have_edh_credentials(), reason='no Earth Data Hub credentials')
def test_integration_real_edh_two_days(tmp_path):
    bbox = {'latN': 42.3, 'latS': 42.0, 'lonW': 23.4, 'lonE': 23.7}   # tiny box around Musala
    rep = fe.fetch_era5_edh('2025-01-10', '2025-01-11', bbox, [600, 700, 850, 925, 1000],
                            output_dir=tmp_path, max_concurrency=8)
    z = xr.open_zarr(rep['zarr'])
    # whole remote chunks are kept; the requested 48 h must be inside and complete
    assert z.sel(time=slice('2025-01-10', '2025-01-11T23')).time.size == 48
    assert z.level.values.tolist() == [600.0, 700.0, 850.0, 925.0, 1000.0]
    assert z.latitude.values.tolist() == [42.5, 42.25, 42.0] and z.longitude.values.tolist() == [23.25, 23.5, 23.75]
    # plausibility: January temperatures (K) and a geopotential column that increases upward
    assert 230 < float(z.t2m.min()) and float(z.t2m.max()) < 295
    zc = z.z.isel(time=0, latitude=0, longitude=0).values
    assert np.all(np.diff(zc) < 0)          # level ascending in hPa -> geopotential decreasing
    assert json.loads((tmp_path / 'edh_blocks' / 'fetch_report.json').read_text())['total_requests_estimate'] > 0


def test_short_fetch_does_not_shrink_store(stores, tmp_path):
    _fetch(stores, tmp_path)                                   # 2020-01-02 .. 01-06
    rep = _fetch(stores, tmp_path, start='2020-01-03', end='2020-01-03')
    z = xr.open_zarr(rep['zarr'])
    assert z.time.size == 6 * 24                               # still the whole fetched period
    assert rep['plev']['n_blocks_fetched'] == 0 and rep['surf']['n_blocks_fetched'] == 0
