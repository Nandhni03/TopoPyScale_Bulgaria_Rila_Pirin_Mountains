"""
Fetch ERA5 from Earth Data Hub (EDH, DestinE) Zarr stores for TopoPyScale.

EDH mirrors ERA5 as analysis-ready Zarr v3 stores (hourly, 0.25 deg, 1940-present). Reading a
small domain from them is much faster than queueing CDS requests: data is read chunk by chunk
over HTTP, many chunks at a time.

Output: a local Zarr store with exactly the layout `topo_scale_zarr.ClimateDownscaler` reads
(the same layout `fetch_era5.convert_netcdf_stack_to_zarr` builds from CDS files):

    dims    time, level (hPa, ascending), latitude (descending), longitude (-180..180, ascending)
    PLEV    z, t, u, v, q, r          (time, level, latitude, longitude)
    SURF    z_surf, d2m, sp, strd, ssrd, tp, t2m   (time, latitude, longitude)

Optionally also yearly SURF_YYYY.nc / PLEV_YYYY.nc files laid out like the CDS path
(`time`, `level`, surface geopotential named `z`), for the classic `topo_scale` downscaler.

How the download works
    1. Both stores are opened lazily (only metadata is read).
    2. The period is split into time blocks aligned to each store's own chunk grid
       (pressure levels: 1440 h = 60 days, single levels: 1080 h = 45 days). Each block reads
       every remote chunk it touches exactly once, so the number of HTTP requests (which count
       against the EDH quota) is predictable: n_blocks x n_variables (x n_levels for PLEV).
    3. A block is loaded with Dask threads (up to `max_concurrency` chunks in flight), cut to
       the domain, and written to its own NetCDF file under a temporary name, then renamed.
       A finished block file therefore means a finished block, and a rerun skips it.
    4. When all blocks exist, they are assembled into the final Zarr store (and/or yearly
       NetCDF files), rechunked for fast per-point reads.

Authentication: the EDH API key is read from the EDH_TOKEN environment variable, or from
~/.netrc (`machine data.earthdatahub.destine.eu  login edh  password <key>`). It is never
written anywhere. EDH needs a *classic* API key; a standard key gets HTTP 426.

S. Filhol / J. Fiddes TopoPyScale; EDH fetcher added 2026 for the Rila-Pirin project.
"""
import json
import math
import netrc
import os
import time as _time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

try:
    from zarr.codecs import BloscCodec
except ImportError:
    from zarr.codecs._blosc import BloscCodec


EDH_HOST = 'data.earthdatahub.destine.eu'
EDH_URLS = {
    'plev': f'https://{EDH_HOST}/era5/era5-pressure-levels-v0.zarr',
    'surf': f'https://{EDH_HOST}/era5/era5-single-levels-atmosphere-v0.zarr',
}
# Variables TopoPyScale needs (EDH short names = CDS short names)
PLEV_VARS = ['z', 't', 'u', 'v', 'q', 'r']
SURF_VARS = ['z', 'd2m', 'sp', 'strd', 'ssrd', 'tp', 't2m']
# Accumulated over the previous hour at each valid_time (CDS convention); see apply_timestep_convention()
ACCUM_VARS = ['tp', 'ssrd', 'strd']

EDH_TIME_DIM = 'valid_time'
EDH_LEVEL_DIM = 'isobaricInhPa'

TRANSIENT_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
AUTH_HTTP_STATUS = {401, 403, 426}


def _log(msg):
    print(f'---> [EDH] {msg}', flush=True)


# ---------------------------------------------------------------- auth / opening

def get_storage_options(url):
    """
    fsspec/aiohttp options for reading `url`. Local paths get none.

    EDH_TOKEN (environment) takes precedence; otherwise aiohttp's trust_env reads ~/.netrc.
    """
    if not str(url).startswith('http'):
        return None
    token = os.environ.get('EDH_TOKEN')
    if token:
        import base64
        basic = base64.b64encode(f'edh:{token}'.encode()).decode()
        return {'client_kwargs': {'headers': {'Authorization': f'Basic {basic}'}, 'trust_env': True}}
    try:
        has_netrc = netrc.netrc().authenticators(EDH_HOST) is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        has_netrc = False
    if not has_netrc:
        raise RuntimeError(
            f'No Earth Data Hub credentials found. Set the EDH_TOKEN environment variable, or add to ~/.netrc:\n'
            f'    machine {EDH_HOST}\n    login edh\n    password <your classic API key>\n'
            f'(create the key at https://platform.destine.eu -> Quota & API Keys).')
    return {'client_kwargs': {'trust_env': True}}


def open_edh_store(kind, url=None):
    """Open an EDH ERA5 store lazily. kind: 'plev' or 'surf'. `url` may be a local path (tests)."""
    url = url or EDH_URLS[kind]
    try:
        ds = xr.open_dataset(url, engine='zarr', chunks={}, storage_options=get_storage_options(url),
                             decode_timedelta=True)
    except Exception as e:
        _raise_if_auth_error(e)
        raise
    return ds


def _http_status(e):
    return getattr(e, 'status', None)


def _raise_if_auth_error(e):
    if _http_status(e) in AUTH_HTTP_STATUS:
        hint = (' EDH returns 426 when the key is a *standard* API key: create a *classic* key.'
                if _http_status(e) == 426 else '')
        raise RuntimeError(f'Earth Data Hub refused the credentials (HTTP {_http_status(e)}).{hint}') from e


def _is_transient(e):
    """Errors worth retrying: timeouts, dropped connections, 5xx/429. Not 404 or auth errors."""
    status = _http_status(e)
    if status is not None:
        return status in TRANSIENT_HTTP_STATUS
    if isinstance(e, FileNotFoundError):
        return False
    try:
        import aiohttp
        if isinstance(e, (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError)):
            return True
    except ImportError:
        pass
    return isinstance(e, (TimeoutError, ConnectionError, OSError))


# ---------------------------------------------------------------- domain / levels / time

def buffer_extent(extent, buffer=0.4):
    """Apply the same buffer as FetchERA5.go_fetch() to a {'latN','latS','lonW','lonE'} extent."""
    return {'latN': extent['latN'] + buffer, 'latS': extent['latS'] - buffer,
            'lonW': extent['lonW'] - buffer, 'lonE': extent['lonE'] + buffer}


def _snap_out(lo, hi, res):
    """Expand [lo, hi] outward to the grid of resolution res (tolerant to float noise)."""
    eps = 1e-6
    return math.floor(lo / res + eps) * res, math.ceil(hi / res - eps) * res


def select_domain(ds, bbox):
    """
    Cut ds to bbox = {'latN','latS','lonW','lonE'} (lon in -180..180 or 0..360).

    The box is expanded outward to whole grid cells, so every point of the domain has ERA5
    neighbours on all sides (TopoPyScale takes the 3x3 cells around each point). Handles the
    store's 0..360 longitudes and boxes crossing the Greenwich meridian. Output longitudes are
    -180..180 ascending (as in CDS files and df_centroids), latitudes descending (as in CDS).
    """
    res = float(abs(ds.latitude.values[1] - ds.latitude.values[0]))
    latS, latN = _snap_out(bbox['latS'], bbox['latN'], res)
    lonW, lonE = _snap_out(bbox['lonW'], bbox['lonE'], res)
    if lonE - lonW >= 360:
        raise ValueError('bbox spans all longitudes')

    lat_desc = ds.latitude.values[0] > ds.latitude.values[-1]
    ds = ds.sel(latitude=slice(latN, latS) if lat_desc else slice(latS, latN))

    w, e = lonW % 360, lonE % 360
    if w <= e:
        ds = ds.sel(longitude=slice(w, e))
    else:  # crosses 0 deg: [w, 360) + [0, e]
        ds = xr.concat([ds.sel(longitude=slice(w, 360)), ds.sel(longitude=slice(0, e))], dim='longitude')
    lon = ((ds.longitude.values + 180) % 360) - 180
    if np.any(np.diff(lon) <= 0):
        raise NotImplementedError('bbox crossing the 180 deg meridian is not supported')
    ds = ds.assign_coords(longitude=lon)
    if not lat_desc:
        ds = ds.sortby('latitude', ascending=False)
    return ds


def resolve_plevels(requested, available):
    """
    Pick the pressure levels to download.

    Uses every available level within [min(requested), max(requested)] hPa. If an end of that
    range is not available, the nearest available level beyond it is added (the next lower
    pressure above the top, the next higher pressure below the bottom), so the column the user
    asked for stays bracketed and vertical interpolation never has to extrapolate.

    Returns (levels_ascending, info_dict).
    """
    requested = sorted(float(p) for p in requested)
    available = sorted(float(p) for p in available)
    lo, hi = requested[0], requested[-1]
    use = [p for p in available if lo <= p <= hi]
    added = []
    if lo not in available:
        above = [p for p in available if p < lo]
        if above:
            use.append(above[-1])
            added.append(above[-1])
    if hi not in available:
        below = [p for p in available if p > hi]
        if below:
            use.append(below[0])
            added.append(below[0])
    use = sorted(set(use))
    missing = [p for p in requested if p not in available]
    info = {'requested': requested, 'used': use, 'missing_from_source': missing, 'added_to_bracket': added}
    if missing:
        _log(f'WARNING: {len(missing)} requested pressure level(s) are not in the EDH store: '
             f'{[int(p) for p in missing]} hPa.')
    if added:
        _log(f'Added level(s) {[int(p) for p in added]} hPa to keep the requested column '
             f'{int(lo)}-{int(hi)} hPa bracketed.')
    _log(f'Pressure levels used: {[int(p) for p in use]} hPa')
    if len(use) < 2:
        raise ValueError(f'Fewer than 2 pressure levels available for {requested}')
    return use, info


def apply_timestep_convention(ds, timestep, time_dim='time'):
    """
    Reduce hourly data to the model timestep ('1h', '3h', '6h').

    1h passes through unchanged. For 3h/6h every variable, including the accumulated ones
    (tp, ssrd, strd), is *sampled* at hours 00, 03, ... (or 00, 06, ...). That is exactly what
    a 3h/6h CDS download contains: ERA5 accumulations at a valid time cover the previous
    1 hour only, whatever timestep is requested. `topo_scale.py` (since commit 91ab64c)
    and the aligned `topo_scale_zarr.py` assume this: radiation / 3600 s, tp x timestep.
    Decision recorded in PROGRESS.md (Phase 0.2, option a).
    """
    step = pd.Timedelta(timestep)
    if step == pd.Timedelta('1h'):
        return ds
    if step not in (pd.Timedelta('3h'), pd.Timedelta('6h')):
        raise ValueError(f'timestep must be 1h, 3h or 6h, got {timestep}')
    hours = int(step / pd.Timedelta('1h'))
    return ds.sel({time_dim: ds[time_dim].dt.hour % hours == 0})


def chunk_aligned_blocks(start, end, origin, chunk_hours, last=None):
    """
    The store's time chunks that intersect [start, end] (inclusive timestamps), as WHOLE chunks.
    origin = first time in the store; chunk k covers origin + [k*chunk_hours, (k+1)*chunk_hours).
    Blocks are never clipped to [start, end]: the remote chunk is downloaded whole anyway, and whole
    chunks give one canonical block file per chunk, so fetches of different periods never overlap.
    Only the store's last (still growing) chunk is clipped to `last`.
    Returns a list of (block_start, block_end) inclusive timestamps.
    """
    start, end, origin = pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(origin)
    h = pd.Timedelta('1h')
    k0 = int((start - origin) // (chunk_hours * h))
    k1 = int((end - origin) // (chunk_hours * h))
    blocks = []
    for k in range(k0, k1 + 1):
        c0 = origin + k * chunk_hours * h
        c1 = c0 + (chunk_hours - 1) * h
        blocks.append((c0, min(c1, pd.Timestamp(last)) if last is not None else c1))
    return blocks


def _time_chunk_hours(ds, var):
    enc = ds[var].encoding
    chunks = enc.get('preferred_chunks', {}).get(EDH_TIME_DIM) or (enc.get('chunks') or [None])[0]
    if not chunks:
        raise ValueError(f'Cannot determine time chunking of {var}')
    return int(chunks)


# ---------------------------------------------------------------- block download

def _standardize(ds, kind):
    """EDH names -> TopoPyScale names. Level ascending (lowest level = level[-1] = highest pressure)."""
    ds = ds.rename({EDH_TIME_DIM: 'time'})
    if kind == 'plev':
        ds = ds.rename({EDH_LEVEL_DIM: 'level'})
        ds = ds.assign_coords(level=ds.level.astype('float64')).sortby('level', ascending=True)
    drop = [c for c in ds.coords if c not in ds.dims]  # e.g. number, expver, step, surface
    return ds.drop_vars(drop)


def _load_with_retry(ds, max_concurrency, max_retries, backoff_s):
    for attempt in range(max_retries + 1):
        try:
            with dask.config.set(scheduler='threads', num_workers=max_concurrency):
                return ds.load()
        except Exception as e:  # noqa: BLE001 - classified below
            _raise_if_auth_error(e)
            if attempt == max_retries or not _is_transient(e):
                raise
            wait = backoff_s * 2 ** attempt
            _log(f'transient error ({type(e).__name__}: {e}); retry {attempt + 1}/{max_retries} in {wait:.0f}s')
            _time.sleep(wait)


def _block_path(block_dir, kind, b0, b1):
    return Path(block_dir) / f'{kind}_{b0:%Y%m%dT%H}_{b1:%Y%m%dT%H}.nc'


def _check_manifest(block_dir, manifest):
    """Refuse to mix blocks fetched with different settings in one directory."""
    f = Path(block_dir) / 'manifest.json'
    if f.exists():
        old = json.loads(f.read_text())
        diff = {k: (old.get(k), v) for k, v in manifest.items() if old.get(k) != v}
        if diff:
            raise ValueError(f'{block_dir} holds blocks fetched with different settings {diff}. '
                             f'Use another directory or delete it.')
    else:
        f.write_text(json.dumps(manifest, indent=2))


def fetch_blocks(kind, start, end, bbox, block_dir, plevels=None, timestep='1h', url=None,
                 max_concurrency=8, max_retries=5, backoff_s=10):
    """
    Download one store (kind 'plev' or 'surf') for [start, end] into block NetCDF files.
    Skips blocks already on disk. Returns (list_of_block_files, info_dict).
    """
    t0 = _time.time()
    block_dir = Path(block_dir)
    block_dir.mkdir(parents=True, exist_ok=True)
    src = open_edh_store(kind, url)
    varlist = PLEV_VARS if kind == 'plev' else SURF_VARS
    missing_vars = [v for v in varlist if v not in src]
    if missing_vars:
        raise KeyError(f'{kind} store lacks variables {missing_vars}')
    ds = src[varlist]

    first, last = pd.Timestamp(ds[EDH_TIME_DIM].values[0]), pd.Timestamp(ds[EDH_TIME_DIM].values[-1])
    if pd.Timestamp(start) < first or pd.Timestamp(end) > last:
        raise ValueError(f'Requested {start} .. {end} but the EDH {kind} store covers {first} .. {last}')

    info = {}
    if kind == 'plev':
        levels, info['plevels'] = resolve_plevels(plevels, ds[EDH_LEVEL_DIM].values)
        ds = ds.sel({EDH_LEVEL_DIM: [lv for lv in ds[EDH_LEVEL_DIM].values if float(lv) in levels]})
    ds = select_domain(ds, bbox)
    info['latitude'] = [float(ds.latitude.max()), float(ds.latitude.min()), int(ds.latitude.size)]
    info['longitude'] = [float(ds.longitude.min()), float(ds.longitude.max()), int(ds.longitude.size)]

    chunk_h = _time_chunk_hours(src, varlist[0])
    blocks = chunk_aligned_blocks(start, end, first, chunk_h, last=last)
    n_fields = len(varlist) * (ds.sizes.get(EDH_LEVEL_DIM, 1))
    todo = [(b0, b1) for b0, b1 in blocks if not _block_path(block_dir, kind, b0, b1).exists()]
    info.update({'n_blocks': len(blocks), 'n_blocks_fetched': len(todo), 'chunk_hours': chunk_h,
                 'estimated_requests': len(todo) * n_fields})
    _log(f'{kind}: {len(blocks)} block(s) of <= {chunk_h} h, {len(blocks) - len(todo)} already on disk, '
         f'{len(todo)} to fetch (~{len(todo) * n_fields} HTTP chunk requests), '
         f'grid {info["latitude"][2]} x {info["longitude"][2]}')

    files = []
    for i, (b0, b1) in enumerate(blocks, 1):
        target = _block_path(block_dir, kind, b0, b1)
        files.append(target)
        if target.exists():
            continue
        tb = _time.time()
        block = ds.sel({EDH_TIME_DIM: slice(b0, b1)})
        block = apply_timestep_convention(block, timestep, time_dim=EDH_TIME_DIM)
        block = _load_with_retry(block, max_concurrency, max_retries, backoff_s)
        block = _standardize(block, kind)
        block.attrs.update({'source': EDH_URLS[kind] if url is None else str(url),
                            'fetched_with': 'TopoPyScale.fetch_era5_edh',
                            'fetch_date': pd.Timestamp.now(tz='UTC').isoformat(),
                            'timestep_convention': f'{timestep}; accumulations = previous 1 h (sampled, CDS-like)'})
        tmp = target.with_suffix('.nc.tmp')
        block.to_netcdf(tmp, engine='netcdf4', encoding={v: {'zlib': True, 'complevel': 4} for v in block.data_vars})
        os.replace(tmp, target)
        _log(f'{kind} block {i}/{len(blocks)} {b0:%Y-%m-%d} .. {b1:%Y-%m-%d %H}h done in {_time.time() - tb:.0f}s')
    info['seconds'] = round(_time.time() - t0, 1)
    return files, info


# ---------------------------------------------------------------- assembly

def _open_blocks(files):
    ds = xr.open_mfdataset(sorted(str(f) for f in files), combine='nested', concat_dim='time',
                           engine='netcdf4', data_vars='minimal', coords='minimal', compat='override')
    # safety net: drop duplicated timesteps (e.g. a block re-fetched after the store's last chunk grew)
    _, keep = np.unique(ds.time.values, return_index=True)
    return ds.isel(time=np.sort(keep))


def _expected_times(start, end, timestep):
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=timestep)


def _check_complete(ds, tvec, what):
    have = pd.DatetimeIndex(ds.time.values)
    missing = tvec.difference(have)
    if len(missing):
        raise ValueError(f'{what}: {len(missing)} timesteps missing, first {missing[:3].tolist()}')


def assemble_zarr(plev_files, surf_files, zarr_path, start, end, timestep='1h', time_chunk=8760):
    """
    Build the ERA5 Zarr store read by topo_scale_zarr.ClimateDownscaler from block files.

    The store holds every timestep present in the given block files (all blocks fetched so far into
    the block directory), so fetching a short period never shrinks a store that covers a longer one.
    [start, end] must be complete; other periods are included as far as they were fetched.
    Written to `<zarr_path>.tmp` then renamed, so a half-written store never replaces a good one.
    Chunks: `time_chunk` steps x all levels x the whole (small) domain, so reading the 3x3 cells
    around one point for the whole period touches few, compact chunks.
    """
    zarr_path = Path(zarr_path)
    tvec = _expected_times(start, end, timestep)
    plev = _open_blocks(plev_files)
    surf = _open_blocks(surf_files).rename({'z': 'z_surf'})
    common = np.intersect1d(plev.time.values, surf.time.values)
    plev, surf = plev.sel(time=common), surf.sel(time=common)
    _check_complete(plev, tvec, 'PLEV')
    _check_complete(surf, tvec, 'SURF')
    ds = xr.merge([plev, surf], compat='override', combine_attrs='drop_conflicts')
    ds.attrs.update({'title': 'ERA5 from Earth Data Hub, cut and formatted for TopoPyScale',
                     'source_plev': EDH_URLS['plev'], 'source_surf': EDH_URLS['surf']})
    chunks = {'time': min(time_chunk, ds.sizes['time']), 'level': ds.sizes['level'],
              'latitude': ds.sizes['latitude'], 'longitude': ds.sizes['longitude']}
    ds = ds.chunk({k: v for k, v in chunks.items()})
    for v in ds.variables:
        ds[v].encoding = {}
    comp = BloscCodec(cname='lz4', clevel=5, shuffle='bitshuffle', blocksize=0)
    encoding = {v: {'compressors': comp} for v in ds.data_vars}
    tmp = zarr_path.with_name(zarr_path.name + '.tmp')
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    ds.to_zarr(tmp, mode='w', zarr_format=3, encoding=encoding, consolidated=True)
    if zarr_path.exists():
        import shutil
        shutil.rmtree(zarr_path)
    os.replace(tmp, zarr_path)
    _log(f'Zarr store written: {zarr_path} ({ds.sizes["time"]} steps, levels {ds.level.values.astype(int).tolist()})')
    return zarr_path


def write_yearly_netcdf(plev_files, surf_files, climate_dir, start, end, timestep='1h', link_to_root=True):
    """
    Write yearly/PLEV_YYYY.nc and yearly/SURF_YYYY.nc in the CDS-path layout (surface geopotential
    stays `z`), and symlink them into climate_dir so `topo_scale.downscale_climate` finds them.
    """
    climate_dir = Path(climate_dir)
    yearly = climate_dir / 'yearly'
    yearly.mkdir(parents=True, exist_ok=True)
    tvec = _expected_times(start, end, timestep)
    plev, surf = _open_blocks(plev_files), _open_blocks(surf_files)
    common = pd.DatetimeIndex(np.intersect1d(plev.time.values, surf.time.values))
    out = []
    for year in sorted(set(common.year)):         # every year with data on disk, not only [start, end]
        ty = common[common.year == year]
        for name, ds in (('PLEV', plev), ('SURF', surf)):
            d = ds.sel(time=ty)
            _check_complete(d, tvec[tvec.year == year], f'{name} {year}')
            if len(ty) < len(pd.date_range(f'{year}-01-01', f'{year}-12-31T23', freq=timestep)):
                _log(f'NOTE: {name}_{year}.nc holds only the {len(ty)} steps fetched so far')
            f = yearly / f'{name}_{year}.nc'
            tmp = f.with_suffix('.nc.tmp')
            d.to_netcdf(tmp, engine='netcdf4', encoding={v: {'zlib': True, 'complevel': 4} for v in d.data_vars})
            os.replace(tmp, f)
            out.append(f)
            if link_to_root:
                link = climate_dir / f.name
                if not link.exists():
                    link.symlink_to(Path('yearly') / f.name)
    _log(f'Yearly NetCDF written: {[f.name for f in out]}')
    return out


# ---------------------------------------------------------------- main entry point

def warn_if_windows_mount(*paths):
    for p in paths:
        if p is not None and str(Path(p).resolve()).startswith('/mnt/'):
            _log(f'WARNING: {p} is on a Windows drive mounted in WSL (/mnt/...). Zarr stores are many '
                 f'small files and are very slow there; use the Linux filesystem (e.g. under ~/).')


def fetch_era5_edh(start, end, bbox, plevels, timestep='1h', output_dir='inputs/climate',
                   output_format='zarr', zarr_name='ERA5.zarr', block_dir=None,
                   max_concurrency=8, max_retries=5, backoff_s=10, time_chunk=8760,
                   urls=None):
    """
    Fetch ERA5 for TopoPyScale from Earth Data Hub.

    Args:
        start, end: first and last day (inclusive; the whole last day is fetched)
        bbox (dict): {'latN','latS','lonW','lonE'}, already buffered (see buffer_extent())
        plevels (list): requested pressure levels [hPa]; see resolve_plevels() for unavailable ones
        timestep (str): '1h', '3h' or '6h' (see apply_timestep_convention())
        output_dir: climate directory (config climate.path)
        output_format (str): 'zarr', 'netcdf' (yearly SURF/PLEV files) or 'both'
        zarr_name (str): name of the Zarr store inside output_dir (config climate.era5.zarr_store)
        block_dir: where block files go (default <output_dir>/edh_blocks); keep it to resume
        max_concurrency (int): max remote chunks downloaded at the same time
        max_retries, backoff_s: retries for transient HTTP errors, exponential backoff
        time_chunk (int): time chunk length of the output Zarr store
        urls (dict): override store URLs {'plev': ..., 'surf': ...} (tests)

    Returns:
        dict with output paths and fetch statistics (also saved as <block_dir>/fetch_report.json)
    """
    if output_format not in ('zarr', 'netcdf', 'both'):
        raise ValueError("output_format must be 'zarr', 'netcdf' or 'both'")
    urls = urls or {}
    output_dir = Path(output_dir)
    block_dir = Path(block_dir) if block_dir else output_dir / 'edh_blocks'
    warn_if_windows_mount(output_dir, block_dir)
    block_dir.mkdir(parents=True, exist_ok=True)

    t_start = pd.Timestamp(start).normalize()
    t_end = pd.Timestamp(end).normalize() + pd.Timedelta('23h')
    _check_manifest(block_dir, {'bbox': {k: round(float(v), 4) for k, v in bbox.items()},
                                'plevels_requested': sorted(float(p) for p in plevels),
                                'timestep': timestep, 'plev_vars': PLEV_VARS, 'surf_vars': SURF_VARS})
    _log(f'Fetching {t_start:%Y-%m-%d} .. {t_end:%Y-%m-%d}, timestep {timestep}, bbox {bbox}')

    kw = dict(bbox=bbox, block_dir=block_dir, timestep=timestep, max_concurrency=max_concurrency,
              max_retries=max_retries, backoff_s=backoff_s)
    plev_files, plev_info = fetch_blocks('plev', t_start, t_end, plevels=plevels, url=urls.get('plev'), **kw)
    surf_files, surf_info = fetch_blocks('surf', t_start, t_end, url=urls.get('surf'), **kw)

    report = {'start': str(t_start), 'end': str(t_end), 'timestep': timestep, 'plev': plev_info, 'surf': surf_info}
    ta = _time.time()
    if output_format in ('zarr', 'both'):
        # assemble from ALL blocks in block_dir (same settings, guaranteed by the manifest)
        plev_files = sorted(block_dir.glob('plev_*.nc'))
        surf_files = sorted(block_dir.glob('surf_*.nc'))
        report['zarr'] = str(assemble_zarr(plev_files, surf_files, output_dir / zarr_name,
                                           t_start, t_end, timestep, time_chunk))
    if output_format in ('netcdf', 'both'):
        plev_files = sorted(block_dir.glob('plev_*.nc'))
        surf_files = sorted(block_dir.glob('surf_*.nc'))
        report['netcdf'] = [str(f) for f in write_yearly_netcdf(plev_files, surf_files, output_dir,
                                                                t_start, t_end, timestep)]
    report['assembly_seconds'] = round(_time.time() - ta, 1)
    report['total_requests_estimate'] = plev_info['estimated_requests'] + surf_info['estimated_requests']
    (block_dir / 'fetch_report.json').write_text(json.dumps(report, indent=2, default=str))   # latest run
    with open(block_dir / 'fetch_history.jsonl', 'a') as fh:                                   # every run
        fh.write(json.dumps(report, default=str) + '\n')
    _log(f'Done. Download {plev_info["seconds"] + surf_info["seconds"]:.0f}s, assembly '
         f'{report["assembly_seconds"]:.0f}s, ~{report["total_requests_estimate"]} chunk requests this run.')
    return report
