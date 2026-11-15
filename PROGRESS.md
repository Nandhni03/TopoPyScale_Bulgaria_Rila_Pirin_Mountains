# PROGRESS: Earth Data Hub ERA5 source + Zarr processing (feature/edh-fetch)

Git: Claude does not commit. Would-be commits are listed in
`../../GIT_PENDING.md` (repo-github/GIT_PENDING.md).

## Setup (2026-09-30), done

- Docker image `toposcale-rila-pirin:fork` (project repo `rila-pirin-downscaling`) now runs
  **this fork** (bind-mounted at `/opt/TopoPyScale`, on `PYTHONPATH`) instead of PyPI
  0.3.3 + sed patches. Package versions frozen from the old working image
  (`docker/requirements.lock.txt`: zarr 3.2.1, xarray 2026.7.0, pandas 3.0.5, dask 2026.7.1).
  Old image kept as `toposcale-rila-pirin:v0.3.3-patched`. Container runs as uid 1000.
- Bulk data on the Linux filesystem: host `repo-github/rila-pirin-data` = container `/data`
  (`terrain/`, `era5/`, `runs/`, `cluster_search/`).
- Old sed patches vs this fork: 5 of 6 are already fixed upstream. The remaining one,
  chained assignment of `point_name` in `topoclass.py` (pandas 3 CoW; left the saved
  cluster map as all `-9999`), is fixed here.
- Terrain: `/data/terrain/ds_param_terrain_only.nc` (pristine; terrain identical to the
  2.3 GB copy used in the pilots) + `mask.tif` (13,945,288 px inside AOI).

## Cluster-number search, DONE (2026-09-30; 5462 s wall, 8 workers)

Script `rila-pirin-downscaling/bulgaria_rila_pirin/scripts/cluster_search.py`; results, figures
(PNG + PDF, 300 dpi) and `run_info.txt` (methods-section details) in `/data/cluster_search/`.
Settings: all 13,945,288 masked 25 m pixels, TopoPyScale's own scaler + mini-batch k-means
(batch 5120), k = 100…2000, elevation weight 1 vs 4, 3 seeds (spread < 1.5 m everywhere).

| k | elev RMSE w=1 | elev RMSE w=4 | slope RMSE w=1 / w=4 | aspect err w=1 / w=4 | median px per cluster |
|---|---|---|---|---|---|
| 100 | 246 m | 91 m | 4.1° / 4.9° | 18° / 26° | 136k (85 km²) |
| 300 | 195 m | 73 m | 3.3° / 4.1° | 14° / 19° | 45k |
| 500 | 175 m | **65 m** | 3.0° / 3.7° | 12° / 16.5° | 27k (17 km²) |
| 800 | 159 m | 59 m | 2.7° / 3.4° | 11° / 15° | 17k |
| 1000 | 153 m | 56 m | 2.6° / 3.3° | 10° / 14° | 13k |
| 2000 | 132 m | 48 m | 2.2° / 2.9° | 9° / 12° | 7k |

Reading:
- **Weight 4 lowers the elevation error ~2.7× at every k** (65 m at k=500 ≈ 0.4 K of lapse-rate
  temperature error, vs 175 m ≈ 1.1 K with weight 1). Weight 4 with k=300 already beats weight 1
  with k=2000.
- **The cost**: slope, aspect and sky-view errors each rise ~20–35 % at the same k (elevation
  dominates the distance). That matters for SW radiation on slopes (illumination angle) and a
  little for LW (svf). Temperature, humidity and precipitation (lapse rate) depend on elevation.
  Weight 4 is a reasonable choice for temperature-driven work (snow, rock glaciers), but the
  thesis should state the trade-off (supplementary figure panels a–c).
- **Diminishing returns** (main figure panel b): with weight 4, each extra 100 clusters removes
  < 2.5 m of RMSE beyond k ≈ 500 and < 1 m beyond k ≈ 1000. Compute cost grows ~linearly with k.
- Davies–Bouldin/Calinski–Harabasz give no clear optimum (terrain is a continuum), same as
  in the earlier search. Elevation RMSE is the physically meaningful metric.
- **Decision on k: the user's.** Phase 3 uses k=500, weight 4 provisionally.

## Phase 0: investigation (no code changes), DONE

### 0.1 EDH stores: DONE (2026-09-30, after the user switched to a classic API key → HTTP 200)
Full dump: `/data/phase0/edh_inspect.log` (script `/data/phase0/edh_inspect.py`; host path
`repo-github/rila-pirin-data/phase0/`). Corrections to the brief are marked **bold**.

| | pressure levels (`era5-pressure-levels-v0.zarr`) | single levels (`era5-single-levels-atmosphere-v0.zarr`) |
|---|---|---|
| dims | `valid_time` 760080, **`isobaricInhPa` 19**, `latitude` 721, `longitude` 1440 | `valid_time`, `latitude`, `longitude` |
| chunks | **1440 × 1 × 60 × 60** (one level per chunk) | 1080 × 60 × 60 |
| time | hourly, 1940-01-01T00 … 2026-09-15T23 | same |
| level coord | **`isobaricInhPa`, int32, ordered 1000 → 1 (descending)** | n/a |
| latitude | float64, **90 → -90 (descending)** | same |
| longitude | float64, 0 → 359.75 | same |
| vars needed | z, t, u, v, q, r: float32, `GRIB_stepType=instant` | z, d2m, sp, t2m: instant; **ssrd, strd, tp: `accum`** (J m-2, m) |
| codec | Blosc | Blosc |
| extra coords | none (no `number`/`expver` in these stores) | none |

- **Level order matters:** both downscalers use `z.isel(level=-1)` as the *lowest* level,
  and the CDS files are ordered 600 → 1000 (checked `PLEV_2022.nc`). The fetcher must
  rename `isobaricInhPa` → `level` and **sort ascending** (600, 700, 850, 925, 1000).
- **Spatial chunks:** the buffered bbox (lat 41.03–42.72, lon 22.57–24.49) lies in lat
  chunk 3 (index 180–239 = 45.0 → 30.25°N) and lon chunk 1 (15.0 → 29.75°E): **one spatial
  chunk per variable (per level)**, confirmed.
- **Time chunks for 2025:** chunk grids start at 1940-01-01T00. PLEV chunks 517–523 (517
  starts 2024-12-05); SURF chunks 689–698 (689 starts 2024-11-20). "60-day blocks" match
  PLEV only; SURF chunks are 45 days, so each store's blocks should follow its own grid.
- **Transfer estimate for 2025:** PLEV 7 time chunks × 6 vars × 5 levels = 210 chunks
  (~20.7 MB each uncompressed); SURF 10 × 7 = 70 chunks (~15.6 MB). About 5.5 GB before
  compression and ~300 requests, far below the 500k/month quota.
- **Side finding (for Phase 3):** the existing CDS file `PLEV_2022.nc` is *not* on the
  native 0.25° grid (lat 42.669, 42.419…; lon 22.390, 22.640…, step 0.2501), because CDS
  regrids when the requested `area` isn't grid-aligned. EDH returns native grid points, so
  an EDH-vs-CDS comparison must request a grid-aligned CDS area, or it will show
  interpolation differences that aren't source differences.

### 0.2 Accumulation convention: DONE, decided
Inputs from the CDS path: `fetch_era5.retrieve_era5` requests ERA5 *reanalysis* at the
listed hours only (`time_step_dict`: 3h → 00,03,…,21). In CDS ERA5 reanalysis, the
accumulated fields (tp [m], ssrd/strd [J m-2]) at hour T are the accumulation over the
**previous 1 hour**, whatever timestep is requested. A 3h CDS download is therefore a
*sample* of 1-hour accumulations, not 3-hour sums.

| | `topo_scale.py` (classic) | `topo_scale_zarr.py` (ClimateDownscaler) |
|---|---|---|
| tp | `tp * tstep * 1000` (l.190) | `tp / tstep * 1000` (l.308) |
| ssrd/strd → W m-2 | `/ 3600` fixed (l.258) | `/ (tstep*3600)` (l.335) |
| assumes input is | 1-h accumulation sampled every tstep | accumulation over the whole tstep |
| at 1h | tp mm/h, SW/LW W m-2 ✓ | identical ✓ |
| at 3h with CDS-style input | SW/LW ✓; tp = mm per 3 h (1-h sample × 3) | SW/LW **÷3 too low**; tp **÷9** vs classic |

- History: `topo_scale.py` originally had the same formulas as the zarr version. Joel
  Fiddes changed it in `91ab64c` (2025-11-08): *"radiation always divided by 3600
  (accumulated over 1hour) and TP is multiplied by timestep to get full budget when
  timesteps are > 1"*. `topo_scale_zarr.py` (arcticsnow, `008a53b`, 2025-08-22) was written
  before that and **never updated**, so it is inconsistent with the CDS inputs at 3h/6h,
  while the classic path is consistent.
- Second issue in the classic path at tstep > 1h: tp becomes mm *per timestep* but keeps
  `units: 'mm hr**-1'`, and the exporters treat it as mm/h (e.g. `topo_export.py:111`
  `/3600` → mm/s; `:202` `*24` → mm/day). So FSM/Cryogrid precipitation would be 3× too
  high at 3h. **Reported only, not changed.**
- The 1-h sample × tstep estimate of precipitation is noisy for convective showers;
  summing the hourly values over the window is more accurate.
- **Our run is hourly (1h), so neither issue affects it.**
- **Decision (user, 2026-09-30): option (a).** For 3h/6h the fetcher *samples* the hourly
  store like CDS does (no aggregation). In Phase 1/2, `topo_scale_zarr.py` gets aligned with
  `91ab64c` (radiation /3600, tp * tstep) so both downscalers agree. The units-label/exporter
  issue stays reported, unchanged.
- The run itself is hourly. Daily products are made *after* downscaling (user +
  supervisor, 2026-09-30).

## Phase 1: EDH fetcher, DONE (2026-09-30)

New module `TopoPyScale/fetch_era5_edh.py`, tests `tests/test_fetch_era5_edh.py`
(18 synthetic-store unit tests + 1 integration test against the real EDH stores; all pass).

### How it works (short version, for colleagues)
1. Open both EDH stores lazily (metadata only).
2. Cut the domain: bbox expanded outward to whole 0.25° cells; EDH 0–360° longitudes converted
   to -180…180 (boxes crossing 0° handled; boxes crossing 180° refused).
3. Split the period into **time blocks aligned to each store's own chunks** (PLEV 60 days, SURF
   45 days). Each block loads every remote chunk it touches once, with Dask threads
   (`max_concurrency`, default 8). Requests per run ≈ blocks × variables (× levels).
4. Each block → `<climate>/edh_blocks/{plev|surf}_<start>_<end>.nc`, written as `.tmp`, then
   renamed. Rerun = skip existing blocks (resume). `manifest.json` refuses to mix blocks fetched
   with other bbox/levels/timestep.
5. Blocks → final `ERA5.zarr` (schema of `topo_scale_zarr`), and/or yearly `SURF_YYYY.nc` /
   `PLEV_YYYY.nc` (CDS layout, `z` kept, root symlinks like `era5_batch_fetch.py` makes).
   `fetch_report.json` records levels used, blocks, estimated requests, timings.

### Decisions and why
- **Blocks staged as NetCDF, then assembled into Zarr** (the brief said "region writes per
  block"). Why: (1) atomic block files make resume trivially correct (a file exists = the
  block is complete); (2) the final store can use a layout that suits the downscaler (time
  chunk 8760 × all levels × whole domain) instead of the block layout, which would mean
  thousands of tiny reads per point. Assembly costs seconds (2025 ≈ 100 MB).
- **Blocks follow each store's own chunk grid** (60 d PLEV, 45 d SURF) rather than one common
  60-day grid, so no remote chunk is read twice. That's what keeps the quota predictable.
- **Level resolution** (`resolve_plevels`): all available levels inside the requested range,
  plus the nearest level beyond each end that isn't available. Your 14 CDS levels → EDH
  600/700/850/925/1000 (both ends exist, nothing added). The 9 missing levels are logged as a
  WARNING. Whether they matter is Phase 3's question.
- **Levels stored ascending** (600 … 1000): both downscalers treat `level[-1]` as the lowest level.
- **Latitude descending, longitude -180…180 ascending**, like CDS files and like `df_centroids.lon`
  (the downscalers find the nearest cell with `abs(lon - row.lon)`).
- **3h/6h = sampling** (option a), in `apply_timestep_convention()`.
- **Retries**: exponential backoff on timeouts/connection errors/408/425/429/5xx. **No retry**
  on 401/403/426 (clear message; 426 = standard key, need a classic key) or 404.
- **Auth**: `EDH_TOKEN` env var (sent as a Basic-auth header) or `~/.netrc` via aiohttp
  `trust_env`. The token is never logged or written.
- **bbox**: as in `FetchERA5.go_fetch`, it comes from the **DEM** lat/lon extent + 0.4°. For
  this DEM that's lat 40.92–42.83, lon 22.39–24.68 → grid 40.75–43.0 N, 22.25–24.75 E
  (10 × 11 cells). The brief's 22.97–24.09 / 41.43–42.32 is the mask AOI, which is smaller.
  Still one EDH spatial chunk.

### Real fetch, 2025 (hourly) → `/data/era5/edh_2025/`
See `fetch.log` and `edh_blocks/fetch_report.json` there. PLEV ≈ 50 s per 60-day block.

## Zarr downscaler (`topo_scale_zarr.py`): bugs found, fixed, parity-tested (2026-09-30)

This goes a bit beyond the brief's Phase 1–2 text, but the zarr pipeline can't run without it.
**Evidence-based fixes only. Physics now identical to the classic path**
(`tests/test_topo_scale_zarr_parity.py`: all 16 output variables within 1e-5 of
`topo_scale.downscale_climate`, 3 points incl. one below z(1000 hPa), with and without the
precip lapse rate on).

Bugs in the 2025 version, with evidence:
1. **Pressure-level humidity overwritten by surface humidity.** Pressure-level and surface
   fields were merged into one dataset, then `mu.dewT_2_q_magnus(subset_interp, var_era_surf)`
   wrote `ds['q']`, replacing the pressure-level `q` (`meteo_util.py:185`,
   `var_era_surf['spec_h'] = 'q'`). The vertical interpolation then used surface q. `w`/`vp`
   collided the same way. The classic path keeps separate datasets. **Fixed: separate datasets.**
2. **Precip lapse rate crashed**: `... / \` followed by `{} (1 - ...)` = calling a dict →
   TypeError whenever `precip_lapse_rate: True` (the project config). **Fixed: classic formula.**
3. **Accumulations**: `/ tstep` and `/ (tstep*3600)` instead of the 91ab64c convention →
   aligned (decision 0.2a). No effect at 1h.
4. float32 arithmetic and no rounding, vs the classic path's float64 and `.round(5)` →
   aligned to the classic path.
5. Errors swallowed (`raise ValueError("Error processing subset")`) → the real traceback is
   reported, with the failing point.
6. `sys` not imported (used by `sys.exit`) → fixed (raises ValueError).
7. Output store rebuilt with `mode='w'` on every run (not resumable) → existing compatible store
   reused; finished points recorded as `<store>.progress/<i>.done` and skipped on rerun.
8. Dask method printed "finished" before the work was gathered → fixed.

Also changed (same numbers, faster): the horizon angle per timestep is taken from the
`hori_azi_*` columns `compute_horizon()` already put in `df_centroids` (same nearest lookup
via `pandas.Index.get_indexer(method='nearest')`, exactly what xarray `.sel(method='nearest')`
does), instead of indexing the 7 GB `da_horizon.nc` per point. The parity test covers both paths.

Wiring bugs in `topoclass.downscale_climate` (zarr branch), fixed in Phase 2:
- passes both `file_pattern` and `store_name` → ClimateDownscaler always raised
- dask branch used a non-existent `self.config.project.dask_worker`
- then reads `down_pt_*.nc` even when the output is a Zarr store

## Daily products (`TopoPyScale/topo_daily.py`), DONE (2026-09-30)

User + supervisor want daily min/mean/max. Decisions (user): **UTC days** (as in the METER.AC
paper). Aggregation happens **after** hourly downscaling. Hourly output is kept as the master
product (models such as FSM/CryoGrid need hourly forcing).
`daily_stats(ds, utc_offset_hours=0)`: t, q → mean/min/max; ws → mean/max (mean of hourly
speeds); u, v, p, SW*, LW → mean; wd → direction of the mean (u,v) vector; tp (and
snowfall/rainfall if present) → daily sum in mm/day. Days with missing steps → NaN.
Tests `tests/test_topo_daily.py` (4, pass).

## Test status (2026-09-30)
`python -m pytest tests/` in the container: **23 passed** (fetcher 19 incl. 1 real-EDH integration,
parity 4) + 4 daily = 27. Found while running the suite: with `fork` start method, worker
processes deadlocked after zarr's I/O thread had started in the parent (the full suite hung,
the parity tests alone passed) → `forkserver`.

## Fixes found while running Phase 3 (2026-09-30, afternoon)

1. **A short fetch shrank the store.** Run A (January) called `get_era5` → fetcher → reassembled
   `ERA5.zarr` for January only, replacing the full-2025 store (the blocks were still on disk).
   Also, block files were named after the *clipped* period, so a later fetch of a different
   period created a second file for the same remote chunk (overlapping times, which
   `open_mfdataset` refuses). **Decision:** one block file = one whole remote chunk (never
   clipped to the request; only the store's still-growing last chunk is clipped). Same number of
   requests, since the remote chunk is downloaded whole anyway. Assembly now uses **all** blocks in the
   block directory, and completeness is checked for the requested period only. Duplicate
   timesteps are dropped when reading blocks, as a safety net. Regression test
   `test_short_fetch_does_not_shrink_store`. Side effect: the 2025 store also holds the
   parts of the boundary chunks outside 2025 (e.g. from 2024-12-05); the downscaler selects its
   period. The 2025 blocks were refetched with the new naming (8 min); the old copy is in
   `/data/era5/edh_2025_old_clipped_blocks/` (can be deleted).
2. **forkserver re-ran the whole pipeline.** With the start method `forkserver`, Python by
   default preloads `__main__` in the server, i.e. re-imports the calling script. My
   `run_pipeline_zarr.py` had no `if __name__ == '__main__':` guard, so the pipeline started a
   second time inside the fork server (seen in `run.log`: steps repeated after "500 points to
   downscale"). Killed. **Fixes:** (a) the script now has the guard; (b) `topo_scale_zarr` sets
   `set_forkserver_preload(['TopoPyScale.topo_scale_zarr'])`, so even unguarded user scripts
   are safe.
3. Run A had been started before fixes to `topo_scale_zarr` (Python had loaded the old module):
   killed after clustering and restarted.

Tests after these fixes: **28 passed** (`python -m pytest tests/`).

## ⚠️ Clustering mask applied to the wrong pixels (upstream bug), FIXED (2026-09-30)

**Found by:** a cluster in run A with centroid sky-view factor 0.0003 → January-mean SW ≈ 0.
No pixel inside `mask.tif` has svf < 0.05; all 335,703 such pixels are in the buffer.

**Evidence** (`/data/phase0/check_cluster_map.py`, run A before the fix):
- clustered pixels = 13,945,288 = mask pixel count, but only **10,451,891 (75 %) inside the mask**;
  3,493,397 clustered pixels were in the buffer, and as many AOI pixels were left out
- saved cluster map vs centroids: elevation correlation **-0.08** (median |diff| 374 m); after
  re-ordering the saved labels from `order='F'` to `order='C'`: correlation **1.000** (1.4 m)

**Cause:** `extract_topo_cluster_param` flattened each raster to a table in the order of its
Dataset's dims, then combined the tables **by position**. The mask read with rasterio has dims
`(x, y)` → column by column. `ds_param` **reloaded from disk** has dims in the file's order. The
Rila-Pirin terrain file (written by an older TopoPyScale) is `(y, x)` → row by row. So the mask
selected pixels at the wrong positions. The write-back of the cluster map used a fixed
`order='F'` (upstream commit 3639188, Oct 2025), which is only right for `(x, y)` tables, so the
map was also scrambled. A freshly computed `ds_param` is `(x, y)`, the same as the mask, so the bug
only appears with a cached/reloaded terrain file (the normal case when reusing `ds_param.nc`).
Hidden until today because the pandas-3 bug wrote `-9999` into every map cell.

**Fix:** one explicit order `(y, x)` for everything (`tu.ds_to_indexed_dataframe(..., dim_order)`,
mask and groups via `.transpose('y','x')`, map written back with `order='C'`).
**Regression test** `tests/test_clustering_mask.py`: clusters a tiny DEM (asymmetric mask) from
a `(y, x)` cached terrain file. Fails on the old code (centroids up to 264 m from their mapped
pixels), passes on the new code. (Note for reruns of such checks: `python -m pytest` puts the
current directory first on `sys.path`; run old-code checks from the old package's directory.)

**Impact:**
- Downscaling physics: none. But **which terrain the clusters represent was wrong**: 25 % of the
  clusters' pixels came from outside the AOI, including artifact pixels with svf ≈ 0.
- Anything mapping clusters back to pixels (cluster maps, `map_variable`, `landform.tif`, rasters
  for the thesis) was scrambled.
- **The user's earlier 500-cluster toposub trial (2023-03-15) used this code path** and was
  affected the same way. The 28-station point runs (`sampling: points`) were not.
- The cluster-search figure is **not** affected (`cluster_search.py` applies the mask by array
  position on the same grid).
- Phase 3 run A was redone with the fixed clustering (old outputs kept in
  `/data/runs/jan2025_edh/superseded_buggy_mask/`).

## Phase 3: validation EDH (5 levels) vs CDS (14 levels), January 2025, DONE (2026-09-30)

Script `rila-pirin-downscaling/bulgaria_rila_pirin/scripts/phase3_validate.py`
(`convert` / `prepare` / `report`). Outputs in `/data/runs/phase3/`: `phase3_report.json`,
`fig_phase3_t_diff_vs_elevation.png/.pdf`. Runs: A = `/data/runs/jan2025_edh` (EDH, 600/700/850/
925/1000 hPa), B = `/data/runs/jan2025_cds` (CDS, 600–1000 hPa, 14 levels). **Same 500 clusters**
(k=500, elevation weight 4, fixed mask), same solar geometry and horizon. Only the ERA5 input differs.

### Why January
Winter valley inversions are where coarse levels should hurt most (brief). 118 of 744 hours
(16 %) had T increasing with height between 1000 and 700 hPa in the AOI-mean CDS profile.

### Q1: same data? **Yes.** EDH − CDS on the 8×9 common cells, 5 common levels, 744 h:
t, t2m, d2m ≤ 0.125 K (RMSE 0.07 K), z ≤ 16 m²/s² (≈ 1.6 m), sp ≤ 36 Pa, u/v ≤ 0.02 m/s,
ssrd/strd ≤ 512 J/m² (≈ 0.14 W/m²), tp ≈ 0. These are the storage-quantisation steps of the
two archives (GRIB packing): identical ERA5.

### Q2/Q3: does dropping 9 levels change the downscaled climate? **Yes, at high elevations.**
EDH − CDS, all 500 clusters × 744 h: t bias −0.15 K, RMSE 0.64 K; LW bias −0.39 W/m², RMSE 2.9;
q RMSE 2.2e-4 kg/kg; ws RMSE 0.67 m/s; tp and SW ≈ unchanged (tp RMSE 4e-5 mm/h; SW p99 0.6 W/m²).

| elevation band | clusters | t bias | t RMSE | t RMSE, inversion hours | LW RMSE |
|---|---|---|---|---|---|
| 0–1000 m | 197 | −0.13 K | 0.65 K | 0.65 K | 2.6 W/m² |
| 1000–1500 m | 148 | +0.05 K | 0.41 K | 0.47 K | 1.8 W/m² |
| 1500–2000 m | 92 | −0.26 K | 0.66 K | (1500–3000 m: 1.29 K, bias −0.89 K) | 3.0 W/m² |
| 2000–3000 m | 63 | −0.52 K | 0.95 K | | 5.3 W/m² |

- **Pattern** (figure): per-cluster bias ≈ 0 at the heights of the EDH levels (≈ 800 m at 925 hPa,
  ≈ 1500 m at 850 hPa) and grows between them. The difference comes from linear interpolation
  across the gaps between levels. Largest gap: **850 → 700 hPa (≈ 1500 → 3080 m)**, which is exactly the
  zone of the high Rila/Pirin terrain. There EDH is **≈ 0.5 K colder on average and 1–1.7 K colder
  during inversion hours** (the CDS 800/775/750 levels resolve the warm layer aloft).
- Low valley points far from any level show ±0.5 K either way.
- One month only. A summer month (well-mixed boundary layer) probably shows smaller differences,
  but that is untested.

### Q4: clusters below z(1000 hPa)
15 clusters (151–320 m, the Struma/Mesta lowlands inside the rectangle AOI) are below the
1000 hPa geopotential during ~36 % of the hours, in **both** datasets (same 1000 hPa field).
The downscaler then uses its fallback for the **whole period** (upstream behaviour): the
nearest level above, per hour, without interpolation. With EDH the level above is 925 hPa
(≈ 800 m) instead of CDS's 975/950 hPa, so these points are **1.6 K colder on average (RMSE
2.1 K, up to 3.5 K)**. (Also: the fallback warning printed by workers was partly lost from
the log. The downscaler now returns and reports the list of fallback points.)

### Side finding: SW spikes at sunrise/sunset (existing physics, both paths)
47 of 372,000 point-hours differ by > 50 W/m² (max 774 W/m², at 15 UTC with the sun at the
horizon, on a sun-facing 25° slope). When mu0 → 0, the direct-beam scaling
`cos_illumination / mu0` and `log(SWtoa / SW_direct)` amplify tiny ssrd values (a few hundred
J/m² in the last sunlit hour), so the 0.1 W/m² archive quantisation becomes a ~700 W/m² spike.
This is a numerical sensitivity of the TopoScale SW scheme (classic and zarr), not of EDH.
**Reported only, not changed** (user rule). A mu0 threshold would be the usual remedy.

### Q5: wall times
| | EDH | CDS |
|---|---|---|
| ERA5 download | **all of 2025: 285 s, 280 requests** | **January only: 3066 s** (11 PLEV + 1 SURF requests, queue) |
| downscaling, 500 clusters × 744 h (20 cores) | 21 s | 17 s |
| clustering (k=500, 13.9 M px), one-off | 383 s | reused |
Downscaling a full year (8760 h) × 500 clusters should take ≈ 3–5 min (per-point cost ~0.5 s
per month per worker). For comparison, the classic path took 2233 s for 500 clusters × **1 day**
(August 2026 run).

### Conclusions / options for the user (decision needed)
1. EDH gives identical ERA5 and is ~130× faster to download (per month of data). Fine for everything that doesn't
   depend on the vertical profile between levels (tp, SW, and points near 925/850 hPa).
2. For the **high terrain (> 1600 m), which matters most for snow and rock glaciers**, the 5 EDH
   levels make January temperatures ≈ 0.5 K colder than with 14 levels (1–1.7 K during
   inversions). Options:
   - **(a) EDH only**: accept and document the difference (simplest).
   - **(b) Hybrid**: EDH for surface + its 5 levels, plus only **800, 775, 750 hPa** from CDS
     (fills the 850→700 gap) and optionally **975, 950 hPa** (low points). With 3–5 levels
     instead of 14, CDS requests can be ~3–4× longer, so about 1 day of queue per ~10 years.
     Run B above is exactly what (b) would give for January.
   - **(c) CDS 14 levels throughout**: slowest.
3. Which is closer to reality can only be decided against observations (e.g. Musala and the
   other METER.AC stations). Neither dataset is "truth"; 14 levels just follow ERA5's own profile
   more closely.
4. The 15 lowland clusters (< 330 m) use the fallback. If the study is about mountains, consider
   restricting the AOI mask to the mountain area. Otherwise option (b) with 975/950 hPa fixes them.

## Open option (future): mask only the mountain area (logged 2026-10-01)

**Possibility, not decided.** The current AOI mask (`/data/terrain/mask.tif`, rectangle
`rila_pirin_PERFECT_RECTANGLE_32634.geojson`, 13,945,288 px) also contains lowlands
(Struma/Mesta valleys, cluster centroids down to ~151 m).

Why it could be worth doing:
- 15 lowland clusters (151–320 m) sit below the 1000 hPa geopotential at ~36 % of hours and use
  the no-vertical-interpolation fallback for the whole period (Phase 3 Q4). With EDH's 5 levels
  they come out 1.6 K colder on average than with CDS's 14 levels (up to 3.5 K).
- The study targets the mountains (snow, rock glaciers), so lowland clusters use up part of the
  cluster budget (k) without being of interest.

How (when decided):
- Build a new mask aligned pixel-for-pixel with the DEM (same bounds/resolution, as
  `scripts/fix_mask.py` does), e.g. from a mountain outline polygon or an elevation threshold.
  The threshold must be chosen by the user and supervisor.
- Point `sampling.toposub.clustering_mask` at it, delete the run's `df_centroids.pck` (and restore
  a clean `ds_param.nc`) so clustering reruns. The terrain parameters and horizon don't change.
- **Rerun the cluster search** (`cluster_search.py`, set `MASK`) because the elevation RMSE vs k
  curve depends on the masked terrain, then re-pick k.
- The ERA5 domain needs no change (derived from the DEM extent, not the mask).

## Data folder moved (2026-10-01, user request)
`~/rila-pirin-data` → `~/nandhni/downscaling-topopyscale-project/repo-github/rila-pirin-data`
(next to the two repos, not inside either). `docker-compose.yml` now mounts `../rila-pirin-data:/data`
(relative to the project repo). Inside the container everything is still `/data/...`, so configs and
scripts are unchanged. Container recreated and checked (ERA5.zarr readable, /data writable).

## Data integrity check (2026-10-06)
User thought `rila-pirin-data` hadn't run. Checked with `/data/phase0/verify_data.py`: everything is
complete. EDH ERA5.zarr is hourly with no gaps (2024-12-05 to 2026-01-28, 0 NaN); CDS January has
14 levels and 744 h; runs A and B have 500 points × 744 h, all `.done`, 0 NaN; the cluster map
labels exactly the 13,945,288 `mask.tif` pixels. Why it looks empty from Windows: only January
2025 has been downscaled so far, the Zarr stores are directories, and `runs/*/outputs/da_horizon.nc`
are symlinks to `/app/...`, which only resolve inside the container. The container had exited
(255) and was restarted.
- **Side finding:** the `mask` variable inside `ds_param.nc` is all ones (24,363,963 px). It is
  *not* the clustering mask. Use `mask.tif` (or `cluster_labels != '-9999'`). Reported only.

## Cluster allocation by elevation, run A (k=500, w=4), 2026-10-06
`/data/phase0/cluster_allocation.py`. Clusters are allocated roughly in proportion to area, but
the elevation error *grows* with elevation, because high terrain is steeper/rougher:

| band (m) | % area | % clusters | km² per cluster | elev RMSE |
|---|---|---|---|---|
| 0-500 | 11.1 | 8.4 | 22.9 | 59 m |
| 500-1000 | 32.1 | 31.0 | 18.0 | 61 m |
| 1000-1500 | 30.4 | 29.6 | 17.9 | 65 m |
| 1500-2000 | 16.8 | 18.4 | 16.0 | 69 m |
| 2000-2500 | 8.5 | 10.4 | 14.3 | 74 m |
| 2500-3000 | 1.0 | 2.2 | 8.2 | 96 m |

So the terrain of most interest (snow, rock glaciers) is represented worst. This motivates the
user's idea of an elevation- or terrain-complexity-dependent cluster density (under discussion).

## Full 2025 run v1 (2026-10-06), DONE
**Decisions (user, 2026-10-06):** change of priorities, so run all of 2025 end to end now. k = 500
("diminishing returns" of the cluster search; with weight 1 the curve has no sharp elbow, so word
it that way). **Elevation weight 1** (authors' default in all their spatial examples; the user finds
weight 4 disproportionate). Keep the code as close as possible to the authors' framework. Products:
**daily** mean/min/max temperature as GeoTIFF. Still open: pressure levels (run uses EDH's 5),
mask (rectangle kept), SW spikes.

Config `rila-pirin-downscaling/bulgaria_rila_pirin/runs/config_2025_edh_v1.yml`, output
`/data/runs/2025_edh_v1`. As in run A, `outputs/ds_param.nc` was seeded from the pristine
`/data/terrain/ds_param_terrain_only.nc` (the authors' code loads an existing ds_param), and the horizon is
now a real file `/data/terrain/da_horizon.nc` (7 GB, byte-identical copy of the old `/app` one)
linked *relatively*, so the link also resolves on the host and in Windows.

Timings: dem_param 17 s, clustering 447 s, solar 27 s, horizon 0.3 s, ERA5 assembly 32 s,
**downscaling 500 pts × 8760 h in 45 s** (20 cores). Total 568 s.

QA (`scripts/qa_run.py`): 500/500 points done, 8760 h with no gaps, 0 NaN in all 13 variables, all
within physical bounds **except SW** (below). Period independence: a January-only run with the same
clusters (`/data/runs/2025_edh_v1_jancheck`) reproduces the January slice **bit-identically**
(max |diff| 0 for all variables).
- 10 points (lowlands) use the below-1000-hPa fallback: 048 050 079 132 164 213 275 281 343 461.
- Annual mean t by centroid band: 0-500 m 15.0 °C; 500-1000 13.5; 1000-1500 10.1; 1500-2000 7.3;
  2000-2500 4.2 °C.
- **Weight 1 consequence:** the highest centroid is at **2397 m** (centroids span 169-2397 m). Terrain at
  2400-2925 m (Musala, Vihren) is merged into lower clusters, so the peaks get the climate of a
  cluster several hundred metres lower. With weight 4 (run A) the top centroid was 2570 m. Measured
  (pixel minus its centroid's elevation; K = indicative lapse-rate equivalent at 6.5 K/km):

  | pixel band | pixels | weight 1 (v1): mean bias / RMSE | weight 4 (run A): mean bias / RMSE |
  |---|---|---|---|
  | 2000-2250 m | 643,891 | +8 m / 187 m | +5 m / 76 m |
  | 2250-2500 m | 546,241 | +128 m (≈ 0.8 K too warm) / 177 m | +7 m / 70 m |
  | 2500-2750 m | 141,612 | +302 m (≈ 2.0 K too warm) / 318 m | +61 m (≈ 0.4 K) / 88 m |
  | 2750-2950 m | 3,586 | +519 m (≈ 3.4 K too warm) / 524 m | +259 m (≈ 1.7 K) / 263 m |

  So weight 1 gives a systematic warm bias in the highest terrain, which weight 4 mostly avoids. This is the
  concrete cost of the authors' default for this domain. Relevant for snow/rock-glacier work and for
  the station validation at Musala. **For the user's decision** (and the adaptive-allocation idea).
- **SW spikes, full year** (`/data/phase0/sw_spikes_2025.py`): 14 of 4.38 M point-hours > 1361 W/m²
  (max 1976), 6 points, all with the sun 1.1-1.5° above the horizon on 26-34° slopes. Changes the daily-mean
  SW on 14 point-days by ≤ 25.6 W/m². Existing TopoSCALE behaviour. **Reported only**; decision #4
  still open. No effect on temperature.

### Daily products + GeoTIFF export
`scripts/export_geotiff.py`: `topo_daily.daily_stats` (UTC days) → `outputs/downscaled/daily.zarr`
(500 × 365). Then each pixel takes its cluster's value via the cluster map, exactly the authors'
`topo_plot.map_variable` lookup (`ds[var].sel(point_name=ds_param.point_name)`, done once as integer
positions). No lapse-rate correction. K → °C. Output: Cloud-Optimised GeoTIFF per statistic and month,
one band per day (band description = date), float32, nodata -9999 outside `mask.tif`, DEM grid and
CRS (**EPSG:32634**, no resampling). Tags record variable, statistic, day definition, k, features,
source. `--epsg 7801` (BGS2005 / CCS2005) optionally writes a bilinear reprojected copy (not run).
`outputs/geotiff/t_{mean,min,max}_daily_2025-MM.tif`, ~0.7 GB per file, ~25 GB in total.
- The old `scripts/paint_back.py` (refits its own unweighted scaler and assigns pixels to the nearest
  centroid) was a workaround for the empty cluster map. It can disagree with the real k-means labels.
  **Superseded.**
- `scripts/check_geotiff.py`: checks grid = DEM, nodata = outside mask in every band, 1000 random
  pixels = their cluster's daily value, and band 1 = the authors' `.sel` expression over the whole field.
- Export done: 36 files (t mean/min/max × 12 months), **22 GB**, about 25 min. `check_geotiff.py`:
  **all 36 pass** (grid = DEM, EPSG:32634; nodata = outside mask in every band; 1000 sampled pixels
  = their cluster's daily value to ≤ 2e-6 °C (float32); band 1 identical (0 diff) to the authors'
  `.sel(point_name=ds_param.point_name)` field). Log: `/data/runs/2025_edh_v1/check_geotiff.log`.
- Quick-look `/data/runs/2025_edh_v1/quicklook_t_mean.png` (15 Jan, 15 Jul daily mean, 100 m overview):
  Rila/Pirin coldest (15 Jan down to -8.4 °C; 15 Jul 11.5 °C), Struma/Mesta valleys warmest (+3.3 / 28.2 °C).

## Station validation 2025 + choice of the elevation weight (2026-10-06), DONE
User: "try to obtain a final good result"; validation over all of 2025. Test suite first: **30 passed**.

Runs (all full-year 2025, k=500, EDH 5 levels, rectangle mask, seed 2; only the elevation weight differs):
`2025_edh_v1` (w1), `2025_edh_w2`, `2025_edh_w4` (clusters reused from run A: identical settings), plus
`2025_edh_stations` = `sampling: points` at the 28 stations' own terrain (no clustering; 39 s). Station-to-series
mapping verified (annual mean t vs own elevation r = -0.999; with names shifted by one r = -0.05).

Tool `scripts/validate_stations.py`, outputs `/data/runs/validation_2025/`. Obs = METER.AC hourly
means, 'complete' hours only; station hours are left-labelled (10:00 = 10:00-10:55), so model hour =
mean(t at HH:00, HH+1:00). 21 stations with >= 500 complete hours (375-2525 m; Tevno ezero 2525 m only
Jul-Oct; Ortsevo and Lovna excluded, no 2025 data).

**1. Against observations (21 stations):** hourly RMSE points 4.01, w1 3.87, w2 3.93, w4 3.91 K; r ≈ 0.89.
Run-vs-run differences are not significant (all 95 % bootstrap CIs include 0). The error is dominated
by model/ERA5 limits: **daily min +4.3 K too warm, daily max -2.6 K too cold** (diurnal range far
too small; valley cold-air pools and nocturnal inversions are not resolved), bias decreasing with
elevation (points: -1.7 K/km; +2.1 K below 1000 m, -0.1 K above 1500 m). Clustered runs look slightly
"better" than points in the valleys only through compensating errors (the cluster centroid is higher
than the valley station, so the lapse rate cools it).

**2. Error added by clustering** (clustered run minus point run at the same 28 locations, full year,
no obs needed; this isolates what k/weights control):

| | mean RMSE | bias < 1000 m | 1000-1500 m | > 1500 m (n=7) | Musala 2925 m (centroid) | Tevno 2525 m |
|---|---|---|---|---|---|---|
| w1 | 0.96 K | -0.37 | +0.11 | **+1.39** | +3.47 K (2319 m) | +1.65 K |
| w2 | 0.75 K | -0.19 | -0.04 | +0.95 | +3.11 K (2388 m) | +1.20 K |
| w4 | **0.54 K** | -0.22 | -0.00 | **+0.57** | +2.31 K (2523 m) | +0.07 K |

Paired over 28 stations: w4 - w1 = **-0.43 K [95 % CI -0.63, -0.23]**, better at 23/28; w2 - w1 -0.22
[-0.42, -0.00]; w4 - w2 -0.21 [-0.38, -0.06]. All significant. Figures `fig_clustering_error.png/.pdf`
and `fig_station_validation.png/.pdf`.

**Conclusion:** observations can't separate the weights (model error ≈ 4 K dominates, quadrature), so
the deciding criterion is clustering fidelity, and there **weight 4 is significantly best**, halving
the clustering error and mostly removing the summit warm bias. The user's concern ("weight 4 is
disproportionate") holds for slope/aspect/svf (+20-35 % error there, cluster search), but for temperature
those cost less than the elevation error. Remaining: even w4 leaves Musala +2.3 K (no centroid above
~2520 m), which is the case for the user's elevation-dependent allocation idea (or a larger k).
**Weight choice is the user's decision.** w4 GeoTIFFs exported as the final candidate
(`/data/runs/2025_edh_w4/outputs/geotiff`); the w1 maps are kept.
- w4 export: 36 files, 19 GB, `check_geotiff.py` **all passed**. Quick-look
  `/data/runs/2025_edh_w4/quicklook_t_mean.png`: valleys follow the river courses (Struma, Mesta) much more
  sharply than with w1; summits colder (15 Jan min -8.9 vs -8.4 °C; 15 Jul min 10.4 vs 11.5 °C).

## Decisions 2026-10-13 (user): settings and plan for the Zarr workflow
Status check first: nothing was produced after the 2026-10-06 entries; this log was complete. The user's own
commits are now recorded in `GIT_PENDING.md` (table "Commit log").

**Clustering: k = 500, elevation weight 4, all other features 1** (user decision, after briefly considering
weight 1 earlier the same day). Evidence behind it:
- Weight: station test above. Clustering error 0.54 K (w4) vs 0.96 K (w1), paired difference -0.43 K
  [95 % CI -0.63, -0.23]; bias above 1500 m +0.57 K vs +1.39 K. Cost: slope/aspect/svf represented 20-35 %
  worse (cluster search), mainly relevant for SW on slopes.
- k: cluster search with weight 4. Elevation RMSE 65 m at k = 500 (about 0.4 K at 6.5 K/km); beyond 500 each
  extra 100 clusters removes < 2.5 m. Wording: "diminishing returns". k = 500 is final for the workflow test;
  the planned clustering-error-vs-k experiment (below) is meant to justify it in the variable that matters.
- Authors' framework: docs and spatial example configs use all weights 1; `topo_sub.scale_df` code default and
  `TopoPyScale_examples/ex1_norway_finse/config_point.yml` use elevation 4. The MATLAB port TopoLAB
  (github.com/krisaalstad/TopoLAB, `src/TopoSUB.m`) uses **no weights** (z-score of svf, northness,
  eastness, sin(slope), elevation, optional x/y; plain `kmeans`, MaxIter 1e4, k = 200 in `main.m`, no k
  selection method, no fixed seed). So weight 4 is our evidence-based choice, not a documented default;
  the thesis must say so. Side notes from TopoLAB: slope enters as sin(slope); the representative point is
  the real grid cell nearest each centroid; `D(these)` at `TopoSUB.m:25` indexes only the first column of
  the Np x Nc distance matrix (looks like a bug in the nearest-cell choice; reading only, not run).

**Monthly processing: option B** (user). One yearly output store, written month by month (time regions),
resume per month. Needs a change in `topo_scale_zarr` (today an existing store must match the run period
exactly, `setup_output_store`). To be implemented with tests showing month-by-month == whole-year.

**New workflow folder:** `rila-pirin-downscaling/zarr_workflow/` (code + configs in git; `data/` git-ignored).
**Fresh run**, nothing copied from earlier runs, outputs not mixed with `/data/runs`. Docker compose kept.
Products: hourly + daily Zarr, daily GeoTIFFs (monthly files, one band per day). MSc-Thesis update later.

**Intermediate files to Zarr:** `ds_param`, `ds_solar`, `da_horizon` are NetCDF today
(`topoclass.py:185/489/574/592`, `solar_geom.py:98`, `topo_param.py:336`). Plan: choose the format by file
suffix, test Zarr == NetCDF.

### ARCO-ERA5 (Google) inspected (metadata only, anonymous GCS), 2026-10-13
- `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`: **all 37 pressure levels** (incl.
  800/775/750, 975/950 missing from EDH), hourly, valid 1940-01-01 to 2026-06-30 (ERA5T to 2026-10-07),
  lat 90 → -90, lon 0-359.75; variables z, t, u, v, q present, `relative_humidity` not under that name; surface
  t2m, d2m, sp, ssrd, strd, tp, z_surf present. Zarr v2 layout despite the name.
- **Chunks: 1 hour x all 37 levels x whole globe.** Measured compressed size for 2025-01-15T12: temperature
  85.1 MB, 2m_temperature 2.3 MB. One year of the 5 pressure variables ≈ 8760 x 5 x ~85 MB ≈ **3.7 TB of
  download** (+ ~0.2 TB surface), vs ~0.1 GB/280 requests on EDH. The older `ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2`
  has the same chunks (checked). The `co/` stores are model-level/native-grid data (not checked further).
- Consequence: ARCO is not practical as the main source for full years from this machine. Possible uses:
  a short test period with all levels (e.g. one inversion-rich month) to quantify the 5-level error, as the
  CDS run B did. **Decision for the user.**

## Implementation 2026-10-13: Zarr caches + yearly store filled month by month
Baseline before changes: test suite **30 passed**.

### Fork changes
1. **Caches as Zarr, by file name** (`ds_param`, `ds_solar`, `da_horizon`; `xxx.zarr` in the config → Zarr
   store, anything else → NetCDF as before).
   - `topo_utils`: `is_zarr`, `dataset_exists` (a store is a directory, so the old `Path.is_file()` checks
     would have recomputed a Zarr cache on every run), `open_dataset`, `open_dataarray`.
   - `topo_export`: `to_zarr` + `save_dataset`. Zarr stores are **packed exactly like the authors'
     `to_netcdf`** (floats → int16 with the same scale_factor/add_offset), so both formats decode to the
     same numbers. Written to `<name>.part` and renamed (an interrupted write never looks complete);
     if the target exists it is loaded first (Topoclass rewrites ds_param after clustering while it is
     still open lazily). `da_horizon` stays unpacked float64, as the authors write it.
   - Wired into `topo_param.compute_dem_param` / `compute_horizon`, `solar_geom.get_solar_geom`,
     `topoclass` (`load_project`, cluster-map save, `compute_solar_geometry`, `compute_horizon`).
   - **Bug in my first version, caught by the new test:** after writing a *new* Zarr ds_param I returned
     the reopened (packed) store, while the authors' NetCDF branch returns the in-memory (unpacked) dataset.
     Clustering then ran on quantized values and 37 of 1800 test pixels got other clusters. Fixed: same
     behaviour as NetCDF (new → in-memory; existing → rewrite + reopen).
   - **Side finding (authors' behaviour, reported only):** `to_netcdf` packs with `n=10` bits
     (`compute_scaling_and_offset(..., n=10)`), i.e. 1023 levels over each variable's range: elevation
     48–2905 m → ~2.8 m steps, aspect ~0.35°. A *fresh* run clusters on unpacked values, a run from a
     *cached* ds_param on packed ones, so the two can give slightly different clusters. All earlier runs
     (from `ds_param_terrain_only.nc`) clustered on packed values; the fresh zarr_workflow run clusters on
     unpacked ones. Expect small differences between them for that reason alone.
2. **Yearly store filled month by month (option B)**, `topo_scale_zarr.ClimateDownscaler(store_start=,
   store_end=)`: the store spans the store period; a run writes only its own time slice (`t_offset`);
   progress markers per run period `<store>.progress/<YYYYMMDDHH>_<YYYYMMDDHH>/<i>.done`. Run period must
   lie inside the store period; an existing store must match the store period. Without the new arguments
   the behaviour is unchanged. Topoclass passes `outputs.file.zarr_store_start/_end` from the config.
3. Tests: `tests/test_zarr_caches.py` (Topoclass steps on a tiny DEM with `.nc` vs `.zarr` names: identical
   terrain values to 1e-9 (last-bit decoding), identical cluster map, centroids, solar, horizon; Zarr caches
   reused by a second instance; packing equal to to_netcdf; rewriting an open store) and
   `tests/test_zarr_store_period.py` (store filled day by day, in reverse order == one run, **bit-identical**;
   per-period resume; refusals). **Suite: 36 passed.**

### Project: `rila-pirin-downscaling/zarr_workflow/`
- `config/config_2025.yml`: k=500, elevation weight 4, EDH (Zarr only), all caches `.zarr`, store period 2025.
- `scripts/run_year.py`: steps 1–5 once (terrain, clusters, solar for the year, horizon, ERA5), then per
  month: downscale its slice → QA (NaN, bounds) → daily stats (`topo_daily`, UTC) into yearly `daily.zarr`
  → daily GeoTIFFs (one file per statistic and month, one band per day) → independent checks →
  `outputs/months/<YYYY-MM>.json` (status, timings, QA, checks). Done months are skipped on rerun.
- `scripts/workflow.py`: those building blocks (GeoTIFF mapping/checks from the validated 2026-10-06 scripts).
- `data/` (git-ignored): `dem/` = symlinks to the DEM, mask, AOI and station list in
  `bulgaria_rila_pirin/inputs` (mask.tif identical to /data/terrain/mask.tif, md5 checked); everything else
  is computed fresh. Logs in `data/logs/`.

### Run log 2026-10-13 (fresh 2025 run, `zarr_workflow/data/logs/run_2025_20261013_1317.log`)
- Year steps (all fresh): compute_dem_param **2757 s** (svf is the slow part, single core), clustering
  387 s, solar (year) 23 s, horizon **1379 s**, EDH fetch 211 s. Cache sizes: ds_param.zarr 155 MB,
  ds_solar.zarr 13 MB, da_horizon.zarr 5.0 GB (NetCDF was 7 GB), ERA5 96 MB.
- January: downscaling 29 s, whole month incl. daily stats, 3 GeoTIFFs and checks **147 s**. QA: 0 NaN in
  all 13 variables; t −11.3…+16.0 °C hourly; 5 SW values > 1400 W/m² (known sunrise/sunset spikes); 14
  lowland clusters use the below-1000-hPa fallback. GeoTIFF checks: 1000 sampled pixels = cluster value to
  4.8e-7 °C, band 1 **identical (0.0)** to the authors' `map_variable` expression. February ok (136 s).
- **Memory incident (my error):** I started the k analysis in parallel with the main run. The main process
  held ~20–23 GB (in-memory horizon + terrain after fresh computation) and the k analysis's 20 workers
  ~2 GB each → 61/62 GB used. Claude Code reaped the background shells (the processes inside the container
  survived); I then stopped the k analysis (kill by PID) → 40 GB free; the main run continued unharmed.
  **Fix for later runs:** `run_year.py` reopens `da_horizon` and `ds_param` lazily from Zarr after the
  year steps (the downscaler uses the `hori_azi_*` columns of df_centroids, not the horizon array).
  Lesson: do not run two 20-worker jobs at once on this 64 GB machine; the k analysis must run after the
  main run (or with fewer workers).
- **Year finished** (total 6516 s ≈ 1 h 49 min, of which svf 46 min and horizon 23 min; each month 134–178 s).
  June was first marked "failed checks": one wind direction = 6.28319 rad (due north, 2π rounded *up* by
  the authors' 5-decimal rounding) exceeded my QA bound 2π+1e-6. **QA bound bug, not a data problem**; bound
  now 2π+1e-5; `scripts/recheck_qa.py` re-evaluated all months from the stored values (nothing recomputed):
  **12/12 ok** (June record keeps the previous status under `previous`).
- Year checks (`data/logs/year_check_*.log`): hourly store 500 × 8760, every hour of 2025, **0 NaN** in all 13
  variables, 12 × 500 progress markers; daily store 500 × 365, 18 variables, 0 NaN in t; GeoTIFFs 3 × 12 files,
  365 bands per statistic with all dates in order, 20.2 GB. Centroids 143–2554 m. Annual mean t by centroid
  band: <500 m 15.7, 500–1000 13.5, 1000–1500 10.2, 1500–2000 7.3, 2000–2500 4.1, >2500 m 2.4 °C.
- **Finding: clustering realization uncertainty.** Fresh run vs the 2026-10-06 w4 run (same k, weights,
  seed, mask, ERA5; stored terrain bit-identical): the only difference is that k-means saw unpacked values
  here and 10-bit-packed values there (elevation ≤ ±1.4 m). That sent k-means to a different, equally good
  partition: only 181/500 centroids coincide, top centroid 2554 vs 2570 m. Per pixel: annual mean t
  new−old **bias +0.06 K, RMSE 0.65 K**, p99 2.8 K, max 4.3 K (15 Jan RMSE 0.63 K, 15 Jul 0.90 K).
  So the domain-level climate is stable but **an individual pixel's value depends on the k-means
  realization by ~0.6–0.9 K RMSE**, the same size as the clustering error at the stations (0.54 K). For
  the thesis: report it; the k analysis should add a seed ensemble at the chosen k to quantify it.
- Viewer: `notebooks/view_results.ipynb` runs on the host in a separate venv `../venv-viewer` (jupyterlab,
  ipywidgets, xarray/zarr/pandas pinned to the container versions, pyarrow 25.0.0 for the pandas-3 pickle);
  all cells run without error (≤ 11 s each); GeoTIFF band vs Zarr map difference 0.0.
- k analysis: stopped after the memory incident (point run had started; resumable). To be restarted
  **alone, after the user's OK**, with 10 workers.
