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
