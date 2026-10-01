# EDH + Zarr pipeline guide

This page explains how to run TopoPyScale with ERA5 read from the **Earth Data Hub (EDH)** Zarr
mirror and downscaled with the **Zarr downscaler** (`topo_scale_zarr`). It was written for a
project in the Rila and Pirin mountains (Bulgaria) and is meant for anyone setting up the same
chain for another study area.

## Why this path

| | CDS path (classic) | EDH + Zarr path |
|---|---|---|
| Download | CDS queue, one request per day (or per 3 days for pressure levels, because of the CDS cost limit) | Direct reads of only the chunks you need, 8 in parallel |
| One year, ~100 km domain | hours to days, depending on the queue | ~8 minutes, ~300 requests |
| Pressure levels | all 37 ERA5 levels | 1000, 925, 850, 700, 600, 500 … 1 hPa (no 975/950/900/875/825/800/775/750/650) |
| Downscaler input | yearly `PLEV_*.nc` / `SURF_*.nc` | one `ERA5.zarr` store |
| Downscaler | `topo_scale.downscale_climate` (writes temporary files per point) | `topo_scale_zarr.ClimateDownscaler` (in memory per point, parallel, **resumable**) |
| Physics | reference | **identical** (checked by `tests/test_topo_scale_zarr_parity.py`) |

Whether the missing pressure levels matter for your area is an empirical question. For
Rila–Pirin it was tested for January 2025 (see the Phase 3 report). Repeat that test if your
domain has strong low-level inversions or points far below 1000 hPa.

## The chain, step by step

```
 DEM (GeoTIFF, metric CRS)                           Earth Data Hub (Zarr v3, HTTPS)
        |                                                        |
 1 compute_dem_param   -> outputs/ds_param.nc                    |
 2 extract_topo_param  -> outputs/df_centroids.pck  (TopoSUB clusters or your points)
 3 compute_solar_geometry -> outputs/ds_solar.nc                 |
 4 compute_horizon     -> outputs/da_horizon.nc  (+ hori_azi_* columns in df_centroids)
        |                                                        |
        |                        5 get_era5 (data_repository: edh)
        |                            edh_blocks/*.nc  (one file per time block; resumable)
        |                            -> ERA5.zarr     (and optionally yearly SURF/PLEV .nc)
        |                                                        |
        +------------------ 6 downscale_climate -----------------+
                               topo_scale_zarr, one worker per core
                               -> outputs/downscaled/down.zarr  (point_ind x time)
                                  down.zarr.progress/<i>.done   (resume markers)
```

Every step writes its result to a file and **skips itself when that file already exists**. To
redo a step, delete its output (e.g. delete `df_centroids.pck` after changing `n_clusters`).

### Step 5: what the EDH fetcher does

1. Opens the two EDH stores lazily (only metadata):
   `era5-pressure-levels-v0.zarr` and `era5-single-levels-atmosphere-v0.zarr`.
2. Takes the DEM's lat/lon extent + 0.4° (same as the CDS path), widened to whole 0.25° cells.
3. Chooses pressure levels: every EDH level inside your `plevels` range; if an end of the range
   isn't available, the next level beyond it is added so the column stays bracketed. Missing
   levels are logged.
4. Finds the **remote time chunks** covering the period (60 days for pressure levels, 45 days
   for single levels). Each chunk becomes one block, always the whole chunk, so every remote chunk
   is read once and fetching another period later never overlaps.
5. Loads each block with 8 parallel downloads, cuts it to the domain, renames to TopoPyScale
   conventions (`valid_time`→`time`, `isobaricInhPa`→`level` sorted ascending, surface
   `z`→`z_surf`, longitudes −180…180), and writes `edh_blocks/<kind>_<start>_<end>.nc`
   (temporary name, then rename). **Interrupt it any time; a rerun resumes.**
6. Assembles **all blocks in the block directory** into `ERA5.zarr` (time chunks of 8760 h, whole
   domain per chunk) and, if asked, yearly NetCDF files for the classic downscaler. Fetching a
   short period never shrinks a store that already covers more.
7. Writes `edh_blocks/fetch_report.json`: levels used, blocks, estimated requests, timings.

### Step 6: what the Zarr downscaler does

For each point (cluster centroid), in parallel on `CPU_cores` workers:

1. read the 3×3 ERA5 cells around the point for the whole period from `ERA5.zarr`
2. inverse-distance-weighted horizontal interpolation
3. vertical interpolation between the two pressure levels bracketing the point's elevation
   (or the lowest level if the point is below the 1000 hPa surface)
4. precipitation lapse rate, wind speed/direction
5. longwave (clear-sky + cloud emissivity, terrain contribution), shortwave (diffuse/direct split,
   illumination angle, horizon shading)
6. write the point's time series into its own slice of `down.zarr`, then mark it done

Points are independent, so no locking is needed. If a run stops, rerun the same command: only
points without a `.done` marker are computed.

## Setting it up for a new study area

1. **Credentials**: an Earth Data Hub **classic** API key in `~/.netrc` (see
   [Datasets](04_datasetSources.md#from-earth-data-hub-destine)). A standard key gives HTTP 426.
2. **Environment**: Python ≥ 3.13, `zarr>=3`, and the TopoPyScale fork with these changes.
   The Rila–Pirin project uses a Docker image that mounts the fork (see the project repo's
   `docker/`).
3. **Paths**: keep `climate.era5.path` and `project.directory` on a Linux filesystem. On WSL,
   **not** under `/mnt/c/…`: Zarr stores are many small files, and the Windows mount is very slow.
4. **Config** (the parts that differ from a CDS config):

```yaml
climate:
  precip_lapse_rate: True
  era5:
    path: /data/era5/my_area          # absolute, Linux filesystem
    timestep: 1h
    plevels: [600, 700, 850, 925, 1000]   # must bracket your highest and lowest terrain
    data_repository: edh
    edh_output_format: zarr           # 'both' also writes yearly SURF/PLEV .nc
    edh_max_concurrency: 8
    zarr_store: ERA5.zarr
project:
  parallelization:
    downscaling_method: multicore
    setting:
      multicore:
        CPU_cores: 20
outputs:
  file:
    zarr_store: down.zarr             # downscaled result as one Zarr store
```

5. **Run** (your own scripts must use an `if __name__ == '__main__':` guard, as for any
   Python multiprocessing code):

```bash
python bulgaria_rila_pirin/scripts/run_pipeline_zarr.py my_config.yml
```

6. **Read the result**:

```python
from TopoPyScale import topo_scale_zarr as tz
ds = tz.open_downscaled_store('outputs/downscaled/down.zarr')   # dims: point_name, time
ds.t.sel(point_name='042').plot()
```

## Choosing pressure levels

The downscaler interpolates between the two levels bracketing each point. The top level's
geopotential must stay above your highest cluster (the code stops with an error otherwise),
and points below the 1000 hPa geopotential use the surface/lowest-level fallback. Check the
typical heights in your ERA5 store: `(ERA5.z / 9.81).mean(['time', 'latitude', 'longitude'])`.
For Rila–Pirin (48–2905 m): 1000 hPa ≈ 150 m, 925 ≈ 800 m, 850 ≈ 1500 m, 700 ≈ 3080 m,
600 ≈ 4300 m.

## Timestep and accumulations

Hourly (`1h`) is recommended. With `3h`/`6h` every variable is **sampled** every 3 or 6 hours,
exactly like a CDS download. The accumulated fields (tp, ssrd, strd) then still hold the
accumulation of the last hour only, and the downscalers convert them with that assumption
(radiation / 3600 s, precipitation × timestep). Aggregate to daily values *after* downscaling
hourly.

## Known limitations

- bboxes crossing the 180° meridian are refused
- the fetcher needs the domain inside the EDH coverage dates (1940 → about 2 weeks before today)
- `split.IO` (time splitting in `Topoclass`) is not compatible with the Zarr path
