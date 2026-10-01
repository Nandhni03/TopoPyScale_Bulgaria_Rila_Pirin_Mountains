"""
Daily statistics from hourly TopoPyScale output.

Aggregate AFTER downscaling hourly: the downscaling steps are non-linear (vertical interpolation,
terrain shading, lapse rates), so downscaling daily means would give different, wrong results.

Days are UTC calendar days (00:00-23:59 UTC) by default. Each variable is aggregated according
to what it physically is:

    t (air temperature)             mean, min, max            K
    q (specific humidity)           mean, min, max            kg/kg
    p (pressure)                    mean                      (as input)
    ws (wind speed)                 mean, max                 m/s   (mean of hourly speeds)
    u, v (wind components)          mean                      m/s
    wd (wind direction)             direction of the mean (u, v) vector    rad
    SW, SW_direct, SW_diffuse, LW   mean flux                 W/m2  (night-time zeros included)
    tp (precipitation)              sum                       mm/day (input mm/h at hourly steps)
    snowfall/rainfall (if present)  sum                       mm/day

Min/max of hourly values slightly underestimate thermometer extremes, which are continuous.
A day is only reported if all its hourly steps are present (otherwise NaN), so partial days at
the ends of a run never look like real daily values.
"""
import numpy as np
import pandas as pd
import xarray as xr

MEAN_MIN_MAX = ['t', 'q']
MEAN_ONLY = ['p', 'u', 'v', 'SW', 'SW_direct', 'SW_diffuse', 'LW', 'w', 'vp']
SUMS = ['tp', 'snowfall', 'rainfall']


def daily_stats(ds, utc_offset_hours=0, steps_per_day=None):
    """
    Args:
        ds (Dataset): hourly (or 3h/6h) downscaled output with a 'time' dimension
        utc_offset_hours (int): 0 for UTC days (default). E.g. 2 for days in UTC+2.
        steps_per_day (int): expected steps per day (default: inferred from the time step)

    Returns:
        Dataset with 'time' = day, variables named <var>_mean, <var>_min, <var>_max, <var>_sum
    """
    step = pd.Timedelta(ds.time.values[1] - ds.time.values[0])
    n_expected = steps_per_day or int(pd.Timedelta('1D') / step)
    hours = step / pd.Timedelta('1h')
    shifted = ds.assign_coords(time=ds.time + np.timedelta64(int(utc_offset_hours), 'h'))
    day = shifted.time.dt.floor('D')
    grp = shifted.groupby(day.rename('day'))
    count = grp.count()
    out = {}

    def full(da, name):
        n = count[name]
        return da.where(n == n_expected)

    for v in MEAN_MIN_MAX:
        if v in ds:
            out[f'{v}_mean'] = full(grp.mean()[v], v)
            out[f'{v}_min'] = full(grp.min()[v], v)
            out[f'{v}_max'] = full(grp.max()[v], v)
    for v in MEAN_ONLY:
        if v in ds:
            out[f'{v}_mean'] = full(grp.mean()[v], v)
    if 'ws' in ds:
        out['ws_mean'] = full(grp.mean()['ws'], 'ws')
        out['ws_max'] = full(grp.max()['ws'], 'ws')
    if 'u' in ds and 'v' in ds:
        um, vm = grp.mean()['u'], grp.mean()['v']
        wd = np.arctan2(-um, -vm)          # same convention as the downscalers (direction wind comes from)
        out['wd_vector_mean'] = full(xr.where(wd < 0, wd + 2 * np.pi, wd), 'u')
    for v in SUMS:
        if v in ds:
            # rate in mm/h at each step -> amount per step = rate * hours per step
            out[f'{v}_sum'] = full(grp.sum()[v] * hours, v)

    res = xr.Dataset(out).rename({'day': 'time'})
    for k in res.data_vars:
        base = k.rsplit('_', 1)[0] if not k.startswith('wd_') else 'wd'
        attrs = dict(ds[base].attrs) if base in ds else {}
        stat = k.rsplit('_', 1)[1] if not k.startswith('wd_') else 'vector_mean'
        if stat == 'sum':
            attrs['units'] = 'mm day**-1'
        attrs['cell_methods'] = f'time: {stat} (daily, {n_expected} steps, UTC{utc_offset_hours:+d})'
        res[k].attrs = attrs
    res.attrs = {**ds.attrs, 'title': 'Daily statistics of TopoPyScale downscaled output',
                 'day_definition': f'calendar day in UTC{utc_offset_hours:+d}; incomplete days are NaN'}
    return res
