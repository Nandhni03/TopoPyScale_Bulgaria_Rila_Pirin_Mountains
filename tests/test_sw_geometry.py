"""
Angle conventions of the shortwave chain: DEM aspect, solar azimuth, illumination and horizon must share one reference
(Dozier/IPW: 0 = south, positive towards east). Synthetic 30 deg slopes at 45.5 N on 21 December.

    python -m pytest tests/test_sw_geometry.py -v
"""
import numpy as np
import pandas as pd
import pytest
from topocalc import gradient, horizon

from TopoPyScale import solar_geom as sg

LAT, LON, SLOPE = 45.5, 24.0, np.radians(30)
DX = 30.0


def north_up_dem(facing, n=21):
    """DEM whose rows go north -> south (y decreasing, as a GeoTIFF), descending towards `facing`."""
    x = np.arange(n) * DX
    y = 1000.0 - np.arange(n) * DX
    X, Y = np.meshgrid(x, y)
    t = np.tan(SLOPE)
    return {'N': (Y - Y.min()) * -t, 'S': (Y - Y.min()) * t, 'E': -X * t, 'W': X * t}[facing] + 1000.0


def dem_aspect(facing):
    """Aspect [rad] the way topo_param.compute_dem_param computes it (dy from the y coordinate: negative)."""
    _, a = gradient.gradient_d8(north_up_dem(facing), DX, -DX)
    return np.deg2rad(a[10, 10])


@pytest.fixture(scope='module')
def solar(tmp_path_factory):
    d = tmp_path_factory.mktemp('solar')
    df = pd.DataFrame({'point_name': ['0'], 'latitude': [LAT], 'longitude': [LON], 'x': [LON], 'y': [LAT],
                       'elevation': [1500.0]})
    ds = sg.get_solar_geom(df, '2024-12-21', '2024-12-21', '1h', '4326', 1, 'ds_solar.nc', d)
    return ds.isel(point_name=0)


def illum(solar, facing):
    return sg.cos_illumination(solar.mu0.values, solar.zenith.values, solar.azimuth.values, SLOPE, dem_aspect(facing))


def test_dem_aspect_convention():
    # 0 = south-facing, 90 = east-facing, 180 = north-facing, 270 = west-facing (degrees, Dozier/IPW reference)
    got = {f: np.rad2deg(dem_aspect(f)) % 360 for f in 'NESW'}
    assert got == pytest.approx({'S': 0.0, 'E': 90.0, 'N': 180.0, 'W': 270.0}, abs=1e-6)
    assert {f: sg.aspect_to_bearing_deg(dem_aspect(f)) for f in 'NESW'} == \
        pytest.approx({'N': 0.0, 'E': 90.0, 'S': 180.0, 'W': 270.0}, abs=1e-6)


def test_solar_azimuth_convention(solar):
    # sun due south at transit (azimuth ~0), east in the morning (> 0), west in the afternoon (< 0)
    noon = int(np.argmax(solar.elevation.values))
    az = np.rad2deg(solar.azimuth.values)
    assert abs(az[noon]) < 15
    assert az[noon - 2] > 0 > az[noon + 2]


def test_south_gets_more_direct_than_north(solar):
    noon = int(np.argmax(solar.elevation.values))
    s, n = illum(solar, 'S'), illum(solar, 'N')
    # noon sun ~21 deg high: a 30 deg north slope is in self-shadow, a south slope faces the sun
    assert np.rad2deg(solar.elevation.values[noon]) < 30
    assert n[noon] == 0.0
    # sun in the slope's fall line: incidence elevation = solar elevation + slope
    assert s[noon] == pytest.approx(np.sin(solar.elevation.values[noon] + SLOPE), abs=0.02)
    assert s.sum() > 5 * max(n.sum(), 1e-9)


def test_east_west_symmetric(solar):
    # Hourly stamps are not symmetric around transit; compare E and W on a fine grid of exact times instead.
    import pvlib
    t = pd.date_range('2024-12-21 06:00', '2024-12-21 15:00', freq='1min', tz='UTC')
    sp = pvlib.solarposition.get_solarposition(t, LAT, LON)
    zen, az = np.radians(sp.zenith.values), np.radians(180 - sp.azimuth.values)   # as get_solar_geom stores it
    mu0 = np.cos(zen) * (np.cos(zen) > 0)
    e = sg.cos_illumination(mu0, zen, az, SLOPE, dem_aspect('E'))
    w = sg.cos_illumination(mu0, zen, az, SLOPE, dem_aspect('W'))
    i0 = int(np.argmax(sp.elevation.values))
    k = np.arange(1, min(i0, len(t) - 1 - i0))
    assert np.allclose(e[i0 - k], w[i0 + k], atol=0.01)     # E in the morning == W in the afternoon
    assert (e[i0 - k] >= w[i0 - k] - 1e-12).all()            # E slope gets the morning sun
    win = np.r_[i0 - k, i0, i0 + k]                          # window centred on transit
    assert e[win].sum() == pytest.approx(w[win].sum(), rel=0.01)


def test_horizon_azimuth_convention():
    # A ridge south of the point must raise the horizon at azimuth 0 (south), the solar-azimuth reference
    n = 101
    z = np.full((n, n), 1000.0)
    z[70:75, :] = 1500.0                                     # rows increase southwards (north-up DEM)
    deg = {az: np.degrees(np.pi / 2 - np.arccos(horizon.horizon(az, z, DX)[50, 50])) for az in (0, 90, 180, -90)}
    assert deg[0] > 30 and deg[180] == pytest.approx(0) and deg[90] == pytest.approx(0)


def test_circular_mean_aspect_wraps_at_south():
    from TopoPyScale import topo_sub as ts
    # cluster 0: pixels 10 deg either side of SOUTH (aspect 0 = 2*pi); cluster 1: either side of north (pi)
    a = np.radians([10, 350, 170, 190])
    df = pd.DataFrame({'cluster_labels': [0, 0, 1, 1], 'aspect': a, 'aspect_cos': np.cos(a), 'aspect_sin': np.sin(a)})
    assert np.degrees(df.groupby('cluster_labels').aspect.mean()[0]) == pytest.approx(180)   # the old, wrong mean
    c = ts.circular_mean_aspect(df)
    assert np.degrees(c.aspect[0]) % 360 == pytest.approx(0, abs=1e-6) or np.degrees(c.aspect[0]) == pytest.approx(360)
    assert np.degrees(c.aspect[1]) == pytest.approx(180)
    assert c.aspect_R.values == pytest.approx(np.cos(np.radians(10)))
