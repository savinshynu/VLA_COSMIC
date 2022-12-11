"""
Author: Paul Demorest
"""
import numpy as np
from astropy import coordinates, time
import astropy.units as u

def vla_uvw(mjd, direction, antpos):
    """
    Calculates and returns uvw in meters for a given time and pointing direction.
    direction is (ra,dec) as tuple in radians.
    antpos is Nant-by-3 array of antenna locations

    returns uvw (m) as Nant-by-3 array, relative to array center
    """

    phase_center = coordinates.SkyCoord(*direction, unit='rad', frame='icrs')

    antpos = np.array(antpos)
    antpos = coordinates.EarthLocation(x=antpos[:,0], y=antpos[:,1], z=antpos[:,2], unit='m')

    datetime = time.Time(mjd,format='mjd')
    #print(datetime)

    # VLA array center location
    tel = coordinates.EarthLocation(x=-1601185.4, y=-5041977.5, z=3554875.9, unit='m')

    tel_p, tel_v = tel.get_gcrs_posvel(datetime)
    antpos_gcrs = coordinates.GCRS(antpos.get_gcrs_posvel(datetime)[0],
                                   obstime = datetime, obsgeoloc = tel_p,
                                   obsgeovel = tel_v)
    tel_gcrs = coordinates.GCRS(tel_p,
                                   obstime = datetime, obsgeoloc = tel_p,
                                   obsgeovel = tel_v)

    uvw_frame = phase_center.transform_to(antpos_gcrs).skyoffset_frame()
    antpos_uvw = antpos_gcrs.transform_to(uvw_frame).cartesian
    tel_uvw = tel_gcrs.transform_to(uvw_frame).cartesian

    # Calculate difference from array center
    bl = antpos_uvw - tel_uvw
    nant = len(antpos_uvw)
    uvw = np.empty((nant,3))
    for iant in range(nant):
        uvw[iant,0] = bl[iant].y.value
        uvw[iant,1] = bl[iant].z.value
        uvw[iant,2] = bl[iant].x.value

    return uvw
