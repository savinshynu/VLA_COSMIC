import numpy as np

def eq2hrz(ra, dec, MJD_time, lat, lon):
    """
    ra,dec is the right ascension and declination in degrees, then comes time in MJD day.
    lat and lon are the station/telscope coordinates. If passing an array of ra,dec also pass an array of MJD time
    returns altitude and azimuth
    """
    RA = ra
    LAT = lat*(np.pi/180.0)
    UT = (MJD_time - int(MJD_time))*24.0
    d = MJD_time - 40000 - 11544
    DEC = dec*(np.pi/180)
    LST = np.mod(100.46 + 0.985647*d + lon + 15.0*UT,360) # calculation of LST and hour angle
    HA = np.mod(LST-RA,360)*(np.pi/180.0)

    ALT = np.arcsin(np.sin(DEC)*np.sin(LAT) + np.cos(DEC)*np.cos(LAT)*np.cos(HA))   #calculation of topocentric coordinates
    az = np.arccos((np.sin(DEC) - np.sin(ALT)*np.sin(LAT))/(np.cos(ALT)*np.cos(LAT)))
        
    try:
       comp1 = np.where((np.sin(HA) >= 0.0))[0]
       comp2 = np.where((np.sin(HA) < 0.0))[0]
       AZ = HA*0.0
       AZ[comp1] = 2*np.pi - az[comp1]
       AZ[comp2] = az[comp2]

    except IndexError:
        if np.sin(HA) >= 0.0:
           AZ = 2*np.pi-az
        elif np.sin(HA) < 0.0:
           AZ=az

    ALT *= 180.0/np.pi
    AZ *= 180.0/np.pi 
    
    return (AZ,ALT)

