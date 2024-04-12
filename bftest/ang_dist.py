import sys
import numpy as np

ra1, dec1, ra2, dec2 = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
ra1 *= 15.0
ra2 *= 15.0

def ang_dist(ra1,dec1,ra2,dec2):
    x = np.sin(dec1*(np.pi/180.0))*np.sin(dec2*(np.pi/180.0))
    y = np.cos(dec1*(np.pi/180.0))*np.cos(dec2*(np.pi/180.0))*np.cos((ra1-ra2)*(np.pi/180.0))
    z = x + y
    ang_dist = np.arccos(z)
    #return ang_dist*(180.0/np.pi)
    print(f"Ang dist in arcmin: {ang_dist*(180.0/np.pi)*60.0}")

ang_dist(ra1,dec1,ra2,dec2)
