"""
Written by Savin Shynu Varghese
A script to calculate the geometric delays using a referene antenna and subtract that from the 
total delays. This is to understand the non-geometric contribution of the delays itself.

Basically extract paramters from metadata and calculate the delays
"""
import sys
import json
import numpy as np
from radec2azalt import eq2hrz
from compute_uvw import vla_uvw 


metafile = sys.argv[1]

fh = open(metafile)
data =  json.load(fh) 

mjd_time_now = data['META']['tnow']
mjd_time_start = data['META']['tstart']
source = data['META']['src']
ra_deg = data['META']['ra_deg']
dec_deg = data['META']['dec_deg']
ants = data['META_arrayConfiguration']['cosmic-gpu-0.1']['ANTNMS00']
ants = ants.split(',')
nants = data['META_arrayConfiguration']['cosmic-gpu-0.1']['NANTS']

print(f"Observing {source}: ra, dec : {ra_deg, dec_deg}")
print(f"The available antennas are {ants}")
ref_ant = input("Enter the reference antenna number: ")

ref_X = data['META_antennaProperties'][ref_ant]['X']
ref_Y = data['META_antennaProperties'][ref_ant]['Y']
ref_Z = data['META_antennaProperties'][ref_ant]['Z']

print(f"Ref ant positions (X,Y,Z) : {ref_X, ref_Y, ref_Z}")

lat_vla, lon_vla = [34.078749, -107.617728]
lat_vla_rad, lon_vla_rad = [lat_vla*(np.pi/180.0), lon_vla*(np.pi/180.0)]
VLA_X = -1601185.4
VLA_Y = -5041977.5
VLA_Z = 3554875.9





az, alt = eq2hrz(ra_deg, dec_deg, mjd_time_start, lat_vla, lon_vla)
print(f"Az, Alt: {az, alt}")
az_rad, alt_rad = [az*(np.pi/180.0), alt*(np.pi/180.0)]

#Converting Azimuth and elevation to ENU system

Se = np.cos(alt_rad)*np.sin(az_rad)
Sn = np.cos(alt_rad)*np.cos(az_rad)
Su = np.sin(alt_rad)
S_enu = np.array([Se, Sn, Su])


c = 3e+8 #speed of light

#An array to store the baseline difference wrt to ref antenna
delay = np.zeros(nants)
XYZ = np.zeros((nants,3))

for i,ant in enumerate(ants):
    X = data['META_antennaProperties'][ant]['X']
    Y = data['META_antennaProperties'][ant]['Y']
    Z = data['META_antennaProperties'][ant]['Z']
    
    XYZ[i,:] = [X + VLA_X, Y + VLA_Y, Z + VLA_Z]
    #Getting the baseline vector
    V_ref_ant = np.array([ref_X-X, ref_Y-Y, ref_Z-Z])
    #d = np.sqrt((X-ref_X)**2+(Y-ref_Y)**2+(Z-ref_Z)**2)
    
    #Converting that to ENU
    #Conversion matrix
    T_mat = np.array([[-np.sin(lon_vla_rad), np.cos(lon_vla_rad), 0], 
                      [-np.cos(lon_vla_rad)*np.sin(lat_vla_rad), -np.sin(lon_vla_rad)*np.sin(lat_vla_rad), np.cos(lat_vla_rad)],
                     [np.cos(lon_vla_rad)*np.cos(lat_vla_rad), np.sin(lon_vla_rad)*np.cos(lat_vla_rad), np.sin(lat_vla_rad)]
                    ])
    
    enu_ant_ref = np.matmul(T_mat, V_ref_ant.T)

    #delay[i] = (d*np.cos(alt*(np.pi/180.0)))/c
    delay = np.dot(enu_ant_ref, S_enu) #/c
    #print(f"{ant}: X,Y,Z = {X, Y, Z}")
    #print(f"{ref_ant}-{ant}, delay = {delay} ns")
    print(f"{ref_ant}-{ant}, delay_distance = {delay} m")
print("Getting UVW terms(in meters) from Paul's scripts")

uvw = vla_uvw(mjd_time_start, (ra_deg*(np.pi/180.0), dec_deg*(np.pi/180.0)), XYZ) 

#for i in range(uvw.shape[0]):
#    print(f"{ants[i]}: X,Y,Z = {XYZ[i,0], XYZ[i,1], XYZ[i,2]}")
#     print(f"{ants[i]}: U,V,W = {uvw[i,0], uvw[i,1], uvw[i,2]}")
#print(delay*1e+9)

for i, ant1 in enumerate(ants):
    for j, ant2 in enumerate(ants):
        if j > i:
           print(f"{ants[i]}-{ants[j]}: U,V,W = {uvw[i,0]-uvw[j,0], uvw[i,1]-uvw[j,1] , uvw[i,2]-uvw[j,2]}") 

           




