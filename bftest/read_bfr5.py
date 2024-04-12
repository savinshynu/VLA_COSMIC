"""
Reads a .bfr5 file containing all the informations requried for
beamforming and print out the details

"""

import sys
import h5py
import numpy as np

filename= sys.argv[1]

f = h5py.File(filename, 'r')

print("Groups")
print(f.keys())

print("diminfo:")
nants = f['diminfo/nants'][()]
npol = f['diminfo']["npol"][()]
nchan = f['diminfo']['nchan'][()]
nbeams = f['diminfo']['nbeams'][()]
ntimes = f['diminfo']['ntimes'][()]
print(f"nants: {nants.shape}, npol :{npol}, nchan:{nchan}, nbeams :{nbeams}, ntimes : {ntimes} ")


print("Telinfo")
ant_pos =  f['telinfo']['antenna_positions'][()]
ant_frame = f['telinfo']['antenna_position_frame'][()]
ant_names = f['telinfo']['antenna_names'][()]
lat,lon,alt = f['telinfo']['latitude'][()], f['telinfo']['longitude'][()], f['telinfo']['altitude'][()]
tel = f['telinfo']['telescope_name'][()]
print(f"ant_names :{ant_names}")
print("Antenna positions \n")
print(ant_pos)
print(f"antenna frame: {ant_frame}")
print(f"lat,lon,alt : {lat,lon,alt}")

print("obsinfo")
freq = f['obsinfo']['freq_array'][()]
phase_ra = f['obsinfo']['phase_center_ra'][()]
phase_dec = f['obsinfo']['phase_center_dec'][()]
print(f"Freq shape: {freq.shape} \n\
        freq: {freq} ")
print(f"phase center ra, dec : {phase_ra, phase_dec}")

beam =  f['beaminfo']
ras = beam['ras'][()]
decs = beam['decs'][()]
for i in range(nbeams):
    print(f"beam {i},  ra, dec : {ras[i],decs[i]}")

#print(np.unique(ras)*(12.0/np.pi))
#print(np.unique(decs)*(180.0/np.pi))

print("Reading cal data")
cal_all = f["calinfo/cal_all"][()]
print(cal_all.shape)
print(cal_all)

delay = f['delayinfo']['delays']
time_array = f["/delayinfo/time_array"][()]
#time_array = f["/delayinfo/jds"][()]
print(time_array[:10]-time_array[0])
print(f"Delays array shape: {delay.shape}")
for i in range(nbeams):
    print("Delay vallues from all antennas at start time")
    print(f"beam{i} at block 0: {delay[0,i,:]}")
    print(f"beam{i} at block 10: {delay[10,i,:]}")
