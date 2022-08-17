from pyuvdata import UVData
import sys
import numpy as np
from matplotlib import pyplot as plt

#The uvh5 file to load to load
filename = sys.argv[1]

uvd = UVData()
uvd.read(filename, fix_old_proj=False)

dat = uvd.data_array
nant_data = uvd.Nants_data
nant_array = uvd.Nants_telescope
ant_names = uvd.antenna_names
nfreqs = uvd.Nfreqs
ntimes = uvd.Ntimes
npols = uvd.Npols
bls = uvd.Nbls
nspws = uvd.Nspws
chan_width = uvd.channel_width
intg_time = uvd.integration_time
source = uvd.object_name
telescope = uvd.telescope_name
pol_array = uvd.polarization_array
freq_array = uvd.freq_array[0,:]
lobs = ntimes*intg_time
#Print out the observation details

print(f" Observations from {telescope}: \n\
         Source observed: {source} \n\
         No. of time integrations: {ntimes} \n\
         Length of time integration: {intg_time[0]} s \n\
         Length of observations: {lobs[0]} s \n\
         No. of frequency channels: {nfreqs} \n\
         Width of frequency channel: {chan_width/1e+3} kHz\n\
         Observation bandwidth: {(freq_array[-1] - freq_array[0])/1e+6} MHz \n\
         No. of spectral windows: {nspws}  \n\
         Polarization array: {pol_array} -5:-8 for (XX, YY, XY and YX)\n\
         No. of polarizations: {npols}   \n\
         Data array shape: {dat.shape} \n\
         No. of baselines: {bls}  \n\
         No. of antennas present in data: {nant_data} \n\
         No. of antennas in the array: {nant_array} \n\
         Antenna name: {ant_names}")

print (np.unique(uvd.ant_1_array))

uvd.write_ms("test_vla.ms", force_phase = True)

#print(uvd.time_array)

# Collect the data for a pair of antennas at a polarization

ant_dat = uvd.get_data(10,10)

print(ant_dat.shape)

resp_rr = np.abs(ant_dat[100,:,0])
resp_ll = np.abs(ant_dat[100,:,1])
plt.plot(freq_array/1e+9, resp_rr, label = "RR")
plt.plot(freq_array/1e+9, resp_ll, label = "LL")
plt.xlabel('Frequency (GHz)')
plt.ylabel('Amplitude (a.u.)')
plt.title('Autocorrelation, Antenna:10')
plt.legend()
plt.show()




#resp_xx = np.absolute(np.mean(tdata[:,:,0,2,2], axis = 0))
#resp_yy = np.absolute(np.mean(tdata[:,:,3,2,2], axis = 0))


#plt.scatter(freq_array/1e+9, resp_xx, marker = '.', label = 'XX')
#plt.scatter(freq_array/1e+9, resp_yy, marker = '.', label = 'YY')
#plt.ylabel('Amplitude')
#plt.xlabel('Frequency (GHz)')
#plt.show()

