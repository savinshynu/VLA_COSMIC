from pyuvdata import UVData
import sys
import numpy as np
from matplotlib import pyplot as plt
import pyuvdata.utils as uvutils

#The uvh5 file to load to load
filename = sys.argv[1]

uvd = UVData()
self.uvd.read(filename, fix_old_proj=False)

dat = self.uvd.data_array
nant_data = self.uvd.Nants_data
nant_array = self.uvd.Nants_telescope
ant_names = self.uvd.antenna_names
nfreqs = self.uvd.Nfreqs
ntimes = self.uvd.Ntimes
npols = self.uvd.Npols
bls = self.uvd.Nbls
nspws = self.uvd.Nspws
chan_width = self.uvd.channel_width
intg_time = self.uvd.integration_time
source = self.uvd.object_name
telescope = self.uvd.telescope_name
pol_array = uvutils.polnum2str(self.uvd.polarization_array)
freq_array = self.uvd.freq_array[0,:]
lobs = ntimes*intg_time
uvw_array = self.uvd.uvw_array
uvw_array = uvw_array.reshape(ntimes,bls, 3)

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
         Polarization array: {pol_array} \n\
         No. of polarizations: {npols}   \n\
         Data array shape: {dat.shape} \n\
         No. of baselines: {bls}  \n\
         No. of antennas present in data: {nant_data} \n\
         No. of antennas in the array: {nant_array} \n\
         Antenna name: {ant_names}")

print (np.unique(self.uvd.ant_1_array))

#self.uvd.write_ms("test_vla.ms", force_phase = True)

#print(self.uvd.time_array)

# Collect the data for a pair of antennas at a polarization

#ant_pair = self.uvd.get_antpairs()

#print(self.uvd.baseline_array)
ant1, ant2 = self.uvd.baseline_to_antnums(self.uvd.baseline_array[:bls])

ant1 = np.array(ant1)
ant2 = np.array(ant2)

for i in range(bls):
    print(f"{ant1[i]}-{ant2[i]}, :U,V,W: {uvw_array[0,i,:]}")



#print(len(ant_pair))

#print(self.uvd.get_baseline_nums())
#print(uvw_array[bi,0,:]/3e+8)  

ant_dat = data = self.uvd.get_data(26, 23)
print(ant_dat.shape)


resp_rr = 10*np.log10(np.abs(ant_dat[0,:,0]))
resp_ll = 10*np.log10(np.abs(ant_dat[0,:,1]))
phase_rr = np.angle(ant_dat[0,:,0], deg = True)
phase_ll = np.angle(ant_dat[0,:,0], deg = True)


plt.rcParams['figure.constrained_layout.use'] = True
plt.subplot(1,2,1)
plt.plot(freq_array/1e+9, resp_rr, label = "RR")
plt.plot(freq_array/1e+9, resp_ll, label = "LL")
plt.xlabel('Frequency (GHz)')
plt.ylabel('Amplitude (a.u.)')
#plt.title('Autocorrelation, Antenna:10')
plt.legend()

plt.subplot(1,2,2)
plt.plot(freq_array/1e+9, phase_rr, '.', label = "RR")
plt.plot(freq_array/1e+9, phase_ll, '.', label = "LL")
plt.xlabel('Frequency (GHz)')
plt.ylabel('Phase (deg)')
#plt.title('Autocorrelation, Antenna:10')
plt.legend()

plt.show()


"""

#resp_xx = np.absolute(np.mean(tdata[:,:,0,2,2], axis = 0))
#resp_yy = np.absolute(np.mean(tdata[:,:,3,2,2], axis = 0))


#plt.scatter(freq_array/1e+9, resp_xx, marker = '.', label = 'XX')
#plt.scatter(freq_array/1e+9, resp_yy, marker = '.', label = 'YY')
#plt.ylabel('Amplitude')
#plt.xlabel('Frequency (GHz)')
#plt.show()
"""
