from pyuvdata import UVData
import sys
from derive_sol import ant2bl, bl2ant, gaincal, applycal
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg

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
         No. of antennas in the array: {nant_array}")


#uvd.write_ms("test_vla.ms", force_phase = True)






bls = uvd.Nbls

data = dat[:, :, :, :]

print(bl2ant(2))

print(bls)
print(uvd.Ntimes)
print(uvd.time_array)



print(data.shape)



axis = 0
ndim = len(data.shape)

(check, nant) = bl2ant(bls)

print (check, nant)

tdata = np.zeros((ntimes,)+data.shape[axis+2:]+(nant, nant),
                     dtype=data.dtype)

print (tdata.shape)

a0, a1 = uvd.baseline_to_antnums(1)

print(a0, a1)


for i in range(bls):
    (a0, a1) = bl2ant(i)
    print (a0, a1)

bsind = np.arange(0, data.shape[0], bls)

for k,ind in enumerate(bsind):
    data_int = data[ind:ind+bls,0, :, :] 
    for i in range(bls):
        (a0, a1) = bl2ant(i)
        #print (a0, a1)
        tdata[k,:,:, a0, a1] = data_int.take(i, axis=axis)
        tdata[k,:,:, a1, a0] = np.conj(data_int.take(i, axis=axis))




resp_xx = np.absolute(np.mean(tdata[:,:,0,2,2], axis = 0))
resp_yy = np.absolute(np.mean(tdata[:,:,3,2,2], axis = 0))


plt.scatter(freq_array/1e+9, resp_xx, marker = '.', label = 'XX')
plt.scatter(freq_array/1e+9, resp_yy, marker = '.', label = 'YY')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (GHz)')
plt.show()

"""
#print (resp_xx)
#print (resp_yy)
#plt.scatter(freq_array/1e+9, resp_xx*(180.0/np.pi), marker = '.', label = 'XX')
#plt.scatter(freq_array/1e+9, resp_yy*(180.0/np.pi), marker = '.', label = 'YY')
#plt.ylabel('Amplitude (a.u.)')
plt.ylabel('Phase (degrees)')
plt.xlabel('Frequency (GHz)')
#plt.title(f'Single integration: {round(intg_time[0],4)} s, baseline: ea6-ea6') 
plt.title(f'Averaged in time, baseline: ea06-ea12')
plt.legend()
plt.show()


(wtmp, vtmp) = linalg.eigh(tdata)

v = vtmp[..., -1].copy()
w = wtmp[..., -1]
print (w[0,0,0], wtmp[0,0,0,-1])
print (wtmp.shape, vtmp.shape, (w.T).shape, (v.T).shape)



caldat = gaincal(data)

np.save('calfile',caldat)

print (caldat.shape)
#print(resp.shape)

#gain_ant = np.abs(resp[10,0,:,0])

#phase_ant = np.angle(resp[:,0,0,0])


#plt.plot(range(len(gain_ant)), gain_ant)
#plt.show() 


applycal(data, caldat)

resp = data[100,0,:,0]

gain_cor = np.abs(resp)
phase_cor = np.angle(resp)*(180.0/np.pi)

plt.plot(range(len(phase_cor)), phase_cor, label = 'After_Cal')
plt.ylabel('Phase (degrees)')
#plt.ylabel('Amplitude (a.u.) in logscale')
plt.xlabel('Frequency Channel')
plt.title('Baseline: 100')
plt.legend()
plt.show()


"""
