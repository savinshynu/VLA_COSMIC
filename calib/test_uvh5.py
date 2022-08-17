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
print(dat.shape)

bls = uvd.Nbls

data = dat[:bls, :, :, :]

print(bl2ant(100))

print(bls)
print(uvd.Ntimes)
#print(uvd.time_array)


"""

print(data.shape)




axis = 0
nbl = data.shape[axis]
ndim = len(data.shape)

(check, nant) = bl2ant(nbl)

print (check, nant)

tdata = np.zeros(data.shape[:axis]+data.shape[axis+1:]+(nant, nant),
                     dtype=data.dtype)

print (tdata.shape)


for i in range(nbl):
        (a0, a1) = bl2ant(i)
        tdata[..., a0, a1] = data.take(i, axis=axis)
        tdata[..., a1, a0] = np.conj(data.take(i, axis=axis))

"""
resp = np.angle(data[1,0,:,0])
plt.plot(range(len(resp)), resp, label = 'Before cal')
#plt.show()
"""
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
