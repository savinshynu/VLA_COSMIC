import sys
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
from sliding_rfi_flagger import flag_rfi

filename = sys.argv[1]
spec = np.load(filename)
spec_pol_c0 = spec[...,0].ravel(order = 'F')
spec_pol_c1 = spec[...,1].ravel(order = 'F')

threshold = 3
bad_chan, bp = flag_rfi(spec_pol_c0, 20, threshold)

print(bad_chan.shape)

plt.plot(np.abs(spec_pol_c0), label = 'data')
plt.plot(np.abs(bp), label = 'bp')
plt.legend()
plt.show()

##Zeroing bad channels
spec_pol_c0[bad_chan[:,0]] = 0

#loading data to GPU
spec_pol_g0 = cp.asarray(spec_pol_c0)
#spec_pol_g1 = cp.asarray(spec_pol_c1)

print(spec_pol_g0.device)

spec_pol0_fft = cp.fft.fft(spec_pol_g0)
spec_pol0_fft = cp.fft.fftshift(spec_pol0_fft)

#spec_pol1_fft = cp.fft.fft(spec_pol_g1)
#spec_pol1_fft = cp.fft.fftshift(spec_pol1_fft)

tfreq = cp.fft.fftfreq(1024*120,8.3333e+3)
tfreq = cp.fft.fftshift(tfreq)

plt.plot(tfreq.get(),cp.abs(spec_pol0_fft).get())
#plt.plot(tfreq.get(), cp.abs(spec_pol1_fft).get()
plt.ylabel("Amplitude")
plt.xlabel("Time delay (s)")
plt.show()
