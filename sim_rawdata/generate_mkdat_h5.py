from astropy import units as u
import setigen as stg
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Tkagg')

nchans = 8192
n_coarse = 512
tbins = 18
df = 16.21246337890625*u.Hz
dt = 15.79032094117647*u.s

fch1 = 747.93359375*u.MHz
#drift_rate = 8*u.Hz/u.s
width = 10*u.Hz
#snr = 100

frame = stg.Frame(fchans = nchans*n_coarse, tchans = tbins, df = df, dt = dt,
                  fch1 = fch1, ascending = True)

noise = frame.add_noise(x_mean=5)

bp = np.ones(nchans)
bottom = np.arange(0,819)
top = np.arange(7373,8192)
fact = bottom*(1/819)
bp[bottom] = fact
bp[top] = np.flip(fact)

bp_resp = np.tile(bp,n_coarse)

#plt.plot(range(len(bp)), bp)
#plt.show()

#signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(20000),
#                                            drift_rate=drift_rate),
#                          stg.constant_t_profile(frame.get_intensity(snr=snr)),
#                          stg.box_f_profile(width=width),
#                          stg.constant_bp_profile(level = 1))

#frame.data = frame.data*bp


freq = np.arange(4096,81920,8192)
#print (freq.shape)

snr_ar = np.linspace(20,100,10)

sig_drift = np.linspace(1,20,10)*u.Hz/u.s
#print (sig_drift)

for i in range(freq.shape[0]):

    signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(freq[i]),
                          drift_rate=sig_drift[i]),
                          stg.constant_t_profile(frame.get_intensity(snr=snr_ar[i])),
                          stg.box_f_profile(width=width),
                          stg.constant_bp_profile(level=1))

#frame.data = frame.data*bp_resp
frame.save_h5(filename='data_setigen_mk.h5')

"""
plt.imshow(frame.get_data(), aspect='auto')
plt.colorbar()
plt.xlabel("Frequency Channels, df = 10Hz")
plt.ylabel("Time Samples, dt = 100 ms ")
plt.title(f"Drift rate = {drift_rate}")
plt.show()

spectrum = frame.integrate()
plt.ylabel('Power (a.u.)')
plt.xlabel('Channels')
plt.plot(range(len(spectrum)), spectrum)
plt.show()
"""
