from astropy import units as u
import setigen as stg
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Tkagg')

nchans = 98304 #100000
n_coarse = 32
tbins = 50
df =  10.172526041666666*u.Hz   #10*u.Hz
dt = 0.098304*u.s   #0.1*u.s

fch1 = 1999.5*u.MHz   #3095.214*u.MHz
#drift_rate = 8*u.Hz/u.s
width = 10*u.Hz
#snr = 100

frame = stg.Frame(fchans = nchans*n_coarse, tchans = tbins, df = df, dt = dt,
                  fch1 = fch1, ascending = True)

noise = frame.add_noise(x_mean=5)

bp = np.ones(nchans)
bottom = np.arange(0,9830) # First 10 percent
top = np.arange(88474,98304) # Last 10 percent
fact = bottom*(1/9830)
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


freq = np.arange(49152,n_coarse*nchans,nchans)
print (freq.shape)

snr_ar = np.linspace(10,50,32)

sig_drift = np.linspace(5,55,32)*u.Hz/u.s
print (sig_drift)

for i in range(freq.shape[0]):

    signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(freq[i]),
                          drift_rate=sig_drift[i]),
                          stg.constant_t_profile(frame.get_intensity(snr=snr_ar[i])),
                          stg.box_f_profile(width=width),
                          stg.constant_bp_profile(level=1))

frame.data = frame.data*bp_resp
frame.data = frame.data.astype('float32')
print(frame.data.dtype)
frame.save_h5(filename='data_setigen_ds32.h5')

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
