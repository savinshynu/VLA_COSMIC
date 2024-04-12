import sys
import numpy as np
from blimpy import Waterfall
import matplotlib
from matplotlib import pyplot as plt
from blimpy.signal_processing import dedoppler_1
matplotlib.use('Tkagg')

plt.rcParams.update({'font.size': 14})

filename = sys.argv[1]

sb1_low = 8420.3962
sb1_high = 8420.3970

sb2_low = 8420.4412
sb2_high = 8420.4420

car_low = 8420.4190
car_high = 8420.4192

fb = Waterfall(filename) #Instance without dedoppler
db = Waterfall(filename) #Instance with dedoppler

#Header info
fb.info()
header = fb.header

if header['foff'] < 0:
    fb.data  = np.flip(fb.data, axis = 2)
    db.data  = np.flip(db.data, axis = 2)
    
#drift rate to dedoppler the data
df_rate = -0.55

#only do this if want to dedopple the data
dedoppler_1(db, df_rate)

#Frequency to subtract
freq_sub = 8420.4191030

#Collecting the data for sidebands, carriers  in data without and with dedoppler

#without dedoppler
freq_sb1, data_sb1 = fb.grab_data(f_start = sb1_low, f_stop = sb1_high, t_start=None, t_stop=None)
freq_sb2, data_sb2 = fb.grab_data(f_start = sb2_low, f_stop = sb2_high, t_start=None, t_stop=None)
freq_car, data_car = fb.grab_data(f_start = car_low, f_stop = car_high, t_start=None, t_stop=None)

#With dedoppler
freq_sb1_corr, data_sb1_corr = db.grab_data(f_start = sb1_low, f_stop = sb1_high, t_start=None, t_stop=None)
freq_sb2_corr, data_sb2_corr = db.grab_data(f_start = sb2_low, f_stop = sb2_high, t_start=None, t_stop=None)
freq_car_corr, data_car_corr = db.grab_data(f_start = car_low, f_stop = car_high, t_start=None, t_stop=None)

tdat = np.arange(fb.data.shape[0])*header['tsamp']

if header['foff'] < 0:
    freq_sb1 = np.flip(freq_sb1)
    freq_sb2 = np.flip(freq_sb2)
    freq_car = np.flip(freq_car)
    freq_sb1_corr = np.flip(freq_sb1_corr)
    freq_sb2_corr = np.flip(freq_sb2_corr)
    freq_car_corr = np.flip(freq_car_corr)

spec_sb1 = 10*np.log10(np.mean(data_sb1, axis = 0))
spec_sb2 = 10*np.log10(np.mean(data_sb2, axis = 0))
spec_car = 10*np.log10(np.mean(data_car, axis = 0))

spec_sb1_corr = 10*np.log10(np.mean(data_sb1_corr, axis = 0))
spec_sb2_corr = 10*np.log10(np.mean(data_sb2_corr, axis = 0))
spec_car_corr = 10*np.log10(np.mean(data_car_corr, axis = 0))


del_freq = header['foff']/2.0
del_tdat = header['tsamp']/2.0

#Time and frequency labelling for waterfall plots
freq_plt_sb1 = np.linspace(freq_sb1[0]-del_freq, freq_sb1[-1]+del_freq, len(freq_sb1)+1)
freq_plt_sb2 = np.linspace(freq_sb2[0]-del_freq, freq_sb2[-1]+del_freq, len(freq_sb2)+1)
freq_plt_car = np.linspace(freq_car[0]-del_freq, freq_car[-1]+del_freq, len(freq_car)+1)
tdat_plt = np.linspace(tdat[0]-del_tdat, tdat[-1]+del_tdat, len(tdat)+1)

fig, axs = plt.subplots(2, 3, sharex  = False, sharey = False, constrained_layout=True, figsize = (12,8))

axs[0,0].plot((freq_sb1-freq_sub)*1e+3, spec_sb1, color='blue', marker='.', linestyle='solid', label = f"Drift rate = 0.0 Hz/s" )
axs[0,0].plot((freq_sb1_corr -freq_sub)*1e+3, spec_sb1_corr, color='black', marker='.', linestyle='solid',label = f"Drift rate = {df_rate} Hz/s" )
axs[0,0].set_ylabel("Power [dB]")
axs[0,0].set_ylim(97,110)
axs[0,0].legend()

axs[0,1].plot((freq_car - freq_sub)*1e+3, spec_car, color='blue', marker='.', linestyle='solid')
axs[0,1].plot((freq_car_corr -freq_sub)*1e+3, spec_car_corr, color='black', marker='.', linestyle='solid')
axs[0,1].set_ylim(97,110)

axs[0,2].plot((freq_sb2-freq_sub)*1e+3, spec_sb2, color='blue', marker='.', linestyle='solid', label = f"Drift rate = 0.0 Hz/s" )
axs[0,2].plot((freq_sb2_corr-freq_sub)*1e+3, spec_sb2_corr, color='black', marker='.', linestyle='solid',label = f"Drift rate = 0.0 Hz/s" )
axs[0,2].set_ylim(97,110)

axs[1,0].pcolormesh((freq_plt_sb1-freq_sub)*1e+3, tdat_plt, data_sb1, cmap = 'jet')
axs[1,0].set_ylabel("Time [s]")
axs[1,1].pcolormesh((freq_plt_car-freq_sub)*1e+3, tdat_plt, data_car, cmap = 'jet')
axs[1,2].pcolormesh((freq_plt_sb2-freq_sub)*1e+3, tdat_plt, data_sb2, cmap = 'jet')

#fig.supxlabel("Frequency (MHz)")
fig.supxlabel(f"Relative Frequency [kHz] from {freq_sub} MHz")
#fig.suptitle("Voyager 1")
plt.savefig("voyager-spec-wf.png", dpi = 300)
plt.show()
plt.close()


