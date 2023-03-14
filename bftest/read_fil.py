import sys
import numpy as np
from blimpy import Waterfall
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Tkagg')

filename = sys.argv[1]

fb = Waterfall(filename)
#fb = Waterfall('/path/to/filterbank.h5') #works the same way
#fb.info()
header = fb.header
print(header)
#data = fb.grab_data(f_start=3031, f_stop=3032,t_start=None, t_stop=None)
data = fb.data
print(fb.data.shape)
data = np.squeeze(fb.data)
freq = np.array(fb.get_freqs())
print(freq)

if header['foff'] < 0:
    data  = np.flip(data, axis = 1)
    freq = np.flip(freq)

lfreq =  3032.2496  #6668.30 #3032.2496
hfreq = 3032.2504  #6668.45 #3032.2504
llim = np.where(np.abs(freq-lfreq)<0.00016)[0]
ulim = np.where(np.abs(freq-hfreq)<0.00016)[0]

llim = int(np.mean(llim))
ulim  = int(np.mean(ulim))

#llim = int(32.005*256000)
#ulim = int(32.010*256000)

print(freq[0], freq[1]-freq[0])
#print(freq[llim], freq[ulim])

data = data[:,llim:ulim]
freq = freq[llim:ulim]
tdat = np.arange(data.shape[0])*header['tsamp']

del_freq = (freq[1] - freq[0])/2.0
del_tdat = (tdat[1]-tdat[0])/2.0

freq_plt = np.linspace(freq[0]-del_freq, freq[-1]+del_freq, len(freq)+1)
tdat_plt = np.linspace(tdat[0]-del_tdat, tdat[-1]+del_tdat, len(tdat)+1)

#print(freq)
print(data.shape)

spec = np.mean(data, axis = 0)
tseries = np.mean(data, axis = 1)

fig = plt.figure(figsize=(10, 10), constrained_layout = True)
#grid = plt.GridSpec(6, 4, fig, hspace=0.4, wspace=0.4)
subfigs = fig.subfigures(3, 1, wspace=0.07, hspace=0.07,  height_ratios = [3,1,1] )

main_ax = subfigs[0].subplots()
#main_ax = fig.add_subplot(grid[:4, :])
#fax = fig.add_subplot(grid[4,:], xticklabels=[], sharex = main_ax)
#tax = fig.add_subplot(grid[5,:], yticklabels=[], sharex = main_ax)


#fax = fig.add_subplot(grid[4,:], xticklabels=[])
#tax = fig.add_subplot(grid[5,:], yticklabels=[])

fax = subfigs[1].subplots()
tax = subfigs[2].subplots()

subfigs[1].supylabel("Power (a.u.)")
subfigs[1].supxlabel("Frequency Channels")
subfigs[1].suptitle("Spectra")

subfigs[2].supylabel("Power (a.u.)")
subfigs[2].supxlabel("Time (s)")
subfigs[2].suptitle("Time Series")

main_ax.pcolormesh(freq_plt, tdat_plt, data, cmap = 'jet')
subfigs[0].supxlabel("Frequency (MHz)")
subfigs[0].supylabel("Time (s)")
subfigs[0].suptitle(f"Waterfall, Source:{header['source_name']}")
fax.plot(freq, spec, color='blue', marker='.', linestyle='solid')
tax.plot(tdat, tseries, color='blue', marker='.', linestyle='solid')
#plt.savefig("maser_low_res_beam2.png")

plt.show()
#plt.close()
