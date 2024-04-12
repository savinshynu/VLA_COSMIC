import sys
import numpy as np
from blimpy import Waterfall
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import median_absolute_deviation as mad
#from blimpy.signal_processing import dedoppler_1
matplotlib.use('Tkagg')

def dedoppler_1(wf, drift_rates, range):
    """
    Simple de-doppler code for a Filterbank or HDF5 file.
    Parameters:
    ----------
    wf : object
        Blimpy Waterfall object, previously instantiated with a loaded data matrix.
    drift_rate : A list of float
        Signal drift rate over time [Hz/s]
    """
    llim, ulim = [range[0], range[1]]
    # Get the time sampling interval in seconda.
    tsamp = wf.header['tsamp']

    # Get the fine channel bandwidth in Hz.
    chan_bw = wf.header['foff'] * 1e6

    # Compute the number of numpy rolls to perform.
    n_rolls = (drift_rates * tsamp) / chan_bw
    #print(n_rolls)
    # For each time-row,
    #      roll all of the data power values in each fine channel frequency column
    #      given by -(n_roll * row number).
    
    snr = np.zeros(len(n_rolls))
    for n, n_roll in enumerate(n_rolls):
        data = wf.data.copy()
        print(data.shape[0])
        
        for k in np.arange(data.shape[0]):
            data[k][0][:] = np.roll(data[k][0][:], -int(n_roll * k))
        
        data = np.squeeze(data)
        spec = np.mean(data[:,llim:ulim], axis = 0)
        snr[n] = (np.max(spec) - np.median(spec))/mad(spec)
        del data    
    plt.plot(drift_rates, snr)
    plt.show()






filename = sys.argv[1]

fb = Waterfall(filename)

#only do this if want to dedopple the data
#dedoppler_1(fb, -0.5)

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

lfreq =  8420.4190  #3032.2496  #6668.30 #3032.2496
hfreq =  8420.4192 #3032.2504  #6668.45 #3032.2504
llim = np.where(np.abs(freq-lfreq)<0.00016)[0]
ulim = np.where(np.abs(freq-hfreq)<0.00016)[0]

llim = int(np.mean(llim))
ulim  = int(np.mean(ulim))

#llim = int(32.005*256000)
#ulim = int(32.010*256000)

drift_rates = np.linspace(-0.8,0.9,50)
print(drift_rates)
dedoppler_1(fb, drift_rates, [llim, ulim])


"""#
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
"""