import sys,os
import time
import glob
import json
from blimpy import GuppiRaw
import numpy as np
import matplotlib
import tqdm as tq
from matplotlib import pyplot as plt
import argparse
import h5py
import matplotlib

matplotlib.use('Tkagg')

class Recipe(object):
    """
    Class to take a bfr5 file and get all the information needed from it
    """
    def __init__(self, filename):
        self.h5 = h5py.File(filename)
        self.ras = self.h5["/beaminfo/ras"][()]
        self.decs = self.h5["/beaminfo/decs"][()]
        self.phase_ra = self.h5['obsinfo/phase_center_ra'][()]
        self.phase_dec = self.h5['obsinfo/phase_center_dec'][()]
        self.obsid = self.h5["/obsinfo/obsid"][()]
        self.src_names = self.h5["/beaminfo/src_names"][()]
        #Delays calculated are in the from (ntimes, nbeams, nants), Here ntimes is calculated
        #for each block in the guppi raw files
        self.delays = self.h5["/delayinfo/delays"][()] 
        self.time_array = self.h5["/delayinfo/time_array"][()]
        self.npol = self.h5["/diminfo/npol"][()]
        self.nbeams = self.h5["/diminfo/nbeams"][()]
        self.cal_all = self.h5["/calinfo/cal_all"][()]
        self.nants = self.h5["/diminfo/nants"][()]
        self.nchan = self.h5["/diminfo/nchan"][()]

        self.antenna_names = [s.decode("utf-8")
                              for s in self.h5["/telinfo/antenna_names"][()]]
        
        # Validate shapes of things
        assert self.delays.shape == (len(self.time_array), self.nbeams, self.nants)
        if self.cal_all.shape != (self.nchan, self.npol, self.nants):
            print("cal_all shape:", self.cal_all.shape)
            print("nchan, npol, nants:", (self.nchan, self.npol, self.nants))
            raise ValueError("unexpected cal_all size")
        
        
    def time_array_index(self, time):
        """Return the index in time_array closest to time.
        The time array is calculated for middle point of each block in Guppi raw file.
        """
        dist = np.abs(self.time_array - time)
        min = np.argmin(dist)
        return min

"""
def normalize(data):
    for ant in range(data.shape[0]):
        for pol in range(data.shape[-1]):
            mean = np.mean(data[ant,:,:,pol])
            std = np.std(data[ant,:,:,pol])
            data[ant,:,:,pol] = (data[ant,:,:,pol] - mean)/std
    
def calculate_weights_from_noise(data):
    weights = np.zeros((data.shape[0], data.shape[-1]))
    for ant in range(data.shape[0]):
        for pol in range(data.shape[-1]):
            weights[ant,pol] = np.median(np.abs(data[ant,:,:,pol]))
    weights[:,0] = (weights[:,0]/weights[:,0].min())
    weights[:,1] = (weights[:,1]/weights[:,1].min())
    
    for ant in range(data.shape[0]):
        for pol in range(data.shape[-1]):
            data[ant,:,:,pol] = data[ant,:,:,pol]/weights[ant,pol]


def calculate_weights_from_signal(data):
    
    weights = np.zeros((data.shape[0], data.shape[-1]))
    for ant in range(data.shape[0]):
        for pol in range(data.shape[-1]):
            spec = np.mean(np.abs(data[ant,:,:,pol]), axis = 0)
            weights[ant,pol] = np.max(spec)
    weights[:,0] = (weights[:,0]/weights[:,0].min())
    weights[:,1] = (weights[:,1]/weights[:,1].min())
    print(weights[:,0])
    print(weights[:,1])

    for ant in range(data.shape[0]):
        for pol in range(data.shape[-1]):
            data[ant,:,:,pol] = data[ant,:,:,pol]/weights[ant,pol]
"""


def coherent_bf_voltage(data, beam, times, freq_array, recp_ob):
    """
    Coherent beamforming function
    Input: channelized data (nants, ntime, nfreqs, npols)
    Also need a unix time array and frquency array of fine channels in MHz
    Returns a numpy array of beamformeed volates power.
    Output dimensions are [time, chan]
    """
    npols = 2
    nants = data.shape[0]

    #Coefficient array which has the same size of the data
    coeff = data*0.0

    #Normalizing the data
    #normalize(data)
    #calculate_weights_from_signal(data)

    # The delay values from bfr5 files are in ns, so converting frequencies to GHz
    freqs = freq_array*1e-3 
    print(freqs)
    
    #Now for each time in the upchannelized data, find the delay values close to the time for each antennas
    #Calculates the phasor information for each time 
    # Apply it for all antennas, frequencies and polarizations

    for timestep, time_value in enumerate(times):
        time_array_index = recp_ob.time_array_index(time_value)
        #print(time_value, time_array_index)
        # The delay values for each antennas for a time and beam
        tau = recp_ob.delays[time_array_index, beam, :]
        #print(tau)
        for ant in range(nants):
            #Calculating the phasor
            angles = tau[ant] * (freqs * 2 * np.pi * 1.0j)
            for pol in range(npols):
                coeff[ant, timestep, :, pol] = np.exp(angles)
    
    #print(coeff[5,2,:,0])
    #print(coeff[10,2,:,0])
    
    #Taking the conjugate of phasor, multiply with the data and summing it across the antenna axis
    bf_volt = (np.conjugate(coeff) * data).sum(axis = 0)

    #Squaring the beamformed voltages and summing across the polarizations to get Stokes I
    bf_pow = (np.abs(bf_volt)**2).sum(axis = 2)

    return bf_pow

def incoherent_bf_voltage(data):
    """
    Input: channelized data (nants, ntime, nfreqs, npols)
    Returns a numpy array of beamformeed volates power.
    Output dimensions are [time, chan]
    """
    #Normalizing the data
    #normalize(data)
    #calculate_weights_from_signal(data)

    #Squaring the data and adding across antenna and polarization axis
    bf_pow = (np.abs(data)**2).sum(axis = (0,3))

    return bf_pow

def plot_antenna_spectra(data):
    
    """
    Input: Channelized FFT data
    plots the individual averaged spectra for antennas
    """
    #calculate_weights_from_signal(data)

    #Normalizing the data
    #normalize(data)

    #Gettng a a complex data here
    for i in range(data.shape[0]):
        #Plotting the single antennna
        spec = (np.abs(data[i,:,:,:])**2).sum(axis = 2)
        pow = np.mean(spec, axis = 0)
        #plt.plot(10*np.log10(pow), label = i)
        plt.plot(pow, label = i)
    plt.legend()
    plt.show()

def plot_antenna_timeseries(data):

    """
    input: Channelized FFT data
    plots the time series from each antennas
    """

    for i in range(data.shape[0]):
        #Plotting the single antennna
        ts = data[i,:,:,:]
        ts = (np.abs(ts)**2).sum(axis = 2)
        pow  = np.mean(ts, axis = 1)
        plt.plot(pow, label = i)
        plt.legend()
    plt.show()

def plot_all(data, freq, tdat, phase = None, point = None, beam = "incoh"):

    """
    Plots the waterfall, averaged spectra and time series
    """

    print("plotting now: wait")
    tdat -= tdat[0]
    spec = np.mean(data, axis = 0)
    tseries = np.mean(data, axis = 1)

    del_freq = (freq[1] - freq[0])/2.0
    del_tdat = (tdat[1]-tdat[0])/2.0

    freq_plt = np.linspace(freq[0]-del_freq, freq[-1]+del_freq, len(freq)+1)
    tdat_plt = np.linspace(tdat[0]-del_tdat, tdat[-1]+del_tdat, len(tdat)+1)

    fig = plt.figure(figsize=(10, 10), constrained_layout = True)
    subfigs = fig.subfigures(3, 1, wspace=0.07, hspace=0.07,  height_ratios = [3,1,1] )

    main_ax = subfigs[0].subplots()

    fax = subfigs[1].subplots()
    tax = subfigs[2].subplots()

    subfigs[1].supylabel("Power (a.u.)")
    subfigs[1].supxlabel("Frequency (MHz)")
    subfigs[1].suptitle("Spectra")

    subfigs[2].supylabel("Power (a.u.)")
    subfigs[2].supxlabel("Time (s)")
    subfigs[2].suptitle("Time Series")

    main_ax.pcolormesh(freq_plt, tdat_plt, data, cmap = 'jet')
    subfigs[0].supxlabel("Frequency (MHz)")
    subfigs[0].supylabel("Time (s)")
    subfigs[0].suptitle(f"Waterfall")
    fax.plot(freq, spec, color='blue', marker='.', linestyle='solid')
    tax.plot(tdat, tseries, color='blue', marker='.', linestyle='solid')
    if beam == "incoh":
        fig.suptitle(f"Pointing: Ra,dec: {point}, Boresight: {phase}")
    else:
        fig.suptitle(f"Pointing: Ra,dec: {np.round(point[0],3), np.round(point[1],3)}, Boresight: {np.round(phase[0],3), np.round(phase[1],3)}")
    #plt.savefig(f"vla_maser-on-bore_beam_{beam}.png")
    plt.savefig(f"maser-off-bore_beam{beam}.png")
    #plt.show()
    plt.close()

def plot_spectra(data, freq, ns_range, sig_range, beam_info):

    """
    Plots the integrated spectra from al the beams
    Pass the array which has stored spectra from al the beams
    """
    print("plotting now: wait")
    
    #print(ns_range)
    #print(sig_range)
    nbeam = data.shape[0]
    beams = list(beam_info.keys())
    print(beams)

    fig = plt.figure(figsize=(10, 6))
    clr = ['b','g','r','c','m','y','k']

    for bm in range(nbeam):
        beam = beams[bm]
        print(f"plotting beam : {beam}")
        spec = data[bm,:]

        #Calculates the noise and peak values and SNR from the data
        noise = np.std(spec[ns_range[0]:ns_range[1]])
       
        peak = np.max(spec[sig_range[0]:sig_range[1]])
       
        snr = peak/noise
        print(f"SNR : {snr}")
        
        plt.plot(freq, 10*np.log10(spec), color=clr[bm], label = beam, marker='.', linestyle='solid')
        #plt.plot(freq, spec, color=clr[bm], label = beam, marker='.', linestyle='solid')
        plt.ylabel("Power (dB)")
        plt.xlabel("Frequency (MHz)")
        plt.legend()
        
        #point = beam_info[beam][0]
        #phase = beam_info[beam][1]

        #if beam == "incoh":
        #    plt.title(f"Pointing: Ra,dec: {point}, Boresight: {phase}, SNR : {snr}")
        #else:
        #    plt.title(f"Pointing: Ra,dec: {np.round(point[0],3), np.round(point[1],3)}, Boresight: {np.round(phase[0],3), np.round(phase[1],3)}, SNR : {snr}")
   
    plt.savefig(f"spectra-on-bore_log.png")
    plt.show()
    #plt.close()


def collect_header(dat_stem):
    """
    Collecting the header information from the first files
    """
    files = sorted(glob.glob(dat_stem+'*.raw')) #Input guppi raw file 
    gob = GuppiRaw(files[0]) #Instantiating a guppi object
    header = gob.read_first_header() # Reading the first header 
    return header

def collect_sing_chan_data(dat_stem, chan_num):
    """
    Collecting the data from a single coarse channel in all the associated the guppi raw files and
    saving them to an array
    Input is the file stem and the coarse channel number (where the signal is expected)
    """


    #files = sorted(glob.glob(dat_stem+'*.raw')) #Input guppi raw file 
    files = sorted(glob.glob(dat_stem+'.0000''*.raw')) #Taking one file for simplicity
    for f,filename in enumerate(files):
        print(filename)
        gob = GuppiRaw(filename) #Instantiating a guppi object
        header = gob.read_first_header() # Reading the first header 
        n_blocks = gob.n_blocks # Number of blocks in the raw file
        guppi_scanid = header.get('OBSID','')
        nant_chans = int(header['OBSNCHAN'])
        nant = int(header['NANTS'])
        nbits = int(header['NBITS'])
        npols = int(header['NPOL'])
        nchan = int(nant_chans/nant)
        freq_mid = header['OBSFREQ'] #stopping frequency 
        chan_timewidth = header['TBIN']
        chan_freqwidth = header['CHAN_BW']
        
        freq_start = freq_mid - ((nchan*chan_freqwidth)/2.0)
        freq_end = freq_mid + ((nchan*chan_freqwidth)/2.0)
    
        fstart_chan = freq_start + chan_num*chan_freqwidth
        fstop_chan = freq_start + (chan_num+1)*chan_freqwidth

        print(fstart_chan, fstop_chan)


        #For the UNIX time stamp calculation
        ntimes = (header["BLOCSIZE"] * 8) // (header["OBSNCHAN"] * header["NPOL"] * 2 * header["NBITS"])
        start_time_unix = header["SYNCTIME"] + header["PKTIDX"] * header.get("TBIN", 1/header["CHAN_BW"]) * ntimes/header.get("PIPERBLK", ntimes)
        times_unix = (start_time_unix) + np.arange(n_blocks*ntimes)*header.get("TBIN", 1/header["CHAN_BW"])
        #print(times_unix.shape)
        #block_time_span_s = header.get("PIPERBLK", ntimes) * header.get("TBIN", 1/header["CHAN_BW"]) * ntimes/header.get("PIPERBLK", ntimes)
        #print(block_time_span_s, start_time_unix)
        #block_unix = (start_time_unix + 0.5 * block_time_span_s) + np.arange(n_blocks)*block_time_span_s
        #print(block_unix)
        
        blocksize = header['BLOCSIZE']
        ntsamp_block = int(blocksize/(nant_chans*npols*2*(nbits/8))) # Number of time samples in the block

        # Collecting data from each block into a big array
        data = np.zeros((nant, int(ntsamp_block*n_blocks), npols), dtype = 'complex64')
        print(data.shape)

        ants1 = header['ANTNMS00']
        ants = ants1.split(',')
        try:
            ants2 = header['ANTNMS01']
            ants += ants2.split(',')
        except KeyError:
            pass

        try:
            ants3 = header['ANTNMS02']
            ants += ants3.split(',')
        except KeyError:
            pass

        if f == 0:
            print("The header info for the file: ")
            print(header)
        print(f"Nblocks: {n_blocks},  Number of blocks to read: {n_blocks}")
           

        
        print("Started collecting data")
        for i in tq.tqdm(range(n_blocks)):
            head_block, data_block = gob.read_next_data_block()
            data_block = data_block.reshape(nant, nchan, ntsamp_block, npols)
            data[:,i*ntsamp_block:(i+1)*ntsamp_block, :] = data_block[:,chan_num,:,:]

        #Concatenating the data from each files into a bigger array, the complex data and time information
        try:
            data_main = np.concatenate((data_main, data), axis = 1)
        except NameError:
            data_main = data
        
        #print(data_main.shape)


        try:
            times_unix_main = np.concatenate((times_unix_main, times_unix), axis = 0)
        except NameError:
            times_unix_main = times_unix
        #print(times_unix_main.shape)

    return data_main, times_unix_main, [fstart_chan, fstop_chan]


def fft_sing_chan(data, times_unix, freqs, header, lfft):


    """
    Upchannelize the voltage data
    Inputs: Voltage data (nant, time (associated with single coarse channel), pols ), time array, (start and stop frequency), header information, length of fft
    output: Channelized data: (nant, tsteps, fine_channels, pols)
    """

    print("fft ing the data now")

    nfine = lfft # Number of fine channels per coarse channel
    
    # trim time to be a multiple of FFT size
    ntime_total = data.shape[1]
    ntime_total -= ntime_total % lfft
    data = data[:, :ntime_total, :]
    npols = 2
    #Restricting unix time as well
    print(data.shape, len(times_unix))
    times_unix  = times_unix[:ntime_total]
    print(len(times_unix))

    freq_end = freqs[1]
    freq_start = freqs[0]

    delt = header['TBIN']
    delf =  header['CHAN_BW']
    nant = int(header['NANTS'])
    # Time, frequency resolution and number of time samples after FFT
    chan_timewidth = delt*nfine
    ntsampfine = int(data.shape[1]/nfine)
    chan_freqwidth = delf/nfine

    #Getting new unix time values
    times_unix_new = np.zeros(ntsampfine)
    time_index = np.arange(0,len(times_unix), nfine)
    print(times_unix[0], times_unix[1])
    for k,m in enumerate(time_index):
        times_unix_new[k] = np.mean(times_unix[m:m+nfine])


    # reshape to [A, Tfine, Tfft, P]
    data = data.reshape(nant, ntsampfine, nfine, npols)

    print(f"FFT of the data ({data.shape} along axis 2)")
    t0 = time.time()
    data = np.fft.fft(data, axis = 2)
    data = np.fft.fftshift(data, axes = 2)
    t1 = time.time()
    print(f"FFT done, took {t1-t0} s")

    print(f"The channelized datashape [A, Tfine, Cfine, P]: {data.shape}") 
    freq_axis = np.linspace(freq_start, freq_end, nfine) #New Upchannelized frequency channels

    return data, times_unix_new, freq_axis





    
if __name__ == "__main__":
    
    #Input file stem
    dat_stem = sys.argv[1]
    #Input bfr5 file for beamforming
    recipe_file =  sys.argv[2]
    
    #Coarse channel to collect the data and upchannelize
    chan_num = 10 #50

    #collecting the header infor and data from all the guppi raw files
    header = collect_header(dat_stem)
    data, times_unix, freq_info = collect_sing_chan_data(dat_stem, chan_num)
    print(data.shape)
    print(freq_info)
    
    #Upchannelizing the data
    data_fft, times_unix_fft, freqs_fft = fft_sing_chan(data, times_unix, freq_info, header, 16384)
    #data_fft, times_unix_fft, freqs_fft = fft_sing_chan(data, times_unix, freq_info, header, 131072)

    #plot_antenna_spectra(data_fft)
    #plot_antenna_timeseries(data_fft)
    

    recp_ob = Recipe(recipe_file)
    

    #Giving a range of frequencies for displaying the data

    lfreq = 6668.3 #3032.2496 #6669.0#3019.7496
    hfreq = 6668.5 #3032.2506 #6669.370#3019.7504
    llim = np.where(np.abs(freqs_fft-lfreq)<0.00016)[0]
    ulim = np.where(np.abs(freqs_fft-hfreq)<0.00016)[0]
    llim = int(np.mean(llim))
    ulim  = int(np.mean(ulim))

    freqs_fft_trim = freqs_fft[llim:ulim]
    
    #Noise range freq in MHz
    ns_beg = 6668.300
    ns_end = 6668.325

    #signal range freq in MHz
    sig_beg = 6668.375
    sig_end = 6668.400

    ns_llim = np.where(np.abs(freqs_fft_trim-ns_beg)<0.00016)[0]
    ns_ulim = np.where(np.abs(freqs_fft_trim-ns_end)<0.00016)[0]
    ns_llim = int(np.mean(ns_llim))
    ns_ulim  = int(np.mean(ns_ulim))

    sig_llim = np.where(np.abs(freqs_fft_trim-sig_beg)<0.00016)[0]
    sig_ulim = np.where(np.abs(freqs_fft_trim-sig_end)<0.00016)[0]
    sig_llim = int(np.mean(sig_llim))
    sig_ulim  = int(np.mean(sig_ulim))

    ns_range = [ns_llim, ns_ulim]
    sig_range = [sig_llim, sig_ulim]

    #Save the spectra
    spec_array = np.zeros((recp_ob.nbeams+1,len(freqs_fft_trim)), dtype = 'float64')


    #Looking at incoherent beam
    print("Forming incoherent beam")
    incoh_pow = incoherent_bf_voltage(data_fft) 
    plot_all(incoh_pow[:, llim:ulim], freqs_fft[llim:ulim], times_unix_fft.copy())
    
    beam_info = {}
    for beam in range(recp_ob.nbeams): 
        print(f"Forming beam number: {beam}")
        pow = coherent_bf_voltage(data_fft, beam, times_unix_fft, freqs_fft, recp_ob)
        spec_array[beam,:] = np.mean(pow[:, llim:ulim], axis = 0)
       
        point = [recp_ob.ras[beam], recp_ob.decs[beam]]
        bore = [recp_ob.phase_ra, recp_ob.phase_dec]
        beam_info[beam] = [point,bore]
        plot_all(pow[:, llim:ulim], freqs_fft[llim:ulim], times_unix_fft.copy(), bore, point, beam)
    
    spec_array[beam+1,:] = np.mean(incoh_pow[:, llim:ulim], axis = 0)
    beam_info['incoh'] = [None, None]
    plot_spectra(spec_array, freqs_fft[llim:ulim], ns_range, sig_range, beam_info)
    