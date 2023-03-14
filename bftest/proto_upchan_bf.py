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
    def __init__(self, filename):
        self.h5 = h5py.File(filename)
        self.ras = self.h5["/beaminfo/ras"][()]
        self.decs = self.h5["/beaminfo/decs"][()]
        self.phase_ra = self.h5['obsinfo/phase_center_ra'][()]
        self.phase_dec = self.h5['obsinfo/phase_center_dec'][()]
        self.obsid = self.h5["/obsinfo/obsid"][()]
        self.src_names = self.h5["/beaminfo/src_names"][()]
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
        """Return the index in time_array closest to time."""
        #dist_tuples = [(i, np.abs(val - time)) for i, val in enumerate(self.time_array)]
        dist = np.abs(self.time_array - time)
        #np.set_printoptions(precision=6)
        #print(dist)
        #i, _ = min(dist_tuples)
        min = np.argmin(dist)
        #print(min, self.time_array[min])
        return min





def coherent_bf_voltage(data, beam, times, freq_array, recp_ob):
    """Returns a numpy array of beamformeed volates power.
    Output dimensions are [time, chan]
    """
    npols = 2
    nants = data.shape[0]
    coeff = data*0.0
    #Converting the frequency to GHz for phasor calculation since delays are in ns
    freqs = freq_array*1e-3 
    print(freqs)
    
    for timestep, time_value in enumerate(times):
        time_array_index = recp_ob.time_array_index(time_value)
        #print(time_value, time_array_index)
        tau = recp_ob.delays[time_array_index, beam, :]
        #print(tau)
        for ant in range(nants):
            angles = tau[ant] * (freqs * 2 * np.pi * 1.0j)
            for pol in range(npols):
                coeff[ant, timestep, :, pol] = np.exp(angles)

    bf_volt = (np.conjugate(coeff) * data).sum(axis = 0)
    #bf_volt = (coeff * data).sum(axis = 0)
    bf_pow = ((np.abs(bf_volt))**2).sum(axis = 2)

    return bf_pow

def incoherent_bf_voltage(data):
    """Returns a numpy array of beamformeed volates power.
    Output dimensions are [time, chan]
    """
    
    bf_pow = ((np.abs(data))**2).sum(axis = (0,3))

    return bf_pow

def plot_antenna(data):

    #Plotting the single antennna
    data = data[0,:,2*98250:2*98350,:]
    pow = (np.abs(data)**2).sum(axis = 2)

    plt.pcolormesh(pow)
    plt.show()

def plot_all(data, freq, tdat, phase = None, point = None, beam = "incoh"):


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
    plt.savefig(f"setigen_beam_{beam}.png")
    #plt.show()
    plt.close()

"""

def coefficients(self, beam):
        #Returns a numpy array of beamforming coefficients.

        
        #This does not conjugate, so it should match up with c++ generateCoefficients.
        #Output dimensions are [time, chan, pol, ant]
        
        recipe_channel_index = self.stamp.schan + self.stamp.coarseChannel

        answer = np.zeros((self.stamp.numTimesteps,
                           self.stamp.numChannels,
                           self.stamp.numPolarizations,
                           self.stamp.numAntennas),
                          dtype=np.cdouble)

        for timestep, time_value in enumerate(self.times()):
            for chan, freq_value in enumerate(self.frequencies()):
                time_array_index = self.recipe.time_array_index(time_value)
                ghz = freq_value * 0.001
                tau = self.recipe.delays[time_array_index, beam, :]
                angles = tau * (ghz * 2 * np.pi * 1.0j)
                for pol in range(self.stamp.numPolarizations):
                    cal = self.recipe.cal_all[recipe_channel_index, pol, :]
                    answer[timestep, chan, pol, :] = cal * np.exp(angles)

        return answer


def beamform_voltage(self, beam):
     #Beamforms, leaving the result in complex voltage space.
        
    # Output dimensions are [time, chan, pol]
    
    coeffs = self.coefficients(beam)
    inputs = self.complex_array()

    # Sum along polarization and antenna
    return (np.conjugate(coeffs) * inputs).sum(axis=3)

def beamform_power(self, beam):
    #Converts voltage to power and combines across polarities.

    #Output dimensions are [time, chan]
    
    voltage = self.beamform_voltage(beam)
    squared = np.square(np.real(voltage)) + np.square(np.imag(voltage))
    return squared.sum(axis=2)

"""
def collect_header(dat_stem):
    files = sorted(glob.glob(dat_stem+'*.raw')) #Input guppi raw file 
    gob = GuppiRaw(files[0]) #Instantiating a guppi object
    header = gob.read_first_header() # Reading the first header 
    return header

def collect_sing_chan_data(dat_stem, chan_num):
    # Collecting the data from the guppi raw files and
    # saving them to an array

    files = sorted(glob.glob(dat_stem+'*.raw')) #Input guppi raw file 
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

        #print(fstart_chan, fstop_chan)


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
        #print(data.shape)

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


def get_all_data(dat_file, band, band_center,lfft,tint):
    # Collecting the data from the guppi raw files and
    # saving them to an array

    filename = dat_file #Input guppi raw file 

    gob = GuppiRaw(filename) #Instantiating a guppi object
    header = gob.read_first_header() # Reading the first header 
    n_blocks = gob.n_blocks    # Number of blocks in the raw file
    guppi_scanid = header['OBSID']
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
    
    #For the UNIX time stamp calculation
    ntimes = (header["BLOCSIZE"] * 8) // (header["OBSNCHAN"] * header["NPOL"] * 2 * header["NBITS"])
    start_time_unix = header["SYNCTIME"] + header["PKTIDX"] * header.get("TBIN", 1/header["CHAN_BW"]) * ntimes/header.get("PIPERBLK", ntimes)
    #block_time_span_s = header.get("PIPERBLK", ntimes) * header.get("TBIN", 1/header["CHAN_BW"]) * ntimes/header.get("PIPERBLK", ntimes)
    #times_unix = (start_time_unix + 0.5 * block_time_span_s) + np.arange(n_blocks)*block_time_span_s
    times_unix = (start_time_unix) + np.arange(n_blocks*ntimes)*header.get("TBIN", 1/header["CHAN_BW"])

    #Get MJD time
    stt_imjd = float(header['STT_IMJD'])
    stt_smjd = float(header['STT_SMJD'])
    stt_offs = float(header['STT_OFFS'])
    mjd_start = stt_imjd + ((stt_smjd + stt_offs)/86400.0)
    mjd_now = mjd_start + (tint/(2*86400.0)) # Getting the middle point of the integration
    
    blocksize = header['BLOCSIZE']
    ntsamp_block = int(blocksize/(nant_chans*npols*2*(nbits/8))) # Number of time samples in the block
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

    print(ants)
    ntsamp_tot = tint/chan_timewidth #Total number of coarse time samples in the integration time.
    n_blocks_read = int(ntsamp_tot/ntsamp_block) # Number of blocks to read
    
    if  n_blocks_read > n_blocks:
        print(f"Warning: Given intg_time > duration of the file: {round(n_blocks*ntsamp_block*chan_timewidth)} s, Using maximum time duration of the file")
        n_blocks_read = n_blocks

    band_percentage = float(band)
    bandcenter_percentage = float(band_center)
    assert band_percentage > 0.0 and band_percentage <= 1.0

    band_bottom = bandcenter_percentage - (0.5 * band_percentage) # center case, default
    band_top = bandcenter_percentage + (0.5 * band_percentage) # center case, default
    assert band_bottom > 0.0 and band_bottom <= 1.0
    assert band_top > 0.0 and band_top <= 1.0

    print("The header info for the file: ")
    print(header)
    print(f"Nblocks: {n_blocks},  Number of blocks to read: {n_blocks_read}")
    print(f"MJD start time: {mjd_start}, MJD time now: {mjd_now}")

    # Collecting data from each block into a big array
    data = np.zeros((nant, nchan, int(ntsamp_block*n_blocks_read), npols), dtype = 'complex64')
    print("Started collecting data")
    for i in tq.tqdm(range(n_blocks_read)):
        head_block, data_block = gob.read_next_data_block()
        data[:,:, i*ntsamp_block:(i+1)*ntsamp_block, :] = data_block.reshape(nant, nchan, ntsamp_block, npols)

    # Upchannelize the data
    print("Starting upchannelization part")

    nfine = lfft # Number of fine channels per coarse channel
    nchan_fine = nchan*nfine
    
    # trim time to be a multiple of FFT size
    ntime_total = data.shape[2]
    ntime_total -= ntime_total % lfft
    data = data[:, :, 0:ntime_total, :]

    #Restricting unix time as well
    print(data.shape, len(times_unix))
    times_unix  = times_unix[:ntime_total]
    print(len(times_unix))

    freq_range = freq_end - freq_start
    freq_end = freq_start + freq_range*band_top
    freq_start = freq_start + freq_range*band_bottom

    # Time, frequency resolution and number of time samples after FFT
    chan_timewidth = chan_timewidth*nfine
    ntsampfine = int(data.shape[2]/nfine)
    chan_freqwidth = chan_freqwidth/nfine

    #Getting new unix time values
    times_unix_new = np.zeros(ntsampfine)
    time_index = np.arange(0,len(times_unix), nfine)
    print(times_unix[0], times_unix[1])
    for k,m in enumerate(time_index):
        times_unix_new[k] = np.mean(times_unix[m:m+nfine])


    #Getting the average time sample values
    # Reshaping the data for FFT
    print(f"Reshaping the data for FFT from {data.shape}")
    # reshape to [A,C,Tfine,Tfft,P]
    data = data.reshape(nant, nchan, ntsampfine, nfine, npols)
    
    ## trim channel to minimum:
    # transpose to [A,Tfine,C,Tfft,P]
    data = np.transpose(data, axes=(0,2,1,3,4))
    # reshape to [A,Tfine,C*Tfft,P]
    data = data.reshape(nant, ntsampfine, nchan_fine, npols)
    # select band
    band_lower = int(nchan_fine*band_bottom)
    band_upper = int(nchan_fine*band_top + 0.5)
    band_length = band_upper-band_lower
    ncoarse_chan_required = int(np.ceil(band_length/nfine))
    band_upper_padding = ncoarse_chan_required*nfine - band_length
    print(f"Selecting [{band_bottom}, {band_top}] of the channels data, range [{band_lower}, {band_upper}).")
    data = data[:, :, band_lower:band_upper+band_upper_padding, :]
    # reshape to [A, Tfine, C, Tfft, P]
    data = data.reshape(nant, ntsampfine, ncoarse_chan_required, nfine, npols)

    print(f"FFT of the data ({data.shape} along axis 3)")
    t0 = time.time()
    data = np.fft.fft(data, axis = 3)
    data = np.fft.fftshift(data, axes = 3)
    t1 = time.time()
    print(f"FFT done, took {t1-t0} s")

    print(f"The channelized datashape [A, Tfine, C, Cfine, P]: {data.shape}")
    data = data.reshape(nant, ntsampfine, ncoarse_chan_required*nfine, npols)
    if band_upper_padding > 0:
        data = data[:, :, 0:-band_upper_padding, :] # discard extra data
    print(f"The channelized datashape collapsed and filtered [A, Tfine, Cfine, P]: {data.shape}")

    nchan = data.shape[2]
    freq_axis = np.linspace(freq_start, freq_end, nchan) #New Upchannelized frequency channels


    return data, times_unix_new, freq_axis


    
if __name__ == "__main__":
    """
    dat_file = sys.argv[1]
    band = float(sys.argv[2])
    band_center = float(sys.argv[3])
    lfft = int(sys.argv[4])
    tint = float(sys.argv[5])
    recipe_file =  sys.argv[6]

    print(dat_file)
    chan_data, times, freqs = get_chan_data(dat_file, band, band_center, lfft, tint)
    print(chan_data.shape, freqs.shape, times.shape)
    print(freqs[0], freqs[-1])
    print(times[0], times[1])

    plot_antenna(chan_data)
    """
    dat_stem = sys.argv[1]
    recipe_file =  sys.argv[2]
    chan_num = 32 #50
    header = collect_header(dat_stem)

    data, times_unix, freq_info = collect_sing_chan_data(dat_stem, chan_num)
    print(freq_info)
    #plt.plot(np.mean(data[0,:,0], '.')
    #plt.show()
    data_fft, times_unix_fft, freqs_fft = fft_sing_chan(data, times_unix, freq_info, header, 262144)
    
    #data_ant = (np.abs(data_fft[0,...])**2).sum(axis = 2)
    #corr = data_fft[0,...]*np.conjugate(data_fft[1,...])
    #corr = np.mean(corr[:,:,0], axis = 0)



    
    #plt.subplot(1,2,1)
    #spec_ant = np.abs(np.mean(data_ant, axis = 0))
    #plt.plot(freqs_fft, spec_ant, '.', linestyle = 'solid')
    #plt.title("single antenna")
    #plt.xlabel("Frequency")
    #plt.ylabel("Power (a.u.)")
    #plt.show()
    
    #print((times_unix_fft - times_unix_fft[0])*1e+3)
    #plot_antenna(data_fft)
    recp_ob = Recipe(recipe_file)
    #np.set_printoptions(precision=6)
    #print(recp_ob.time_array-recp_ob.time_array[0])
    #print(times_unix_fft-times_unix_fft[0])

    lfreq = 3032.2496 #6669.0#3019.7496
    hfreq = 3032.2506 #6669.370#3019.7504
    llim = np.where(np.abs(freqs_fft-lfreq)<0.00016)[0]
    ulim = np.where(np.abs(freqs_fft-hfreq)<0.00016)[0]
    llim = int(np.mean(llim))
    ulim  = int(np.mean(ulim))

    #Looking at incoherent beam
    print("Forming incoherent beam")
    incoh_pow = incoherent_bf_voltage(data_fft) 
    plot_all(incoh_pow[:, llim:ulim], freqs_fft[llim:ulim], times_unix_fft.copy())

    for beam in range(recp_ob.nbeams): 
        print(f"Forming beam number: {beam}")
        pow = coherent_bf_voltage(data_fft, beam, times_unix_fft, freqs_fft, recp_ob)

        """
        spec_bf = np.abs(np.mean(pow, axis = 0))    
        plt.subplot(1,2,2)
        plt.plot(freqs_fft, spec_bf, '.', linestyle ='solid')
        plt.title("BF data")
        plt.xlabel("Frequency")
        plt.show()
        """
        point = [recp_ob.ras[beam], recp_ob.decs[beam]]
        bore = [recp_ob.phase_ra, recp_ob.phase_dec]

        plot_all(pow[:, llim:ulim], freqs_fft[llim:ulim], times_unix_fft.copy(), bore, point, beam)