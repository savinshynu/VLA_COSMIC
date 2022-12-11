"""
Written by Savin Shynu Varghese:
    
A script to load dual antenna data from a single guppi raw file,
upchannelize the data from each antenna, conducts autocorrelation,
cross correlation and make diagnostic plots.

Currently set up to take one raw file. Uses blimpy for reading raw files and 
numpy for FFT (which is slower). Takes 3 minutes on average to do the FFT for a single file
data
"""

import sys,os
import time
import json
from blimpy import GuppiRaw
import numpy as np
import matplotlib
import tqdm as tq
from matplotlib import pyplot as plt
import argparse
from sliding_rfi_flagger import flag_rfi
from compute_uvw import vla_uvw
from mad import median_absolute_deviation as mad


# matplotlib.use('Tkagg')

def main(
    dat_file,
    log_file,
    band,
    band_center,
    lfft,
    tint,
    time_delay,
    savefig_directory,
    crosscorr_channel_time_promptplot,
    autocorr_show_phase = False,
    autocorr_cross_polarizations = False
):
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

    freq_range = freq_end - freq_start
    freq_end = freq_start + freq_range*band_top
    freq_start = freq_start + freq_range*band_bottom

    # Time, frequency resolution and number of time samples after FFT
    chan_timewidth = chan_timewidth*nfine
    ntsampfine = int(data.shape[2]/nfine)
    chan_freqwidth = chan_freqwidth/nfine


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

    source_file_name = os.path.basename(dat_file)
    plot_id = f"{source_file_name}_{freq_axis[0]:0.3f}-{freq_axis[-1]:0.3f}"

    if autocorr_cross_polarizations:
        autocorr_mean_dict = {
            ants[ant1]: np.mean(data[ant1,...]*np.conjugate(data[ant1,:,:,-1::-1]), axis = 0, keepdims=False)
            for ant1 in range(0, nant)
        }
    else:
        autocorr_mean_dict = {
            ants[ant1]: np.mean(data[ant1,...]*np.conjugate(data[ant1,...]), axis = 0, keepdims=False)
            for ant1 in range(0, nant)
        }
    plot_autocorrelations(
        autocorr_mean_dict, # {ant_name: [Chan, Pol]}
        freq_axis,
        plot_id = plot_id,
        savefig_directory = savefig_directory,
        omit_phase = not autocorr_show_phase
    )

    if crosscorr_channel_time_promptplot:
        track_chan = int(input(f"Enter the channel (enumeration) to track [0={freq_axis[0]}, {nchan}={freq_axis[1]}]:"))
        #chan = np.where((abs(freq-chan) < 0.002))[0]
        #if len(chan) == 0:
        #   sys.exit("No such channel exist, cannot track")
        #print(chan)
    

    #Check if the GUPPI file and logfile has the same scanid, otherwise derived value will be wrong
    if guppi_scanid != log_scanid(log_file):
       sys.exit("Make sure the observation log files correspond to the guppi files")

    #Opening a file to save the delays, geodelays and non-geometric delays for each baselines
    dh = open(f"delays_{source_file_name}.csv", "w")
    dh.write(",".join(
            [
                "Baseline",
                "total_pol0",
                "total_pol1",
                "geo",
                "non-geo_pol0",
                "non-geo_pol1"
            ]
        )+"\n")
        
    # Geometric delay terms
    geo_delays = calculate_geo_delay(log_file, mjd_now)

    #Seperating out the data from two antennas into a different array and changing their order
    
    nrows = 2
    if time_delay:
        nrows += 1
    for ant1 in range(0, nant):
        for ant2 in range(ant1+1, nant):
            baseline_str = f"{ants[ant1]}-{ants[ant2]}"
            crosscorr = data[ant1,...]*np.conjugate(data[ant2,...])
            crosscorr_mean = np.mean(crosscorr, axis = 0, keepdims=False) # Cross correlations

            #Plotting the phase and amplitude of the cross correlation
            fig, axs = plt.subplots(nrows, 2, constrained_layout=True, figsize = (10,4*nrows))

            _plot_crosscorrelations(
                autocorr_mean_dict[ants[ant1]], # [Chan, Pol]
                autocorr_mean_dict[ants[ant2]], # [Chan, Pol]
                crosscorr_mean, # [Chan, Pol]
                baseline_str,
                freq_axis,
                axs
            )
            
            if time_delay:
                crosscorr_ifft_power_pol0, crosscorr_ifft_power_pol1, tlags = proc_time_delay(
                    crosscorr_mean,
                    chan_freqwidth,
                    nchan_fine,
                    nfine
                )

                baseline_time_delays, snr = _plot_time_delay(
                    crosscorr_ifft_power_pol0,
                    crosscorr_ifft_power_pol1,
                    tlags,
                    axs[2,0], axs[2,1],
                    baseline_str
                )

                geo_baseline = -(geo_delays[ant1] - geo_delays[ant2]) # sign flipping the delay
                non_geo_baseline = [baseline_time_delays[0] - geo_baseline, baseline_time_delays[1] - geo_baseline]
                #dh.write(f"{baseline_str}   {round(baseline_time_delays[0],3)}   {round(baseline_time_delays[1],3)}   {round(geo_baseline,3)}   {round(non_geo_baseline[0],3)}   {round(non_geo_baseline[1],3)} \n")
                dh.write(f"{baseline_str},{baseline_time_delays[0]:+012.03f},{baseline_time_delays[1]:+012.03f},{geo_baseline:+012.03f},{non_geo_baseline[0]:+012.03f},{non_geo_baseline[1]:+012.03f},{snr[0]:+012.03f}, {snr[1]:+012.03f} \n")

                print(f"Time delay for {baseline_str}: {round(baseline_time_delays[0],3), round(baseline_time_delays[1],3)} (ns), Geo delays: {round(geo_baseline,3)} (ns)")
            
            fig.suptitle(f"File: {plot_id}")

            if savefig_directory is None:
                plt.show()
            else:
                filename = os.path.join(savefig_directory, f"cross_corr_{plot_id}_{baseline_str}.png")
                plt.savefig(filename, dpi = 150)
                plt.close()

            if crosscorr_channel_time_promptplot:
                plot_crosscorrelation_time(
                    crosscorr[:,track_chan,:], # [Time, Pol]
                    baseline_str,
                    chan_timewidth,
                    plot_id,
                    savefig_directory = savefig_directory
                )
    dh.close()
    print(f"Plotted: {plot_id}")


def plot_autocorrelations(
    autocorr_mean_dict, # {ant_name: [Chan, Pol]}
    freq_axis,
    plot_id,
    savefig_directory = None,
    omit_phase = True,
    ncol = 4
):
    nant = len(autocorr_mean_dict)

    nrow = int(np.ceil(nant/ncol))
    if not omit_phase:
        nrow *= 2
    fig, axs = plt.subplots(
        nrow,
        ncol,
        constrained_layout=True,
        figsize = (3*ncol,4*nrow),
        sharex=True,
        sharey=True
    )

    row_index = 0
    col_index = 0
    for ant_name, autocorr_mean in autocorr_mean_dict.items():

        axs[row_index,col_index].plot(freq_axis, 10*np.log10(np.abs(autocorr_mean[:,0])), label = 'pol 0')
        axs[row_index,col_index].plot(freq_axis, 10*np.log10(np.abs(autocorr_mean[:,1])), label = 'pol 1')
        if col_index == 0:
            axs[row_index,col_index].set_ylabel("Amplitude log scale (a.u. dB)")
        if row_index == nrow-1:
            axs[row_index,col_index].set_xlabel("Frequency (MHz)")
        axs[row_index,col_index].set_title(f"Amplitude : {ant_name}")
        axs[row_index,col_index].legend()

        if not omit_phase:
            col_index += 1
            if col_index == ncol:
                col_index = 0
                row_index += 1
            axs[row_index,col_index].plot(freq_axis, np.angle(autocorr_mean[:,0], deg = True),  '.', label = 'pol 0')
            axs[row_index,col_index].plot(freq_axis, np.angle(autocorr_mean[:,1], deg = True), '.', label = 'pol 1')
            if col_index <= 1:
                axs[row_index,col_index].set_ylabel("Phase (degrees)")
            if row_index == nrow-1:
                axs[row_index,col_index].set_xlabel("Frequency (MHz)")
            axs[row_index,col_index].set_title(f"Phase: {ant_name}")
            axs[row_index,col_index].legend()

        col_index += 1
        if col_index == ncol:
            col_index = 0
            row_index += 1

    fig.suptitle(f"Autocorrelations: {plot_id}")

    if savefig_directory is None:
        plt.show()
    else:
        filename = os.path.join(savefig_directory, f"auto_corr_{plot_id}.png")
        plt.savefig(filename, dpi = 150)
        plt.close()


def _plot_crosscorrelations(
    autocorr_mean1, # [Chan, Pol]
    autocorr_mean2, # [Chan, Pol]
    crosscorr_mean, # [Chan, Pol]
    baseline_str,
    freq_axis,
    axes_2by2
):
    #sqrt_autos_pol0 = np.sqrt(autocorr_mean1[:,0]*autocorr_mean2[:,0])
    #sqrt_autos_pol1 = np.sqrt(autocorr_mean1[:,1]*autocorr_mean2[:,1])
    #mean_crosscorr_pol0_coeff = crosscorr_mean[:,0]/sqrt_autos_pol0
    #mean_crosscorr_pol1_coeff = crosscorr_mean[:,1]/sqrt_autos_pol1

    #Plotting the phase and amplitude of the cross correlation
    axes_2by2[0,0].plot(freq_axis, np.angle(crosscorr_mean[:,0], deg = True),  '.', label = 'pol 0')
    axes_2by2[0,0].set_ylabel("Phase (degrees)")
    axes_2by2[0,0].set_xlabel("Frequency (MHz)")
    axes_2by2[0,0].set_title(f"Crosscorrelation : {baseline_str}")
    axes_2by2[0,0].legend()

    axes_2by2[0,1].plot(freq_axis, np.angle(crosscorr_mean[:,1], deg = True), '.', label = 'pol 1')
    axes_2by2[0,1].set_ylabel("Phase (degrees)")
    axes_2by2[0,1].set_xlabel("Frequency (MHz)")
    axes_2by2[0,1].set_title(f"Crosscorrelation : {baseline_str}")
    axes_2by2[0,1].legend()

    axes_2by2[1,0].plot(freq_axis, 10*np.log10(np.abs(crosscorr_mean[:,0])), label = 'pol 0')
    axes_2by2[1,0].set_ylabel("Amplitude (dB)")
    axes_2by2[1,0].set_xlabel("Frequency (MHz)")
    axes_2by2[1,0].set_title(f"Crosscorrelation: {baseline_str}")
    axes_2by2[1,0].legend()

    axes_2by2[1,1].plot(freq_axis, 10*np.log10(np.abs(crosscorr_mean[:,1])), label = 'pol 1')
    axes_2by2[1,1].set_ylabel("Amplitude (dB)")
    axes_2by2[1,1].set_xlabel("Frequency (MHz)")
    axes_2by2[1,1].set_title(f"Crosscorrelation : {baseline_str}")
    axes_2by2[1,1].legend()


def plot_crosscorrelations(
    autocorr_mean1, # [Chan, Pol]
    autocorr_mean2, # [Chan, Pol]
    crosscorr_mean, # [Chan, Pol]
    baseline_str,
    freq_axis,
    plot_id,
    savefig_directory = None
):
    #Plotting the phase and amplitude of the cross correlation
    fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize = (10,8))

    _plot_crosscorrelations(
        autocorr_mean1, # [Chan, Pol]
        autocorr_mean2, # [Chan, Pol]
        crosscorr_mean, # [Chan, Pol]
        baseline_str,
        freq_axis,
        axs
    )
    
    fig.suptitle(f"File: {plot_id}")

    if savefig_directory is None:
        plt.show()
    else:
        filename = os.path.join(savefig_directory, f"cross_corr_{plot_id}_{baseline_str}.png")
        plt.savefig(filename, dpi = 150)
        plt.close()


def proc_time_delay(
    crosscorr_mean,
    chan_freqwidth,
    final_nchan,
    nfine,
    rfi_removal_threshold=3
):
    # Conduct an ifft of the crosscorrelated spectra to get the time delay plots
    nchan = crosscorr_mean.shape[0]
    #Conducting a step of RFI removal using sliding median window before doing ifft
    # Threshold for RFI removal
    rfi_removal_threshold = 3
    
    #Getting bad channels
    bad_chan0 = flag_rfi(np.abs(crosscorr_mean[:,0]), int(nfine/6), rfi_removal_threshold)
    bad_chan1 = flag_rfi(np.abs(crosscorr_mean[:,1]), int(nfine/6), rfi_removal_threshold)
    
    # print(bad_chan0.shape[0], bad_chan1.shape[0])
    
    ##Zeroing bad channels
    crosscorr_mean[bad_chan0[:,0],0] = 0
    crosscorr_mean[bad_chan1[:,0],1] = 0

    #FFT of the spectra
    mean_crosscorr_pol0_ifft = np.fft.ifft(crosscorr_mean[:,0], n = final_nchan)
    mean_crosscorr_pol1_ifft = np.fft.ifft(crosscorr_mean[:,1], n = final_nchan) 
    
    #FFT shift of the data
    mean_crosscorr_pol0_ifft = np.fft.ifftshift(mean_crosscorr_pol0_ifft)
    mean_crosscorr_pol1_ifft = np.fft.ifftshift(mean_crosscorr_pol1_ifft)

    #Defining  total frequency channels and fine channel bandwidths in Hz to get the time lags
    tlags = np.fft.fftfreq(final_nchan,chan_freqwidth*1e+6)
    tlags = np.fft.fftshift(tlags)*1e+9 #Converting the time lag into ns
    crosscorr_ifft_power_pol0 = 10*np.log(np.abs(mean_crosscorr_pol0_ifft))
    crosscorr_ifft_power_pol1 = 10*np.log(np.abs(mean_crosscorr_pol1_ifft))
    return crosscorr_ifft_power_pol0, crosscorr_ifft_power_pol1, tlags


def _plot_time_delay(
    crosscorr_ifft_power_pol0,
    crosscorr_ifft_power_pol1,
    tlags,
    ax0, ax1,
    baseline_str
):
    tmax_pol0 = np.argmax(crosscorr_ifft_power_pol0)
    sig_pol0 = crosscorr_ifft_power_pol0[tmax_pol0]
    noise_pol0 = mad(crosscorr_ifft_power_pol0)
    median_pol0 = np.median(crosscorr_ifft_power_pol0)
    snr_pol0 = (sig_pol0-median_pol0)/noise_pol0

    tmax_pol1 = np.argmax(crosscorr_ifft_power_pol1)
    sig_pol1 = crosscorr_ifft_power_pol1[tmax_pol1]
    noise_pol1 = mad(crosscorr_ifft_power_pol1)
    median_pol1 = np.median(crosscorr_ifft_power_pol1)
    snr_pol1 = (sig_pol1-median_pol1)/noise_pol1

    ax0.plot(tlags, crosscorr_ifft_power_pol0, label = 'pol 0')
    ax0.set_ylabel("Power (a.u) log scale")
    ax0.set_xlabel(f"Time lags (delta t = {tlags[1] -tlags[0]}) ns)")
    ax0.set_title(f"{baseline_str}: time delay = {tlags[tmax_pol0]} ns")
    ax0.legend()

    ax1.plot(tlags, crosscorr_ifft_power_pol1, label = 'pol 1')
    ax1.set_ylabel("Power (a.u) log scale")
    ax1.set_xlabel(f"Time lags (delta t = {tlags[1] -tlags[0]} ns)")
    ax1.set_title(f"{baseline_str}: time delay = {tlags[tmax_pol1]} ns")
    ax1.legend()
    return (tlags[tmax_pol0], tlags[tmax_pol1]), (snr_pol0, snr_pol1)


def plot_time_delay(
    crosscorr_mean, # [Chan, Pol]
    baseline_str,
    chan_freqwidth,
    plot_id,
    savefig_directory = None,
):
    crosscorr_ifft_power_pol0, crosscorr_ifft_power_pol1, tlags = proc_time_delay(
        crosscorr_mean,
        chan_freqwidth,
        nchan_fine,
        nfine
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True, figsize = (10,8))

    tdelay_pol0, tdelay_pol1 = _plot_time_delay(
        crosscorr_ifft_power_pol0,
        crosscorr_ifft_power_pol1,
        tlags,
        ax0, ax1,
        baseline_str
    )

    fig.suptitle(f"File: {plot_id}")

    if savefig_directory is None:
        plt.show()
    else:
        filename = os.path.join(savefig_directory, f"time_delay_{plot_id}_{baseline_str}.png")
        plt.savefig(filename, dpi = 150)
        plt.close()
    return (tdelay_pol0, tdelay_pol1)


def plot_crosscorrelation_time(
    crosscorr_time, # [Time, Pol]
    baseline_str,
    chan_timewidth,
    plot_id,
    savefig_directory = None,
):
    
    fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True, figsize = (10,8))
    
    ax0.plot(np.angle(crosscorr_time[:,0] ,deg = True), '.', label = 'pol 0')
    ax0.set_ylabel("Phase (degrees)")
    ax0.set_xlabel(f"Time samples (delta t = {chan_timewidth} s)")
    ax0.set_title(f"Crosscorrelation : {baseline_str}")
    ax0.legend()

    ax1.plot(np.angle(crosscorr_time[:,1] ,deg = True), '.', label = 'pol 1')
    ax1.set_ylabel("Phase (degrees)")
    ax1.set_xlabel(f"Time samples (delta t = {chan_timewidth} s)")
    ax1.set_title(f"Crosscorrelation : {baseline_str}")
    ax1.legend()
    
    fig.suptitle(f"File: {plot_id}")

    if savefig_directory is None:
        plt.show()
    else:
        plt.savefig(f"chan_trac_{plot_id}.png", dpi = 150)
        plt.close()

def log_scanid(metafile):
    
    fh = open(metafile)
    data = json.load(fh)
    scanid = data['META']['scanid']
    fh.close()
    return scanid

def calculate_geo_delay(metafile, mjd_time):

    fh = open(metafile)
    data =  json.load(fh)
    
    #scanid = data['META']['scanid']
    #mjd_time_now = data['META']['tnow']
    #mjd_time_start = data['META']['tstart']
    #source = data['META']['src']
    ra_deg = data['META']['ra_deg']
    dec_deg = data['META']['dec_deg']
    

    ants_log1 = data['META_arrayConfiguration']['cosmic-gpu-0.1']['ANTNMS00']
    ants_log = ants_log1.split(',')
    
    try:
        ants_log2 = data['META_arrayConfiguration']['cosmic-gpu-0.1']['ANTNMS01']
        ants_log += ants_log2.split(',')
    except KeyError:
        pass
    
    try:
        ants_log3 = data['META_arrayConfiguration']['cosmic-gpu-0.1']['ANTNMS02']
        ants_log += ants_log3.split(',')
    except KeyError:
        pass
    
    nants_log = data['META_arrayConfiguration']['cosmic-gpu-0.1']['NANTS']

    # The VLA array center coordinates in ECEF
    VLA_X = -1601185.4
    VLA_Y = -5041977.5
    VLA_Z = 3554875.9

    XYZ = np.zeros((nants_log,3))

    for i,ant in enumerate(ants_log):
        X = data['AntennaProperties'][ant]['X']
        Y = data['AntennaProperties'][ant]['Y']
        Z = data['AntennaProperties'][ant]['Z']

        XYZ[i,:] = [X + VLA_X, Y + VLA_Y, Z + VLA_Z]
    
    fh.close()
    #Calculating the uvw values
    uvw = vla_uvw(mjd_time, (ra_deg*(np.pi/180.0), dec_deg*(np.pi/180.0)), XYZ)
    
    # Returning only the W term which are the projected baselines towards the source
    # converting the distance(m) to ns (d/c)*1e+9 == d*(10/3)
    return uvw[:,2]*(10.0/3.0)



if __name__ == '__main__':
    
    # Argument parser taking various arguments
    parser = argparse.ArgumentParser(
        description='Reads guppi rawfiles, upchannelize, conducts auto and crosscorrelation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--dat-file', type = str, required = True, help = 'GUPPI raw file to read in')
    parser.add_argument('-l','--log-file', type = str, required = True, help = 'Log-metafile to calculate the geometric delays')
    parser.add_argument('-b','--band', type = float, required = False, default = 1.0,  help = 'Bandwidth to plot specified as a decimal percentage [0.0, 1.0], default:1.0')
    parser.add_argument('-bc','--band-center', type = float, required = False, default = 1.0,  help = 'Bandwidth center to plot specified as a decimal percentage [0.0, 1.0]-`band`, default:0.5')
    parser.add_argument('-f','--lfft', type = int, required = True, default = 120,  help = 'Length of FFT, default:120')
    parser.add_argument('-i', '--tint', type = float, required = True, help = 'Time to integrate in (s), default: whole file duration')
    parser.add_argument('-td', '--time-delay', action = 'store_true', help = 'If there are fringes, plot/save the time delay plot. An RFI filtering is conductted before the IFFT')
    parser.add_argument('-o', '--savefig-directory', type = str, default = None, help = 'Save plots to this directory instead of plotting')
    parser.add_argument('-t', '--track', action = 'store_true', help = 'Track a channel as a function of time, need to enter a RFI free channel after inspection')
    parser.add_argument('-ap', '--autocorr-show-phase', action = 'store_true', help = 'Don\'t omit the phase in the autocorrelation plot')
    parser.add_argument('-ac', '--autocorr-cross-pols', action = 'store_true', help = 'Cross the polarizations for the autocorrelations')

    args = parser.parse_args()

    main(
        args.dat_file,
        args.log_file,
        args.band,
        args.band_center,
        args.lfft,
        args.tint,
        args.time_delay,
        args.savefig_directory,
        args.track,
        autocorr_show_phase = args.autocorr_show_phase,
        autocorr_cross_polarizations = args.autocorr_cross_pols
    )

