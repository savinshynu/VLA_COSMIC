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
from blimpy import GuppiRaw
import numpy as np
import matplotlib
import tqdm as tq
from matplotlib import pyplot as plt
import argparse
from sliding_rfi_flagger import flag_rfi

# matplotlib.use('Tkagg')

def main(args):
    
    tbeg = time.time()

    # Collecting the data from the guppi raw files and
    # saving them to an array

    filename = args.dat_file #Input guppi raw file 

    gob = GuppiRaw(filename) #Instantiating a guppi object

    header = gob.read_first_header() # Reading the first header 
    n_blocks = gob.n_blocks    # Number of blocks in the raw file
    obsid = header['OBSID']
    nant_chans = int(header['OBSNCHAN'])
    nant = int(header['NANTS'])
    nbits = int(header['NBITS'])
    npols = int(header['NPOL'])
    nchan = int(nant_chans/nant)
    freq_mid = header['OBSFREQ'] #stopping frequency 
    del_t = header['TBIN']
    del_f = header['CHAN_BW']
    freq_start = freq_mid - ((nchan*del_f)/2.0)
    freq_end = freq_mid + ((nchan*del_f)/2.0)
    freq = np.arange(freq_start, freq_end, del_f) #Coarse channels
    blocksize = header['BLOCSIZE']
    ntsamp_block = int(blocksize/(nant_chans*npols*2*(nbits/8))) # Number of time samples in the block
    ants = header['ANTNMS00']
    ants = ants.split(',')
    ntsamp_tot = args.tint/del_t #Total number of coarse time samples in the integration time.
    n_blocks_read = int(ntsamp_tot/ntsamp_block) # Number of blocks to read
    
    if  n_blocks_read > n_blocks:
        print(f"Warning: Given intg_time > duration of the file: {round(n_blocks*ntsamp_block*del_t)} s, Using maximum time duration of the file")
        n_blocks_read = n_blocks

    print("The header info for the file: ")
    print(header)
    print(f"Nblocks: {n_blocks},  Number of blocks to read: {n_blocks_read}")
   

    # Collecting data from each block into a big array
    data = np.zeros((nant, nchan, int(ntsamp_block*n_blocks_read), npols), dtype = 'complex64')
    print("Started collecting data")
    for i in tq.tqdm(range(n_blocks_read)):
        head_block, data_block = gob.read_next_data_block()
        data[:,:, i*ntsamp_block:(i+1)*ntsamp_block, :] = data_block.reshape(nant, nchan, ntsamp_block, npols)

    #Seperating out the data from two antennas into a different array and changing their order
    for ant1 in range(0, nant):
        for ant2 in range(ant1+1, nant):
            plot_func(
                np.transpose(data[ant1,...], axes = (1,0,2)),
                np.transpose(data[ant2,...], axes = (1,0,2)),
                args,
                [ants[ant1], ants[ant2]],
                npols,
                nchan,
                del_t,
                del_f,
                freq_start,
                freq_end,
                obsid
            )
    
    tend = time.time()
    print(f"Total processing time: {(tend-tbeg)/60.0} min")

def plot_func(
    data1,
    data2,
    args,
    ants,
    npols,
    nchan,
    del_t,
    del_f,
    freq_start,
    freq_end,
    obsid
):
    antpair_str = f"{ants[0]}-{ants[1]}"
    print(f"Processing data from baseline {antpair_str}")
    print(f"The data shape of single antenna data: {data1.shape}")

    #Upchannelization part
    print("Starting upchannelization part")

    nfine = args.lfft # Number of fine channels per coarse channel

    # nfine has to be a factor of the total time samples
    ## so trim it
    ntime_total = data1.shape[0]
    ntime_total -= ntime_total % nfine
    data1 = data1[0:ntime_total, ...]
    data2 = data2[0:ntime_total, ...]

    check =  data1.shape[0] % nfine
    if  check != 0:
        sys.exit(f"The total time samples {data1.shape[0]} should be divisible by number of fine channels {nfine}")


    # Time, frequency resolution and number of time samples after FFT
    del_t_new = del_t*nfine
    ntsamp_new = int(data1.shape[0]/nfine)
    del_f_new = del_f/nfine
    freq_new = np.arange(freq_start, freq_end, del_f_new) #New Upchannelized frequency channels

    # Reshaping the data for FFT
    print("Reshaping the data for FFT")
    chan_dat1 = data1.reshape(int(data1.shape[0]/nfine), nfine, nchan, npols)
    chan_dat2 = data2.reshape(int(data2.shape[0]/nfine), nfine, nchan, npols)


    # Conduct the FFT of the data and an fftshift after that
    print("FFT of the data from each antenna,this might take a while (~3 min))")

    t0 = time.time()
    chan_dat1 = np.fft.fft(chan_dat1, axis = 1) 
    chan_dat1 = np.fft.fftshift(chan_dat1, axes = 1)

    chan_dat2 = np.fft.fft(chan_dat2, axis = 1) 
    chan_dat2 = np.fft.fftshift(chan_dat2, axes = 1)
    t1 = time.time()
    print(f"FFT done, took {t1-t0} s")
    

    print(f"The channelized datashape: {chan_dat1.shape}")

    print("Calculating auto and crosscorrelation")

    #autocorrelation part
    autocorr1 = chan_dat1*np.conjugate(chan_dat1) # antenna1
    autocorr2 = chan_dat2*np.conjugate(chan_dat2) # antenna2

    # Cross correlation part
    cross_corr = chan_dat1*np.conjugate(chan_dat2) # First antenna data times the conjugate of the second antenna

    t2 = time.time() 
    print(f"Correlation done in {t2-t1} s")


    print(f"Averaging data")    

    # Average spectra across specified time intervals
    mean_autocorr_spec1 = np.mean(autocorr1, axis = 0) #antenna 1 autocorrelation
    mean_autocorr_spec2 = np.mean(autocorr2, axis = 0) #antenna 2 autocorrelation

    mean_crosscorr_spec = np.mean(cross_corr, axis = 0) # Cross correlations

    #mid freq range
    midi = int((nchan*nfine/2) - 16*nfine)
    mide = int((nchan*nfine/2) + 16*nfine) 

    #Plotting the phase and amplitude of the autocorrelation for 2 antennas

    print('Plotting the phase and amplitude of the autocorrelation')
    fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize = (12,10))

    axs[0,0].plot(freq_new, 10*np.log10(np.abs(mean_autocorr_spec1[...,0].flatten(order = 'F'))), label = 'pol 0')
    axs[0,0].plot(freq_new, 10*np.log10(np.abs(mean_autocorr_spec1[...,1].flatten(order = 'F'))), label = 'pol 1')
    axs[0,0].set_ylabel("Amplitude log scale (a.u.)")
    axs[0,0].set_xlabel("Frequency (MHz)")
    #axs[0,0].set_xlim(freq_new[midi], freq_new[mide])
    axs[0,0].set_title(f"Autocorrelation : {ants[0]}")
    axs[0,0].legend(loc = 'upper right')

    axs[0,1].plot(freq_new, np.angle(mean_autocorr_spec1[...,0].flatten(order = 'F'), deg = True),  '.', label = 'pol 0')
    axs[0,1].plot(freq_new, np.angle(mean_autocorr_spec1[...,1].flatten(order = 'F'), deg = True), '.', label = 'pol 1')
    axs[0,1].set_ylabel("Phase (degrees)")
    axs[0,1].set_xlabel("Frequency (MHz)")
    axs[0,1].set_xlim(freq_new[midi], freq_new[mide])
    axs[0,1].set_title(f"Autocorrelation : {ants[0]}")
    axs[0,1].legend(loc = 'upper right')

    axs[1,0].plot(freq_new, 10*np.log10(np.abs(mean_autocorr_spec2[...,0].flatten(order = 'F'))), label = 'pol 0')
    axs[1,0].plot(freq_new, 10*np.log10(np.abs(mean_autocorr_spec2[...,1].flatten(order = 'F'))), label = 'pol 1')
    axs[1,0].set_ylabel("Amplitude log scale (a.u.)")
    axs[1,0].set_xlabel("Frequency (MHz)")
    #axs[1,0].set_xlim(freq_new[midi], freq_new[mide])
    axs[1,0].set_title(f"Autocorrelation : {ants[1]}")
    axs[1,0].legend(loc = 'upper right')

    axs[1,1].plot(freq_new, np.angle(mean_autocorr_spec2[...,0].flatten(order = 'F'), deg = True),  '.', label = 'pol 0')
    axs[1,1].plot(freq_new, np.angle(mean_autocorr_spec2[...,1].flatten(order = 'F'), deg = True), '.', label = 'pol 1')
    axs[1,1].set_ylabel("Phase (degrees)")
    axs[1,1].set_xlabel("Frequency (MHz)")
    axs[1,1].set_xlim(freq_new[midi], freq_new[mide])
    axs[1,1].set_title(f"Autocorrelation : {ants[1]}")
    axs[1,1].legend(loc = 'upper right')

    fig.suptitle(f"File: {args.dat_file} \n ObsID: {obsid}")

    if args.plot:
        plt.show()
    else:
        plt.savefig(f"auto_corr_{os.path.basename(args.dat_file)}_{antpair_str}.png", dpi = 150)
        plt.close()


    #Plotting the phase and amplitude of the cross correlation

    print('Plotting the phase and amplitude of the cross correlation')

    fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize = (12,10))

    axs[0,0].plot(freq_new, np.angle(mean_crosscorr_spec[...,0].flatten(order = 'F'), deg = True),  '.', label = 'pol 0')
    axs[0,0].set_ylabel("Phase (degrees)")
    axs[0,0].set_xlabel("Frequency (MHz)")
    axs[0,0].set_xlim(freq_new[midi], freq_new[mide])
    axs[0,0].set_title(f"Crosscorrelation : {antpair_str}")
    axs[0,0].legend(loc = 'upper right')

    axs[0,1].plot(freq_new, np.angle(mean_crosscorr_spec[...,1].flatten(order = 'F'), deg = True), '.', label = 'pol 1')
    axs[0,1].set_ylabel("Phase (degrees)")
    axs[0,1].set_xlabel("Frequency (MHz)")
    axs[0,1].set_xlim(freq_new[midi], freq_new[mide])
    axs[0,1].set_title(f"Crosscorrelation : {antpair_str}")
    axs[0,1].legend(loc = 'upper right')

    axs[1,0].plot(freq_new, 10*np.log10(np.abs(mean_crosscorr_spec[...,0].flatten(order = 'F'))), label = 'pol 0')
    axs[1,0].set_ylabel("Amplitude (a.u.)")
    axs[1,0].set_xlabel("Frequency (MHz)")
    axs[1,0].set_title(f"Crosscorrelation : {antpair_str}")
    #axs[1,0].set_xlim(freq_new[midi], freq_new[mide])
    axs[1,0].legend(loc = 'upper right')

    axs[1,1].plot(freq_new, 10*np.log10(np.abs(mean_crosscorr_spec[...,1].flatten(order = 'F'))), label = 'pol 1')
    axs[1,1].set_ylabel("Amplitude (a.u.)")
    axs[1,1].set_xlabel("Frequency (MHz)")
    #axs[1,1].set_xlim(freq_new[midi], freq_new[mide])
    axs[1,1].set_title(f"Crosscorrelation : {antpair_str}")
    axs[1,1].legend(loc = 'upper right')
    
    fig.suptitle(f"File: {args.dat_file} \n ObsID: {obsid}")

    if  args.plot:
        plt.show()
    else:
        plt.savefig(f"cross_corr_{os.path.basename(args.dat_file)}_{antpair_str}.png", dpi = 150)
        plt.close()

    

    #Conduct an ifft of the crosscorrelated spectra to get the time delay plots
    
    if  args.time_delay:
        
        print('ifft of the crosscorrelated spectra')    
        mean_crosscorr_pol0 = mean_crosscorr_spec[...,0].flatten(order = 'F')
        mean_crosscorr_pol1 = mean_crosscorr_spec[...,1].flatten(order = 'F')

        spec_mean_crosscorr_pol0 = np.abs(mean_crosscorr_pol0)
        spec_mean_crosscorr_pol1 = np.abs(mean_crosscorr_pol1)
        
        #Conducting a step of RFI removal using sliding median window before doing ifft
        # Threshold for RFI removal
        threshold = 3
        
        #Getting bad channels
        bad_chan0 = flag_rfi(spec_mean_crosscorr_pol0, int(nfine/6), threshold)
        bad_chan1 = flag_rfi(spec_mean_crosscorr_pol1, int(nfine/6), threshold)
        
        print(bad_chan0.shape[0], bad_chan1.shape[0])
        
        ##Zeroing bad channels
        mean_crosscorr_pol0[bad_chan0[:,0]] = 0
        mean_crosscorr_pol1[bad_chan1[:,0]] = 0

        #FFT of the spectra
        mean_crosscorr_pol0_ifft = np.fft.ifft(mean_crosscorr_pol0)
        mean_crosscorr_pol1_ifft = np.fft.ifft(mean_crosscorr_pol1) 
        
        #FFT shift of the data
        mean_crosscorr_pol0_ifft = np.fft.ifftshift(mean_crosscorr_pol0_ifft)
        mean_crosscorr_pol1_ifft = np.fft.ifftshift(mean_crosscorr_pol1_ifft)
        


        #Defining  total frequency channels and fine channel bandwidths in Hz to get the time lags
        tlags = np.fft.fftfreq(nchan*nfine,(del_f/nfine)*1e+6)
        tlags = np.fft.fftshift(tlags)*1e+9 #Converting the time lag into us
        tmax_pol0 = np.argmax(10*np.log(np.abs(mean_crosscorr_pol0_ifft)))
        tmax_pol1 = np.argmax(10*np.log(np.abs(mean_crosscorr_pol1_ifft)))
          
        print('Plotting time delay') 
        fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True, figsize = (10,8))

        ax0.plot(tlags, 10*np.log(np.abs(mean_crosscorr_pol0_ifft)), label = 'pol 0')
        ax0.set_ylabel("Power (a.u) log scale")
        ax0.set_xlabel(f"Time lags (delta t = {tlags[1] -tlags[0]}) ns)")
        ax0.set_title(f"{antpair_str}: time delay = {tlags[tmax_pol0]} ns")
        ax0.legend(loc = 'upper right')

        ax1.plot(tlags, 10*np.log(np.abs(mean_crosscorr_pol1_ifft)), label = 'pol 1')
        ax1.set_ylabel("Power (a.u) log scale")
        ax1.set_xlabel(f"Time lags (delta t = {tlags[1] -tlags[0]} ns)")
        ax1.set_title(f"{antpair_str}: time delay = {tlags[tmax_pol1]} ns")
        ax1.legend(loc = 'upper right')

        fig.suptitle(f"File: {args.dat_file} \n ObsID: {obsid}")

        if args.plot:
            plt.show()
        else:
            plt.savefig(f"time_delay_{os.path.basename(args.dat_file)}_{antpair_str}.png", dpi = 150)
            plt.close()


    #Proceed if needed to track a channel as a function of time
    if args.track:

        #Tracking a channel as a function of time
        chan = int(input("Enter the channel (MHz) to track:"))
        #chan = np.where((abs(freq_new-chan) < 0.002))[0]
        #if len(chan) == 0:
        #   sys.exit("No such channel exist, cannot track")
        #print(chan)
        cs = int(chan/nfine)
        fn = int(chan % nfine)
    
        fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True, figsize = (10,8))
        
        ax0.plot(np.angle(cross_corr[:,fn,cs,0] ,deg = True), '.', label = 'pol 0')
        ax0.set_ylabel("Phase (degrees)")
        ax0.set_xlabel(f"Time samples (delta t = {del_t_new} s)")
        ax0.set_title(f"Crosscorrelation : {antpair_str}")
        ax0.legend(loc = 'upper right')

        ax1.plot(np.angle(cross_corr[:,fn,cs,1] ,deg = True), '.', label = 'pol 1')
        ax1.set_ylabel("Phase (degrees)")
        ax1.set_xlabel(f"Time samples (delta t = {del_t_new} s)")
        ax1.set_title(f"Crosscorrelation : {antpair_str}")
        ax1.legend(loc = 'upper right')
       
        fig.suptitle(f"File: {args.dat_file} \n ObsID: {obsid}")

        if args.plot:
            plt.show()
        else:
            plt.savefig(f"chan_trac_{os.path.basename(args.dat_file)}_{antpair_str}.png", dpi = 150)
            plt.close()


if __name__ == '__main__':
    
    # Argument parser taking various arguments
    parser = argparse.ArgumentParser(
        description='Reads guppi rawfiles, upchannelize, conducts auto and crosscorrelation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--dat_file', type = str, required = True, help = 'GUPPI raw file to read in')
    parser.add_argument('-f','--lfft', type = int, required = True, default = 120,  help = 'Length of FFT, default:120')
    parser.add_argument('-i', '--tint', type = float, required = True, help = 'Time to integrate in (s), default: whole file duration')
    parser.add_argument('-td', '--time_delay', action = 'store_true', help = 'If there are fringes, plot/save the time delay plot. An RFI filtering is conductted before the IFFT')
    parser.add_argument('-p', '--plot', action = 'store_true', help = 'plot the figures, otherwise save figures to working directory')
    parser.add_argument('-t', '--track', action = 'store_true', help = 'Track a channel as a function of time, need to enter a RFI free channel after inspection')

    args = parser.parse_args()


    main(args)

