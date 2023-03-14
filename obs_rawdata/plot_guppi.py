import sys,os
import time
import json
from blimpy import GuppiRaw
import numpy as np
import matplotlib
import tqdm as tq
from matplotlib import pyplot as plt
import argparse


matplotlib.use('Tkagg')

def main(dat_file,tint):
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

   # Plotting the data fromt the first 2 antennas:

    plt.plot(np.abs(np.mean(data[0,31:32,:100,0], 0)), '.', label = "ea00")
    plt.plot(np.abs(np.mean(data[1,31:32,:100,0], 0)), '.', label = "ea01")
    plt.title("Time Series")
    plt.xlabel("Time samples")
    plt.ylabel("Power (a.u.)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    dat_file = sys.argv[1]
    print(dat_file)
    tint = 0.5
    main(dat_file,tint)
