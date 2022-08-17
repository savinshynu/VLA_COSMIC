"""
Written by Savin Shynu Varghese:
    
A script to load dual antenna data from a single guppi raw file,
upchannelize the data from each antenna, conducts autocorrelation,
cross correlation and make diagnostic plots.

Currently set up to take one raw file. Uses blimpy for reading raw files and 
numpy for FFT (which is slower). 
"""

import sys,os
import time
from blimpy import GuppiRaw
import numpy as np
import matplotlib
import tqdm as tq
from matplotlib import pyplot as plt
matplotlib.use('Tkagg')
import argparse

# Collecting the data from the guppi raw files and
# saving them to an array

filename = sys.argv[1] #Input guppi raw file 

gob = GuppiRaw(filename) #Instantiating a guppi object

header = gob.read_first_header() # Reading the first header 
n_blocks = gob.n_blocks    # Number of blocks in the raw file
nant_chans = int(header['OBSNCHAN'])
nant = int(header['NANTS'])
nbits = int(header['NBITS'])
npols = int(header['NPOL'])
nchan = int(nant_chans/nant)
del_t = header['TBIN']
del_f = header['CHAN_BW']
blocksize = header['BLOCSIZE']
ntsamp_block = int(blocksize/(nant_chans*npols*2*(nbits/8))) # Number of time samples in the block
ants = header['ANTNMS00']
ants = ants.split(',')

print("The header info for the file: ")
print(header)
print(f"Nblocks: {n_blocks}")


# Collecting data from each block into a big array
data = np.zeros((nant, nchan, int(ntsamp_block*n_blocks), npols), dtype = 'complex64')
print("Started collecting data")
for i in tq.tqdm(range(n_blocks)):
    head_block, data_block = gob.read_next_data_block()
    data[:,:, i*ntsamp_block:(i+1)*ntsamp_block, :] = data_block.reshape(nant, nchan, ntsamp_block, npols)



#Seperating out the data from two antennas into a different array and changing their order
data1 = np.transpose(data[0,...], axes = (1,0,2))
data2 = np.transpose(data[1,...], axes = (1,0,2))

print(f"The data shape of single antenna data: {data1.shape}")

#Delete the data array
del data


#Upchannelization part
print("Starting upchannelization part")

nfine = 120 # Number of fine channels per coarse channel

# nfine has to be a factor of the total time samples

check =  data1.shape[0] % nfine
if  check != 0:
    sys.exit(f"The total time samples {data1.shape[0]} should be divisible by number of fine channels {nfine}")

# Reshaping the data for FFT
print("Reshaping the data for FFT")
chan_dat1 = data1.reshape(int(data1.shape[0]/nfine), nfine, nchan, npols)
chan_dat2 = data2.reshape(int(data2.shape[0]/nfine), nfine, nchan, npols)


# Conduct the FFT of the data and an fftshift after that
print("FFT of the data from each antenna,this might take a while)")

t0 = time.time()
chan_dat1 = np.fft.fft(chan_dat1, axis = 1) 
chan_dat1 = np.fft.fftshift(chan_dat1, axes = 1)

chan_dat2 = np.fft.fft(chan_dat2, axis = 1) 
chan_dat2 = np.fft.fftshift(chan_dat2, axes = 1)
t1 = time.time()
print(f"FFT done, took {t1-t0} s")

print(f"The channelized datashape: {chan_dat1.shape}")


#autocorrelation part
autocorr1 = chan_dat1*np.conjugate(chan_dat1)
autocorr2 = chan_dat2*np.conjugate(chan_dat2)

# Cross correlation part
cross_corr = chan_dat1*np.conjugate(chan_dat2) # First antenna data times the conjugate of the second antenna


#plt.plot(np.abs(np.mean(cross_cor[:,:,:,0], axis = 0).T.flatten()+np.mean(cross_cor[:,:,:,1], axis = 0).T.flatten()))
#plt.plot(np.abs(np.mean(chan_dat1[:,:,0], axis = 0)))
#plt.show()


# Averaged spectra across time
mean_autocorr_spec1 = np.mean(autocorr1, axis = 0) 
mean_autocorr_spec2 = np.mean(autocorr2, axis = 0)

mean_crosscorr_spec = np.mean(cross_corr, axis = 0)


#Plotting the phase and amplitude of the autocorrelation of the first antenna

plt.plot(np.angle(mean_autocorr_spec1[...,0].flatten(order = 'F'), deg = True),  '.', label = 'pol 0')
plt.plot(np.angle(mean_autocorr_spec1[...,1].flatten(order = 'F'), deg = True), '.', label = 'pol 1')
plt.ylabel("Phase (degrees)")
plt.xlabel("Frequency channels")
plt.title(f"Autocorrelation : {ants[0]}")
plt.legend()
plt.show()

plt.plot(np.abs(mean_autocorr_spec1[...,0].flatten(order = 'F')), label = 'pol 0')
plt.plot(np.abs(mean_autocorr_spec1[...,1].flatten(order = 'F')), label = 'pol 1')
plt.ylabel("Amplitude (a.u.)")
plt.xlabel("Frequency channels")
plt.title(f"Autocorrelation : {ants[0]}")
plt.legend()
plt.show()



plt.plot(np.angle(mean_crosscorr_spec[...,0].flatten(order = 'F'), deg = True),  '.', label = 'pol 0')
#plt.plot(np.angle(mean_crosscorr_spec[...,1].flatten(order = 'F'), deg = True), '.', label = 'pol 1')
#plt.plot(phase_avg, '.', label = 'pol 0')
plt.ylabel("Phase (degrees)")
plt.xlabel("Frequency channels")
plt.title(f"Crosscorrelation : {ants[0]}- {ants[1]}")
plt.legend()
plt.show()




plt.plot(np.abs(mean_crosscorr_spec[...,0].flatten(order = 'F')), label = 'pol 0')
#plt.plot(np.abs(mean_crosscorr_spec[...,1].flatten(order = 'F')), label = 'pol 1')
plt.ylabel("Amplitude (a.u.)")
plt.xlabel("Frequency channels")
plt.title(f"Crosscorrelation : {ants[0]}-{ants[1]}")
plt.legend()
plt.show()


#Tracking a channel as a function of time
chan = int(input("Enter the channel number to track:"))

cs = int(chan/nfine)
fn = int(chan % nfine)

plt.plot(np.angle(cross_corr[:,fn,cs,0] ,deg = True), '.', label = 'pol 0')
plt.ylabel("Phase (degrees)")
plt.xlabel("Time samples (delta t = 0.12 ms)")
plt.title(f"Crosscorrelation : {ants[0]}-{ants[1]}")
plt.legend()
plt.show()



np.save("mean_spectra_"+os.path.basename(filename), mean_crosscorr_spec)
"""
# Let's do one more fft of the averaged crosscorrelated spectra

cross_spec_pol0 = mean_crosscorr_spec[...,0].flatten(order = 'F')
fft_cross_spec_pol0 = np.fft.fft(cross_spec_pol0)
fft_cross_spec_pol0 = np.fft.fftshift(fft_cross_spec_pol0)

plt.plot(np.abs(fft_cross_spec_pol0))
plt.show()

"""


