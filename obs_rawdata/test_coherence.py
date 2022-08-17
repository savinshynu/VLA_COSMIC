import sys
from blimpy import GuppiRaw
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Tkagg')

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

print("The header info for the file: ")
print(header)
print(f"Nblocks: {n_blocks}")

# Collecting data from each block into a big array
data = np.zeros((nant, nchan, int(ntsamp_block*n_blocks), npols), dtype = 'complex64')
#data = np.zeros((nant, nchan, int(ntsamp_block*2),npols), dtype = 'complex64')
for i in range(n_blocks):
    #for i in range(2):
    head_block, data_block = gob.read_next_data_block()
    #print(head_block['BLKSTART'])
    #print(head_block['NPKT'])
    #print(i)
    data[:,:, i*ntsamp_block:(i+1)*ntsamp_block, :] = data_block.reshape(nant, nchan, ntsamp_block, npols)


# Plotting the average spectra from each antenna in first polarization

avg_spec1 = np.mean(data[0,:,:,0], axis = 1)
avg_spec2 = np.mean(data[1,:,:,0], axis = 1)

plt.plot(np.abs(avg_spec1), label = 'ea11')
plt.plot(np.abs(avg_spec2), label = 'ea22')
plt.title("Averages spectra")
plt.legend()
plt.show()



# Cross correlation part

# The array to save the correlated voltages
cross_corr = np.zeros(data.shape[1:], dtype = 'complex64')

for i in range(cross_corr.shape[1]):
    for j in range(npols):
        cross_corr[:,i,j] = data[0,:,i,j]*np.conjugate(data[0,:,i,j])


# Averaged spectra across time
mean_cross_spec_pol1 = np.mean(cross_corr[:,:,0], axis = 1)
mean_cross_spec_pol2 = np.mean(cross_corr[:,:,1], axis = 1)


plt.plot(np.angle(mean_cross_spec_pol1, deg = True),  '.', label = 'pol 0')
plt.plot(np.angle(mean_cross_spec_pol2, deg = True), '.', label = 'pol 1')
plt.ylabel("Phase (degrees)")
plt.xlabel("Frequency channels")
plt.title("Autocorrelation : ea11")
#plt.title("Crosscorrelation : ea11-ea22")
plt.legend()
plt.show()

plt.plot(np.abs(mean_cross_spec_pol1), label = 'pol 0')
plt.plot(np.abs(mean_cross_spec_pol2), label = 'pol 1')
plt.ylabel("Amplitude (a.u.)")
plt.xlabel("Frequency channels")
plt.title("Autocorrelation : ea11")
#plt.title("Crosscorrelation : ea11-ea22")
plt.legend()
plt.show()

