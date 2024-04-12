"""
Prints the Guppi raw header files

"""


import sys
from blimpy import GuppiRaw

filename = sys.argv[1] #Input guppi raw file 

gob = GuppiRaw(filename) #Instantiating a guppi object
header = gob.read_first_header() # Reading the first header 
n_blocks = gob.n_blocks    # Number of blocks in the raw file
guppi_scanid = header.get('OBSID', None)
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

src_name = header['SRC_NAME']
ra = header['RA_STR']
dec = header['DEC_STR']
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


print("The header info for the file: ")
print(header)
print("Some useful metadata \n")
print(f"No. of blocks: {n_blocks} \n\
        Blocksize: {blocksize} \n\
        No. of time samples in a block: {ntsamp_block} \n\
        Guppi_scanid: {guppi_scanid} \n\
        Source :{src_name} \n\
        Ra: {ra}, Dec :{dec} \n\
        Number of antenna channels: {nant_chans} \n\
        Number of antennas: {nant} \n\
        Antenna list : {ants} \n\
        Data Bitsize: {nbits} \n\
        Number of pols : {npols} \n\
        Number of channels: {nchan} \n\
        Center frequency : {freq_mid} \n\
        Time resolution: {chan_timewidth} s \n\
        Frequency resolution = {chan_freqwidth} MHz \n\
        Starting frequency : {freq_start} MHz \n\
        Stopping frequency: {freq_end} MHz\n\
        MJD start time: {mjd_start} \n\
         ") 

