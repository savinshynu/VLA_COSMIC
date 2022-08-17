from astropy import units as u
import numpy as np
import setigen as stg
import os
#os.environ['SETIGEN_ENABLE_GPU'] = '1'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'



n_pols = 2
n_bits = 8
n_ant = 2 #27
n_chan_perant = 1024 #128
n_time = 1024 #32768 #1024


sample_rate = 2.048e9 #64e+6 #2.048e9

# Channel bandwidth  = sample_rate/(num_branches)  or sample_rate/(2*n_chan_perant)

# Adjust the filesize here. The idea is to specifiy the filesize and send out maximum blocks as possible
# Each block can be configured in a flexible way with varying antennas, channels and time samples

file_bytesize = (2**30) #1.0*(2**30)
block_bytesize = (n_time*n_ant*n_chan_perant*n_pols*2*n_bits)//8

print(block_bytesize)

n_obschan = n_ant*n_chan_perant
blocks_per_file = file_bytesize // block_bytesize

#delays = np.array([int(i*5e-9*sample_rate) for i in range(n_ant)]) # as samples
delays = np.array([int(i*5e-9*sample_rate) for i in range(n_ant)]) # as samples
antenna_array = stg.voltage.MultiAntennaArray(num_antennas=n_ant,
                                    sample_rate=sample_rate*u.Hz,
                                    fch1=2*u.GHz,
                                    ascending=True,
                                    num_pols=2,
                                    delays=delays)



freq = np.linspace(2000e6,2032e6,32)*u.Hz


#snr_ar = np.linspace(10,50,32)

sig_drift = np.linspace(5,55,32)*u.Hz/u.s

for stream in antenna_array.bg_streams:
        stream.add_noise(v_mean=0,
                        v_std=1)
        
        for i in range(freq.shape[0]):
            stream.add_constant_signal(f_start=freq[i],
                                drift_rate=sig_drift[i],
                                level=0.002)

digitizer = stg.voltage.RealQuantizer(target_fwhm=32,
                                      num_bits=8)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=8,
                                             num_branches=n_chan_perant*2)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32,
                                           num_bits=n_bits)


#set the num_chans to specifiy how much channels to write to 
#num_chans=  n_chan_perant usuallly edit if needed
rvb = stg.voltage.RawVoltageBackend(antenna_array,
                                    digitizer=digitizer,
                                    filterbank=filterbank,
                                    requantizer=requantizer,
                                    start_chan=0,
                                    num_chans=  n_chan_perant,
                                    block_size=block_bytesize,
                                    blocks_per_file=blocks_per_file,
                                    num_subblocks=32)

"""
rvb.record(
    output_file_stem='synth_guppi_postage',
    num_blocks=(2**29)//block_bytesize,
    length_mode='num_blocks',
    header_dict={
        'TELESCOP': 'SETIGEN',
        'OBSID'   : 'SYNTHETIC',
    },
    verbose=False,
    load_template=False
)
"""


rvb.record(
            output_file_stem='synth_coh_test',
            obs_length = 5.0,
            length_mode='obs_length',
            header_dict={
                        'TELESCOP': 'SETIGEN',
                        'OBSID'   : 'SYNTHETIC',
                        },
            verbose=False,
            load_template=False
            )

