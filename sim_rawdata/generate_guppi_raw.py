from astropy import units as u
import numpy as np
import setigen as stg
import os
#os.environ['SETIGEN_ENABLE_GPU'] = '1'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'



n_pols = 2
n_bits = 8
n_ant = 22
n_chan_perant = 64 #1024 #128
n_time = 32768 #1024


sample_rate = 128e+6 #2.048e9

# Channel bandwidth  = sample_rate/(num_branches)  or sample_rate/(2*n_chan_perant)

# Adjust the filesize here. The idea is to specifiy the filesize and send out maximum blocks as possible
# Each block can be configured in a flexible way with varying antennas, channels and time samples

file_bytesize = 1.17e+10#(2**30) #1.0*(2**30)
block_bytesize = (n_time*n_ant*n_chan_perant*n_pols*2*n_bits)//8

print(block_bytesize)

n_obschan = n_ant*n_chan_perant
blocks_per_file = int(file_bytesize // block_bytesize)

print(blocks_per_file)


#delays = np.array([int(i*5e-9*sample_rate) for i in range(n_ant)]) # as samples
delays = np.array([int(i*0.0) for i in range(n_ant)]) # as samples
#delays = np.zeros(n_ant)
#antenna_array = stg.voltage.MultiAntennaArray(num_antennas=n_ant,
#                                    sample_rate=sample_rate*u.Hz,
#                                    fch1=3*u.GHz,
#                                    ascending=True,
#                                    num_pols=2,
#                                    delays=delays)

antenna_array = stg.voltage.MultiAntennaArray(num_antennas=n_ant,
                                    sample_rate=sample_rate*u.Hz,
                                    fch1=3*u.GHz,
                                    ascending=True,
                                    num_pols=2, delays = delays)


#freq = np.linspace(3000e+6,3064e+6,64)*u.Hz
freq = np.array([3032.25, 3019.75, 3042.25])*1e+6*u.Hz
print(freq)
"""
#snr_ar = np.linspace(10,50,32)

sig_drift = np.linspace(5,55,32)*u.Hz/u.s
for stream in antenna_array.bg_streams:
        stream.add_noise(v_mean=0,
                        v_std=1)
        
        for i in range(freq.shape[0]):
            stream.add_constant_signal(f_start=freq[i],
                                drift_rate=sig_drift[i],
                                level=0.002)
"""
#The expected signal level calculated for SNR 10,6,20
sig_level = [0.0004449990963094901, 0.00034469481781680114, 0.0006293237572446521]
sig_drift = np.array([4,2,8])*u.Hz/u.s

means = np.random.rand(n_ant)
sigs = np.random.rand(n_ant)
#changing the range
sigs_new = ((sigs - 0)*0.2)/1 + 0.9
means_new = ((means - 0)*0.2)/1 - 0.1

print(sigs_new)
print(means_new)

for i,stream in enumerate(antenna_array.bg_streams):
    stream.add_noise(v_mean=means_new[i],v_std=sigs_new[i])
    #stream.add_constant_signal(f_start=3032e+6*u.Hz, drift_rate=4*u.Hz/u.s, level = 0.001)
    #for i,f in enumerate(freq):
        #    stream.add_constant_signal(f_start=f, drift_rate=sig_drift[i], level = sig_level[i])
    stream.add_constant_signal(f_start=freq[0], drift_rate=4*u.Hz/u.s, level = 0.001)
    stream.add_constant_signal(f_start=freq[1], drift_rate=2*u.Hz/u.s, level = 0.0005)
    stream.add_constant_signal(f_start=freq[2], drift_rate=8*u.Hz/u.s, level = 0.002)


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
header_dict = {'INSTANCE': 0, 'DATADIR': '/mnt/buf0', 'BINDHOST': 'enp97s0f1', 'HPCONFIG': 'IBV-FTP-RAW', 'PROJID': 'delay_modeling_comm_Bconfig', 'BACKEND': 'GUPPI', 'DIRECTIO': 1, 'IBVPKTSZ': '42,16,8192', 'CUDADEV': 0, 
'BINDPORT': 50000, 'TELESCOP': 'VLA', 'IBVSTAT': 'running', 'MAXFLOWS': 16, 'NETSTAT': 'receiving', 'DAQSTATE': 'armed', 'BLOCSIZE': 184549376, 'NANTS': 22, 'NBITS': 8, 'NPOL': 2, 'OBSBW': 64, 
'CHAN_BW': 1, 'OBSNCHAN': 1408, 'OVERLAP': 0, 'PKTFMT': 'ATASNAPV', 'TBIN': 1e-06, 'OBS_MODE': 'RAW', 'NDROP': 1024, 'PKTSTART': 1674172056680875, 'PKTSTOP': 1674172061680875,
 'OBSID': 'TCOS0001_S_3000.59963.98997351852.2.1', 'IBVSNIFF': 50000, 'OBSSTAT': 'processing', 'STTVALID': 1, 'OBSDONE': 0, 'OBSNPKTS': 0, 'OBSNDROP': 0, 'OBSNXCES': 0, 'OBSBLKPS': 30, 
 'OBSBLKMS': 0.027229, 'DAQPULSE': 'Thu Jan 19 23:47:36 2023', 'SYNCTIME': 0, 'OBSFREQ': 3000, 'SIDEBAND': 1, 'TUNING': 'AC_8BIT', 'PKTNTIME': 32, 'PKTNPOL': 2, 'FENCHAN': 1024, 'RA_STR': 3.330044444444444,
  'DEC_STR': 41.511694444444444, 'SCHAN': 480, 'NCHAN': 64, 'PKTNCHAN': 64, 'ANTNMS00': 'ea01,ea02,ea03,ea04,ea05,ea07,ea10,ea11,ea13,ea14,ea15,ea17,ea18', 
  'ANTNMS01': 'ea19,ea21,ea22,ea23,ea24,ea25,ea26,ea27,ea28', 
  'ANTFLG00': '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0', 'NETBUFST': '1/9', 'OBSINFO': 'VALID', 'PIPERBLK': 32768, 'PKTSIZE': 8208, 'XPCTPPS': 687500, 'XPCTGBPS': 45.375, 'IBVBUFST': '1/9',
   'NETTHRDS': 12, 'IBVGBPS': 43.311889, 'IBVPPS': 656240.747, 'IBVBLKMS': 49.15253067, 'NPKTS': 2789786419, 'NEXCESS': 0, 'PHYSPKPS': 656253.75, 'PHYSGBPS': 5.386530876, 'RUSHBLKS': 0,
    'NETBLKPS': 20.906316757, 'NETBLKMS': 12.679693222, 'PKTIDX': 1674172056666112, 'BLKSTART': 1674172056666112, 'BLKSTOP': 1674172056698880, '~ANPKT00': 1024, '~ANPKT01': 1024, '~ANPKT02': 1024,
     '~ANPKT03': 1024, '~ANPKT04': 1024, '~ANPKT05': 1024, '~ANPKT06': 1024, '~ANPKT07': 1024, '~ANPKT08': 1024, 
'~ANPKT09': 1024, '~ANPKT10': 1024, '~ANPKT11': 1024, '~ANPKT12': 0, '~ANPKT13': 1024, '~ANPKT14': 1024, '~ANPKT15': 1024, '~ANPKT16': 1024, '~ANPKT17': 1024, '~ANPKT18': 1024, 
'~ANPKT19': 1024, '~ANPKT20': 1024, '~ANPKT21': 1024, 'SRC_NAME': '3C84', 'STT_IMJD': 59963, 'STT_SMJD': 85657, 'STT_OFFS': -0.319124937, 'NPKT': 21504, 'DROPSTAT': '1024/22528'}


#sig_level = stg.voltage.get_level(snr=20,
#                                     raw_voltage_backend=rvb,
#                                     fftlength=256000,
#                                     obs_length=5.0)

#print(sig_level)


rvb.record(
            output_file_stem='/mnt/slow/savin_vla_analysis/simdata/setigen_df-sig-noise',
            obs_length = 5.0,
            length_mode='obs_length',

            header_dict= header_dict,
            verbose=False,
            load_template=False
            )

