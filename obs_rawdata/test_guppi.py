import os
os.environ['SETIGEN_ENABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from astropy import units as u
import blimpy as bl
import setigen as stg
import cupy as xp

matplotlib.use('Tkagg')

n_pols = 2
n_bits = 8
#n_ant = 64
#n_chan_perant = 128
n_time = 1024

sample_rate = 2.048e9



antenna = stg.voltage.Antenna(sample_rate=sample_rate,
                              fch1=2*u.GHz,
                              ascending=True,
                              num_pols= n_pols)


antenna.x.add_noise(v_mean=0,
                    v_std=1)


#print(antenna.x.shape)


v = antenna.x.get_samples(2000)
v = xp.asnumpy(v) 
print(v.shape)
plt.figure(figsize=(10, 5))
plt.plot(range(v.shape[0]), v)
plt.xlabel('Sample')
plt.ylabel('V')
plt.show()

