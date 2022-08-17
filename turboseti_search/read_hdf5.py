import sys #os,glob
import numpy as np
import matplotlib
from blimpy import Waterfall
from matplotlib import pyplot as plt
matplotlib.use('Tkagg')



filename = sys.argv[1]

#fb =  Waterfall(filename,t_start= 0, t_stop = 6)
fb = Waterfall(filename)
fb.info()
data = np.squeeze(fb.data, axis = 1)

print (data.shape)

spectra = np.mean(data, axis = 0)


#plt.rcParams.update({'font.size': 13})

plt.plot(range(spectra.shape[0]), spectra)
plt.xlabel('Channels')
plt.ylabel('Power (a.u)')
plt.show()
#plt.savefig('spectra', format = 'png')


