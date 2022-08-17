import sys
import blimpy as bl
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Tkagg')

filename = sys.argv[1]


wf = bl.Waterfall(filename)
#print(wf.data.shape)

print(wf.header)

#plt.figure(figsize=(10, 6))
#wf.plot_waterfall()
#plt.show()

data= wf.data[:,:,:]



#plt.figure(figsize=(10, 6))
#wf.plot_spectrum()
#plt.show()

for pol in range(data.shape[1]):
    plt.plot(range(data.shape[2]), np.abs(np.mean(data[:12,pol,:], axis = 0)), label = "pol: %d" %pol)

plt.xlabel("Frequency Channels")
plt.ylabel("Power (a.u.)")
plt.legend()
plt.show()
