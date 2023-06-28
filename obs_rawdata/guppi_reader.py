"""
Reads the guppi raw files
"""
import sys
from blimpy import GuppiRaw
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Tkagg')

filename = sys.argv[1] #Input guppi raw file 
gr = GuppiRaw(filename) #Instantiating a guppi object


head1, data1 = gr.read_next_data_block()
print(head1)
print (data1.shape, data1.dtype)
#print(data1[:3,:2,0])

data1 = data1.reshape(2, 1024, 15360, 2)

#No reshaping
#print(data1[:4,:4])
# First reshaping
#print(data1.reshape(2,1024,30720)[0,:4,:4])
# Second reshaping
#print(data1.reshape(2,1024,15360,2)[0,:4,:2,:])

#print(data1.reshape

#print(data1.shape)

dat_spec = data1[0, :, 0, 0]
plt.plot(range(dat_spec.shape[0]), np.abs(np.mean(data1[0,:,:100,0], axis = 1)))
plt.plot(range(dat_spec.shape[0]), np.abs(np.mean(data1[1,:,:100,0], axis = 1)))
plt.show()

#print(gr.read_next_data_block_shape())
#head2, datax, datay = gr.read_next_data_block_int8_2x()

#print(head2)
#print(datax.dtype, datay.dtype)
#print(datax[:3,:2,0], datay[0:3,0:2])
"""
d = np.zeros(1, dtype='complex64')

for header, data_x, data_y in  gr.get_data():
    #print(header)
    print(data_x.dtype, data_y.dtype)
    #data_x = data_x.view('complex64')
    #data_y = data_y.view('complex64')
    #print(data_x[1,1,:], data_y[1,1,:])

    dn = np.append(data_x, data_y, axis=2)
    print(dn.shape)
    dn = dn.astype('float32')
    dn = dn.view('complex64')
    print(dn.shape, dn.dtype)


    #if dn.shape != d.shape:
    #   d = np.zeros(d.shape, dtype='float32')

    #   d[:] = dn

"""

#print(header)
#print (data.shape)
#print (gr.n_blocks)

#ds = data[:,:,0]

#spec = np.abs(np.mean(ds, axis = 1))

#plt.plot(range(len(spec)), spec)
#plt.show()
