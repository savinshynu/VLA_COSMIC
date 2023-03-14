import sys
import h5py

filename= sys.argv[1]

f = h5py.File(filename, 'r')

print("Groups")
print(f.keys())

delay = f['delayinfo']['delays']
beam =  f['beaminfo']
ras = beam['ras']
decs = beam['decs']
print(ras[:],decs[:])
print(delay.shape)

print(delay[10,0,:])
print(delay[10,1,:])
print(delay[10,2,:])
