import sys
import numpy as np
from matplotlib import pyplot as plt
import sdmpy
from sdmpy import calib
import time

#Enter the directory containing data and metadata
dirname = sys.argv[1]

sdm = sdmpy.SDM(dirname, use_xsd=False)

for scan in sdm.scans(): 
    print (scan.idx,scan.source,scan.intents)

scan_int = sdm.scan(5)

print("Metadata of this scan")

print(f" Source: {scan_int.source} \n\
         Field : {scan_int.field} \n\
         Ra, dec: {scan_int.coordinates} \n\
         Intent: {scan_int.intents} \n\
         Antennas: {scan_int.antennas} \n\
         Start MJD time: {scan_int.startMJD} \n\
         Stop  MJD time: {scan_int.endMJD} \n\
         No. of Integrations: {scan_int.numIntegration} \n\
         SPW : {scan_int.spws} \n\
         Ref frequencies :{scan_int.reffreqs}")


#Duration of scan
dur = (scan_int.endMJD - scan_int.startMJD)*24.0*3600.0

#Binary data for this scan
bd = scan_int.bdf 

bls = scan_int.baselines
freqs = scan_int.freqs().flatten()/1e+9

print(freqs.shape)
#print(freqs[0,:])

data = bd.get_data() # Read full visibility data array for the scan

data_avg = np.mean(data, axis=0)

nbls = data_avg.shape[0]
grid = int(np.ceil(np.sqrt(nbls)))

fig, axs = plt.subplots(grid, grid, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,12))
for i in range(grid):
    for j in range(grid):
        rbl = (i*grid)+j
        if rbl < nbls:

           #Picking the baseline
           data_bls0_rr = data_avg[rbl,:,0,:,0].flatten()
           #data_bls0_ll = data_avg[rbl,:,0,:,1].flatten()

           axs[i,j].plot(freqs, np.angle(data_bls0_rr, deg = True), '.', label = "RR")
           #axs[i,j].plot(freqs, np.angle(data_bls0_ll, deg = True), '.',  label = "LL")

           #axs[i,j].set_ylabel("Phase (degrees)")
           #axs[i,j].set_xlabel("Frequency (GHz)")
           axs[i,j].set_title(f"{bls[rbl]}")
           axs[i,j].legend(loc = 'upper right')
           
fig.suptitle("Before calibration")
fig.supylabel("Phase (degrees)")
fig.supxlabel("Frequency (GHz)")
plt.show()

print(data.shape)

#Looking at data before calibration

print("Deriving Calibrations now")
t1 = time.time()
gainsol = calib.gaincal(data, axis = 1, ref = 0)
t2 = time.time()

print(f"Took {t2-t1}s for getting solution from {dur}s of data")

print(gainsol.shape)

#Applying the solutions now:
calib.applycal(data, gainsol, axis=1)

#plotting after calibrations

cal_data_avg = np.mean(data, axis=0)

fig, axs = plt.subplots(grid, grid, sharex = True, sharey = True, constrained_layout=True, figsize = (12,12))
for i in range(grid):
    for j in range(grid):
        rbl = (i*grid)+j
        if rbl < nbls:

           #Picking the baseline
           cal_data_bls0_rr = cal_data_avg[rbl,:,0,:,0].flatten()
           #cal_data_bls0_ll = cal_data_avg[rbl,:,0,:,1].flatten()

           axs[i,j].plot(freqs, np.angle(cal_data_bls0_rr,  deg = True), '.', label = "RR")
           #axs[i,j].plot(freqs, np.angle(cal_data_bls0_ll,  deg = True), '.',  label = "LL")

           #axs[i,j].set_ylabel("Phase (degrees)")
           #axs[i,j].set_xlabel("Frequency (GHz)")
           axs[i,j].set_title(f"{bls[rbl]}")
           axs[i,j].legend(loc = 'upper right')

fig.suptitle("After calibration}")
fig.supylabel("Phase (degrees)")
fig.supxlabel("Frequency (GHz)")
plt.show()




"""

print(gainsol.shape)

gainsol_tavg = np.mean(gainsol, axis = 0)
gainsol_tavg = gainsol_tavg.reshape(gainsol_tavg.shape[0], gainsol_tavg.shape[1]*gainsol_tavg.shape[3], gainsol_tavg.shape[4])

for i in range(gainsol_tavg.shape[0]):
    gain = gainsol_tavg[i,:,:]
    #amp = np.abs(gain)
    phase_rr  = np.angle(gain[:,0], deg = True)
    phase_ll  = np.angle(gain[:,1], deg = True)
    #amp_rr  = 10*np.log10(np.abs(gain[:,0]))
    #amp_ll  = 10*np.log10(np.abs(gain[:,1]))
    amp_rr  = np.abs(gain[:,0])
    amp_ll  = np.abs(gain[:,1])

    if i==1:
        plt.subplot(1,2,1)
        plt.plot(freqs, amp_rr, '.', label = 'RR')
        #plt.plot(freqs, amp_ll, '.', label = 'LL')
        plt.ylabel("Amplitude (deg)")
        plt.xlabel("Frequency (GHz)")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(freqs, phase_rr, '.', label = 'RR')
        #plt.plot(freqs, phase_ll, '.', label = 'LL')
        plt.ylabel("Phase (deg)")
        plt.xlabel("Frequency (GHz)")
        plt.legend()
plt.suptitle(f"After Calibrations,{scan_int.antennas[1]} wrt {scan_int.antennas[0]}")
plt.show()

"""


