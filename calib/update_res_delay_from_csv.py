#Get the residula delays from the .csv files and write out a latest delay file
import csv
import glob
import sys, os
from matplotlib import pyplot as plt
import numpy as np

#data_path1 = "/home/svarghes/benchmark_test/rawdata/AC_delay_new/delays_guppi*599*"
#data_path2 = "/home/svarghes/benchmark_test/rawdata/BD_delay_new/delays_guppi*599*"

data_path1 = sys.argv[1]
data_path2 = sys.argv[2]

files1 = sorted(glob.glob(data_path1))
files2 = sorted(glob.glob(data_path2))

#Dictionary to store all the collected values
#Dictionary of the order
# main_dict = {'ant':{'mjd_time':{ source : '', 'AC':{ 'delay_pol0: '', 'delay_pol1':'' }, BD :{'delay_pol0' : '', 'delay_pol1' : ''}}}

data_dict = {}
avg_dict = {}
for i in range(1,29):
    ant = "ea"+str(i).zfill(2)
    data_dict[ant] = {}
    avg_dict[ant] = {}

print(data_dict)

#Reference antenna
ref = 'ea10'

for f,filename in enumerate(files1):
    print(f"Getting AC values from {filename}")
    basename_split1 = os.path.basename(filename).split('_')
    mjd_time = float(basename_split1[1]) + float(basename_split1[2])/86400.0
    source = basename_split1[4]
    #freq = float(basename_split1[7].split['-'][0])
    with open(filename, mode='r') as csv_file1: 
        csv_reader1 = csv.DictReader(csv_file1)
        for line in csv_reader1:
            bls = line['Baseline']
            res0 = float(line['res_pol0'])
            res1 = float(line['res_pol1'])
            
            ants_base = bls.split('-')
    
            if ref in ants_base:
                #print(ants_base)
                #print(ref)
                if ants_base[0] ==  ref:
                    res0 = -res0
                    res1 = -res1
                ants_base.remove(ref)
                ant_new = ants_base[0]
                #print(ant_new)    
                data_dict[ant_new][mjd_time] = {'source': source, 'AC':{ 'pol0': res0, 'pol1': res1}}    


    basename_split2 =  os.path.basename(files2[f]).split('_')
    if basename_split1[:7] == basename_split2[:7]:
        print(f"Getting BD values from {files2[f]}")

        with open(files2[f], mode='r') as csv_file2: 
            csv_reader2 = csv.DictReader(csv_file2)
            for line in csv_reader2:
                #print(line['Baseline'], line['non-geo_pol0'])
                bls = line['Baseline']
                res0 = float(line['res_pol0'])
                res1 = float(line['res_pol1'])
                
                ants_base = bls.split('-')
        
        
                if ref in ants_base:
                    if ants_base[0] ==  ref:
                        res0 = -res0
                        res1 = -res1
                    ants_base.remove(ref)
                    ant_new = ants_base[0]    
                    data_dict[ant_new][mjd_time]['BD'] = {'pol0': res0, 'pol1': res1}
    else:
        print("No matching files to get the BD values") 

for key in data_dict.keys():
    print(key, data_dict[key])


r = np.random.rand(28)
g = np.random.rand(28)
b = np.random.rand(28)
clr = np.round(np.linspace(0.2,0.8, 28),2)

#Writing averaged values into a csv file
dh = open(f"latest_res_delays.csv", "w")
dh.write(",".join(
            [
                "antenna",
                "IF0",
                "IF1",
                "IF3",
                "IF4"
            ]
        )+"\n")

print("Plotting now")
#Plotting the delay values from the dictionary
fig, axs = plt.subplots(5, 5, sharex  = True, sharey = False, constrained_layout=True, figsize = (12,12))
row = 0
col = 0
for i,key in enumerate(data_dict.keys()):
    data_ant = data_dict[key]
    if data_ant:
        times = data_ant.keys()
        times_ar = np.zeros(len(times))
        vals = np.zeros((len(times),4)) # Storing the values into an array
        for i, time in enumerate(times):
            times_ar[i] = time
            ac0 = data_ant[time]['AC']['pol0']
            ac1 = data_ant[time]['AC']['pol1']
            bd0 = data_ant[time]['BD']['pol0']
            bd1 = data_ant[time]['BD']['pol1']
            vals[i,:] = [ac0, ac1, bd0, bd1]
        ac0_avg, ac1_avg, bd0_avg, bd1_avg   = np.mean(vals,axis=0)
        dh.write(f"{key},{ac0_avg:+012.03f},{bd0_avg:+012.03f},{ac1_avg:+012.03f},{bd1:+012.03f}\n")
        min = np.min(vals)
        max = np.max(vals)

        #plot the AC values

        #axs[row, col].plot(np.round(times_ar,2), vals[:,0], '.', color = 'b', markerfacecolor = 'none', label = '0') #plotting single pol values
        #axs[row, col].plot(np.round(times_ar,2), vals[:,1], '.', color = 'r', markerfacecolor = 'none', label = '1') #plotting single pol values
        
        #plot the BD values
        axs[row, col].plot(np.round(times_ar,2), vals[:,2], '.', color = 'b', markerfacecolor = 'none', label = '0') #plotting single pol values
        axs[row, col].plot(np.round(times_ar,2), vals[:,3], '.', color = 'r', markerfacecolor = 'none', label = '1') #plotting single pol values
        #axs[row, col].set_xlabel('MJD time')
        #axs[row, col].set_ylim(min-50, max+300)
        axs[row, col].legend(loc='upper right')
        axs[row, col].set_title(f"antenna: {key}")
        col += 1
        if col == 5:
           row += 1
           col = 0 
        #plt.plot(time, val1, 's', color = clr[i], label = key)
    else:
        dh.write(f"{key},{0:+012.03f},{0:+012.03f},{0:+012.03f},{0:+012.03f}\n")    
dh.close()
fig.supxlabel("MJD time")
fig.supylabel("Residual delay(ns)")
#fig.suptitle("Non-geometrical delays per antenna vs time, IF:AC")
fig.suptitle("Non-geometrical delays per antenna vs time, IF:BD")
#plt.legend(loc='upper right')
plt.show()
   
