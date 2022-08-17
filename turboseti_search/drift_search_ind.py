import os,glob,sys
import time
import numpy as np
import matplotlib
from blimpy import Waterfall
from matplotlib import pyplot as plt
import turbo_seti.find_doppler.seti_event as turbo
import turbo_seti.find_event as find
from turbo_seti.find_doppler.find_doppler import FindDoppler
from turbo_seti.find_event.find_event_pipeline import find_event_pipeline
from turbo_seti.find_event.plot_event_pipeline import plot_event_pipeline
#matplotlib.use('Tkagg')

data_dir =  sys.argv[1] #'/home/cosmic/savin/seti_search/MK/MK_dat/MK_HighFreqRes/file1'

# Create a simple .lst file of the .h5 files in the data directory
h5_list = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
    
# This writes the .h5 files into a .lst, as required by the find_event_pipeline:
h5_list_path = os.path.join(data_dir,'h5_files.lst')
with open(h5_list_path, 'w') as f:
    for h5_path in h5_list:
        f.write(h5_path + '\n')

t1 = time.time()
# Running turboseti on  all the .h5 files
for h5file in h5_list:

    print(h5file)
    fb = Waterfall(h5file)
    nint = fb.data.shape[0]
    if nint > 4: # Only runnin the doppler search if the number of time integration > 4

       doppler = FindDoppler(h5file,
                      max_drift = 10, # Max drift rate = 4 Hz/second
                      min_drift = 0,
                      gpu_backend = True,
                      gpu_id = 0,
                      snr = 10,      # Minimum signal to noise ratio = 10:1
                      n_coarse_chan = 1 , # Blimpy calculate this for earlier BL data, make sure each file has correct coarse channels
                     )
       doppler.search()
t2 = time.time()

print(f"Elapsed time is {(t2-t1)/60.0} minutes")


# Create a simple .lst file of the .dat files in the data directory
dat_list = sorted(glob.glob(os.path.join(data_dir, '*.dat')))
    
# This writes the .dat files into a .lst, as required by the find_event_pipeline:
dat_list_path = os.path.join(data_dir, 'dat_files.lst')
with open(dat_list_path, 'w') as f:
    for dat_path in dat_list:
        f.write(dat_path + '\n')


csvf_path = os.path.join(data_dir, 'found_event_table.csv')
find_event_pipeline(dat_list_path, 
                    filter_threshold = 1 , 
                    number_in_cadence = len(dat_list), 
                    csv_name=csvf_path, 
                    saving=True)

# and finally we plot
plot_event_pipeline(csvf_path, # full path of the CSV file built by find_event_pipeline()
                    h5_list_path, # full path of text file containing the list of .h5 files
                    filter_spec='f{}'.format(3), # filter threshold
                    user_validation=False) # Non-interactive
