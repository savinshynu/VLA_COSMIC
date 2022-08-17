import os,glob
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

data_dir = '/home/cosmic/savin/seti_search/MK/MK_dat/MK_HighFreqRes/'

#files = sorted(glob.glob(data_dir+'*.fil'))
#filename = '/home/cosmic/savin/seti_search/MK/file1/blpn51_f65536_t32_guppi_59109_53457_003376_J1644-4559_0001-ics.rawspec.0000.h5'
#fb =  Waterfall(filename,t_start= 0, t_stop = 6)


#print(matplotlib.get_backend())
#fb = Waterfall(filename)
#fb.info()
#data = fb.data

#print (data.shape)

#dat = data[:,0,0:1000]

#plt.pcolormesh(dat, cmap = 'jet')
#plt.show()

#spectra = np.mean(dat, axis = 0)


#plt.rcParams.update({'font.size': 13})

#plt.plot(range(spectra.shape[0]), spectra)
#plt.xlabel('Channels')
#plt.ylabel('Power (a.u)')
#plt.show()
#plt.savefig('spectra', format = 'png')

"""
for filename in files:

    print(filename)

    os.system("fil2h5 " + filename)


# Create a simple .lst file of the .h5 files in the data directory
h5_list = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
    
# This writes the .h5 files into a .lst, as required by the find_event_pipeline:
h5_list_path = os.path.join(data_dir,'h5_files.lst')
with open(h5_list_path, 'w') as f:
    for h5_path in h5_list:
        f.write(h5_path + '\n')



for h5file in h5_list:

    print(h5file)
    fb = Waterfall(h5file)
    nint = fb.data.shape[0]
    if nint > 4:

       doppler = FindDoppler(h5file,
                      max_drift = 4, # Max drift rate = 4 Hz/second
                      snr = 10,      # Minimum signal to noise ratio = 10:1
                      out_dir = data_dir # This is where the turboSETI output files will be stored.
                     )
       doppler.search()

"""

# Create a simple .lst file of the .dat files in the data directory
dat_list1 = sorted(glob.glob(os.path.join(data_dir, '*.dat')))
    
for dat_file in dat_list1:
    dat_list = sorted(glob.glob(os.path.join(data_dir, dat_file)))
    print (dat_list)
    
    # This writes the .dat files into a .lst, as required by the find_event_pipeline:
    dat_list_path = os.path.join(data_dir, 'dat_files.lst')
    with open(dat_list_path, 'w') as f:
       for dat_path in dat_list:
            f.write(dat_path + '\n')


    hfile = os.path.splitext(dat_file)[0]+'.h5' 
    # Create a simple .lst file of the .h5 files in the data directory
    h5_list = sorted(glob.glob(os.path.join(data_dir, hfile)))
    
    # This writes the .h5 files into a .lst, as required by the find_event_pipeline:
    h5_list_path = os.path.join(data_dir,'h5_files.lst')
    with open(h5_list_path, 'w') as f:
       for h5_path in h5_list:
           f.write(h5_path + '\n')

    csvf_path = os.path.join(data_dir, dat_file+'_found_event_table.csv')
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
