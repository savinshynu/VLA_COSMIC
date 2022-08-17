import glob
import os
import sys
import argparse
import numpy as np
from copy import deepcopy
from plot_events_grid import plot_candidate_events
import matplotlib
matplotlib.use('Tkagg')


TOL = 0.01 # checking the tolerance for the comparison

class FilterEvents():

    def __init__(self, data_dir,  SNR=10, check_zero_drift=False, filter_threshold=3):

        self.data_dir = data_dir
        self.SNR = SNR
        self.check_zero_drift = check_zero_drift
        self.filter_threshold = filter_threshold
        #self.nbeams = nbeams
        self.dat_tab, self.dat_array = self.collect_hits()
        self.event_list = []
        self.filtered_event_list = []

    def collect_hits(self):

        """
        Function to collect hits from all the .dat file corresponding to n number of coherent beams 
        for a particular observation
        """
        # Making a list of all .dat files from all the beams 
        dat_file_list = sorted(glob.glob(os.path.join(self.data_dir, '*.dat')))

        # Reading the data files and concatenating their header info into a dictionary and hit info into an array
        dat_tab, dat_array = self.read_files(dat_file_list)

        # Getting a list of events based on the filter threshold
        #cand_list = self.filter_event(dat_tab, dat_array, self.filter_threshold, self.SNR, self.check_zero_drift, self.nbeams)

        return dat_tab, dat_array




    def read_files(self, dat_file_list):
        """
        Loading the .dat file info into a dictionary and data in an array
        """

        dat_tab = [] #list to hold dictionaries for each files
        num = 0
        for dat_file in dat_file_list:

            print(dat_file)
        
            file_dat = open(dat_file.strip())
            hits = file_dat.readlines()

            # Get info from the .dat file header
            FileID = hits[1].strip().split(':')[-1].strip()
            Source = hits[3].strip().split(':')[-1].strip()

            MJD = hits[4].strip().split('\t')[0].split(':')[-1].strip()
            RA = hits[4].strip().split('\t')[1].split(':')[-1].strip()
            DEC = hits[4].strip().split('\t')[2].split(':')[-1].strip()

            DELTAT = hits[5].strip().split('\t')[0].split(':')[-1].strip()  # s
            DELTAF = hits[5].strip().split('\t')[1].split(':')[-1].strip()  # Hz

            max_drift_rate = hits[5].strip().split('\t')[2].split(':')[-1].strip() # Hz/s
            obs_length = hits[5].strip().split('\t')[3].split(':')[-1].strip() # s
        
            #getting beam id from the file tag
            beamid = int(os.path.splitext(os.path.basename(dat_file))[0].split('_')[-1][1:])
            dat_dict = {"FileID" : FileID, "Source": Source, "MJD" : MJD, "RA": RA, "DEC": DEC, "DELTAT":  DELTAT, "DELTAF": DELTAF, "max_drift_rate": max_drift_rate, "obs_length": obs_length, "beamid": beamid }
       
            dat_tab.append(dat_dict)
            file_dat.close()

            #Loading .dat file as numpy array
            dat = np.loadtxt(dat_file, skiprows = 8)
            #print(dat.shape)

            #checking for dat files without any events and change shape of array in that case
            if len(dat.shape) == 1:
                if dat.size == 0 :
                    continue
                else:
                    dat = np.reshape(dat,(1,dat.size))
        
            #adding beam info into the array

            dat_add = np.concatenate((dat, np.reshape(np.ones(dat.shape[0])*beamid, (dat.shape[0],1))), axis = 1)
        
            #Concatenating the beam info from each file into dat_array
            try:
                dat_array = np.concatenate((dat_array, dat_add), axis = 0)
            except NameError:
                dat_array = dat_add
            num += 1    
    
        if num == 0:
            sys.exit("No hits any of the beams, nothing to compare")
        else:
            return dat_tab, dat_array

     
    def is_close(self, event1, event2):
        """
        Comparing two event array by cheching their drfit rate, freq_start and freq_stop
        """
        if np.isclose(event1[1],event2[1], TOL) and np.isclose(event1[6], event2[6], TOL) and np.isclose(event1[7], event2[7], TOL):
           
           return True
        else:
           return False  

    
    def in_list(self, event):

        """
        Checking to see if the event is alreay in the list
        """
        check = False
        for event_dict in self.event_list:
            event_comp = event_dict['event_dat']
            if self.is_close(event, event_comp):
               check = True
               break
        
        return check


    def filter_events(self):
    
        """
        dat_tab: Contains the header info from each .dat file
        dat_array: An array containing the hits and it's properties from all beam data
        dat_arraay format: # Top_Hit_#     Drift_Rate      SNR     Uncorrected_Frequency   Corrected_Frequency     Index   
        freq_start      freq_end        SEFD    SEFD_freq       Coarse_Channel_Number   Full_number_of_hits beam_id
        
        filter_threshold: 1 >> Filter all events above  SNR cut and appearing in more than 5 beams
                        : 2 >> Filter all evnents above SNR cut and appearing in less than 5 beams
                        : 3 >> Filter all events above SNR cut and appearing in one beam >> special candidate event
        """
        #adjusting for SNR
        SNR_dat_array = self.dat_array[self.dat_array[:,2] > self.SNR]

        #Filtering events with zero drift rate
        if self.check_zero_drift:
           adj_dat_array = SNR_dat_array[SNR_dat_array[:,1] != 0]
        else:
           adj_dat_array = SNR_dat_array
    

        for i in range(adj_dat_array.shape[0]):

            #Starting with each event, finding the beam_id, comparing with other
            #events with a different beam_id
          

            event = adj_dat_array[i,:] #starting with each event
            
            if not self.in_list(event):

                curr_beam = event[12] #beam id of the event

                #Collecting the header info of the file using the beam info
                #deltaf = self.dat_tab[int(curr_beam)]["DELTAF"]
                #obs_length = self.dat_tab[int(curr_beam)]["obs_length"]
                #drift_res = abs(float(deltaf))/float(obs_length)
                #print(deltaf, obs_length, drift_res)
        
                comp_array = adj_dat_array[adj_dat_array[:,12] != curr_beam]

                event_array = np.tile(event, (comp_array.shape[0],1))
        
                diff_array = np.absolute(comp_array - event_array)
       
                ind = np.where((diff_array[:,1] < TOL) & (diff_array[:,6] < TOL) & (diff_array[:,7] < TOL))[0]
                
                event_dict = deepcopy(self.dat_tab[int(curr_beam)])
                event_dict['event_dat'] = event # event_array
                event_dict['in_beams'] = len(ind)  # number of beams in which the event appears
                
                self.event_list.append(event_dict)

        #print(self.event_list)

        if self.filter_threshold == 1:
           for final_event in self.event_list:
               if final_event['in_beams'] == 0:
                   self.filtered_event_list.append(final_event) 

        elif self.filter_threshold == 2:
           for final_event in self.event_list:
                 if final_event['in_beams'] > 0 and final_event['in_beams'] < 6:
                    self.filtered_event_list.append(final_event)

        else:
           for final_event in self.event_list:
               if final_event['in_beams'] >= 6:
                  self.filtered_event_list.append(final_event) 
        
        #returning the filtered list depending on the number of beams in which they appeared  
        return self.filtered_event_list


def main(args):

    if os.path.isdir(args.data_dir):
       cand_ob = FilterEvents(args.data_dir,  args.SNR, args.check_zero_drift, args.filter_threshold)
       cand_list = cand_ob.filter_events()
       #for cand in cand_list:
       #    print(cand['event_dat'])
       #print(len(cand_list))
       if args.plot:
          h5file_list = sorted(glob.glob(args.data_dir+'*.h5'))
          plot_candidate_events(cand_list, h5file_list, plot_dir = args.plot_dir,  offset = 0)  

    else:
        sys.exit("Directory does not exist")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Threshold based filtering for hits from the #number of beam data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', type = str,  help = 'Directory containing the .dat files form turboSETI search')
    parser.add_argument('-s','--SNR', type = float, required = False, default = 10.0, help = 'SNR threshold for filtering events')
    parser.add_argument('-c','--check_zero_drift', action = 'store_true', help = 'Use in order to filter zero drift events, default: False')
    parser.add_argument('-f', '--filter_threshold', type = int, required = False, default = 3, help = 'Filter threshold for filtering the events')
    parser.add_argument('-p', '--plot', action = 'store_true', help = 'plot the filtered candidates')
    parser.add_argument('-pd','--plot_dir', type = str, required = False, default = None,  help = 'Directory outputting the plot files')

    args = parser.parse_args()

    main(args)

