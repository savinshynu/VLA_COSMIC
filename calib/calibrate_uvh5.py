import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
import pyuvdata.utils as uvutils
from pyuvdata import UVData
from sdmpy import calib

class calibrate_uvh5:

    def __init__(self, datafile):

        #Initializing the pyuvdata object and reading the files
        self.uvd = UVData()
        self.uvd.read(datafile, fix_old_proj=False)
        self.metadata = self.get_metadata()
        self.vis_data = self.get_vis_data()
        


    def get_metadata(self):

        nant_data = self.uvd.Nants_data
        nant_array = self.uvd.Nants_telescope
        ant_names = self.uvd.antenna_names
        nfreqs = self.uvd.Nfreqs
        ntimes = self.uvd.Ntimes
        npols = self.uvd.Npols
        bls = self.uvd.Nbls
        nspws = self.uvd.Nspws
        chan_width = self.uvd.channel_width
        intg_time = self.uvd.integration_time
        source = self.uvd.object_name
        telescope = self.uvd.telescope_name
        pol_array = uvutils.polnum2str(self.uvd.polarization_array)
        freq_array = self.uvd.freq_array[0,:]
        lobs = ntimes*intg_time
        metadata  = {'nant_data' : nant_data,
        'nant_array' : nant_array,
        'ant_names' : ant_names, 
        'nfreqs' : nfreqs,
        'ntimes' : ntimes,
        'npols': npols,
        'nbls' : bls,
        'nspws' : nspws,
        'chan_width': chan_width,
        'intg_time' : intg_time,
        'lobs' : lobs,
        'source': source,
        'telescope' : telescope,
        'pol_array' : pol_array,
        'freq_array' : freq_array}
        return metadata

    def print_metadata(self):
          #Print out the observation details
        meta = self.metadata
        print(f" Observations from {meta['telescope']}: \n\
                Source observed: {meta['source']} \n\
                No. of time integrations: {meta['ntimes']} \n\
                Length of time integration: {meta['intg_time'][0]} s \n\
                Length of observations: {meta['lobs'][0]} s \n\
                No. of frequency channels: {meta['nfreqs']} \n\
                Width of frequency channel: {meta['chan_width']/1e+3} kHz\n\
                Observation bandwidth: {(meta['freq_array'][-1] - meta['freq_array'][0])/1e+6} MHz \n\
                No. of spectral windows: {meta['nspws']}  \n\
                Polarization array: {meta['pol_array']} \n\
                No. of polarizations: {meta['npols']}   \n\
                Data array shape: {self.vis_data.shape} \n\
                No. of baselines: {meta['nbls']}  \n\
                No. of antennas present in data: {meta['nant_data']} \n\
                No. of antennas in the array: {meta['nant_array']} \n\
                Antenna name: {meta['ant_names']}")


    def get_uvw_data(self):
        uvw_array = self.uvd.uvw_array
        uvw_array = uvw_array.reshape(self.metadata['ntimes'], self.metadata['bls'], 3)
        return uvw_array

    def get_vis_data(self):
        vis = self.uvd.data_array
        new_shape = (self.metadata['nbls'], self.metadata['ntimes'])+vis.shape[1:]
        return vis.reshape(new_shape)

    #def derive_delay(self):
       

    def derive_phase(self):
        gainsol = calib.gaincal(self.vis_data, axis = 0, ref = 0)
        return gainsol

    #def apply_delay(self):

    def apply_phase(self, gainsol):
        calib.applycal(self.vis_data, gainsol, axis=0)
        return self.vis_data

    def plot_solutions(self, data):
        data_avg = np.squeeze(np.mean(data, axis=1))
        ant1, ant2 = self.uvd.baseline_to_antnums(self.uvd.baseline_array[:self.metadata['nbls']])
        
        nbls = self.metadata['nbls']
        grid = int(np.ceil(np.sqrt(nbls)))


        fig, axs = plt.subplots(grid, grid, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,12))
        for i in range(grid):
            for j in range(grid):
                rbl = (i*grid)+j
                if rbl < nbls:

                    #Picking the baseline
                    data_bls0_rr = data_avg[rbl,:,0]
                    #data_bls0_ll = data_avg[rbl,:,0,:,1].flatten()

                    axs[i,j].plot(self.metadata['freq_array']/1e+9, np.angle(data_bls0_rr, deg = True), '.', label = "RR")
                    #axs[i,j].plot(freqs, np.angle(data_bls0_ll, deg = True), '.',  label = "LL")

                    #axs[i,j].set_ylabel("Phase (degrees)")
                    #axs[i,j].set_xlabel("Frequency (GHz)")
                    axs[i,j].set_title(f"ea{ant1[rbl]} - ea{ant2[rbl]}")
                    axs[i,j].legend(loc = 'upper right')
           
        #fig.suptitle("Before calibration")
        fig.supylabel("Phase (degrees)")
        fig.supxlabel("Frequency (GHz)")
        plt.show()

def main(args):
    cal_ob = calibrate_uvh5(args.dat_file)
    cal_ob.print_metadata()
    #cal_ob.plot_solutions(cal_ob.vis_data)
    gain = cal_ob.derive_phase()
    cal_ob.apply_phase(gain)
    cal_ob.plot_solutions(cal_ob.vis_data)


if __name__ == '__main__':
    
    # Argument parser taking various arguments
    parser = argparse.ArgumentParser(
        description='Reads UVH5 files, derives delay and gain calibrations, apply to the data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--dat_file', type = str, required = True, help = 'UVH5 file to read in')
    
    #parser.add_argument('-p', '--plot', action = 'store_true', help = 'plot the figures, otherwise save figures to working directory')

    args = parser.parse_args()
    main(args)



