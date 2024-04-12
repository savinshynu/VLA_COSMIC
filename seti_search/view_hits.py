import sys
import viewer as seticore_viewer

hits_filepath = sys.argv[1]
#hits_filepath = "/mnt/slow/savin_vla_analysis/bfdata/search/out_bf_synth_df-sig_multi.seticore.hits"
f = open(hits_filepath+'.txt', "w")
for hit_enum, hit in enumerate(seticore_viewer.read_hits(hits_filepath)):
    f.write(f"Hit #{hit_enum}: \n")
    f.write(f"\tBeam: {hit.signal.beam} \n")
    f.write(f"\tStarting frequency: {hit.signal.frequency} \n")
    f.write(f"\tDrift rate: {hit.signal.driftRate} Hz/s \n")
    f.write(f"\tDrift steps: {hit.signal.driftSteps} \n")
    f.write(f"\tTime steps: {hit.signal.numTimesteps} \n")
    f.write(f"\tFrequency channel: {hit.signal.index} of Coarse-Channel {hit.signal.coarseChannel} \n")
    f.write(f"\tSNR: {hit.signal.snr} \n")
    f.write(f"\tPower: {hit.signal.power} ({hit.signal.incoherentPower} incoherent) \n")
    
    f.write(f"\t\tSource name: {hit.filterbank.sourceName} \n")
    f.write(f"\t\tFch1: {hit.filterbank.fch1*1e6} Hz \n")
    f.write(f"\t\tChannel BW: {hit.filterbank.foff*1e6} Hz \n")
    f.write(f"\t\tTstart: {hit.filterbank.tstart} MJD \n")
    f.write(f"\t\tChannel timespan: {hit.filterbank.tsamp} s \n")
    f.write(f"\t\tRA: {hit.filterbank.ra} hours \n")
    f.write(f"\t\tDEC: {hit.filterbank.dec} degrees \n")
    f.write(f"\t\tTelescope ID: {hit.filterbank.telescopeId} \n")
    f.write(f"\t\tNumber of timesteps: {hit.filterbank.numTimesteps} \n")
    f.write(f"\t\tNumber of channels: {hit.filterbank.numChannels} \n")
    f.write(f"\t\tBeam: {hit.filterbank.beam} \n")

f.close()