# VLA_COSMIC
Set of scripts used in the commissioning of COSMIC

btest:
  Contains the scripts used for the beamforming tests
  upchan_bf.py -- A Python based single coarse channel, upchannelizer beamformer
Calib:
  Calibration routines used for deriving the gain calibration
mcast:
  Broadcasting the metadata information from the VLA multicast systems to COSMIC redis server postprocessing
obs_rawdata:
  Scritps to play with the guppi raw files, conducting delay calibration, UVW calculation and plotting autocorrelations, crosscorrelations and 
  crosscorrelation coefficients
sim_rawdata:
  Scripts to genereate guppi raw files from setigen
seti_search:
  Scripts to look at the hits and stamps
turboseti_seach:
  Scripts to conduct turboseti searches on filterbank and h5 files.
