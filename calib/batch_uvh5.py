import os
import sys
import glob

path = sys.argv[1]
files = sorted(glob.glob(path+"*BD*.uvh5"))

outdir = "/mnt/cosmic-storage-1/data0/savin/check_calib/uvh5_files/check_rfi_filt1/"
for filename in files:
    print(filename)
    os.system(f"python ~/VLA_COSMIC/calib/calibrate_uvh5.py -d  {filename} -o {outdir}")
