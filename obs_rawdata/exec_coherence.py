#!/usr/bin/env python3

import sys
import glob
import os

inp_dir1 = '/mnt/buf0/delay_modeling_comm_hal/GUPPI/'
inp_path1 = os.path.join(inp_dir1, 'guppi_5992*.0000.raw')
raw_files1 = sorted(glob.glob(inp_path1))

inp_dir2 = '/mnt/buf1/delay_modeling_comm_hal/GUPPI/'
inp_path2 = os.path.join(inp_dir2, 'guppi_5992*.0000.raw')
raw_files2 = sorted(glob.glob(inp_path2))

meta_dir = '/home/svarghes/VLA_COSMIC/obs_rawdata/json/'
meta_path = os.path.join(meta_dir, '*.json')
metafiles = sorted(glob.glob(meta_path))

out_path = '/mnt/slow/savin_vla_analysis/'

for file1 in raw_files1:
    print(f"Processing {file1}")
    file2 = os.path.join(inp_dir2, os.path.basename(file1))
    if file2 not in raw_files2:
        print(f"No tuning 2 file: {file2}")
    #meta_file = os.path.join(meta_dir, os.path.basename(file1).split('.')[0]+'_metadata.json')
    meta_file = os.path.join(meta_dir, os.path.basename(file1)[:-9]+'_metadata.json')
    if meta_file in metafiles:
        print (meta_file)
    else:
        print("No meta json file")

    if file2 in raw_files2 and meta_file in metafiles:
        print(f"Both tuning files present for {os.path.basename(file1)}, also the metafile is present ")

        os.mkdir(os.path.join(out_path, os.path.basename(file1).split('.')[0]))
        out1 = os.path.join(out_path, os.path.basename(file1).split('.')[0], '0')
        os.mkdir(out1)
        out2 = os.path.join(out_path, os.path.basename(file1).split('.')[0], '1')
        os.mkdir(out2)
        
        cmd1 = f"python ~/VLA_COSMIC/obs_rawdata/upchan_coherence_opt2.py -f 256 -i 1 -td -bc 0.4 -b 0.06 -o {out1} -d {file1}   -l {meta_file}"
        cmd2 = f"python ~/VLA_COSMIC/obs_rawdata/upchan_coherence_opt2.py -f 256 -i 1 -td -bc 0.4 -b 0.06 -o {out2} -d {file2}   -l {meta_file}"
        
        print("Processing the tuning1 now")
        os.system(cmd1)
        print("Processing the tuning2 now")
        os.system(cmd2)

    else:
        print(f"cannot process {file2}")
        continue        
