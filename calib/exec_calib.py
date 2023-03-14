#!/usr/bin/env python3

import sys
import glob
import os


keys = ['uvh5_59949*']

for key in keys:

    inp_dir1 = '/mnt/buf0/uvh5_commensal/uvh5/'
    inp_path1 = os.path.join(inp_dir1, key)
    cor_files1 = sorted(glob.glob(inp_path1))

    inp_dir2 = '/mnt/buf1/uvh5_commensal/uvh5/'
    inp_path2 = os.path.join(inp_dir2, key)
    cor_files2 = sorted(glob.glob(inp_path2))


    out_path = '/mnt/slow/savin_vla_analysis/cordata'

    for file1 in cor_files1:
        print(f"Processing {file1}")
        file2 = os.path.join(inp_dir2, os.path.basename(file1))
    
        if file2 not in cor_files2:
            print(f"No tuning 2 file: {file2}")
            continue
    
        else:
            print(f"Both tuning files present, processing {file2}")

            os.mkdir(os.path.join(out_path, os.path.basename(file1).split('.')[0]))
            out1 = os.path.join(out_path, os.path.basename(file1).split('.')[0], '0')
            os.mkdir(out1)
            out2 = os.path.join(out_path, os.path.basename(file1).split('.')[0], '1')
            os.mkdir(out2)
        
            cmd1 = f"python  ~/VLA_COSMIC/calib/calibrate_uvh5.py -d {file1} -o {out1}"
            cmd2 = f"python  ~/VLA_COSMIC/calib/calibrate_uvh5.py -d {file2} -o {out2}"
        
            print("Processing the tuning1 now")
            os.system(cmd1)
            print("Processing the tuning2 now")
            os.system(cmd2)

         
