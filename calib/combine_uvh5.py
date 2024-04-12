"""
Given a path containing multiple UVH5 files (from each node), combine them into a 
single UVh5 file for AC and BD tuning
"""
import sys, os
import glob
from pyuvdata import UVData

path = sys.argv[1]

files1 = sorted(glob.glob(path+"*AC*.uvh5"))
files2 = sorted(glob.glob(path+"*BD*.uvh5"))

print(files1)
if len(files1) > 0:
    print("Getting all the AC uvh5 files")
    outfile1 = os.path.splitext(os.path.basename(files1[0]))[0]+"_comb.uvh5"
    for filename1 in files1:
        print(filename1)
        uvd_ac = UVData()
        uvd_ac.read(filename1, fix_old_proj=False, fix_use_ant_pos=False)

        try:
            uvd_main_ac += uvd_ac
        except NameError:
            uvd_main_ac = uvd_ac

    uvd_main_ac.write_uvh5(outfile1)
    
print(files2)
if len(files2) > 0:
    print("Getting all the BD uvh5 files")
    outfile2 = os.path.splitext(os.path.basename(files2[0]))[0]+"_comb.uvh5"
    for filename2 in files2:
        print(filename2)
        uvd_bd = UVData()
        uvd_bd.read(filename2, fix_old_proj=False, fix_use_ant_pos=False)

        try:
            uvd_main_bd += uvd_bd
        except NameError:
            uvd_main_bd = uvd_bd

    uvd_main_bd.write_uvh5(outfile2)