# needs fixing

import os, glob

filepath = "/mnt/slow/savin_vla_analysis/rawdata/TCOS*"

dirs = glob.glob(filepath)

for dirname in dirs:
    print(dirname)

    path1 = dirname+'*.csv'
    if len(glob.glob(path1)) == 1:
       print(f"copying AC from {path1}")
       os.system(f"cp  {path1} /home/svarghes/benchmark_test/rawdata/B_config/AC_delay")
    
    path2 = dirname+'/1/*.csv'
    if len(glob.glob(path2)) == 1:
       print(f"copying BD from {path2}") 
       os.system(f"cp  {path2} /home/svarghes/benchmark_test/rawdata/B_config/BD_delay")
