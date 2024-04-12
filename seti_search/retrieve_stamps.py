import os
import glob
import sys
import viewer as vi
import numpy as np
import pandas as pd

picklefile = sys.argv[1]
list = pd.read_pickle(picklefile)
hitfiles = list['File Path']
hitfreqs = list['Frequency']
fh = open("stamp_list.txt", "w")
num = 0 
for f,filename in enumerate(hitfiles):
    hitfreq = hitfreqs[f]
    print(hitfreq, filename)
    stampname = os.path.splitext(filename)[0]+'.0000.stamps'
    print(stampname)
    if os.path.exists(stampname):
        for (i, st) in enumerate(vi.read_stamps(stampname)):
            print(f"stamp {i} from {stampname}")
            times = st.times()
            freq = st.frequencies()
            print(freq[0], freq[-1])
            if hitfreq > freq[0] and hitfreq < freq[-1]:
                print(f"Hit found in stamp{i} of {stampname}")
                fh.write(f"{filename},{stampname},{i} \n")
                num += 1
    else:
        print(f"{stampname} cannot be found")    
fh.close()
print(f"Total number of stamps recovered: {num}")