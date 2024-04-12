import os
import glob
import sys
import viewer as vi
import numpy as np

filestem = sys.argv[1]
hitfreq = sys.argv[2]
filenames = sorted(glob.glob(filestem+"*.stamps"))
#filenames = sorted(glob.glob(filestem))
#stamp = vi.read_stamps(filename)
#stamp = vi.Stamp(filename)
print(filenames)
for filename in filenames:
    print(filename)
    num_stamps = vi.stamps_per_file(filename)
    print(f"There are {num_stamps} stamps in the file")
 
    for (i, st) in enumerate(vi.read_stamps(filename)):
        print(f"stamp {i} from {filename}")
        st.get_metadata()
        outfile = os.path.splitext(os.path.basename(filename))[0]+ f"_stamp_{i}.png" 
        #st.show_antennas(outfile)
        times = st.times()
        freq = st.frequencies()
        if hitfreq in freq:
            print(f"Hit found in stamp{i}")
        #data = st.complex_array()
        #print(len(times))
        print(freq[0], freq[-1])
        #print(data.shape)

#print(stamp)
#stamps = vi.read_diff_stamps(filename)
#print(stamps)