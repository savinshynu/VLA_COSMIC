import os
import sys
import viewer as vi
import numpy as np

filename  = sys.argv[1]
#stamp = vi.read_stamps(filename)
#stamp = vi.Stamp(filename)


for (i, st) in enumerate(vi.read_stamps(filename)):
    print(f"stamp {i} from {filename}")
    outfile = os.path.splitext(os.path.basename(filename))[0]+ f"_stamp_{i}.png" 
    st.show_antennas(outfile)
    times = st.times()
    freq = st.frequencies()
    data = st.complex_array()
    print(len(times))
    print(freq)
    print(data.shape)

#print(stamp)
#stamps = vi.read_diff_stamps(filename)
#print(stamps)