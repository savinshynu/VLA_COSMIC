import numpy as np

def comp_bp_coarse(data):

    med_spectra =  np.median(data, axis = 0)
    smth = med_spectra*0.0
    winSize = 1000 
    
    #Compute the smoothed bandpass model
    for i in range(smth.size):
        mn = max([0, i-winSize/2])
        mx = min([i+winSize/2+1, smth.size])
        smth[i] = np.median(med_spectra[int(mn):int(mx)])      
           
    return smth
