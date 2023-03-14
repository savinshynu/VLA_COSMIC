import sys
import os
import shutil

#base = sys.argv[1]

filename  = '/mnt/slow/savin_vla_analysis/msdata/uvh5_59937_82548_51023026977_J0319+4130_0001.ms'
base = os.path.basename(os.path.splitext(filename)[0])
print(base)

#LOB
plotms(vis=filename, xaxis='frequency', yaxis='phase', antenna='!*&&&', correlation='xx,yy', 
        avgtime='1200', iteraxis='baseline', gridrows=5, gridcols=4, coloraxis='corr', titlefont=12, 
        xaxisfont=11, yaxisfont=11, plotfile=base+'.png', highres=True, width=1500, height=1200, plotrange=[0, 0, -180, 180], showgui=False)


#gain K(delay) for refant ea10
gaincal(filename, caltable="cal.K", refant="ea10", gaintype="K")
applycal(filename, gaintable="cal.K", calwt=False)
split(filename, outputvis=base+'_k.ms', datacolumn='corrected')

#gain G for refant ea10
gaincal(base+'_k.ms', caltable="cal.G", refant="ea10", gaintype="G")
applycal(base+'_k.ms', gaintable="cal.G", calwt=False)
split(base+'_k.ms', outputvis=base+'_kg.ms', datacolumn='corrected')

#gain BP for refant ea10
bandpass(base+'_kg.ms', caltable="cal.BP", refant="ea10", bandtype="B", minblperant=6)
applycal(base+'_kg.ms', gaintable="cal.BP", calwt=False)
split(base+'_kg.ms', outputvis=base+'_kgbp.ms', datacolumn='corrected')

plotms(vis=base+'_kgbp.ms', xaxis='frequency', yaxis='phase', antenna='!*&&&', correlation='xx,yy', 
        avgtime='1200', iteraxis='baseline', gridrows=5, gridcols=4, coloraxis='corr', titlefont=12, 
        xaxisfont=11, yaxisfont=11, plotfile=base+'_kgbp.png', highres=True, width=1500, height=1200, plotrange=[0, 0, -180, 180], showgui=False)

print("CASA work complete")
