
#plotting script based on turboseti modified for commensal systems

r'''
'''

import sys
import os
from os import mkdir
from os.path import dirname, abspath, isdir
import gc
#import logging
#logger_plot_event_name = 'plot_event'
#logger_plot_event = logging.getLogger(logger_plot_event_name)
#logger_plot_event.setLevel(logging.INFO)

# Plotting packages import
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.use('agg')

import numpy as np
from astropy.time import Time
import blimpy as bl
from blimpy.utils import rebin

import qrcode

# preliminary plot arguments
fontsize=16
font = {'family' : 'DejaVu Sans',
'size' : fontsize}
MAX_IMSHOW_POINTS = (4096, 1268)


def overlay_drift(ax, f_event, f_start, f_stop, drift_rate, t_duration, offset=0, alpha=1, color='#cc0000'):
    r'''
    Creates a dashed red line at the recorded frequency and drift rate of
    the plotted event - can overlay the signal exactly or be offset by
    some amount (offset can be 0 or 'auto').
    '''
    # determines automatic offset and plots offset lines
    if offset == 'auto':
        offset = ((f_start - f_stop) / 10)
        ax.plot((f_event - offset, f_event),
                 (10, 10),
                 "o-",
                 c=color,
                 lw=2,
                 alpha=alpha)

    # plots drift overlay line, with offset if desired
    ax.plot((f_event + offset, f_event + drift_rate/1e6 * t_duration + offset),
             (0, t_duration),
             c=color,
             ls='dashed', lw=2,
             alpha=alpha)


def plot_waterfall(ax, wf, beamid, f_start=None, f_stop=None, **kwargs):
    r"""
    Plot waterfall of data in a .fil or .h5 file.
    Parameters
    ----------
    wf : blimpy.Waterfall object
        Waterfall object of an H5 or Filterbank file containing the dynamic spectrum data.
    source_name : str
        Name of the target.
    f_start : float
        Start frequency, in MHz.
    f_stop : float
        Stop frequency, in MHz.
    kwargs : dict
        Keyword args to be passed to matplotlib imshow().
    Notes
    -----
    Plot a single-panel waterfall plot (frequency vs. time vs. intensity)
    for one of the on or off observations in the cadence of interest, at the
    frequency of the expected event. Calls :func:`~overlay_drift`
    """
    # prepare font
    matplotlib.rc('font', **font)

    # Load in the data from fil
    plot_f, plot_data = wf.grab_data(f_start=f_start, f_stop=f_stop)

    # Make sure waterfall plot is under 4k*4k
    dec_fac_x, dec_fac_y = 1, 1

    # rebinning data to plot correctly with fewer points
    try:
        if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
            dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]
        if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
            dec_fac_y =  int(np.ceil(plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]))
        plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)
    except Exception as ex:
        print('\n*** Oops, grab_data returned plot_data.shape={}, plot_f.shape={}'
              .format(plot_data.shape, plot_f.shape))
        print('Waterfall info for {}:'.format(wf.filename))
        wf.info()
        raise ValueError('*** Something is wrong with the grab_data output!') from ex


    # determine extent of the plotting panel for imshow
    extent=(plot_f[0], plot_f[-1], (wf.timestamps[-1]-wf.timestamps[0])*24.*60.*60, 0.0) 
    print (extent)
    # plot and scale intensity (log vs. linear)
    kwargs['cmap'] = kwargs.get('cmap', 'viridis')
    plot_data = 10.0 * np.log10(plot_data)

    # get normalization parameters
    vmin = plot_data.min()
    vmax = plot_data.max()
    normalized_plot_data = (plot_data - vmin) / (vmax - vmin)

    # display the waterfall plot
    this_plot = ax.imshow(normalized_plot_data,
        aspect='auto',
        rasterized=True,
        interpolation='nearest',
        extent=extent,
        **kwargs
    )

    # add plot labels
    #plt.xlabel("Frequency [Hz]",fontdict=font)
    #plt.ylabel("Time [s]",fontdict=font)
    

    # add source name
    #ax = plt.gca()
    ax.text(0.03, 0.8, 'B:'+beamid, transform=ax.transAxes, bbox=dict(facecolor='white'))

    del plot_f, plot_data
    gc.collect()

    return this_plot, normalized_plot_data


def make_waterfall_plots(h5file_list, candidate,  offset=0, plot_dir=None, **kwargs):
    r'''
    Makes waterfall plots of an event for an entire on-off cadence.
    Parameters
    ----------
    h5file_list : str
        List of h5 files in the cadence.
    source_name : str
        Name of the on_source target.
    f_start : float
        Start frequency, in MHz.
    f_stop : float
        Stop frequency, in MHz.
    drift_rate : float
        Drift rate in Hz/s.
    f_mid : float
        <iddle frequency of the event, in MHz.
    kwargs : dict
        Keyword args to be passed to matplotlib imshow().
    Notes
    -----
    Makes a series of waterfall plots, to be read from top to bottom, displaying a full cadence
    at the frequency of a recorded event from find_event. Calls :func:`~plot_waterfall`
    '''
    

    source_name = candidate['Source']
    f_mid = candidate['event_dat'][3] # Collecting info from the event_dat array within the dictionary
    drift_rate = candidate['event_dat'][1]

    # calculate the length of the total cadence from the fil files' headers
    first_h5 = bl.Waterfall(h5file_list[0], load_data=False)
    tfirst = first_h5.header['tstart']
    t_elapsed = first_h5.n_ints_in_file * first_h5.header['tsamp']


    # calculate the width of the plot based on making sure the full drift is visible
    bandwidth = 2.4 * abs(drift_rate)/1e6 * t_elapsed
    bandwidth = np.max((bandwidth, 500./1e6))

    # Get start and stop frequencies based on midpoint and bandwidth
    f_start, f_stop = np.sort((f_mid - (bandwidth/2),  f_mid + (bandwidth/2)))


    #global logger_plot_event

    # prepare for plotting
    matplotlib.rc('font', **font)


    #Setting up the grid
    #grid = gridspec.GridSpec(nrows=4, ncols=3)
    # set up the sub-plots
        
    n_plots = int(np.sqrt(len(h5file_list)))
    #fig = plt.subplots(n_plots, n_plots, sharex=True, sharey=True,figsize=(3*n_plots, 3*n_plots))
    fig = plt.figure(figsize=(5*n_plots, 3*n_plots),constrained_layout=True)

    
    plot_data_list = []

    #Adding the header info
    #plt.subplot(grid[:1, :2])

    subfigs = fig.subfigures(2, 1, wspace=0.07, height_ratios = [1,3] )
 
    ax_top = subfigs[0].subplots(1, 2)
    subfigs[0].suptitle("Candidate information", size=15, fontweight='bold')


    ax_top[0].axis('off')
    #ax_top[0].text(0.5, 1.00, "Candidate information", size=15, fontweight='bold')
    ax_top[0].text(0.1, 0.70, "Source: " + source_name, size = 13)
    ax_top[0].text(0.1, 0.55, "RA: " + candidate['RA'], size = 13)
    ax_top[0].text(0.1, 0.40, "Dec: " + candidate['DEC'], size = 13)
    ax_top[0].text(0.1, 0.25, "MJD Time: " + candidate['MJD'], size = 13)
    ax_top[0].text(0.7, 0.70, "Observation Length: " + candidate['obs_length']+ " s", size = 13)
    ax_top[0].text(0.7, 0.55, "Drift Rate: " + str(round(drift_rate,6)) + " Hz/s", size = 13)
    ax_top[0].text(0.7, 0.40, "Start Frequency: " + str(round(f_start,6)) + " MHz", size = 13)
    ax_top[0].text(0.7, 0.25, "End Frequency: " + str(round(f_stop,6)) + " MHz", size = 13)

    #Adding the QR code thing
    #plt.subplot(grid[:1, 2:])
    message_list = ("Source: " + source_name,  
              "RA: " + candidate['RA'],  
              "Dec: " + candidate['DEC'], 
              "MJD Time: " + candidate['MJD'],
              "Observation Length: " + candidate['obs_length']+ " s", 
              "Drift Rate: " + str(round(drift_rate,6)) + " Hz/s", 
              "Start Frequency: " + str(round(f_start,6)) + " MHz",
              "End Frequency: " + str(round(f_stop,6)) + " MHz" )
    message = '\n'.join(message_list)
              
    img = qrcode.make(message)
    img.thumbnail((400, 400))   
    ax_top[1].imshow(img, cmap='gray')
    ax_top[1].axis('off')

    # get directory path for storing PNG files
    if plot_dir is None:
        dirpath = dirname(abspath(h5file_list[0])) + '/'
        #mkdir(dirpath)
    else:
        if not isdir(plot_dir):
            mkdir(plot_dir)
        dirpath = plot_dir


    # read in data for the first panel
    max_load = bl.calcload.calc_max_load(h5file_list[0])
    #print('plot_event make_waterfall_plots: max_load={} is required for {}'.format(max_load, fil_file_list[0]))
    wf1 = bl.Waterfall(h5file_list[0], f_start=f_start, f_stop=f_stop, max_load=max_load)
    t0 = wf1.header['tstart']
    plot_f1, plot_data1 = wf1.grab_data()

    if plot_data1.shape[0] > MAX_IMSHOW_POINTS[0] or plot_data1.shape[1] > MAX_IMSHOW_POINTS[1]:
       sys.exit('Needs rebinning of the data or increase the plot size')

    """
    # rebin data to plot correctly with fewer points
    dec_fac_x, dec_fac_y = 1, 1
    if plot_data1.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = plot_data1.shape[0] / MAX_IMSHOW_POINTS[0]
    if plot_data1.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y =  int(np.ceil(plot_data1.shape[1] /  MAX_IMSHOW_POINTS[1]))
    plot_data1 = rebin(plot_data1, dec_fac_x, dec_fac_y)
    """

    mid_f = np.abs(f_start+f_stop)/2.

    # More overall plot formatting, axis labelling
    factor = 1e6
    units = 'Hz'

    #ax = plt.gca()
    #ax.get_xaxis().get_major_formatter().set_useOffset(False)
    xloc = np.linspace(f_start, f_stop, 5)
    xticks = [round(loc_freq) for loc_freq in (xloc - mid_f)*factor]
    if np.max(xticks) > 1000:
       xticks = [xt/1000 for xt in xticks]
       units = 'kHz'



    #subplots = []
    del wf1, plot_f1, plot_data1
    gc.collect()

    #row, col = [0,0]
    
    subfigs[1].suptitle('Waterfall Plots', size=14)  
    subfigs[1].supxlabel("Rel. Freq. [%s] from %f MHz"%(units,mid_f),fontdict=font, size=14)
    subfigs[1].supylabel('Time (s)', size=14)
    
    ax_bot = subfigs[1].subplots(n_plots, n_plots, sharex=True, sharey = True)
    #print(ax_bot)
    # Fill in each subplot for the full plot
    
    ii = 0
    for row in range(n_plots):
        for col in range(n_plots):

            #for ii, filename in enumerate(h5file_list):
            
            filename = h5file_list[ii]

            beamid = os.path.splitext(os.path.basename(filename))[0].split('_')[-1][1:]
           
            # identify panel
            #subplot = plt.subplot(n_plots, n_plots, ii + 1)
            #subplots.append(subplot)

            #plt.subplot(grid[1+row, col])

            """
            col += 1

            if (ii+1) % 3 == 0:
            row += 1
            col = 0
    
            if col % (n_plots-1) == 0:
            col = 0
            row += 1
            """
            #filename = h5file_list[ii]

            # read in data
            max_load = bl.calcload.calc_max_load(filename)
            #print('plot_event make_waterfall_plots: max_load={} is required for {}'.format(max_load, filename))
            wf = bl.Waterfall(filename, f_start=f_start, f_stop=f_stop, max_load=max_load)
            # make plot with plot_waterfall
            this_plot, plt_data = plot_waterfall(ax_bot[row][col], wf,
                                   beamid,
                                   f_start=f_start,
                                   f_stop=f_stop,
                                   **kwargs)
            
            print(plt_data.shape)
            plot_data_list.append(plt_data)
            # calculate parameters for estimated drift line
            t_duration = (wf.n_ints_in_file) * wf.header['tsamp']
            f_event = f_mid # + drift_rate / 1e6 * t_elapsed

            # plot estimated drift line
            overlay_drift(ax_bot[row][col], f_event, f_start, f_stop, drift_rate, t_duration, offset)

            # Title the full plot
            #if ii < n_plots:
            #    plot_title = "%s \n $\\dot{\\nu}$ = %2.3f Hz/s, MJD:%5.5f" % (source_name, drift_rate, t0)
            #    plt.title(plot_title)
        
            # Format full plot
            #if ii < len(fil_file_list)-1:
            #    plt.xticks(np.linspace(f_start, f_stop, num=4), ['','','',''])

        

            # More overall plot formatting, axis labelling
            factor = 1e6
            units = 'Hz'

            #ax = plt.gca()
            #ax.get_xaxis().get_major_formatter().set_useOffset(False)
            xloc = np.linspace(f_start, f_stop, 5)
            xticks = [round(loc_freq) for loc_freq in (xloc - mid_f)*factor]
            if np.max(xticks) > 1000:
                xticks = [xt/1000 for xt in xticks]
                units = 'kHz'
        
            ax_bot[row][col].set_xticks(xloc, xticks)
            #plt.xlabel("Rel. Freq. [%s] from %f MHz"%(units,mid_f),fontdict=font) 
         
            ii += 1
            del wf
            gc.collect()

    


    #Estimation of average
    avg_plt_dat = np.zeros((len(plot_data_list), plot_data_list[0].shape[0], plot_data_list[0].shape[1]))
    for p, pltd in enumerate(plot_data_list):
        avg_plt_dat[p,:,:] = pltd
    avg_plt = np.mean(avg_plt_dat, axis = 0 ) 
    
    """
    # display the waterfall plot
    new_plot = ax.imshow(avg_plt,
        aspect='auto',
        rasterized=True,
        interpolation='nearest',
        extent=extent,
        **kwargs
    )
    
    """
     

    subfigs[1].colorbar(this_plot, shrink=0.8, ax=ax_bot, label='Normalized Power (Arbitrary Units)')

    # More overall plot formatting, axis labelling
    #factor = 1e6
    #units = 'Hz'

    #ax = plt.gca()
    #ax.get_xaxis().get_major_formatter().set_useOffset(False)
    #xloc = np.linspace(f_start, f_stop, 5)
    #xticks = [round(loc_freq) for loc_freq in (xloc - mid_f)*factor]
    #if np.max(xticks) > 1000:
    #    xticks = [xt/1000 for xt in xticks]
    #    units = 'kHz'
    #plt.xticks(xloc, xticks)
    #plt.xlabel("Relative Frequency [%s] from %f MHz"%(units,mid_f),fontdict=font)


    # Title of the plot
    #plot_title = "%s \n $\\dot{\\nu}$ = %2.3f Hz/s, MJD:%5.5f" % (source_name, drift_rate, t0)
    #plt.title(plot_title)
    
    #fig[0].supxlabel("Rel. Freq. [%s] from %f MHz"%(units,mid_f),fontdict=font)
    #fig[0].supylabel('Time (s)')
    #fig[0].suptitle(source_name)

    # Add colorbar
    #cax = fig[0].add_axes([0.91, 0.11, 0.03, 0.8])
    #fig[0].colorbar(this_plot,cax=cax,label='Normalized Power (Arbitrary Units)')
    #fig[0].colorbar(this_plot, label='Normalized Power (Arbitrary Units)')
    # Adjust plots
    #plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.05, hspace = 0.05)
    #fig[0].tight_layout()

    # save the figures
    path_png = dirpath + source_name + '_dr_' + "{:0.2f}".format(drift_rate) + '_freq_' "{:0.6f}".format(f_start) + ".png"
    plt.savefig(path_png, bbox_inches='tight')

    # show figure before closing if this is an interactive context
    mplbe = matplotlib.get_backend()
    if mplbe != 'agg':
        plt.show()


    #plt.show() # for now
    
    # close all figure windows
    plt.close('all')

    #return subplots


def plot_candidate_events(event_list, h5file_list, plot_dir,  offset = 0, **kwargs):
    r'''
    >>> plot_event.plot_candidate_events(event_list, h5file_list, offset=0)
    '''
    #global logger_plot_event

    len_events = len(event_list)
    if len_events < 1:
       sys.exit("No events to plot from the candidate list")
    for i in range(0, len_events-1):
        candidate = event_list[i]

        """
        source_name = candidate['Source']
        f_mid = candidate['event_dat'][3] # Collecting info from the event_dat array within the dictionary
        drift_rate = candidate['event_dat'][1]

        # calculate the length of the total cadence from the fil files' headers
        first_h5 = bl.Waterfall(h5file_list[0], load_data=False)
        tfirst = first_h5.header['tstart']
        t_elapsed = first_h5.n_ints_in_file * first_h5.header['tsamp']


        # calculate the width of the plot based on making sure the full drift is visible
        bandwidth = 2.4 * abs(drift_rate)/1e6 * t_elapsed
        bandwidth = np.max((bandwidth, 500./1e6))

        # Get start and stop frequencies based on midpoint and bandwidth
        f_start, f_stop = np.sort((f_mid - (bandwidth/2),  f_mid + (bandwidth/2)))
        """

        # Pass info to make_waterfall_plots() function
        make_waterfall_plots(h5file_list,
                             candidate,
                             offset=offset,
                             plot_dir=plot_dir,
                             **kwargs)
