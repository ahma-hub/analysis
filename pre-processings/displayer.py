################################################################################
import numpy            as np
import matplotlib, sys
import argparse

from tqdm              import tqdm
from signal_processing import unpackData

## to avoid bug when it is run without graphic interfaces
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except ImportError:
    # print ('Warning importing GTK3Agg: ', sys.exc_info()[0])
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    
################################################################################
def display_matrix (t, f, mat, path_save, bandwidth = None):
################################################################################
# display_matrix
# display the matrix mat, with y_axis = f, x_axis = t, if a path_save is given
# the figure will be saved otherwise it will appear in a pop'up, bandwidth are
# highlighted if given
#
# inputs:
# - t: time axis,
# - f: frequency axis
# - mat: the matrix to display
# - path_save: to save the figure (if None, pop'up)
# - bandwidth: highlighted bandwidth
################################################################################
    # from: https://matplotlib.org/examples/pylab_examples/scatter_hist.html

    # definitions for the axes
    left, bottom  = 0.1, 0.1
    height, width = 0.5, 0.5

    size_hist = 0.2
    colorbar_size = 0.05

    spacing = 0.005
    spacing_colorbar = 0.05


    rect_colorbar = [left, bottom, colorbar_size, height]
    rect_scatter  = [left + colorbar_size + spacing_colorbar, bottom, width, height]

    rect_histx    = [left + colorbar_size + spacing_colorbar, bottom + height + spacing, width, size_hist]
    rect_histy    = [left + colorbar_size + spacing_colorbar + width + spacing, bottom, size_hist, height]


    # start with a rectangular Figure
    fig = plt.figure (figsize = (16, 9))

    ax_scatter = plt.axes (rect_scatter)
    ax_scatter.tick_params (direction='in', top=True, right=True)

    ax_histx = plt.axes (rect_histx)
    ax_histx.tick_params (direction='in', labelbottom=False)

    ax_histy = plt.axes (rect_histy)
    ax_histy.tick_params (direction='in', labelleft=False)

    ax_colorbar = plt.axes (rect_colorbar)
    ax_colorbar.tick_params (direction='in', labelright=False)



    im = ax_scatter.imshow (mat, cmap = 'Blues', interpolation ='none', aspect='auto',
                            origin ='lower',
                            extent = [t.min (), t.max (), f.min (), f.max ()])

    plt.colorbar(im, cax = ax_colorbar)
    ax_colorbar.yaxis.set_ticks_position('left')

    ax_scatter.set_xlabel ('Time (s)')
    ax_scatter.set_ylabel ('Freq (MHz)')

    ax_histx.plot (t, mat.mean (0))
    ax_histx.set_ylabel ('NICV')

    ax_histy.plot (mat.mean (1), f)
    ax_histy.set_xlabel ('NICV')

    ax_histx.set_xlim(ax_scatter.get_xlim ())
    ax_histy.set_ylim(ax_scatter.get_ylim ())

    # projection of axis
    if (bandwidth is not None):
        i = 0
        while (i < len (bandwidth) - 1):
            j = i
            while (bandwidth [j] - bandwidth [j + 1] == 1):
                j += 1
            ax_histy.axhspan (f [bandwidth [i]], f [bandwidth [j]], color = 'red')

            i = j

    ## save if a file is provided
    if (path_save == None):
        plt.show ()
    ## otherwise save it
    else:
        plt.savefig (f'{path_save}')
        plt.close ()


################################################################################
def display_trace (trace, path_save):
################################################################################
# display_trace
# display the trace
#
# inputs:
# - trace: to display
# - path_save: to save the figure (if None, pop'up)
################################################################################
    fig, axs = plt.subplots (1, figsize = (16, 9), sharex = True)

    axs.plot (trace)

     ## save if a file is provided
    if (path_save == None):
        plt.show ()
    ## otherwise save it
    else:
        plt.savefig (f'{path_save}')
        plt.close ()
    
################################################################################
if __name__ == '__main__':
################################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument ('--display_trace', action='store', type=str,
                         default = None,
                         dest='path_trace',
                         help='Absolute path to the trace to display')

    parser.add_argument ('--display_lists', action='store', type=str,
                         default=None, 
                         dest='path_lists',
                         help='Absolute path to the list to display')

    parser.add_argument ('--list_idx', action='store', type=int,
                         default=-1, 
                         dest='list_idx',
                         help='which list to display (all = -1, learning: 0, validating: 1, testing: 2)')
    
    parser.add_argument ('--metric', action='store', type=str,
                         default='mean',
                         dest='metric',
                         help='Applied metric for the display of set (mean, std, means, stds)')

    parser.add_argument ('--extension', default='dat',
                         type = str, dest = 'extension',
                         help = 'extensio of the raw traces ')

    parser.add_argument ('--path_save', default=None,
                         type = str, dest = 'path_save',
                         help = 'Absolute path to save the figure (if None, display in pop\'up)')
        
    
    # parser.add_argument('--log-level', default=logging.INFO,
    #                     type=lambda x: getattr(logging, x),
    #                     help = "Configure the logging level: DEBUG|INFO|WARNING|ERROR|FATAL")
    
    args, unknown = parser.parse_known_args ()
    assert len (unknown) == 0, f"[WARNING] Unknown arguments:\n{unknown}\n"

    ## if one trace is given 
    if (args.path_trace is not None):
        trace = unpackData (args.path_trace, args.extension)

        ## check if it is a [t, f, mat] file
        if (len (trace) == 3):
            t, f, trace = trace

            display_matrix (t, f, trace, args.path_save, None)

        else:
            display_trace (trace, args.path_save)
        
    ## if a list is given 
    elif (args.path_lists is not None):
        ## all learning, validating and testing
        if (args.list_idx == -1):

            lists = np.load (args.path_lists, allow_pickle = True)
            lists = lists [0] + lists [1] + lists [2]
            
        else:
            lists = np.load (args.path_lists, allow_pickle = True) [args.list_idx]

        
        trace = unpackData (lists [0], args.extension)
        ## check if it is a [t, f, mat] file and mean computed
        if (len (trace) == 3 and args.metric == 'mean'):
            acc_x = trace [2]
            for q in tqdm (range (1, len (lists)), desc = 'mean (stft)'):
                acc_x += unpackData (lists [q], args.extension)[2]

            res = acc_x/len (lists)
            display_matrix (trace [0], trace [1], res, args.path_save, None)

        elif (len (trace) == 3 and args.metric == 'means'):
            res = np.zeros (len (lists))
            res [0] = trace [2].mean ()
            for q in tqdm (range (1, len (lists)), desc = 'means (stft)'):
                res [q] = unpackData (lists [q], args.extension)[2].mean ()

            display_trace (res, args.path_save)
            
        ## if [t, f, mat] and std
        elif (len (trace) == 3 and args.metric == 'std'):
            acc_x  = trace [2]
            acc_xx = trace [2]**2
            for q in tqdm (range (1, len (lists)), desc= 'std (stft)'):
                tmp = unpackData (lists [q], args.extension)[2]
                acc_x  += tmp
                acc_xx += tmp**2
                
            res = np.sqrt (acc_xx/len (lists) - (acc_x/len (lists))**2 )
            display_matrix (trace [0], trace [1], res, args.path_save, None)

        elif (len (trace) == 3 and args.metric == 'stds'):
            res = np.zeros (len (lists))
            res [0] = trace [2].std ()
            for q in tqdm (range (1, len (lists)), desc = 'stds (stft)'):
                res [q] = unpackData (lists [q], args.extension)[2].std ()

            display_trace (res, args.path_save)
            
        ## if traces are 1D and mean
        elif (args.metric == 'mean'):
            acc_x = trace 
            for q in tqdm (range (1, len (lists)), desc = 'mean (trace)'):
                acc_x += unpackData (lists [q], args.extension)

            res = acc_x/len (lists)
            display_trace (res, args.path_save)

        elif (args.metric == 'means'):
            res = np.zeros (len (lists))
            res [0] = trace.mean ()
            for q in tqdm (range (1, len (lists)), desc = 'means (trace)'):
                res [q] = unpackData (lists [q], args.extension).mean ()

            display_trace (res, args.path_save)
            
        ## if traces are 1D and mean
        elif (args.metric == 'std'):
            acc_x  = trace
            acc_xx = trace**2
            for q in tqdm (range (1, len (lists)), desc = 'std (trace)'):
                tmp = unpackData (lists [q], args.extension)
                acc_x  += tmp
                acc_xx += tmp**2
                
            res = np.sqrt (acc_xx/len (lists) - (acc_x/len (lists))**2)
            display_trace (res, args.path_save)

        else:
            res = np.zeros (len (lists))
            res [0] = trace.std ()
            for q in tqdm (range (1, len (lists)), desc = 'stds (trace)'):
                res [q] = unpackData (lists [q], args.extension).mean ()

            display_trace (res, args.path_save)
