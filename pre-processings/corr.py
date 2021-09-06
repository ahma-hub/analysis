################################################################################
from  list_manipulation  import get_tag

import numpy            as np
import argparse

import logging
import sys, os, glob, re

from   tqdm             import tqdm
from   displayer        import display_matrix

import matplotlib
## to avoid bug when it is run without graphic interfaces
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except ImportError:
    # print ('Warning importing GTK3Agg: ', sys.exc_info()[0])
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

################################################################################
def compute_corr (path_lists, path_acc, path_save, scale = 'normal',
                  bandwidth_nb = 0, metric = 'corr_max', time_limit = 1):
################################################################################
# compute_corr
# compute the corr of the given data clustered by using the labels
# inputs:
#  - path_lists: path to the lists of the data (cf list_manipulation)
#  - path_acc: path to the directory that contain the accumulators (cf accumulators)
#  - path_save: where the plotting will be save, (if log is DEBUG a window will pop'up)
#  - scale: normal of log scale
#  - bandwidth_nb: nb of bandwidth to extract
#  - metric: metric to use for the bandwidth selection 
#  - time_limit: percentage of the trace (from the begining) 
# 
# output:
#  - t, f, corr: the time axis, the frequency axis and the corrs values
################################################################################
    # load lists
    [x_train_filelist, x_val_filelist, _, y_train, y_val, _]\
            = np.load (path_lists, allow_pickle = True)

    list_files = x_train_filelist + x_val_filelist
    y = y_train + y_val

    # get names of the accumulators
    names = [get_tag (f) for f in list_files]
    
    unique_names, idx_names = np.unique (names, return_inverse = True)
    
    # creat the list of acc_names
    acc_names = []
    counts_acc = []

    for i in range (len (unique_names)):
        # get the two corresponding accumulators {acc_x, acc_xx} and
        # the size of each acc.

        reg_exp = re.compile (path_acc + re.escape (unique_names [i])  + r'_\d+_'+ 'acc_x.npy')
        tmp_acc_x  = list (filter (reg_exp.match, glob.glob (path_acc + '/*'))) [0]

        reg_exp = re.compile (path_acc + re.escape (unique_names [i])  + r'_\d+_'+ 'acc_xx.npy')
        tmp_acc_xx = list (filter (reg_exp.match, glob.glob (path_acc + '/*'))) [0]

        acc_names.append ([tmp_acc_x, tmp_acc_xx])
        counts_acc.append (int (tmp_acc_x.split ('_')[-3]))
        
    # get list of labels (with no repeatition)
    unique_y = np.unique (y)
    
    # some acc needed to compute the corr
    count = 0

    # corr_acc = np.float64 (0.)

    acc_x  = np.float64 (0.)
    acc_xx = np.float64 (0.)

    acc_y  = np.float64 (0.)
    acc_yy = np.float64 (0.)
    
    acc_xy = np.float64 (0.)
    
    # for all labels
    for i in tqdm (range (len (unique_y)), desc='CORR (%s groups)'%len(unique_y)):
        ## get the number of differents accumultors are needed for the current label
        # 1) get the position of the current label
        idx = np.where (np.array (y) == unique_y [i])[0]
  
        
        # 2) get the names (unique)
        current_names_idx = np.unique (idx_names [idx])
        
        for j in range (len (current_names_idx)):

            t, f, tmp_acc_x  = np.load (acc_names [current_names_idx [j]][0],  allow_pickle=True)
            t, f, tmp_acc_xx = np.load (acc_names [current_names_idx [j]][1], allow_pickle=True)

            # nb of samples
            D = int (time_limit*tmp_acc_x.shape [0])
            
            ## update accumulators
            acc_x  += tmp_acc_x [:D, :]
            acc_xx += tmp_acc_xx [:D, :]

            acc_xy += i*tmp_acc_x [:D, :]

            acc_y  += i*float (counts_acc [current_names_idx [j]])
            acc_yy += (i**2)*float (counts_acc [current_names_idx [j]])
            
            count  += float (counts_acc [current_names_idx [j]])
        
        
    res = (count*acc_xy - acc_x*acc_y)/(np.sqrt (count*acc_xx - acc_x**2) * np.sqrt (count*acc_yy - acc_y**2))
    res = np.abs (res)

    # revome potential Nan caused of constant distribution
    np.nan_to_num (res, copy = False, nan = 0)

    ## get the bandwidth with the highest mean
    if (bandwidth_nb != 0):
        if ('max' in metric):
            bandwidth = np.argsort (res.max (1)) [-bandwidth_nb:]
        else:
            bandwidth = np.argsort (res.mean (1)) [-bandwidth_nb:]

        ## then sort from the smallest to the biggest frequency values
        bandwidth = np.sort (bandwidth) 

    else:
        bandwidth = None
        
    # display
    if ((logging.root.level < logging.INFO) or (path_save is not None)):
        if (scale == 'log'):
            display_matrix (t, f, np.log (res), path_save, bandwidth)
        else:
            display_matrix (t, f, res, path_save, bandwidth)
        
    return t, f, res, bandwidth


################################################################################
if __name__ == '__main__':
################################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument ('--acc', action='store', type=str,
                        dest='path_acc',
                        help='Absolute path of the accumulators directory')

    parser.add_argument ('--lists', action = 'store',
                         type = str, dest = 'path_lists',
                         help = 'Absolute path to a file containing the main lists')

    parser.add_argument ('--plot', action = 'store', default=None,
                         type = str, dest = 'path_to_plot',
                         help = 'Absolute path to the file where to save the plot '
                         + '(/!\ \'.png\' expected at the end of the filename)')

    parser.add_argument('--scale', action='store', type=str, default='normal',
                        dest='scale',
                        help='scale of the plotting: normal|log')
        
    parser.add_argument('--bandwidth_nb', default=0,
                        type=int,
                        help = "display the nb of selected bandwidth, by default no bandwidth selected")

    parser.add_argument('--metric', action='store', type=str, default='nicv_max',
                            dest='metric', help='Metric used to select bandwidth: {corr}_{mean, max} ')
      
    parser.add_argument('--log-level', default=logging.INFO,
                        type=lambda x: getattr(logging, x),
                        help = "Configure the logging level: DEBUG|INFO|WARNING|ERROR|FATAL")

    args, unknown = parser.parse_known_args ()
    assert len (unknown) == 0, f"[WARNING] Unknown arguments:\n{unknown}\n"
        
    logging.basicConfig (level=args.log_level)

    compute_corr (args.path_lists,
                  args.path_acc,
                  args.path_to_plot,
                  args.scale,
                  args.bandwidth_nb,
                  args.metric)
