################################################################################
from  list_manipulation  import get_tag
from  signal_processing  import stft

import numpy            as np
import argparse

import itertools
import logging
import sys, os, glob, re


from   tqdm             import tqdm
from   argparse         import ArgumentParser
from   displayer        import display_matrix


import matplotlib, sys
## to avoid bug when it is run without graphic interfaces
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except ImportError:
    # print ('Warning importing GTK3Agg: ', sys.exc_info()[0])
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

################################################################################
def compute_nicv  (path_lists, path_acc, path_save):
################################################################################
# compute_nicv
# compute the nicv of the given data clustered by using the labels
# inputs:
#  - path_lists: path to the lists of the data (cf list_manipulation)
#  - path_acc: path to the directory that contain the accumulators (cf accumulators)
#  - path_save: where the plotting will be save, if log is DEBUG
#
# output:
#  - t, f, nicv: the time axis, the frequency axis and the nicvs values
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
    
    # some acc needed to compute the nicv
    count = 0

    nicv_acc = np.float64 (0.)

    acc_x  = np.float64 (0.)
    acc_xx = np.float64 (0.)
        
    # for all labels
    for i in tqdm (range (len (unique_y)), desc='NICV (%s groups)'%len(unique_y)):
        ## get the number of differents accumultors are needed for the current label
        # 1) get the position of the current label
        idx = np.where (np.array (y) == unique_y [i])[0]
  
        
        # 2) get the names (unique)
        current_names_idx = np.unique (idx_names [idx])

        current_count = 0.
        current_acc = np.float64 (0.)
        
        for j in range (len (current_names_idx)):

            t, f, tmp_acc_x  = np.load (acc_names [current_names_idx [j]][0],  allow_pickle=True)
            t, f, tmp_acc_xx = np.load (acc_names [current_names_idx [j]][1], allow_pickle=True)

            ## update accumulators
            acc_x  += tmp_acc_x
            acc_xx += tmp_acc_xx

            current_acc += tmp_acc_x

            current_count += float (counts_acc [current_names_idx [j]])
            count         += float (counts_acc [current_names_idx [j]])
        
        nicv_acc += (current_acc**2)/current_count
        
    res = ((nicv_acc/count) - (acc_x/count)**2)/\
        (acc_xx/count - (acc_x/count)**2)

    # revome potential Nan caused of constant distribution
    np.nan_to_num (res, copy = False, nan = 0)

       
    # display
    if ((logging.root.level < logging.INFO) or (path_save is not None)):
        display_matrix (t, f, res, path_save)
        
    return t, f, res

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

    parser.add_argument ('--plot', action = 'store', default = None,
                         type = str, dest = 'path_to_plot',
                         help = 'Absolute path to a previously saved NICV in order to display it')
    
    parser.add_argument('--log-level', default=logging.INFO,
                        type=lambda x: getattr(logging, x),
                        help = "Configure the logging level: DEBUG|INFO|WARNING|ERROR|FATAL")

    args, unknown = parser.parse_known_args ()
    assert len (unknown) == 0, f"[ERROR] Unknown arguments:\n{unknown}\n"
    
    logging.basicConfig(level=args.log_level)

    compute_nicv (args.path_lists,
                  args.path_acc,
                  args.path_to_plot)
    
