################################################################################
import sys, os, glob
sys.path.append (os.path.join (os.path.dirname (__file__), '..', 'oscilloscopes', ))

# from   utils             import unpackData

import argparse

from   nicv import compute_nicv
from   signal_processing import stft, unpackData

import numpy as np
import copy

import matplotlib, sys
## to avoid bug when it is run without graphic interfaces
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print ('Warning importing GTK3Agg: ', sys.exc_info()[0])
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
    
from tqdm               import tqdm
import logging
from pathlib            import Path

################################################################################
def compute_poi (path_lists, path_acc, nb_of_bandwidth) :
################################################################################
# compute_poi
# compute the index of the sorted Points-Of-Interests
# inputs:
#  - path_lists: list of files (cf list_manipulation)
#  - path_acc: path to the file where the accumulators are located
#  - nb_of_bandwidth: nb of bandwidth to conserve
# outputs:
#  - poi: 2D array of the concerved indexes
#  - t, f: t-axis and f-axis of the nicv (= spectrograms)
################################################################################

    t, f, nicv = compute_nicv (path_lists, path_acc)

    # in order to remove border effects
    delta = 1
    nicv [:delta, :] = 0
    nicv [-delta:, :] = 0
    
    nicv [:, :delta] = 0
    nicv [:, -delta:] = 0

    # compute the sum over the time
    idx = np.argsort (nicv.sum (1))

    # concerve the highest values
    tmp = np.zeros (nicv.shape)
    tmp [idx [-nb_of_bandwidth:], :] = 1
    
    points = np.where (tmp == 1)
    
    # display
    if (logging.root.level < logging.INFO):
        fig, axs = plt.subplots (1, 2, figsize = (16, 9), sharex=True, sharey=True)

        axs [1].scatter(t [points [1]], f [points [0]], marker='.')

        # axs [1].set_ylabel('Frequency [Hz]')
        axs [1].set_xlabel('Time [s]')
        axs [1].set_title ('Selected Frequencies')

        ## hack for the paper
        # Create a Rectangle patch
        delta = 0.01e6
        rect = Rectangle((0 , np.unique (f [points [0]])[0] - delta),
                         2.5, np.unique (f [points [0]])[3] - np.unique (f [points [0]])[0] + 2* delta, linewidth=1, edgecolor='r',facecolor='none')
        
        axs [1].add_patch(rect)
        axs [1].text(t [int (len (t)/2)], np.unique (f [points [0]])[3]  + 2* delta, "4 frequencies", color = 'red')

        # Create a Rectangle patch
        rect = Rectangle((0 , np.unique (f [points [0]])[4] - delta),
                         2.5, np.unique (f [points [0]])[5] - np.unique (f [points [0]])[4] + 2* delta, linewidth=1, edgecolor='r',facecolor='none')
        
        axs [1].add_patch(rect)
        axs [1].text(t [int (len (t)/2)], np.unique (f [points [0]])[5]  + 2* delta, "2 frequencies", color = 'red')
        
        # Create a Rectangle patch
        rect = Rectangle((0 , np.unique (f [points [0]])[6] - delta),
                         2.5, np.unique (f [points [0]])[8] - np.unique (f [points [0]])[6] + 2* delta, linewidth=1, edgecolor='r',facecolor='none')
        
        axs [1].add_patch(rect)
        axs [1].text(t [int (len (t)/2)], np.unique (f [points [0]])[8]  + 2* delta, "3 frequencies", color = 'red')

        
        # Create a Rectangle patch
        rect = Rectangle((0 , np.unique (f [points [0]])[9] - delta),
                         2.5, 2*delta, linewidth=1, edgecolor='r',facecolor='none')
        
        axs [1].add_patch(rect)
        axs [1].text(t [int (len (t)/2)], np.unique (f [points [0]])[9]  + 2* delta, "1 frequencies", color = 'red')

        
        im = axs [0].imshow (nicv, cmap = 'Blues', interpolation ='none', aspect='auto',
                             origin ='lower',
                             extent = [t.min (), t.max (), f.min (), f.max ()])

        fig.colorbar(im, ax = [axs [0]], location='left')

        axs [0].set_ylabel('Frequency [Hz]')
        axs [0].set_xlabel('Time [s]')

        axs [0].set_title ('NICV')


        plt.subplots_adjust(wspace=0.05, hspace=0.05, left = 0.22)# , left = 0.1, right = 0.12)
        plt.show ()

    return t, f, points


################################################################################
def generate_dataset (t, f, poi, path_lists, path_output_lists, path_output_traces, freq, window, overlap):
################################################################################
# generate_dataset
# from raw data to extracted bandwidth
#  - t, f: t-axis and f-axis of the nicv (= spectrograms)
#  - poi: 2D array of the concerved indexes
#  - path_lists: lists of files and labels
#  - path_output_lists: path where the new list will be saved
#  - path_output_traces: directory where the extracted bandwidth will be saved
#  - freq, window, overlap: parameters of the stft
################################################################################
    # load lists
    [x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test]\
        = np.load (path_lists, allow_pickle = True)

    ## learning
    # traces = np.zeros ((len (poi [0]), len (x_train_filelist)))

    unique_f = np.uint64 (np.unique (poi [0]))
    
    new_train_list = ['']*len (x_train_filelist)
    new_val_list   = ['']*len (x_val_filelist)
    new_test_list  = ['']*len (x_test_filelist)
    
    ## learning
    for i in tqdm (range (len (x_train_filelist)), desc = 'learning data'):
        tmp_stft =  stft (unpackData (x_train_filelist [i], device = 'pico'),\
                              freq, window, overlap, False) [-1][unique_f, :]

        tmp_path = path_output_traces + '/' + x_train_filelist [i].split ('/')[-1]
        new_train_list [i] = tmp_path
        
        np.save (tmp_path, tmp_stft, allow_pickle = True)
        
    ## validating
    for i in tqdm (range (len (x_val_filelist)), desc = 'validating data'):
        tmp_stft =  stft (unpackData (x_val_filelist [i], device = 'pico'),\
                              freq, window, overlap, False) [-1][unique_f, :]
    
        tmp_path = path_output_traces + '/' + x_train_filelist [i].split ('/')[-1]
        new_val_list [i] = tmp_path
        
        np.save (tmp_path, tmp_stft, allow_pickle = True)
    
    ## testing
    for i in tqdm (range (len (x_test_filelist)), desc = 'testing data'):
        tmp_stft =  stft (unpackData (x_test_filelist [i], device = 'pico'),\
                              freq, window, overlap, False) [-1][unique_f, :]

        tmp_path = path_output_traces + '/' + x_train_filelist [i].split ('/')[-1]
        new_test_list [i] = tmp_path

        np.save (tmp_path, tmp_stft, allow_pickle = True)
    
    ## save poi
    np.save (path_output_traces + '/poi', [t, f, poi], allow_pickle = True)

    ## save new list
    np.save (path_output_lists,
             [new_train_list, new_val_list, new_test_list, y_train, y_val, y_test],
             allow_pickle = True)
 
################################################################################
if __name__ == '__main__':
################################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument ('--acc', action='store', type=str,
                         dest='path_acc',
                         help='Absolute path of the accumulators directory')

    parser.add_argument ('--lists', action='store', type=str,
                         # default='../../Accumulators-2.43s-2Mss-paper-stft/',
                         dest='path_lists',
                         help='list of the file')

    parser.add_argument('--nb_of_bandwidth', action='store', type=int,
                        default=26,
                        dest='nb_of_bandwidth',
                        help='number of bandwidth extract')

    parser.add_argument('--log-level', default=logging.INFO,
                        type=lambda x: getattr(logging, x),
                        help = "Configure the logging level: DEBUG|INFO|WARNING|ERROR|FATAL")

    parser.add_argument ('--output_traces', action='store', type=str,
                         default=None,
                         dest='path_output_traces',
                         help='Absolute path to the directory where the traces will be saved')

    parser.add_argument ('--output_lists', action='store', type=str,
                         default=None,
                         dest='path_output_lists',
                         help='Absolute path to the files where the new lists will be saved')
    
    parser.add_argument('--freq', type=float, default= 2e6, dest='freq',
                        help='Frequency of the acquisition in Hz')

    parser.add_argument('--window', type=float, default= 10000,
                        dest='window', help='Window size for STFT')

    parser.add_argument('--overlap', type=float, default= 10000/8,
                        dest='overlap', help='Overlap size for STFT')

    parser.add_argument('--dev', action='store', type=str, default='pico',
                        dest='device', help='Used device under test')
    
    args, unknown = parser.parse_known_args ()
    logging.basicConfig(level=args.log_level)
        
    t, f, poi = compute_poi (args.path_lists, args.path_acc, args.nb_of_bandwidth)
    
    generate_dataset (t, f, poi, args.path_lists, args.path_output_lists,
                      args.path_output_traces, args.freq, args.window, args.overlap)
    
