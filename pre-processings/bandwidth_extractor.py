"""
 File: bandwidth_extractor.py 
 Project: analysis 
 Last Modified: 2021-8-2
 Created Date: 2021-8-2
 Copyright (c) 2021
 Author: AHMA project (Univ Rennes, CNRS, Inria, IRISA)
"""

################################################################################
import sys, os, glob
sys.path.append (os.path.join (os.path.dirname (__file__), '..', 'oscilloscopes', ))

import argparse

from   nicv import compute_nicv
from   corr import compute_corr

from   signal_processing import stft, unpackData
from   displayer         import display_matrix

import numpy as np

import matplotlib
## to avoid bug when it is run without graphic interfaces
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print ('Warning importing GTK3Agg: ', sys.exc_info()[0])
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


from tqdm                  import tqdm
import logging
import concurrent.futures
    

################################################################################
def extract_bandwidth (lists, path_acc, metric, nb_of_bandwidth, plot_save) :
################################################################################
# compute_poi
# compute the index of the sorted Points-Of-Interests
# inputs:
#  - lists: list of files (cf list_manipulation)
#  - path_acc: path to the file where the accumulators are located
#  - metric: nicv_max, ncv_mean, corr_max, corr_mean 
#  - nb_of_bandwidth: nb of bandwidth to conserve
#  - plot_save: location to save the plot
# outputs:
#  - poi: 2D array of the concerved indexes
#  - t, f: t-axis and f-axis of the nicv/corr (= spectrograms)
################################################################################
    ## deal with special case nb_of_bandwidth = -1
    ## it means taht all bandwidth are concerved
    if (nb_of_bandwidth == -1):
        list_acc = glob.glob (path_acc + '/*.npy') ## load one accumulator
        
        t, f, acc = np.load (list_acc [0], allow_pickle = True)
        bandwidth = np.array (range (len (f)))

        return t, f [bandwidth], bandwidth
    
    ## to avoid termediate display (corr or nicv) save and modify
    ## logging level, plotting happens only for 'DEBUG' level
    logging_save = logging.root.level
    logging.basicConfig (level = 'INFO')
    
    for i in range (len (lists)):
        if ('nicv' in metric):
            t, f, tmp_res, _ = compute_nicv (lists [i],
                                          path_acc,
                                          None) # does not save the display
        else:
            t, f, tmp_res, _ = compute_corr (lists [i],
                                      path_acc,
                                      None) # does not save the display

        ## recenter
        tmp_res = np.abs (tmp_res - tmp_res.sum (0).sum ()\
                          /(tmp_res.shape [0]*tmp_res.shape [1]))

        ## ranks: sort values and sum ranks
        if (i == 0):
            # ranks = np.zeros (tmp_res.shape, dtype = np.float64)
            maximums = tmp_res
        else:
            ## concerve maximums
            maximums = np.maximum (tmp_res, maximums)

        # ## sort and sum
        # idx = np.dstack (np.unravel_index (np.argsort (tmp_res.ravel()), tmp_res.shape)) [0]
        # ranks [idx [:, 0], idx [:, 1]] = range (tmp_res.shape [0]*tmp_res.shape [1])        

    ## get the bandwidth with the highest mean
    if ('max' in metric):
        bandwidth = np.argsort (maximums.max (1)) [-nb_of_bandwidth:]
    else:
        bandwidth = np.argsort (maximums.mean (1)) [-nb_of_bandwidth:]

    ## then sort from the smallest to the biggest frequency values
    bandwidth = np.sort (bandwidth) 
    
    ## restore the looging value*
    logging.basicConfig (level = logging_save)

    if ((logging.root.level < logging.INFO) or (plot_save is not None)):
        ###################################################
        # the plot using the stft_investigation 
        ##################################################
        display_matrix (t, f, maximums, plot_save, bandwidth)

        
    return t, f [bandwidth], bandwidth


################################################################################
def generate_dataset_thread (path_in, path_out, bandwidth, freq, window,
                             overlap, device, duration, pos):
################################################################################
# generate_dataset_thread
# generate a traces from a given batch, extract and save bandwidth from each files
#
# inputs:
#  - path_in: list of the input files
#  - path_out: list of the output files
#  - bandwidth: bandwidth to extract 
#  - freq, window, overlap: parameters of the stft
#  - device: to know which file type is used
#  - duration: to fixe the duration in second (for none constant size traces)
#  - pos: use by the progress bar
################################################################################

    for i in tqdm (range (len (path_in)), position = pos):
        trace = unpackData (path_in [i], device = device)
        
        if (duration is not None):
            if (len (trace) < int (freq*duration)):
                trace = np.pad (trace, (0, int (duration * freq - len (trace))), mode = 'constant')

            elif (len (trace) > int (freq*duration)):
                trace = trace [:int (duration * freq)]

        print (path_in [i], path_out [i])
        t, f, tmp_stft =  stft (trace,\
                                freq, window, overlap, False) #  [-1][bandwidth, :]

        tmp_stft = tmp_stft [bandwidth, :]

        np.save (path_out [i], [t, f [bandwidth], tmp_stft], allow_pickle = True)

################################################################################
def generate_dataset (t, f, bandwidth, paths_lists, path_output_lists,
                      path_output_traces, freq, window, overlap, nb_of_threads,
                      device, duration):
################################################################################
# generate_dataset
# from raw data to extracted bandwidth
#  - t, f: t-axis and f-axis of the nicv (= spectrograms)
#  - bandwidth: concerved frequencies indexes
#  - path_lists: lists of files and labels
#  - path_output_lists: path where the new list will be saved
#  - path_output_traces: directory where the extracted bandwidth will be saved
#  - freq, window, overlap: parameters of the stft
#  - nb_of_threads: nb of thread to use during the extraction
#  - device: to know which file type is used
#  - duration: to fixe the duration in second (for none constant size traces)
################################################################################

    filelist = set ([])
    outputs = set ([])

    for j in range (len (paths_lists)):
        # load lists
        [x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test]\
            = np.load (paths_lists [j], allow_pickle = True)

        new_train_list = ['']*len (x_train_filelist)
        new_val_list   = ['']*len (x_val_filelist)
        new_test_list  = ['']*len (x_test_filelist)

        ## learning
        for i in tqdm (range (len (x_train_filelist)), desc = 'learning data'):
            tmp_path = path_output_traces + '/' \
                + '.'.join (x_train_filelist [i].split ('/')[-1].split ('.')[:-1])\
                + '.npy'
            new_train_list [i] = tmp_path

        ## validating
        for i in tqdm (range (len (x_val_filelist)), desc = 'validating data'):
            tmp_path = path_output_traces + '/' \
                + '.'.join (x_val_filelist [i].split ('/')[-1].split ('.')[:-1])\
                + '.npy'
            new_val_list [i] = tmp_path

        ## testing
        for i in tqdm (range (len (x_test_filelist)), desc = 'testing data'):
            tmp_path = path_output_traces + '/' \
                + '.'.join (x_test_filelist [i].split ('/')[-1].split ('.')[:-1])\
                + '.npy'
            new_test_list [i] = tmp_path

        ## save new list: save in the given directory adding 'extracted_bd_'
        np.save (path_output_lists + '/' + 'extracted_bd_' + paths_lists [j].split ('/') [-1], 
                 [new_train_list, new_val_list, new_test_list, y_train, y_val, y_test],
                 allow_pickle = True)

        if (j == 0):
            filelist = x_train_filelist + x_val_filelist + x_test_filelist
            outputs  = new_train_list + new_val_list + new_test_list
    
    ## prepare the arguments [path_in, path_out, bandwidth, freq, window, overlap, position]
    args = []
    filelist = np.array (list (filelist))
    outputs = np.array (list (outputs))

    
    set_size = int (len (filelist)/nb_of_threads) + 1
    for i in range (nb_of_threads):

        current_filelist = filelist [i*set_size: min ((i + 1)*set_size, len (filelist))]
        current_output   = outputs  [i*set_size: min ((i + 1)*set_size, len (filelist))]

        args.append ([current_filelist, current_output, bandwidth, freq,
                      window, overlap, device, duration, i])

    with concurrent.futures.ProcessPoolExecutor (max_workers =  nb_of_threads) as executor:
        ## the * is needed otherwise it does not work
        futures = {executor.submit (generate_dataset_thread, *arg): arg for arg in args}

        
    ## save the extracted bandwidth
    # t, f : original dimensions
    # new_t, new_f: conserved indexes (all time samples are conserved)
    # /!\ if mulitples "extracted_bandwidth_axis" are in the same directory,
    # it will overright
    np.save (path_output_lists + '/extracted_bandwidth_axis',
             [t, f, np.array (range (len (t))), bandwidth],\
             allow_pickle = True)


################################################################################
if __name__ == '__main__':
################################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument ('--acc', action='store', type=str,
                         dest='path_acc',
                         help='Absolute path of the accumulators directory')

    parser.add_argument ('--lists', nargs = '+',
                         dest='lists',
                         default = [],
                         help='Absolute path to all the lists (for each scenario)'
                         +'. /!\ The data in the first one must contain all traces.')

    parser.add_argument ('--plot', action = 'store',
                         type = str, dest = 'path_to_plot',
                         help = 'Absolute path to a file to save the plot')
        
    parser.add_argument('--nb_of_bandwidth', action='store', type=int,
                        default=100,
                        dest='nb_of_bandwidth',
                        help='number of bandwidth to extract (-1 means that'
                        + ' all bandwidth will be concerved)')

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

    parser.add_argument('--overlap', type=float, default= 0,
                        dest='overlap', help='Overlap size for STFT')

    parser.add_argument('--device', action='store', type=str, default='pico',
                        dest='device', help='Used device under test')

    parser.add_argument('--metric', action='store', type=str, default='nicv_max',
                        dest='metric', help='Metric to use for the PoI selection: {nicv, corr}_{mean, max} ')

    parser.add_argument('--core', action='store', type=int, default=6,
                        dest='core',
                        help='Number of core to use for multithreading')

    parser.add_argument('--duration', action='store', type=float, default=None,
                        dest='duration',
                        help='to fixe the duration of the input traces '\
                        + '(padded if input is short and cut otherwise)')

    
    args, unknown = parser.parse_known_args ()
    assert len (unknown) == 0, f"[WARNING] Unknown arguments:\n{unknown}\n"
    
    logging.basicConfig(level=args.log_level)

    ## extract the bandwidth
    t, f, bandwidth = extract_bandwidth (args.lists, args.path_acc, args.metric,
                                         args.nb_of_bandwidth, args.path_to_plot)

    ## generate the dataset conserving only the bandwidth
    generate_dataset (t, f, bandwidth,
                      args.lists,
                      args.path_output_lists,
                      args.path_output_traces,
                      args.freq,
                      args.window,
                      args.overlap,
                      args.core,
                      args.device,
                      args.duration)
    
