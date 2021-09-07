"""
 File: accumulator.py 
 Project: analysis 
 Last Modified: 2021-8-2
 Created Date: 2021-8-2
 Copyright (c) 2021
 Author: AHMA project (Univ Rennes, CNRS, Inria, IRISA)
"""

################################################################################
from  list_manipulation   import get_tag

from   tqdm               import tqdm
from   signal_processing  import stft, unpackData

import multiprocessing    as mltp
import concurrent.futures
import numpy              as np
import argparse


################################################################################
def acc_batch (name,
               filenames,
               pos):
################################################################################
# acc_batch
# compute the sum and the sum of the square by set. The accumulators are save in
# acc_x and acc_xx and then saved in the given file. no stft applied.
# inputs:
#  - name: output to save the files
#  - filenames: files to read
#  - pos: position in the screen for the tqdm progress bar
# 
# outputs:
#  - {acc_x, acc_xx}: no returned value, but the results are stored in the
#  accumulators
################################################################################

    acc_x  = np.float64 (0.)
    acc_xx = np.float64 (0.)

    for i in tqdm (range (len (filenames)), position = pos, desc ='%s'%pos, leave = False):
        t, f, tmp_stft = np.load (filenames [i], allow_pickle = True)

        acc_x  += tmp_stft
        acc_xx += tmp_stft**2
   
    np.save (name + 'acc_x.npy',  [t, f, acc_x],  allow_pickle= True)
    np.save (name + 'acc_xx.npy', [t, f, acc_xx], allow_pickle= True)

    
################################################################################
def acc_stft_batch (name,
                    filenames,
                    freq,
                    window,
                    overlap, pos, duration, device):
################################################################################
# acc_stft_batch
# compute the sum and the sum of the square by set. The accumulators are save in
# acc_x and acc_xx and then save the given file, stft apllied.
# inputs:
#  - name: output to save the files
#  - filenames: files to read
#  - freq: frequency of the sampling rate
#  - windows: size of the windows (for the stft)
#  - overlap: size of the overlap for the stft
#  - pos: position in the screen for the tqdm progress bar
#  - duration: to fixe the duration in second (for none constant size traces)
#  - device: type of files
# 
# outputs:
#  - {acc_x, acc_xx} save in name (no returned arguments)
################################################################################

    acc_x  = np.float64 (0.)
    acc_xx = np.float64 (0.)

    try:
        for i in tqdm (range (len (filenames)), position = pos, desc ='%s'%pos, leave = False):
            trace = unpackData (filenames [i], device = device)
            if (duration is not None):
                if (len (trace) < int (freq*duration)):
                    trace = np.pad (trace, (0, int (duration * freq - len (trace))), mode = 'constant')

                elif (len (trace) > int (freq*duration)):
                    trace = trace [:int (duration * freq)]
        
            t, f, tmp_stft = stft (trace, freq, window, overlap, False)
        
            acc_x  += tmp_stft
            acc_xx += tmp_stft**2
            
        np.save (name + 'acc_x.npy',  [t, f, acc_x],  allow_pickle= True)
        np.save (name + 'acc_xx.npy', [t, f, acc_xx], allow_pickle= True)

    except Exception as e:
        print(e, name)
        
################################################################################
def acc_stft_by_sets (path_lists,
                      freq,
                      window,
                      overlap,
                      output,
                      nb_of_threads,
                      no_stft,
                      duration,
                      device):
################################################################################
# acc_fft_by_sets
# compute the sum and the sum of the square by set of the spectrogram. The
# accumulators are save in output (if the directory does not exist it will be
# created). To creat the accumulators, only the learning and the validating data
# are used
# inputs:
#  - path_main_list: list of the traces
#  - {freq, window, overlap}: parameters of the stft
#  - output: the directory where the accumulators will be
#  saved (npy format);
#  - nb_max_of_thread: set the number max of thread that will be
#  used. the number of thread is egual to the minimum between the number of core
#  less one and the given parameter
#  - no_stft: at true if no need of stft  on the data (it means that data are npy). 
#  otherwise the data are raw data (need to be unpacked)
#  - duration: to fixe the duration in second (for none constant size traces)
#  - device: type of files
################################################################################
    ## load the main list
    [x_train_filelist, x_val_filelist, _, _, _, _] \
         = np.load (path_lists, allow_pickle = True)

    ## merge training and validating 
    x_file_list = np.array (x_train_filelist + x_val_filelist)
    y = np.array ([get_tag (f) for f in x_file_list])
    
    ## nb os sets
    unique_y = np.unique (y)
    nb_of_sets = len (unique_y)

    print (unique_y)
    
    # arguments for the multithreading
    args = []
    nb_of_threads = min (mltp.cpu_count () - 2, nb_of_threads)

    if (not no_stft): ## stft is needed
        for i in range (nb_of_sets):
            idx = np.where (y == unique_y [i])[0]

            current_file_list = x_file_list [idx]

            current_name = output + '/' + unique_y [i] + '_%s_'%len (idx)
        
            args.append ([current_name,
                          current_file_list,
                          freq, window, overlap,
                          i%nb_of_threads, duration, device])
        
        with concurrent.futures.ProcessPoolExecutor (max_workers =  nb_of_threads) as executor:
            ## the * is needed otherwise it does not work
            futures = {executor.submit (acc_stft_batch, *arg): arg for arg in args}



    else: ## no need of stft
        for i in range (nb_of_sets):
            idx = np.where (y == unique_y [i])[0]

            current_file_list = x_file_list [idx]

            current_name = output + '/' + unique_y [i] + '_%s_'%len (idx)
            args.append ([current_name,
                          current_file_list,
                          i%nb_of_threads])
        
        with concurrent.futures.ProcessPoolExecutor (max_workers =  nb_of_threads) as executor:
            ## the * is needed otherwise it does not work
            futures = {executor.submit (acc_batch, *arg): arg for arg in args}

################################################################################
if __name__ == '__main__':
################################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument ('--lists', action='store', type=str,
                         dest='path_lists',
                         help='Absolute path of the lists '\
                         + '(cf. list_manipulation.py -- using '\
                         + ' a main list will help) trace directory')

    parser.add_argument('--output', action='store', type=str,
                        dest='output_path',
                        help='Absolute path of the output directory')

    parser.add_argument('--no_stft', default=False, dest='no_stft', action='store_true',
                        help='If no stft need to be applyed on the listed data')

    parser.add_argument('--freq', type=float, default= 2e6, dest='freq',
                        help='Frequency of the acquisition in Hz')

    parser.add_argument('--window', type=float, default= 10000,
                        dest='window', help='Window size for STFT')

    parser.add_argument('--overlap', type=float, default= 0, # 10000/8,
                        dest='overlap', help='Overlap size for STFT')
    
    parser.add_argument('--core', action='store', type=int, default=6,
                        dest='core',
                        help='Number of core to use for multithreading accumulation')

    parser.add_argument('--duration', action='store', type=float, default=None,
                        dest='duration',
                        help='to fixe the duration of the input traces '\
                        + '(padded if input is short and cut otherwise)')

    parser.add_argument('--device', action='store', type=str, default='pico',
                        dest='device',
                        help='to fixe the duration of the input traces '\
                        + '(padded if input is short and cut otherwise)')
       
    
    args, unknown = parser.parse_known_args ()
    assert len (unknown) == 0, f"[WARNING] Unknown arguments:\n{unknown}\n"
    
    # compute the accumulation
    acc_stft_by_sets  (path_lists          = args.path_lists,
                       freq                = args.freq,
                       window              = args.window,
                       overlap             = args.overlap,
                       output              = args.output_path,
                       nb_of_threads       = args.core,
                       no_stft             = args.no_stft,
                       duration            = args.duration,
                       device              = args.device)
