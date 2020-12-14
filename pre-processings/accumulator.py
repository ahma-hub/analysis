################################################################################
from  list_manipulation   import get_tag

from   tqdm               import tqdm
from   signal_processing  import stft, unpackData


import multiprocessing    as mltp
import concurrent.futures
import numpy              as np
import functools
import operator
import time
import argparse
import logging
import glob, os

################################################################################
def acc_stft_batch (name,
                    filenames,
                    freq,
                    window,
                    overlap):
################################################################################
# acc_stft_batch
# compute the sum and the sum of the square by set. The accumulators are save in
# acc_x and acc_xx (manager.list for the multithreading)
# inputs:
#  - filenames: files to read
#  - {freq, window, overlap}: parameters of the stft
#  - {acc_x, acc_xx}: accumulators (manager.list ())
#  - indexes: indexes to fill inside the accumulators
#  - (diasble) device: device under test (example: 'pico')
#  - pbar: update the ongoing process (progress value, manager.value)
#  - pbar_up: frequency of the update of the pbar (int value)
#
# outputs:
#  - {acc_x, acc_xx}: no returned value, but the results are stored in the
#  accumulators
################################################################################
    
    
    t, f, tmp_stft = stft (unpackData (filenames [0], device = 'pico'),\
                     freq, window, overlap, False) 

    acc_x  = tmp_stft
    acc_xx = tmp_stft**2

    
    for i in range (1, len (filenames)):
        
        tmp_stft =  stft (unpackData (filenames [i], device = 'pico'),\
                             freq, window, overlap, False) [-1]

        acc_x  += tmp_stft
        acc_xx += tmp_stft**2

    np.save (name + 'acc_x.npy',  [t, f, acc_x],  allow_pickle= True)
    np.save (name + 'acc_xx.npy', [t, f, acc_xx], allow_pickle= True)
        
## disable here 
# ################################################################################
# def get_size_batch (filenames,
#                     shift_idx,
#                     device,
#                     send_end,
#                     pbar,
#                     pbar_up):
# ################################################################################
# # get_size_batch
# # read the traces in the batch in order to get the smallest/biggest trace
# # inputs:
# #  - filenames: files to read
# #  - shift_idx: shift of the current batch
# #  - device: device under test (example: 'pico')
# #  - send_end: to store the results
# #  - pbar: update the ongoing process (progress value, manager.value)
# #  - pbar_up: frequency of the update of the pbar (int value)
# #
# # outputs:
# #  - size, idx
# ################################################################################
    
#     for i in range (len (filenames)):
#         current_trace = unpackData (filenames [i], device)
        
#         if (i == 0):
#             bigger  = len (current_trace)
#             smaller = len (current_trace)

#             idx_bigger  = i + shift_idx
#             idx_smaller = i + shift_idx

#         else:
#             if (bigger < len (current_trace)):
#                 bigger     = len (current_trace)
#                 idx_bigger = i + shift_idx

#             elif (smaller > len (current_trace)):
#                 smaller     = len (current_trace)
#                 idx_smaller = i + shift_idx

#         # check to update the progress bar
#         if (pbar is not None and (i + 1)%pbar_up == 0):
#             pbar.value += pbar_up

#     send_end.send ([smaller, idx_smaller, bigger, idx_bigger])

## disable here 
# ################################################################################
# def get_size (batch_filenames,
#               shift_idx,
#               device):
# ################################################################################
# # get_size
# # compute the sum and the sum of the square by set. The accumulators are save in
# # acc_x and acc_xx (manager.list for the multithreading)
# # inputs:
# #  - filenames: files to read
# #  - {freq, window, overlap}: parameters of the stft
# #  - indexes: indexes to fill inside the accumulators
# #  - device: device under test (example: 'pico')
# #  - pbar: update the ongoing process (progress value, manager.value)
# #  - pbar_up: frequency of the update of the pbar (int value)
# #
# # outputs:
# #  - {acc_x, acc_xx}: no returned value, but the results are stored in the
# #  accumulators
# ################################################################################
#     nb_of_threads = len (batch_filenames)

#     # compute the nb_of_traces
#     nb_of_traces = 0
#     for i in range (nb_of_threads):
#         nb_of_traces += len (batch_filenames [i])
    
#     # pbar for all the thread // do not update after the same number of iteration to limit the
#     # concurent access to the pbar_value
#     # each thread will update the progress bar with a frequency
#     # \in [nb_of_threads : 3*nb_of_threads]
#     pbar_ups = np.array (range (nb_of_threads, 3*nb_of_threads))
#     np.random.shuffle (pbar_ups)
#     pbar_ups = pbar_ups [:nb_of_threads]
#     print ('updates per batch: ', pbar_ups)

#     manager =  mltp.Manager ()
#     # # creat a shared list to get the results
#     # sizes = manager.list ()
#     # for i in range (nb_of_threads):
#     #     sizes.append (np.zeros (4))

#     pbar_value = manager.Value ('i', 0)
#     pbar = tqdm (total = nb_of_traces, desc='reading and get size', unit='feat.', leave = True)

#     jobs    = []
#     pip_res = []
#     for i in range (nb_of_threads):
#         recv_end, send_end = mltp.Pipe(False)
#         process = mltp.Process (target = get_size_batch,
#                                 args = (batch_filenames [i], shift_idx [i], device, send_end,  
#                                         pbar_value, pbar_ups [i]))

        
#         jobs.append (process)
#         pip_res.append (recv_end)
#         process.start ()
        
#     # all threads are running, check every second to update the progress bar
#     old_value = pbar_value.value
#     while (pbar_value.value < nb_of_traces):
#         time.sleep (1)
#         # check if the has been updated
#         if (old_value != pbar_value.value):
#             pbar.update (pbar_value.value - old_value)
#             old_value = pbar_value.value

#         # check if all job are finished
#         tmp = 0
#         for p in jobs:
#             tmp += int (p.is_alive ())
#         if (tmp == 0):
#             break

#     pbar.close ()

#     # get the global {min, max, idxs}
#     for i in range (nb_of_threads):
#         sizes = pip_res [i].recv ()
       
#         if i == 0:
#             smaller = sizes [0]
#             idx_smaller = sizes [1]

#             bigger = sizes [2]
#             idx_bigger = sizes [3]

#         else :
#             if (bigger < sizes [2]):
#                 bigger = sizes [2]
#                 idx_bigger = sizes [3]
#             elif (smaller > sizes [0]):
#                 smaller = sizes [0]
#                 idx_smaller = sizes [1]

#     return smaller, idx_smaller, bigger, idx_bigger

################################################################################
def acc_stft_by_sets (path_lists,
                      freq,
                      window,
                      overlap,
                      output,
                      # same_size,
                      nb_of_threads):
                      # device
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
#  - (disable) same_size: if none we assume that all traces have same size,
#  if 0 we padd,  if 1 we cut.
#  - nb_max_of_thread: set the number max of thread that will be
#  used. the number of thread is egual to the minimum between the number of core
#  less one and the given parameter
#  - (disable) device: device under test (example: 'pico')
################################################################################
    ## load the main list
    [x_train_filelist, x_val_filelist, _, y_train, y_val, _] \
         = np.load (path_lists, allow_pickle = True)

    ## merge training and validating 
    x_file_list = np.array (x_train_filelist + x_val_filelist)
    y = np.array ([get_tag (f) for f in x_file_list])
    
    ## nb os sets
    unique_y = np.unique (y)
    nb_of_sets = len (unique_y)

    # arguments for the multithreading
    args = []

    for i in range (nb_of_sets):
        idx = np.where (y == unique_y [i])[0]

        current_file_list = x_file_list [idx]

        current_name = output + '/' + unique_y [i] + '_%s_'%len (idx)
        args.append ([current_name, current_file_list, freq, window, overlap])


    nb_of_threads = min (mltp.cpu_count () - 2, nb_of_threads)
    with concurrent.futures.ProcessPoolExecutor (max_workers =  nb_of_threads) as executor:
        ## the * is needed otherwise it does not work
        futures = {executor.submit (acc_stft_batch, *arg): arg for arg in args}

        kwargs = {
            'total': len (futures),
            'unit': 'sample',
            'unit_scale': True,
            'leave': True,
            'desc': 'acc (%s)'%nb_of_threads
        }

        for f in tqdm (concurrent.futures.as_completed (futures), **kwargs):
            pass
 
################################################################################
if __name__ == '__main__':
################################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument ('--lists', action='store', type=str,
                         dest='path_lists',
                         help='Absolute path of the lists (cf. list_manipulation.py -- using a main list will help) trace directory')

    parser.add_argument('--output', action='store', type=str,
                        dest='output_path',
                        help='Absolute path of the output directory')

    parser.add_argument('--freq', type=float, default= 2e6, dest='freq',
                        help='Frequency of the acquisition in Hz')

    parser.add_argument('--window', type=float, default= 10000,
                        dest='window', help='Window size for STFT')

    parser.add_argument('--overlap', type=float, default= 10000/8,
                        dest='overlap', help='Overlap size for STFT')

    parser.add_argument('--core', action='store', type=int, default=4,
                         dest='core',
                         help='Number of core to use for multithreading accumulation')

    ## disable, not needed here -- we assume that all traces have the same size
    # parser.add_argument('--same_size', action='store', type=int, default=None,
    #                      dest='same_size',
    #                      help='Use when the traces do not have the same size, 0 padd, 1 cut')

    ## disable, not needed here -- we assume that all traces have the same format (from picoscope)
    # parser.add_argument('--dev', action='store', type=str, default='pico',
    #                     dest='device', help='Used device under test')

    
    # parser.add_argument ('--log-level', default=logging.INFO,
    #                      type=lambda x: getattr (logging, x),
    #                      help = "Configure the logging level: DEBUG|INFO|WARNING|ERROR|FATAL")
    
    args, unknown = parser.parse_known_args ()

    # compute the accumulation
    acc_stft_by_sets  (path_lists          = args.path_lists,
                       freq                = args.freq,
                       window              = args.window,
                       overlap             = args.overlap,
                       output              = args.output_path,
                       # same_size         = args.same_size,
                       nb_of_threads       = args.core,
                       # device            = args.device
                       )
