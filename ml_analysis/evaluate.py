"""
 File: evaluate.py 
 Project: analysis 
 Last Modified: 2021-8-2
 Created Date: 2021-8-2
 Copyright (c) 2021
 Author: AHMA project (Univ Rennes, CNRS, Inria, IRISA)
"""

################################################################################
import sys, os, glob
import logging
import numpy                       as np
import joblib
import argparse
import time

from tqdm                          import tqdm
from datetime                      import datetime
from sklearn.svm                   import SVC
from sklearn.naive_bayes           import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing         import LabelEncoder
from sklearn.metrics               import classification_report


sys.path.append(os.path.join (os.path.dirname (__file__), "../pre-processings/"))
from nicv import compute_nicv
from corr import compute_corr
from list_manipulation import get_tag


################################################################################
import numpy            as np
import matplotlib, sys
import logging

## to avoid bug when it is run without graphic interfaces
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except ImportError:
    # print ('Warning importing GTK3Agg: ', sys.exc_info()[0])
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt



################################################################################
def load_traces (files_list, bandwidth, time_limit):
################################################################################
# load_traces
# load all the traces listed in 'files_lists', the 2D-traces are flattened
#
# input:
#  + files_list: list of the filenames
#  + bandwidth: selected bandwidth
#  + time_limit: percentage of the trace to concerve
#
# output:
#  + traces: array containing the traces (dimension: DxQ, D number of features,
#  Q number of samples)
################################################################################
    ## get dimension
    tmp_trace = np.load (files_list [0], allow_pickle = True)[-1][bandwidth, :]

    ## takes only half of the features
    D = int (tmp_trace.shape [1]*time_limit)
    tmp_trace = tmp_trace [:, :D].flatten ()
    
    traces = np.zeros ((len (tmp_trace), len (files_list)))
    traces [:, 0] = tmp_trace 

    for i in tqdm (range (1, traces.shape [1])):
        traces [:, i] = np.load (files_list [i], allow_pickle = True)[-1][bandwidth, :D].flatten ()

    return traces

################################################################################
def mean_by_label (traces, labels, mean_size):
################################################################################
# mean_by_label
# mean traces per label, it means the input traces are mean by batch of 'mean_size' 
# of the same label
# 
# input:
#  + traces: array of traces (DxQ)
#
# output:
#  + traces: mean traces (dimension: DxQ, D number of features,
#  Q number of samples)
################################################################################

    unique = np.unique (labels)

    tmp_res = []
    tmp_labels = []
    count = 0

    for i in tqdm (range (len (unique)), desc = 'meaning (%s)'%mean_size):
        idx = np.where (labels == unique [i])[0]
        
        for j in range (0, len (idx) - mean_size, mean_size):
            tmp_labels.append (unique [i])
            current_res = 0.
            for k in range (mean_size):
                current_res += traces [:, idx [j + k]]
            tmp_res.append (current_res/mean_size)

    return np.array (tmp_res).T, tmp_labels

################################################################################
def evaluate (path_lists, log_file, mean_sizes, nb_of_bd, path_acc,
              time_limit, metric):
################################################################################
# mean_by_label
# compute the LDA + BN and LDA + SVM machine learning algorithm
# 
# input:
#  + path_lists: path of the lists 
#  + log_file: where the results are saved
#  + mean_sizes: numbers of mean sizes to try
#  + nb_of_bd: nb of frequency band to conserve
#  + path_acc: directory where acculmulators are
#  + time_limit: percentage of the trace (from the begining)
#  - metric: metric to use for the bandwidth selection
################################################################################
    ## logging exp in file
   
    today = datetime.now ()
    d1 = today.strftime ("%d.%m.%Y - %H:%M:%S")

    file_log = open (log_file, 'a')
    file_log.write ('-'*80 + '\n')
    file_log.write (d1 + '\n')
    file_log.write ('path_lists: %s\n'%str (path_lists)\
                    + 'log_file: %s\n'%str (log_file)\
                    + 'model_lda: None\n'\
                    + 'model_svm: None\n'\
                    + 'model_nb: None\n'\
                    + 'means: %s\n'%str (mean_sizes)\
                    + 'nb_of_bd: %s\n'%str (nb_of_bd)\
                    + 'path_acc: %s\n'%str (path_acc)\
                    + 'time_limit: %s\n'%str (time_limit)\
                    + 'metric: %s\n'%str (metric))
    
    file_log.write ('-'*80 + '\n')
    file_log.close ()

    ## get indexes
    if ('nicv' in metric):
        t, f, nicv, bandwidth = compute_nicv (path_lists, path_acc, None,\
                                              bandwidth_nb = nb_of_bd,
                                              time_limit = time_limit,
                                              metric = metric)
    else:
        t, f, nicv, bandwidth = compute_corr (path_lists, path_acc, None,\
                                              bandwidth_nb = nb_of_bd,
                                              time_limit = time_limit,
                                              metric = metric)

    ## load lists
    [x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test] \
         = np.load (path_lists, allow_pickle = True)

    learning_traces = load_traces (x_train_filelist, bandwidth, time_limit)
    validating_traces = load_traces (x_val_filelist, bandwidth, time_limit)

    ## learning
    traces = np.zeros ((learning_traces.shape [0], learning_traces.shape [1] + validating_traces.shape [1]))
    traces [:, :learning_traces.shape [1]] = learning_traces 
    traces [:, learning_traces.shape [1]:] = validating_traces 
    
        
    learning_labels = y_train
    validating_labels = y_val

    labels = np.concatenate ((learning_labels, validating_labels))
       
    ## projection
    t0 = time.time ()
    clf = LinearDiscriminantAnalysis ()
  
    transformed_traces = clf.fit_transform (traces.T, list (labels))
    ## save LDA
    tmp = path_lists.split ('=')[-1].split ('.')[0]
    joblib.dump (clf, '/'.join (path_lists.split ('/')[:-1]) + f'/LDA_tagmpas={tmp}_{nb_of_bd}bd.jl')    
    
    file_log = open (log_file, 'a')
    file_log.write ('LDA (compuation): %s seconds\n'%(time.time () - t0))
    file_log.close ()   

    ## learning on projection
    t0 = time.time ()

    gnb  = GaussianNB ()
    gnb.fit (transformed_traces, labels)
    ## save GN
    tmp = path_lists.split ('=')[-1].split ('.')[0]
    joblib.dump (gnb, '/'.join (path_lists.split ('/')[:-1]) + f'/NB_tagmpas={tmp}_{nb_of_bd}bd.jl')    
    
    file_log = open (log_file, 'a')   
    file_log.write ('NB (computation): %s seconds\n'%(time.time () - t0))
    file_log.close ()
    
    t0 = time.time ()
    svc = SVC ()
    svc.fit (transformed_traces, labels)

    ## save SVM
    tmp = path_lists.split ('=')[-1].split ('.')[0]
    joblib.dump (svc, '/'.join (path_lists.split ('/')[:-1]) + f'/SVM_tagmpas={tmp}_{nb_of_bd}bd.jl')
    
    file_log = open (log_file, 'a')   
    file_log.write ('SVM (computation): %s seconds\n'%(time.time () - t0))
    file_log.close ()    
    
    ## now and testing
    testing_traces = load_traces (x_test_filelist,  bandwidth, time_limit)
    testing_labels = y_test
    
    ## no means
    ## projection LDA
    t0 = time.time ()
    X = clf.transform (testing_traces.T)

    # save transformed traces
    tmp = path_lists.split ('=')[-1].split ('.')[0]
    np.save ('/'.join (path_lists.split ('/')[:-1]) + f'/transformed_traces_tagmaps={tmp}_{nb_of_bd}bd.npy',
             X, allow_pickle = True)

    
    file_log = open (log_file, 'a')
    file_log.write ('transform (size: %s): %s seconds\n'%(str(testing_traces.shape), str (time.time () - t0)))
    file_log.close ()

    ## NB 
    t0 = time.time ()

    predicted_labels = gnb.predict (X)
    file_log = open (log_file, 'a')
    file_log.write ('Test NB  (size: %s) [%s seconds]:\n'%(str (X.shape), str (time.time () - t0)))
    file_log.write (f'{classification_report (list (testing_labels), predicted_labels, digits = 4, zero_division = 0)}')
    file_log.close ()

    ## SVM
    t0 = time.time ()

    predicted_labels = svc.predict (X)
    file_log = open (log_file, 'a')
    file_log.write ('Test SVM  (size: %s) [%s seconds]:\n'%(str (X.shape), str (time.time () - t0)))
    file_log.write (f'{classification_report (list (testing_labels), predicted_labels, digits = 4, zero_division = 0)}')
    file_log.close ()
    
    for mean_size in mean_sizes:
        file_log = open (log_file, 'a')
        file_log.write ('compute with %s per mean\n'%mean_size)
        file_log.close ()

        X, y = mean_by_label (testing_traces, np.array (testing_labels), mean_size)
        
        ## LDA on means
        t0 = time.time ()
        X = clf.transform (X.T)
        file_log = open (log_file, 'a')
        file_log.write ('transform (size: %s): %s seconds\n'%(str(testing_traces.shape), str (time.time () - t0)))
        file_log.close ()

        ## NB
        t0 = time.time ()
        predicted_labels = gnb.predict (X)
        file_log = open (log_file, 'a')
        file_log.write (f'NB - mean {mean_size}:\n {classification_report (list (y), predicted_labels, digits = 4, zero_division = 0)}')
        file_log.close ()

        # SVM
        t0 = time.time ()
        predicted_labels = svc.predict (X)
        file_log = open (log_file, 'a')
        file_log.write (f'SVM - mean {mean_size}:\n {classification_report (list (y), predicted_labels, digits = 4, zero_division = 0)}')
        file_log.close ()

        
################################################################################
if __name__ == '__main__':
################################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument ('--lists', action = 'store',
                         type = str, dest = 'path_lists',
                         help = 'Absolute path to a file containing the lists')

    parser.add_argument("--mean_size", default = [2,3,4,5,6,7,9,10],
                        action = 'append', dest = 'mean_sizes',
                        help = 'Size of each means')
    
    parser.add_argument('--log-file', default = 'log-evaluation.txt',
                         dest = 'log_file',
                         help = 'Absolute path to the file to save results')

    
    parser.add_argument ('--acc', action='store', type=str,
                         dest='path_acc',
                         help='Absolute path of the accumulators directory')

    parser.add_argument('--nb_of_bandwidth', action='store', type=int,
                        default=20,
                        dest='nb_of_bandwidth',
                        help='number of bandwidth to extract')

    parser.add_argument ('--time_limit', action ='store', type = float, default = 1,
                         dest = 'time_limit',
                         help = 'percentage of time to concerve (from the begining)')

    parser.add_argument('--metric', action='store', type=str, default='nicv_max',
                        dest='metric', help='Metric to use for select bandwidth: {nicv, corr}_{mean, max} ')
        
        
    args, unknown = parser.parse_known_args ()
    assert len (unknown) == 0, f"[WARNING] Unknown arguments:\n{unknown}\n"
    
    args, unknown = parser.parse_known_args ()

    # test (args, GaussianNB (), False)
    evaluate (args.path_lists,
              args.log_file,
              args.mean_sizes,
              args.nb_of_bandwidth,
              args.path_acc,
              args.time_limit,
              args.metric)
