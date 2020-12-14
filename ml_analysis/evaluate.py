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



################################################################################
def load_traces (files_list):
################################################################################
# load_traces
# load all the traces listed in 'files_lists', the 2D-traces are flattened
#
# input:
#  + files_list: list of the filenames
#
# output:
#  + traces: array containing the traces (dimension: DxQ, D number of features,
#  Q number of samples)
################################################################################
    ## get dimension
    tmp_trace = np.load (files_list [0]).flatten ()
    
    traces = np.zeros ((len (tmp_trace), len (files_list)))
    traces [:, 0] = tmp_trace

    for i in range (1, traces.shape [1]):
        traces [:, i] = np.load (files_list [i]).flatten ()

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
def evaluate (path_lists, log_file, model_lda, model_svm, model_nb,
              path_transformed_traces, mean_sizes):
################################################################################
# mean_by_label
# compute the LDA + BN and LDA + SVM machine learning algorithm
# 
# input:
#  + path_lists: path of the lists 
#  + log_file: where the results are saved
#  + model_{lda, svm, nb}: prevously saved {LDA, SVM, NB}-model
#  + path_transformed_traces: path to the transformed learning traces (validate + train)
#  + mean_sizes: numbers of mean sizes to try.
################################################################################
    ## logging exp in file
   
    today = datetime.now ()
    d1 = today.strftime ("%d.%m.%Y - %H:%M:%S")

    file_log = open (log_file, 'a')
    file_log.write ('-'*80 + '\n')
    file_log.write (d1 + '\n')
    file_log.write ('path_lists: %s\n'%str (path_lists)\
                    + 'log_file: %s\n'%str (log_file)\
                    + 'model_lda: %s\n'%str (model_lda)\
                    + 'model_svm: %s\n'%str (model_svm)\
                    + 'model_nb: %s\n'%str (model_nb)\
                    + 'means; %s\n'%str (mean_sizes))
    file_log.write ('-'*80 + '\n')
    file_log.close ()

    ## load lists
    [x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test] \
         = np.load (path_lists, allow_pickle = True)

    # # raw data (need only if models are not loaded)
    if (path_transformed_traces is None):
        learning_traces = load_traces (x_train_filelist)
        validating_traces = load_traces (x_val_filelist)

        ## learning
        traces = np.zeros ((learning_traces.shape [0], learning_traces.shape [1] + validating_traces.shape [1]))
        traces [:, :learning_traces.shape [1]] = learning_traces 
        traces [:, learning_traces.shape [1]:] = validating_traces 

    else:
        transformed_traces = np.load (path_transformed_traces, allow_pickle = True)
        
        
    learning_labels = y_train
    validating_labels = y_val

    labels = np.concatenate ((learning_labels, validating_labels))
   
    
    ## projection
    t0 = time.time ()
    if (model_lda is None):
        clf = LinearDiscriminantAnalysis()
        transformed_traces = clf.fit_transform (traces.T, list (labels))

        file_log = open (log_file, 'a')
        file_log.write ('LDA (compuation): %s seconds\n'%(time.time () - t0))
        file_log.close ()

        ## save LDA
        joblib.dump (clf, path_lists + '_LDA.jl')    

        # save transformed traces
        np.save (path_lists + '_transformed_traces', transformed_traces, allow_pickle = True)
                    
    else:
        clf = joblib.load (model_lda)

        if (path_transformed_traces is None):
            transformed_traces = clf.transform (traces.T)
            # once the traces has been transformed del the raw one in order to save RAM
            del traces

            # save transformed traces
            np.save (path_lists + '_transformed_traces', transformed_traces, allow_pickle = True)
            
        else:
            transformed_traces = np.load (path_transformed_traces, allow_pickle = True)
            
        file_log = open (log_file, 'a')
        file_log.write ('LDA (loading): %s seconds\n'%(time.time () - t0))
        file_log.close ()
        

    ## learning on projection
    t0 = time.time ()
    if (model_nb is None):
        gnb  = GaussianNB ()
        gnb.fit (transformed_traces, labels)

        file_log = open (log_file, 'a')   
        file_log.write ('NB (computation): %s seconds\n'%(time.time () - t0))
        file_log.close ()

        ## save LDA
        joblib.dump (gnb, path_lists + '_NB.jl')
        
    else :
        gnb = joblib.load (model_nb)
        file_log = open (log_file, 'a')   
        file_log.write ('NB (loading): %s seconds\n'%(time.time () - t0))
        file_log.close ()
        
    t0 = time.time ()
    if (model_svm is None):
        svc = SVC ()


        svc.fit (transformed_traces, labels)

        file_log = open (log_file, 'a')   
        file_log.write ('SVM (computation): %s seconds\n'%(time.time () - t0))
        file_log.close ()

        ## save LDA
        joblib.dump (svc, path_lists + '_SVM.jl')
        
    else :
        svc = joblib.load (model_svm)
        file_log = open (log_file, 'a')   
        file_log.write ('SVM (loading): %s seconds\n'%(time.time () - t0))
        file_log.close ()
        
    
    ## now and testing
    testing_traces = load_traces (x_test_filelist)
    testing_labels = y_test
    
    res = []
    ## no means
    t0 = time.time ()
    X = clf.transform (testing_traces.T)
    
    file_log = open (log_file, 'a')
    file_log.write ('transform (size: %s): %s seconds\n'%(str(testing_traces.shape), str (time.time () - t0)))
    file_log.close ()
    
    t0 = time.time ()
    res.append (gnb.score (transformed_traces, list (labels)))
    res.append (gnb.score (X, list (testing_labels)))
    file_log = open (log_file, 'a')
    file_log.write ('Test NB  (size: %s) [%s seconds]: %s [%s]\n'%(str (X.shape), str (time.time () - t0), str (res  [-1]), str (res  [-2])))
    file_log.close ()

    t0 = time.time ()
    res.append (svc.score (transformed_traces, list (labels)))
    res.append (svc.score (X, list (testing_labels)))
    file_log = open (log_file, 'a')
    file_log.write ('Test SVM (size: %s) [%s seconds]: %s [%s]\n'%(str (X.shape), str (time.time () - t0), str (res  [-1]), str (res  [-2])))
    file_log.close ()
    
    for mean_size in mean_sizes:
        file_log = open (log_file, 'a')
        file_log.write ('compute with %s per mean\n'%mean_size)
        file_log.close ()

        X, y = mean_by_label (testing_traces, np.array (testing_labels), mean_size)
        
        ## no means
        t0 = time.time ()
        X = clf.transform (X.T)
        file_log = open (log_file, 'a')
        file_log.write ('transform (size: %s): %s seconds\n'%(str(testing_traces.shape), str (time.time () - t0)))
        file_log.close ()
        
        t0 = time.time ()
        res.append (gnb.score (X, list (y)))
        file_log = open (log_file, 'a')
        file_log.write ('Test NB   (size: %s) [%s seconds]: %s\n'%(str (X.shape), str (time.time () - t0), str (res  [-1])))
        file_log.close ()
        
        t0 = time.time ()
        res.append (svc.score (X, list (y)))
        file_log = open (log_file, 'a')
        file_log.write ('Test SVM   (size: %s) [%s seconds]: %s\n'%(str (X.shape), str (time.time () - t0), str (res  [-1])))
        file_log.close ()

    
################################################################################
if __name__ == '__main__':
################################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument ('--lists', action = 'store',
                         type = str, dest = 'path_lists',
                         help = 'Absolute path to a file containing the lists')

    parser.add_argument ('--model_lda', action = 'store', type=str,
                         dest = 'model_lda',
                         help = 'Absolute path to the file where the LDA model has been previously saved')

    parser.add_argument ('--model_nb', action = 'store', type=str,
                         dest = 'model_nb',
                         help = 'Absolute path to the file where the NB model has been previously saved')

    parser.add_argument ('--model_svm', action = 'store', type=str,
                         dest = 'model_svm',
                         help = 'Absolute path to the file where the SVM model has benn previously saved')

    parser.add_argument ('--transformed_traces', action = 'store', type=str,
                         dest = 'path_transform_traces',
                         help = 'Absolute path to the file where the transformed learning traces has been previously saved')

    parser.add_argument("--mean_size", default = [2,3,4,5,6,7,9,10],
                        action = 'append', dest = 'mean_sizes',
                        help = 'Size of each means')
    
    parser.add_argument('--log-file', default = 'log-evaluation.txt',
                         dest = 'log_file',
                         help = 'Absolute path to the file to save results')

    args, unknown = parser.parse_known_args ()

    # test (args, GaussianNB (), False)
    evaluate (args.path_lists,
              args.log_file,
              args.model_lda,
              args.model_svm,
              args.model_nb,
              args.path_transform_traces,
              args.mean_sizes)
    
    
