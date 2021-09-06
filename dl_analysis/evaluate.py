from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse

import sys
import os
sys.path.append(os.path.join (os.path.dirname (__file__), "../pre-processings/"))

from nicv import compute_nicv


def create_dataset(filelist, labels, bandwidth):
    batch_x = np.asarray([np.load(f, allow_pickle=True)[-1][bandwidth, :] for f in filelist])      
    batch_x = batch_x/np.max(batch_x) #normalization
    batch_x = batch_x[:,:,:,np.newaxis]
    dataset = tf.data.Dataset.from_tensor_slices((batch_x, labels))
    return dataset


def evaluate(model, testdataset, p_true):

    p_pred = model.predict(testdataset)
    y_pred = p_pred.argmax(axis=1)
    y_true = p_true.argmax(axis=1)
    accuracy = (y_true == y_pred).mean() 
    confusion = confusion_matrix(y_true, y_pred)

    return accuracy, confusion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', type=str, default='../pretrained_models/CNN/binary_classification.h5', dest='load', help='load a pretrained NN model from a hdf5 file, default=../pretrained_models/CNN/binary_classification.h5')
    parser.add_argument('--list', action='store', type=str, default='lists_selected_bandwidth/files_lists_tagmap=binary_classification.npy', dest='path_lists', help='load the file list corresponding to the experiments, default=lists_selected_bandwidth/files_lists_tagmap=binary_classification.npy')
    parser.add_argument('--band', action='store', type=int, dest='nb_of_bd', help='number of selected bandwidth')
    parser.add_argument ('--acc', action='store', type=str,
                         dest='path_acc',
                         help='Absolute path of the accumulators directory')

    args = parser.parse_args()

    ## get indexes
    _, _, nicv = compute_nicv (args.path_lists, args.path_acc, None)
    bandwidth = np.argsort (nicv.mean (1))[-args.nb_of_bd:]

    ## then sort from the smallest to the biggest
    bandwidth = np.sort (bandwidth)

    ## load lists
    [_, _, x_test_filelist, y_train, _, y_test] = np.load (args.path_lists, allow_pickle = True)
   
    le = LabelEncoder()
    y_train = keras.utils.to_categorical(le.fit_transform(y_train))

    y_test = keras.utils.to_categorical(le.transform(y_test))

    model = keras.models.load_model(args.load)
    print(model.summary())

    test_dataset = create_dataset(x_test_filelist, y_test, bandwidth)
    test_dataset = test_dataset.batch(100)

    accuracy, confusion = evaluate(model, test_dataset, y_test)
    
    print("Testing accuracy: {}%".format(accuracy*100))
    #print("Confusion matrix: {}".format(confusion))

    log_file = "dl_analysis/evaluation_log_DL.txt"
    file_log = open (log_file, 'a')
    file_log.write ("------------------------------\n")
    file_log.write ("Preloaded model from: {}\n".format(args.load))
    file_log.write ("File list: {}\n".format(args.path_lists))
    file_log.write ("Nr of bandwidth: {}\n".format(args.nb_of_bd))
    file_log.write ("Testing accuracy: {}%\n".format(accuracy*100))
    file_log.write ("------------------------------\n")
    file_log.close ()
