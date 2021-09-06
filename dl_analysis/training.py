from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Flatten, AveragePooling2D
from tqdm import trange
import numpy as np
import os, sys, argparse

sys.path.append(os.path.join (os.path.dirname (__file__), "../pre-processings/"))

from nicv import compute_nicv


def load_datafile(datafile, bandwidth):
    Sxx = np.load (datafile, allow_pickle = True)[-1][bandwidth, :]
    return Sxx

def get_data_dimension(datafile, bandwidth):
    return load_datafile(datafile, bandwidth).shape

def create_dataset(filelist, labels, bandwidth):
    batch_x = np.asarray([load_datafile(f, bandwidth=bandwidth) for f in filelist])
        
    batch_x = batch_x/np.max(batch_x) #normalization
    batch_x = batch_x[:,:,:,np.newaxis]

    dataset = tf.data.Dataset.from_tensor_slices((batch_x, labels))
    return dataset


class NN(object):
    def __init__(self, nb_rows, nb_columns, nb_labels, arch, batch_size):
        self.data_nb_rows = nb_rows
        self.data_nb_columns = nb_columns
        self.data_nb_channels = 1
        self.batch_size = batch_size
        input_shape = (self.data_nb_rows, self.data_nb_columns, self.data_nb_channels )

        self.model = keras.models.Sequential()
        
        if arch == "cnn":
            # Block 1
            self.model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding="same", input_shape=input_shape))
            self.model.add(MaxPooling2D(pool_size=2, padding='same'))
            # Block 2
            self.model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding="same"))
            self.model.add(MaxPooling2D(pool_size=2, padding='same'))
            # Block 3
            self.model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding="same"))
            self.model.add(MaxPooling2D(pool_size=2, padding='same'))
            # Block 4
            self.model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding="same"))
            self.model.add(MaxPooling2D(pool_size=2, padding='same'))
            # Block 5
            self.model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding="same"))
            self.model.add(MaxPooling2D(pool_size=2, padding='same'))
            #
            self.model.add(Flatten())
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(nb_labels, activation='softmax'))

        elif arch=="mlp":
            self.model.add(Flatten(input_shape=input_shape))
            #
            self.model.add(Dense(500))
            self.model.add(tf.keras.layers.LeakyReLU())
            #
            self.model.add(Dense(200))
            self.model.add(tf.keras.layers.LeakyReLU())
            #
            self.model.add(Dense(100))
            self.model.add(tf.keras.layers.LeakyReLU())
            self.model.add(Dense(nb_labels, activation='softmax'))
        keras.backend.clear_session()
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


    def train(self, training_data, validation_data, class_weights, nb_epochs=10, save_kernel='', verbose = True,):
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=save_kernel, monitor='val_accuracy', mode='max',verbose=verbose, save_best_only=True)
        history = self.model.fit(training_data, epochs=nb_epochs, class_weight=class_weights, verbose=verbose, validation_data=validation_data, callbacks=[checkpointer])
        return history

def main(input_file_list, path_acc, nb_epochs, batch_size, nb_of_bandwidth, arch, save, verbose):
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:
            print(e)

    print("Using {} number(s) of bandwidth(s)".format(nb_of_bandwidth))
    print("Loading data...")

    [ x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test ] = np.load (input_file_list, allow_pickle = True)
        
    le = LabelEncoder()
    y_train = keras.utils.to_categorical(le.fit_transform(y_train))
    y_test = keras.utils.to_categorical(le.transform(y_test))
    y_val = keras.utils.to_categorical(le.transform(y_val))
    nb_labels = len(le.classes_)
    
    ## get indexes
    _, _, nicv, bandwidth = compute_nicv (input_file_list, path_acc, path_save = None,\
                                              bandwidth_nb = nb_of_bandwidth )

    train_dataset = create_dataset(x_train_filelist, y_train, bandwidth=bandwidth)
    validation_dataset = create_dataset(x_val_filelist, y_val, bandwidth=bandwidth)
    train_dataset = train_dataset.shuffle(len(x_train_filelist)).batch(batch_size)
    validation_dataset = validation_dataset.shuffle(len(x_val_filelist)).batch(batch_size)
    
    nb_rows, nb_columns = get_data_dimension(x_train_filelist[0], bandwidth=bandwidth)
    classes = dict(zip(le.classes_, le.transform(le.classes_)))
    d_class_weights = {}

    print("Batch size: {}".format(batch_size))
    best_training_accuracy  = None
    history = None
    nn = NN(nb_rows, nb_columns, nb_labels, arch, batch_size)

    if verbose:
        print(nn.model.summary())
        
    print("Training the model...")
    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))
    history = nn.train(train_dataset, validation_dataset, d_class_weights, nb_epochs, save_kernel=save, verbose=verbose)
    #best_training_accuracy = max(history.history["accuracy"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store', type=str, default='lists_selected_bandwidth/files_lists_tagmap=executable_classification.npy', dest='path_lists', help='load the file list corresponding to the experiments, default=lists_selected_bandwidth/files_lists_tagmap=binary_classification.npy')
    parser.add_argument('--band', action='store', type=int, dest='nb_of_bd', help='number of selected bandwidth')
    parser.add_argument ('--acc', action='store', type=str, dest='path_acc', help='Absolute path of the accumulators directory')
    parser.add_argument('--epochs', action='store', type=int, dest='nb_of_epochs', help='number of epochs')
    parser.add_argument('--batch', action='store', type=int, dest='batch_size', help='batch size')
    parser.add_argument ('--arch', action='store', type=str, dest='arch', help='neural network architecture (cnn / mlp)')
    parser.add_argument ('--save', action='store', type=str, dest='save', help='filename to save model')
    args = parser.parse_args()
    
    main(args.path_lists, args.path_acc, args.nb_of_epochs, args.batch_size, args.nb_of_bd, args.arch,  args.save, verbose = True)
