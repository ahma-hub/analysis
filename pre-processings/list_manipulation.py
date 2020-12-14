################################################################################
import argparse
import numpy                   as np
import logging
import random
import glob, os

from   tabulate                import tabulate
from   tqdm                    import tqdm
from   sklearn.model_selection import train_test_split
from   tabulate                import tabulate


################################################################################
def change_directory (path_lists, new_dir):
################################################################################
# change_directory
# change the path of the directory where the traces are stored, the given file
# is over-written
#
# input:
#  + new_dir: new directory
################################################################################
    [x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test] \
        = np.load (path_lists, allow_pickle = True)

    for i in range (len (x_train_filelist)):
        x_train_filelist [i] = new_dir + '/' + os.path.basename (x_train_filelist [i])


    for i in range (len (x_val_filelist)):
        x_val_filelist [i] = new_dir + '/' + os.path.basename (x_val_filelist [i])


    for i in range (len (x_test_filelist)):
        x_test_filelist [i] = new_dir + '/' + os.path.basename (x_test_filelist [i])


    np.save (path_lists,
             [x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test],
             allow_pickle = True)


################################################################################
def get_tag (f):
################################################################################
# get_tag
# [from neural-network/utils]
# get list of labels from a list of file
#
# input:
#  + f: list of files
#
# output:
# + list of tags
################################################################################
    return "-".join(os.path.basename(f).split("-")[:-1])

################################################################################
def display_tabular (table, header):
################################################################################
# display_tabular
# display the data as tabular
#
# inputs:
#  + table: tabular to display
#  + header: header of the tabular
################################################################################
    print(tabulate (table, headers= header))

################################################################################
def display_list (y_train, y_val, y_test):
################################################################################
# display_list
# display the the content of a list
#
# inputs:
#  + y_{train, val, test} labels of a list
################################################################################
    y_unique = np.unique (y_train + y_val + y_test)

    lines = []
    for i in range (len (y_unique)):
        line = [y_unique [i]]

        idx = np.where (np.array (y_train)  == y_unique [i])[0]
        line.append (len (idx))

        idx = np.where (np.array (y_val)  == y_unique [i]) [0]
        line.append (len (idx))


        idx = np.where (np.array (y_test)  == y_unique [i]) [0]
        line.append (len (idx))


        lines.append (line)

    print(tabulate (lines, headers= ['label', 'train', 'val', 'test']))


################################################################################
def compute_main_list (data, extension = 'dat'):
################################################################################
# compute_main_list
# [from neural-network/utils]
# Label the data and separate it in train and test dataset.
# inputs:
#  - datadir: path to the directory containing all data
#  - extension: type of file in datadir
#
# outputs:
#    lists: {filelist, labels} x {learning, validating, testing}
################################################################################

    if (not type (data) is list):
        filelist = glob.glob (data + "/**/*.%s"%extension, recursive = True)
    else:
        filelist = data

    random.shuffle (filelist)

    # get labels
    y = [get_tag (f) for f in filelist]

    x_train_filelist, x_test_filelist, y_train, y_test\
        = train_test_split (filelist, y, test_size=0.2)

    x_train_filelist, x_val_filelist, y_train, y_val\
            = train_test_split (x_train_filelist, y_train, test_size=0.2)

    return x_train_filelist, x_val_filelist, x_test_filelist,\
        y_train, y_val, y_test


################################################################################
def parse_data (filelist, tagmap):
################################################################################
# parse_data_dir
# [from neural-network/utils]
# Label the data and separate it in train and test dataset.
# It is possible to provide a path to a file containing a tag map.
# Each line of this file should be formatted like this: tag <space>
# corresponding_label <space> dataset
# Dataset value are 0: not used, 1: train only, 2: test only, 3: train and test
# inputs:
#  + filelist: list of the filename
#  + tagmap: tagmap used for the labeling
#
# outputs:
#  + x_{train, test}_filelist: list files for testing and training (not split yet
#    into training and validating)
#  + x_trainandtest_filelist: list that could be part of the testing and learning
#  + y_{train, test, trainandtest}: labels of the list above
################################################################################

    random.shuffle (filelist)

    tm = open(tagmap)
    label = {}
    group = {}
    count = {}
    for l in tm:
        try:
            tag, l, g = l.strip("\n").split(",")
        except:
            tag, l, g, c = l.strip("\n").split(",")
            count[tag] = int(c)
        g = int(g)
        label[tag] = l
        group[tag] = g
    tm.close()

    #compute for each file its group and label
    x_train_filelist, x_test_filelist, x_trainandtest_filelist = [], [], []
    y_train, y_test, y_trainandtest = [], [], []
    for f in filelist:
        tag = get_tag(f)
        if tag in count:
            if count[tag] == 0:
                continue
        if tag in group.keys():
            g = group[tag]
            if g == 1:
                x_train_filelist.append(f)
                y_train.append(label[tag])
            elif g == 2:
                x_test_filelist.append(f)
                y_test.append(label[tag])
            elif g == 3:
                x_trainandtest_filelist.append(f)
                y_trainandtest.append(label[tag])
            else:
                continue
            if tag in count: count[tag] -= 1


    return x_train_filelist, x_test_filelist, x_trainandtest_filelist,\
        y_train, y_test, y_trainandtest

################################################################################
if __name__ == '__main__':
################################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument ('--raw', action = 'store', type = str,
                         dest = 'path_raw',
                         help = 'Absolute path to the raw data directory')

    parser.add_argument ('--tagmap', action = 'store',
                         type = str, dest = 'path_tagmap',
                         help = 'Absolute path to a file containing the tag map')

    parser.add_argument ('--save', action = 'store',
                         type = str, dest = 'path_save',
                         help = 'Absolute path to a file to save the lists')

    parser.add_argument ('--main-lists', action = 'store',
                         type = str, dest = 'path_main_lists',
                         help = 'Absolute path to a file containing the main lists')

    parser.add_argument ('--extension', default='dat',
                         type = str, dest = 'extension',
                         help = 'extensio of the files in \'datadir\' ')

    parser.add_argument ('--log-level', default=logging.INFO,
                         type=lambda x: getattr (logging, x),
                         help = "Configure the logging level: DEBUG|INFO|WARNING|ERROR|FATAL")

    parser.add_argument ('--lists', action = 'store', type = str,
                         dest = 'path_lists',
                         help = 'Absolute path to a file containing lists')

    parser.add_argument ('--new_dir', action = 'store', type = str,
                         dest = 'path_new_dir',
                         help = 'Absolute path to the raw data, to change in a given file lists')

    args, unknown = parser.parse_known_args ()
    logging.basicConfig (level = args.log_level)

    main_list = False

    ## change the directory of a list
    if ((args.path_lists is not None) and (args.path_new_dir is not None)):
        change_directory (args.path_lists, args.path_new_dir)


    ## generate main list (not tag needed)
    if (args.path_raw is not None and args.path_tagmap is None):
        # split testing and learning
       x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test \
           = compute_main_list (args.path_raw, extension = args.extension)

       main_list = True

       if (logging.root.level < logging.INFO):
            print ('main generated list:')
            display_list (y_train, y_val, y_test)


    ## otherwise it is loaded
    elif (args.path_main_lists is not None):
        [x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test]\
            = np.load (args.path_main_lists, allow_pickle = True)

        main_list = True
        if (logging.root.level < logging.INFO):
            print ('main loaded list:')
            display_list (y_train, y_val, y_test)

    ## creat list from the main list
    if (args.path_tagmap is not None and main_list):
        x_train_filelist, _, x_trainandtest_filelist,\
            y_train, _, y_trainandtest = parse_data (x_train_filelist + x_val_filelist,
                                                            args.path_tagmap)

        x_train_filelist = x_train_filelist + x_trainandtest_filelist
        y_train = y_train + y_trainandtest

        # creat the validation
        x_train_filelist, x_val_filelist, y_train, y_val\
            = train_test_split (x_train_filelist, y_train, test_size=0.2)

        # compute testing
        _, x_test_filelist, x_trainandtest_filelist,\
            _, y_test, y_trainandtest = parse_data (x_test_filelist,
                                                    args.path_tagmap)

        x_test_filelist = x_test_filelist + x_trainandtest_filelist
        y_test = y_test+ y_trainandtest

        if (logging.root.level < logging.INFO):
            print ('new list:')
            display_list (y_train, y_val, y_test)


        # ## test witht the paper values
        # tmp = args.path_tagmap.split ('/')[-1].split ('.')[0]
        # tmp = '../neural-network/files_lists/files_lists_tagmap=' + tmp + '_with_var.npy'
        # [x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test]\
        #     = np.load (tmp, allow_pickle = True)

        # print ('paper list:')
        # display_list (y_train, y_val, y_test)

    ## generate list from raw data and tagmap
    elif (args.path_tagmap):
        # get the list of files
        filelist = glob.glob (args.path_raw + "/**/*.%s"%args.extension, recursive = True)

        # creat lists
        x_train_filelist, x_test_filelist, x_trainandtest_filelist,\
        y_train, y_test, y_trainandtest = parse_data (x_train_filelist + x_val_filelist,
                                                            args.path_tagmap)

        # split x_trainandtest
        if len(x_trainandtest_filelist) > 0:
            x_train_filelist_tmp, x_test_filelist_tmp, y_train_tmp, y_test_tmp\
                = train_test_split(x_trainandtest_filelist, y_trainandtest, test_size=0.2)

            x_train_filelist += x_train_filelist_tmp
            x_test_filelist  += x_test_filelist_tmp

            y_train += y_train_tmp
            y_test  += y_test_tmp

        # generate validating set
        x_train_filelist, x_test_filelist, y_train, y_test = train_test_split(filelist, y_train, test_size=0.2)


        if (logging.root.level < logging.INFO):
            print ('generated list:')
            display_list (y_train, y_val, y_test)


    ## save the computed
    if (args.path_save is not None):
        np.save (args.path_save,
                 [x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test],
                 allow_pickle = True)
