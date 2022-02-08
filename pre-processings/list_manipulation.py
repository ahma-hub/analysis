"""
 File: list_manipulation.py 
 Project: analysis 
 Last Modified: 2022-7-2
 Created Date: 2022-7-2
 Copyright (c) 2021
 Author: AHMA project (Univ Rennes, CNRS, Inria, IRISA)
"""

################################################################################
import argparse
import numpy                   as np
import logging
import random
import glob, os

from   sklearn.model_selection import train_test_split
from   tabulate                import tabulate
from   tqdm                    import tqdm


################################################################################
class bcolors:
################################################################################
# class bcolors
# use to get colors iun the terminal
#
################################################################################

    black='\033[30m'
    red='\033[31m'
    green='\033[32m'
    orange='\033[33m'
    blue='\033[34m'
    purple='\033[35m'
    cyan='\033[36m'
    lightgrey='\033[37m'
    darkgrey='\033[90m'
    lightred='\033[91m'
    lightgreen='\033[92m'
    yellow='\033[93m'
    lightyellow = '\u001b[33;1m'
    lightblue='\033[94m'
    pink='\033[95m'
    lightcyan='\033[96m'

    endc = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'


    whitebg = '\u001b[47;1m'
    resetbg = '\u001b[0m'
    
################################################################################
def change_directory (path_lists, new_dir):
################################################################################
# change_directory
# change the path of the directory where the traces are stored
#
# input:
#  + path_lists: file containing tthe lists
#  + new_dir: new directory
################################################################################
    [x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test] \
        = np.load (path_lists, allow_pickle = True)

    for i in range (len (x_train_filelist)):
        x_train_filelist [i] = new_dir +  '/' + os.path.basename (x_train_filelist [i])
        ## hack to modify the extension
        # x_train_filelist [i] = new_dir +  '/' + '.'.join (os.path.basename (x_train_filelist [i]).split ('.')[:-1]) + '.npy'


    for i in range (len (x_val_filelist)):
        x_val_filelist [i] = new_dir +  '/' + os.path.basename (x_val_filelist [i])
        ## hack to modify the extension
        # x_val_filelist [i] = new_dir +  '/' + '.'.join (os.path.basename (x_val_filelist [i]).split ('.')[:-1]) + '.npy'


    for i in range (len (x_test_filelist)):
        x_test_filelist [i] = new_dir +  '/' + os.path.basename (x_test_filelist [i])
        ## hack to modify the extension
        # x_test_filelist [i] = new_dir + '/' + '.'.join (os.path.basename (x_test_filelist [i]).split ('.')[:-1]) + '.npy'


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
def display_list (x_train, x_val, x_test, y_train, y_val, y_test):
################################################################################
# display_list
# display the the content of a list:
#
# inputs:
#  + x_{train, val, test} filenames to get the tags
#  + y_{train, val, test} labels of a list
################################################################################

    y_unique = np.unique (np.array (list (y_train) + list (y_val) + list (y_test)))
    tags = [np.array ([get_tag (f) for f in list (x_train)]),
            np.array ([get_tag (f) for f in list (x_val)]),
            np.array ([get_tag (f) for f in list (x_test)])]

    lines = []
    last_line = ['-', 'total', 0, 0, 0, 0, 0]

    y_idx_lines = []
    tags_idx_lines = []
    
    for i in range (len (y_unique)):
        line = [i, f'{bcolors.bold}{y_unique [i]}{bcolors.endc}']
        count = 0
        
        current_tags = []
        
        ## train 
        idx = np.where (np.array (y_train)  == y_unique [i])[0]
        line.append (len (idx))
        last_line [2] += len (idx)
        count += len (idx)
        
        current_tags.append (tags [0][idx])

        ## val
        idx = np.where (np.array (y_val)  == y_unique [i]) [0]
        line.append (len (idx))
        last_line [3] += len (idx)
        count += len (idx)

        current_tags.append (tags [1][idx])
                
        ## train + val
        line.append (count)
        last_line [4] += count
        
        ## test
        idx = np.where (np.array (y_test)  == y_unique [i]) [0]
        line.append (len (idx))
        last_line [5] += len (idx)
        count += len (idx)
        
        current_tags.append (tags [2][idx])
        
        ## totlal
        line.append (count)
        last_line [6] += count

        lines.append (line)
        y_idx_lines.append (len (lines) - 1)

        tmp_tags_idx_lines = []
        
        current_unique_tags = np.unique (np.concatenate ((current_tags [0], current_tags [1], current_tags [2])))

        for j in range (len (current_unique_tags)):
            
            line = [f'{i}-{j}', f'{current_unique_tags [j]}']
            tag_count = 0
            
            ## train 
            idx = np.where (np.array (tags [0])  == current_unique_tags [j])[0]
            line.append (len (idx))
            tag_count += len (idx)
        
            ## val
            idx = np.where (np.array (tags [1])  == current_unique_tags [j]) [0]
            line.append (len (idx))
                        
            ## train + val
            tag_count += len (idx)
            line.append (tag_count)
                    
            ## test
            idx = np.where (np.array (tags [2])  == current_unique_tags [j]) [0]
            line.append (len (idx))
                    
            ## totlal
            tag_count += len (idx)
            line.append (tag_count)

            lines.append (line)

            tmp_tags_idx_lines.append (len (lines) - 1)
        
        tags_idx_lines.append (tmp_tags_idx_lines)
        

    ## add the percentages
    lines_array = np.array (np.array (lines)[:, 2:], dtype = np.uint32)
    
    for i in range (lines_array.shape [1]): ## for each colunm
        for j in range (len (y_idx_lines)): ## for each label
            current_percentage = 100*lines_array [y_idx_lines [j], i]/lines_array [y_idx_lines, i].sum ()
            lines [y_idx_lines [j]][i + 2] = f'{bcolors.lightblue}{lines [y_idx_lines [j]][i + 2]}{bcolors.endc} {bcolors.lightgreen}[{current_percentage:.2f}]{bcolors.endc}'
            
            for k in range (len (tags_idx_lines [j])): # for each tags
                current_percentage_local = 100*lines_array [tags_idx_lines [j][k], i]/lines_array [tags_idx_lines [j], i].sum ()
                current_percentage_global = 100*lines_array [tags_idx_lines [j][k], i]/lines_array [y_idx_lines, i].sum ()

                lines [tags_idx_lines [j][k]][i + 2] = f'{bcolors.blue}{lines [tags_idx_lines [j][k]][i + 2]}{bcolors.endc}'\
                    +f' {bcolors.green}[{current_percentage_global:.2f}]{bcolors.endc}'\
                    +f' {bcolors.yellow}[{current_percentage_local:.2f}]{bcolors.endc}'
                

    for j in range (len (y_idx_lines)):
        lines [y_idx_lines [j]][0]  = f'{bcolors.whitebg} {lines [y_idx_lines [j]][0]} {bcolors.resetbg}'
                
    for i in range (2, len (last_line) - 1):
        last_line [i] = f'{bcolors.blue}{last_line [i]}{bcolors.endc} {bcolors.green}[{100*last_line [i]/last_line [-1]:.2f}]{bcolors.endc}'
        
    last_line [-1] = f'{bcolors.lightred}{last_line [-1]}{bcolors.endc}'

    lines.append (last_line)

    print(tabulate (lines, headers= [f'{bcolors.bold}idx{bcolors.endc}\nsub-idx',
                                     f'{bcolors.bold}label{bcolors.endc}\ntag',
                                     f'{bcolors.bold}train nbr [gobal %] {bcolors.endc}\nnbr [global %] [local %]',
                                     f'{bcolors.bold}val nbr [gobal %] {bcolors.endc}\nnbr [global %] [local %]',
                                     f'{bcolors.bold}test nbr [gobal %] {bcolors.endc}\nnbr [global %] [local %]',
                                     f'{bcolors.bold}total nbr [gobal %] {bcolors.endc}\nnbr [global %] [local %]'], tablefmt="grid"))

    
# ################################################################################
# def display_list (y_train, y_val, y_test):
# ################################################################################
# # display_list
# # display the the content of a list
# #
# # inputs:
# #  + y_{train, val, test} labels of a list
# ################################################################################
#     y_unique = np.unique (y_train + y_val + y_test)

#     lines = []

#     for i in range (len (y_unique)):
#         line = [i, y_unique [i]]
#         count = 0

#         idx = np.where (np.array (y_train)  == y_unique [i])[0]
#         line.append (len (idx))
#         count += len (idx)

#         idx = np.where (np.array (y_val)  == y_unique [i]) [0]
#         line.append (len (idx))
#         count += len (idx)
#         line.append (count)

#         idx = np.where (np.array (y_test)  == y_unique [i]) [0]
#         line.append (len (idx))
#         count += len (idx)
#         line.append (count)

#         lines.append (line)

#     print(tabulate (lines, headers= ['idx', 'label', 'train', 'val', 'train + val', 'test', 'total']))



################################################################################
def compute_main_list (data, extension, nb_of_traces_per_label):
################################################################################
# compute_main_list
# [from neural-network/utils]
# Label the data and separate it in train and test dataset.
# inputs:
#  - datadir: path to the directory containing all data
#  - extension: type of file in datadir
#  - nb_of_traces_per_label: nb of traces per labels
#
# outputs:
# - lists: {filelist, labels} x {learning, validating, testing}
################################################################################

    if (not type (data) is list):
        filelist = glob.glob (data + "/**/*.%s"%extension, recursive = True)
    else:
        filelist = data

    ## sanity check
    clean_file = []
    empty_file = []
    for i in tqdm (filelist, desc = 'sanity check', leave = False):
        if (os.stat (i).st_size == 0):
            empty_file.append (i)
        else:
            clean_file.append (i)

    if (len (empty_file) != 0):
        print (f'[EMPTY FILES]: {empty_file} ({len (empty_file)})')

    filelist = clean_file
        
    random.shuffle (filelist)
    
    # get labels 
    y = np.array ([get_tag (f) for f in filelist])

    # if a limit is needed
    if (nb_of_traces_per_label is not None):
        unique_y = np.unique (y)
        new_y = []
        new_filelist = []
        filelist = np.array (filelist)
        for u in unique_y:
            idx = np.where (y == u)[0]
            new_y += list (y [idx [:nb_of_traces_per_label]])
            new_filelist += list (filelist [idx [:nb_of_traces_per_label]])

        y = new_y
        filelist = new_filelist
    
    x_train_filelist, x_test_filelist, y_train, y_test\
        = train_test_split (filelist, y, test_size=0.2)

    x_train_filelist, x_val_filelist, y_train, y_val\
            = train_test_split (x_train_filelist, y_train, test_size=0.2)


    return x_train_filelist, x_val_filelist, x_test_filelist,\
        y_train, y_val, y_test


################################################################################
def parse_data (filelist, tagmap):
################################################################################
# parse_data
# [from neural-network/utils]
# Label the data and separate it in train and test dataset.
# It is possible to provide a path to a file containing a tag map.
# Each line of this file should be formatted like this: tag <space>
# corresponding_label <space> dataset
# Dataset value are 0: not used, 1: train only, 2: test only, 3: train and test
# inputs:
#  - filelist: list of the filename
#  - tagmap: tagmap used for the labeling
#
# outputs:
#  - x_train_filelist, x_test_filelist, x_trainandtest_filelist: files lists
#  - y_train, y_test, y_trainandtest: labels
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
                         help = 'extensio of the raw traces ')

    parser.add_argument ('--log-level', default=logging.INFO,
                         type=lambda x: getattr (logging, x),
                         help = "Configure the logging level: DEBUG|INFO|WARNING|ERROR|FATAL")

    parser.add_argument ('--lists', action = 'store', type = str,
                         dest = 'path_lists',
                         help = 'Absolute path to a file containing lists')

    parser.add_argument ('--new_dir', action = 'store', type = str,
                         dest = 'path_new_dir',
                         help = 'Absolute path to the raw data, to change in a given file lists')

    parser.add_argument('--nb_of_traces_per_label', action='store', type=int,
                        default=None,
                        dest='nb_of_traces_per_label',
                        help='number of traces to keep per label')


    args, unknown = parser.parse_known_args ()
    assert len (unknown) == 0, f"[WARNING] Unknown arguments:\n{unknown}\n"

    logging.basicConfig (level = args.log_level)

    if (logging.root.level < logging.INFO):
        print ("argument:")
        for arg in vars(args):
            print (f"{arg} : {getattr (args, arg)}")

    main_list = False
    ## change the directory of a list
    if ((args.path_lists is not None) and (args.path_new_dir is not None)):
        change_directory (args.path_lists, args.path_new_dir)


    ## generate main list (not tag needed)
    if (args.path_raw is not None and args.path_tagmap is None):
        # split testing and learning
        x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test \
            = compute_main_list (args.path_raw, extension = args.extension,
                                 nb_of_traces_per_label = args.nb_of_traces_per_label)

        main_list = True

        if (logging.root.level < logging.INFO):
            print ('main generated list:')
            display_list (x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test)


    ## otherwise it is loaded
    elif (args.path_main_lists is not None):
        [x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test]\
            = np.load (args.path_main_lists, allow_pickle = True)

        main_list = True
        if (logging.root.level < logging.INFO):
            print ('main loaded list:')
            display_list (x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test)
        
    ## creat list from the main list
    if (args.path_tagmap is not None and main_list):
        x_train_filelist, x_del_0, x_trainandtest_filelist,\
            y_train, y_del_0, y_trainandtest = parse_data (x_train_filelist + x_val_filelist,
                                                            args.path_tagmap)

        x_train_filelist = x_train_filelist + x_trainandtest_filelist
        y_train = y_train + y_trainandtest

        # creat the validation
        x_train_filelist, x_val_filelist, y_train, y_val\
            = train_test_split (x_train_filelist, y_train, test_size=0.2)

        # compute testing
        x_del_1, x_test_filelist, x_trainandtest_filelist,\
            y_del_1, y_test, y_trainandtest = parse_data (x_test_filelist,
                                                    args.path_tagmap)

        x_test_filelist = x_test_filelist + x_trainandtest_filelist
        y_test = y_test+ y_trainandtest

        if (logging.root.level < logging.INFO):
            print ('new list:')
            display_list (x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test)

            # if (len (x_del_0) != 0):
            #     print (f'deleted from training: {len (x_del_0)}')
            #     display_list (y_del_0, [], [])

            # if (len (x_del_1) != 0):
            #     print (f'deleted from testing: {len (x_del_1)}')
            #     display_list ([], [], y_del_1)



    ## generate list from raw data and tagmap
    elif (args.path_tagmap is not None):
        # get the list of files
        filelist = glob.glob (args.path_raw + "/**/*.%s"%args.extension, recursive = True)

        # creat lists
        x_train_filelist, x_test_filelist, x_trainandtest_filelist,\
        y_train, y_test, y_trainandtest = parse_data (filelist, args.path_tagmap)

        # split x_trainandtest
        if (len (x_trainandtest_filelist) > 0):
            print (f'{len (x_trainandtest_filelist)}, {np.unique (y_trainandtest, return_counts = True)[1].sum ()}')

            x_train_filelist_tmp, x_test_filelist_tmp, y_train_tmp, y_test_tmp\
                = train_test_split(x_trainandtest_filelist, y_trainandtest, test_size=0.2)

            x_train_filelist += x_train_filelist_tmp
            x_test_filelist  += x_test_filelist_tmp

            y_train += y_train_tmp
            y_test  += y_test_tmp

        # generate validating set
        x_train_filelist, x_val_filelist, y_train, y_val = train_test_split (x_train_filelist, y_train, test_size=0.2)


        if (logging.root.level < logging.INFO):
            print ('generated list:')
            display_list (x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test)
            
    ## display a provided list
    elif ((args.path_lists is not None) and  (logging.root.level < logging.INFO)):
        [x_train_filelist, x_val_filelist, x_test_filelist,
         y_train, y_val, y_test] = np.load (args.path_lists, allow_pickle = True)

        display_list (x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test)
            

    ## save the computed
    if (args.path_save is not None):
        np.save (args.path_save,
                 [x_train_filelist, x_val_filelist, x_test_filelist, y_train, y_val, y_test],
                 allow_pickle = True)

