"""
 File: read_logs.py
 Project: analysis
 Last Modified: 2021-8-2
 Created Date: 2021-8-2
 Copyright (c) 2021
 Author: AHMA project (Univ Rennes, CNRS, Inria, IRISA)
"""

import numpy            as np
from   tqdm             import tqdm
import argparse
from   tabulate                import tabulate
from io import StringIO

import matplotlib, sys
## to avoid bug when it is run without graphic interfaces
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except ImportError:
    # print ('Warning importing GTK3Agg: ', sys.exc_info()[0])
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

################################################################################
class Exp:
################################################################################
# to store the data of an experience (1 call of evaluation)
################################################################################
    def __init__ (self):# (self, tag, means, nb_of_bd):

        ## for the "meta" data
        # example:
        # 02.09.2021 - 13:30:41
        # path_lists: lists/extracted_bd_files_lists_tagmaps=executable_classification.npy
        # log_file: log-evaluation.txt
        # model_lda: None
        # model_svm: None
        # model_nb: None
        # means: [2, 3, 4, 5, 6, 7, 9, 10]
        # nb_of_bd: 6
        # path_acc: acc_stft/
        # time_limit: 0.5
        # metric: nicv_max

        self.date       = None
        self.hour       = None
        self.path_lists = None
        self.log_file   = None
        self.model_lda  = None
        self.model_svm  = None
        self.model_nb   = None
        self.means      = None
        self.nb_of_bd   = None
        self.path_acc   = None
        self.time_limit = None
        self.metric     = None

        self.LDA_duration = None
        self.NB_duration  = None
        self.SVM_duration = None

        self.means_vect  = None
        self.nb_of_means = None

        ## to store the results
        for i in ['LDA', 'SVM', 'NB']:
            setattr (self, f'{i}_duration', None)

        for i in ['SVM', 'NB']:
            setattr (self, f'{i}_acc', [])

            for j in ['macro', 'weighted']:
                setattr (self, f'{i}_{j}_precision', [])
                setattr (self, f'{i}_{j}_recall', [])
                setattr (self, f'{i}_{j}_f1', [])

    def init_vectors (self):
        ## [2, -2] to remove blank '[' and ']'
        self.means_vect = [int (i) for i in self.means[2:-2].split (',')]
        self.nb_of_means = len (self.means)



################################################################################
def read_log (path):
################################################################################
# read_log
# extract all usefull information from log
#
# input:
#  + path: location of the log
#
# output:
#  + res: list of Exp
################################################################################
    f = open (path, 'r')
    lines = f.readlines ()

    res = []
    new_item = False
    reading_data = True

    current_res = None

    for line in lines:
        ## start a new experience
        if (10*'-' in line):
            if (reading_data): # save
                new_item = True
                reading_data = False
                if (current_res):
                    res.append (current_res)

                current_res = Exp ()

            else :
                new_item = False
                reading_data = True

        ## reading the header
        elif (new_item):
            if (not current_res.date):
                tmp_split = line.split ('-')

                current_res.date = tmp_split [0]
                current_res.hour = tmp_split [1]
            else:
                tmp_split = line.split (':')
                setattr (current_res, tmp_split [0],  tmp_split [1])
                if (tmp_split [0] == 'means' and 'None' not in current_res.means):
                    current_res.init_vectors ()


        ## reading the data
        elif (reading_data):
            tmp_split = line.split ()

            ## save the times
            if (len (tmp_split) > 1 and tmp_split [0] == 'LDA' and tmp_split [1] == '(compuation):'): # = {LDA, NB, SVM} (computation
                setattr (current_res, f'{tmp_split [0]}_duration', float (tmp_split [2]))

            ## Test {NB, SVM}
            elif (len (tmp_split) > 1 and (tmp_split [0] == 'Test')):
                  current_exp = tmp_split [1]

            # {NB, SVM}{-}{mean}{val:}
            elif (len (tmp_split) > 1 and (tmp_split [0] == 'NB' or tmp_split [0] == 'SVM')):
                  current_exp = tmp_split [0]

            # {NB, SVM}_acc
            elif (len (tmp_split) > 1 and tmp_split [0] == 'accuracy'):
                getattr (current_res, f'{current_exp}_acc').append (float (tmp_split [1]))

            # {NB, SVM}_{macro, weighted}_{precision, recall, f1}
            elif (len (tmp_split) > 1 and (tmp_split [0] == 'macro' or tmp_split [0] == 'weighted')):
                getattr (current_res, f'{current_exp}_{tmp_split [0]}_precision').append (float (tmp_split [2]))
                getattr (current_res, f'{current_exp}_{tmp_split [0]}_recall').append (float (tmp_split [3]))
                getattr (current_res, f'{current_exp}_{tmp_split [0]}_f1').append (float (tmp_split [4]))

    f.close ()

    ## add the last one
    res.append (current_res)

    return res

################################################################################
def display_results (exps, output):
################################################################################
# display_results
# display the results from a log
#
# input:
#  + exps: list of Exp (cf read_log
#  + output: path to the file to save the figure (if None, a pop'up will be open)
################################################################################

    unique_tags = np.unique (np.array ([i.path_lists for i in exps]))

    bds = np.unique (np.array ([int (i.nb_of_bd) for i in exps]))

    if ('None' not in exps [0].means): ## no meaning
        means = exps [0].means_vect ## /!\ all exp must have the means
        means = [1] + means # add the non-averaged


        nb_max_of_means = max ([len (i.means_vect) for i in exps])
        means = []
        for i in exps:
            means += i.means_vect

        means = np.sort (np.unique (means))

        means_NB = np.zeros ((nb_max_of_means + 1, len (unique_tags)))
        means_SVM = np.zeros ((nb_max_of_means + 1, len (unique_tags)))

    else: # No means
        means = []

    tags =  np.unique (np.array ([i.path_lists.split ('=')[-1].split ('.')[0] for i in exps]))

    tabular = []


    font_size = 29
    params = {'legend.fontsize': font_size,
              'axes.labelsize':  font_size,
              'axes.titlesize':  font_size,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size}

    plt.rcParams.update (params)

    fig, axs = plt.subplots (nrows = 1, ncols= 2, figsize = (16, 16), sharex=True, sharey=True)

    ## Construct the tabular
    for i in range (len (unique_tags)):
        current_row = [f'{tags [i]}']
        current_nb = [0, 0]
        current_svm = [0, 0]


        for j in range (len (exps)):
            if (unique_tags [i] == exps [j].path_lists):
                if (len (exps [j].NB_acc)> 0 and exps [j].NB_acc [0] > current_nb [0]):
                    current_nb [0] = exps [j].NB_acc [0]
                    current_nb [1] = j

                    if (len (means) != 0):
                        means_NB   [0, i] = current_nb [0]
                        for k in range (len (exps [j].means_vect)):
                            means_NB  [k + 1, i] = max (means_NB [k + 1, i], exps [j].NB_acc [k + 1])

                if (len (exps [j].SVM_acc) > 0 and exps [j].SVM_acc [0] > current_svm [0]):
                    current_svm [0] = exps [j].SVM_acc [0]
                    current_svm [1] = j

                    if (len (means) != 0):
                        means_SVM  [0, i] = current_svm [0]
                        for k in range (len (exps [j].means_vect)):
                            means_SVM [k + 1, i] = max (means_SVM [k + 1, i], exps [j].SVM_acc [k + 1])


        current_row += [ # NB
            f'{current_nb [0]}', f'{exps [current_nb [1]].nb_of_bd}',\
            f'{exps [current_nb [1]].NB_macro_precision [0]}/{exps [current_nb [1]].NB_weighted_precision [0]}',\
            f'{exps [current_nb [1]].NB_macro_recall [0]}/{exps [current_nb [1]].NB_weighted_recall [0]}',\
            f'{exps [current_nb [1]].NB_macro_f1 [0]}/{exps [current_nb [1]].NB_weighted_f1 [0]}',\
            # SVM
             f'{current_svm [0]}', f'{exps [current_svm [1]].nb_of_bd}',\
            f'{exps [current_svm [1]].SVM_macro_precision [0]}/{exps [current_svm [1]].SVM_weighted_precision [0]}',\
            f'{exps [current_svm [1]].SVM_macro_recall [0]}/{exps [current_svm [1]].SVM_weighted_recall [0]}',\
            f'{exps [current_svm [1]].SVM_macro_f1 [0]}/{exps [current_svm [1]].SVM_weighted_f1 [0]}']



        tabular.append (current_row)


    # display means
    if (len (means) != 0):
        for i in range (len (tags)):
            axs [0].plot ([1] + list (means), means_NB [:, i], label = tags [i], linestyle = '-')#, marker='.')
            axs [0].grid (True)
            axs [0].set_ylabel ('accuracy')
            axs [0].set_title ('NB')
            axs [0].set_xticks ([1] + list (means))
            axs [0].set_xticklabels (['1'] + [str (m) for m in means])


            axs [1].plot ([1] + list (means), means_SVM [:, i], label = tags [i], linestyle = '-')#, marker='.')
            axs [1].grid (True)
            axs [1].set_title ('SVM')
            axs [1].set_xticks ([1] + list (means))
            axs [1].set_xticklabels (['1'] +[str (m) for m in means])


        handles, labels = axs [0].get_legend_handles_labels()
        fig.legend (handles, labels, loc='upper center', ncol=2)

        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        plt.xlabel ('Number of traces t per mean')

        ## to save in file
        if (output):
            plt.savefig (output, format = 'pdf')
        else:
            plt.show ()


    print(tabulate (tabular,
                    headers= ['exp',
                              'LDA + NB ',
                              '#bd',
                              'precision (macro/weigh)',
                              'recall (macro/weigh)',
                              'f1 (macro/weigh)',
                              'LDA + SVM',
                              '#bd',
                              'precision (macro/weigh)',
                              'recall (macro/weigh)',
                              'f1 (macro/weigh)']))


################################################################################
if __name__ == '__main__':
################################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument ('--path', action='store', type=str,
                        dest='path',
                        help='Absolute path to the log file')

    parser.add_argument ('--plot', action = 'store', default = None,
                         type = str, dest = 'path_to_plot',
                         help = 'Absolute path to save the plot')

    args, unknown = parser.parse_known_args ()
    assert len (unknown) == 0, f"[WARNING] Unknown arguments:\n{unknown}\n"

    ## read and parse the log file
    results = read_log (args.path)

    ## display the results tabular in the terminal and the figure (in a pop'up
    # of in a pdf file)
    display_results (results, args.path_to_plot)
