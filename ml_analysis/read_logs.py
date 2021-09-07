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
        self.means_vect  = [int (i) for i in self.means[2:-2].split (',')]
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
                if (tmp_split [0] == 'means'):
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
    means = exps [0].means_vect ## /!\ all exp must have the means
    means = [1] + means # add the non-averaged

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
    # axs [0].axvspan (0.8, 1.2, facecolor = 'blue', alpha=0.2)
    # axs [1].axvspan (0.8, 1.2, facecolor = 'blue', alpha=0.2)


    for i in range (len (unique_tags)):
        nb   = np.zeros ((len (means), len (bds)))
        nb_val   = np.zeros ((len (means), len (bds)))

        nb_precision_macro = np.zeros ((len (means), len (bds)))
        nb_precision_weigh = np.zeros ((len (means), len (bds)))
        nb_recall_macro = np.zeros ((len (means), len (bds)))
        nb_recall_weigh = np.zeros ((len (means), len (bds)))
        nb_f1_macro = np.zeros ((len (means), len (bds)))
        nb_f1_weigh = np.zeros ((len (means), len (bds)))

        svm   = np.zeros ((len (means), len (bds)))
        svm_val = np.zeros ((len (means), len (bds)))
        svm_precision_macro = np.zeros ((len (means), len (bds)))
        svm_precision_weigh = np.zeros ((len (means), len (bds)))
        svm_recall_macro = np.zeros ((len (means), len (bds)))
        svm_recall_weigh = np.zeros ((len (means), len (bds)))
        svm_f1_macro = np.zeros ((len (means), len (bds)))
        svm_f1_weigh = np.zeros ((len (means), len (bds)))

        count = 0
        row = ['_'.join (unique_tags [i].split ('_')[-3:])]
        for j in range (len (exps)):
            if (unique_tags [i] == exps [j].path_lists):

                nb  [:, count] = exps [j].NB_acc
                # nb_val  [:, count] = exps [j].NB_val

                nb_precision_macro  [:, count] = exps [j].NB_macro_precision
                nb_precision_weigh  [:, count] = exps [j].NB_weighted_precision
                nb_recall_macro  [:, count] =  exps [j].NB_macro_recall
                nb_recall_weigh  [:, count] =  exps [j].NB_weighted_recall
                nb_f1_macro  [:, count] =  exps [j].NB_macro_f1
                nb_f1_weigh  [:, count] =  exps [j].NB_weighted_f1

                svm [:, count] = exps [j].SVM_acc
                # svm_val [:, count] = exps [j].SVM_val

                svm_precision_macro  [:, count] = exps [j].SVM_macro_precision
                svm_precision_weigh  [:, count] = exps [j].SVM_weighted_precision
                svm_recall_macro  [:, count] =  exps [j].SVM_macro_recall
                svm_recall_weigh  [:, count] =  exps [j].SVM_weighted_recall
                svm_f1_macro  [:, count] =  exps [j].SVM_macro_f1
                svm_f1_weigh  [:, count] =  exps [j].SVM_weighted_f1

                count += 1

        # print (unique_tags [i])


        # idx_nb = np.unravel_index (current_nb.argmax (), current_nb.shape)
        # print (bds [np.argmax (current_nb [0, :])], current_nb [0, :].max ())
        # print (means [idx_nb [0]], bds [idx_nb [1]], current_nb.max ())

        # idx_nb = np.unravel_index (current_svm.argmax (), current_svm.shape)
        # print (bds [np.argmax (current_svm [0, :])], current_svm [0, :].max ())
        # print (means [idx_nb [0]], bds [idx_nb [1]], current_svm.max ())

        idx_bd = np.argmax (nb [0, :])
        row.append (f'{nb [0, :].max ()}') # No val (auto-testing) anymore '[{nb_val [0, idx_bd]:.4f}]')
        row.append (f'#bds: {bds [idx_bd]}')

        row.append (f'{nb_precision_macro [0, idx_bd]}/{nb_precision_weigh [0, idx_bd]}')
        row.append (f'{nb_recall_macro [0, idx_bd]}/{nb_recall_weigh [0, idx_bd]}')
        row.append (f'{nb_f1_macro [0, idx_bd]}/{nb_f1_weigh [0, idx_bd]}')

        idx_bd = np.argmax (svm [0, :])
        row.append (f'{svm [0, :].max ()}') # No val (auto-testing) anymore [{svm_val [0, idx_bd]:.4f}]')
        row.append (f'#bds: {bds [idx_bd]}')

        row.append (f'{svm_precision_macro [0, idx_bd]}/{svm_precision_weigh [0, idx_bd]}')
        row.append (f'{svm_recall_macro [0, idx_bd]}/{svm_recall_weigh [0, idx_bd]}')
        row.append (f'{svm_f1_macro [0, idx_bd]}/{svm_f1_weigh [0, idx_bd]}')

        tabular.append (row)

        tmp_nb = []
        tmp_svm = []
        for j in range (len (means)):
            tmp_nb.append (nb [j, :].max ())
            tmp_svm.append (svm [j, :].max ())


        # axs [0].plot (bds, nb [0, :].T, label = tags [i])# (
        axs [0].plot (means, tmp_nb, label = tags [i], linestyle = '-')#, marker='.')
        axs [0].grid (True)
        axs [0].set_ylabel ('accuracy')#, fontweight='bold')
        axs [0].set_title ('NB')#, fontweight='bold')
        axs [0].set_xticks (means)
        # axs [0].set_xticks (bds)
        axs [0].set_xticklabels ([str (m) for m in means])
        # axs [0].set_xticklabels ([str (m) for m in bds])


        # axs [1].plot (bds, svm [0, :].T, label = tags [i])
        axs [1].plot (means, tmp_svm, label = tags [i], linestyle = '-')#, marker='.')
        axs [1].grid (True)
        axs [1].set_title ('SVM')
        axs [1].set_xticks (means)
        axs [1].set_xticklabels ([str (m) for m in means])
        # axs [1].set_xticks (bds)
        axs [0].set_xticklabels ([str (m) for m in means])
        # axs [1].set_xticklabels ([str (m) for m in bds])



    handles, labels = axs [0].get_legend_handles_labels()
    fig.legend (handles, labels, loc='upper center', ncol=2)

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # plt.xlabel ('Number of traces t per mean')
    plt.xlabel ('Number of traces bds')

    fig.subplots_adjust (wspace=0.025, top=0.80)


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

    ## to save in file
    if (output):
        plt.savefig (output, format = 'pdf')
    else:
        plt.show ()




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
