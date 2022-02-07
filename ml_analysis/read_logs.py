"""
 File: read_logs.py
 Project: analysis
 Last Modified: 2022-2-7
 Created Date: 2022-2-7
 Copyright (c) 2021
 Author: AHMA project (Univ Rennes, CNRS, Inria, IRISA)
"""

################################################################################
import argparse
import re
import sys
from io import StringIO

import matplotlib
import numpy as np
from tqdm import tqdm


## to avoid bug when it is run without graphic interfaces
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import matplotlib.patches as patches
import tabulate

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import itertools
import numpy as np

matplotlib.use("pdf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 8,
    "font.size": 8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})
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
GRADIENT_COLOR = [bcolors.orange,
                  bcolors.lightyellow,
                  bcolors.yellow,
                  bcolors.lightgreen,
                  bcolors.green,
                  bcolors.lightblue,
                  bcolors.blue,
                  bcolors.lightred,
                  bcolors.red]

################################################################################
def get_color (val):
########################################*########################################
# get_color
# return a string embedded the given val to be written in color in the terminal
    threshold = 0.90
    if (val < threshold):
        return f'{GRADIENT_COLOR [0]}{100*val:.1f}{bcolors.endc}'
    else:
        return f'{GRADIENT_COLOR [int ( (val - threshold)*(len (GRADIENT_COLOR) - 2) / (1 - threshold)) + 1]}{100*val:.2f}{bcolors.endc}'


################################################################################
def display_matrix_exp (grid_x, grid_y, data_nb, data_svm, file_name, **kwargs):
################################################################################
# display_matrix_exp
# from https://stackoverflow.com/questions/38323669/pyplot-imshow-for-rectangles
#
# input:
#  - grid_{x, y}: axis (with one point more than data_*)
#  - data_{nb, svm}: results of the experiments
#  - file_name: name of figure to save
#  - kwargs: additionnal parameters for ""PatchCollection""
################################################################################

    vmin = kwargs.pop("vmin", min (data_nb.min (), data_svm.min ()))
    vmax = kwargs.pop("vmax", 100)

    data_nb = np.array(data_nb).reshape(-1)
    # there should be data for (n-1)x(m-1) cells
    assert (grid_x.shape[0] - 1) * (grid_y.shape[0] - 1) == data_nb.shape[0],\
        "Wrong number of data points. grid_x=%s, grid_y=%s, data=%s" % (grid_x.shape, grid_y.shape, data_nb.shape)

    ptchs = []
    for j, i in itertools.product (range (len (grid_y) - 1), range (len (grid_x) - 1)):
        xy = grid_x [i], grid_y [j]
        width = grid_x [i+1] - grid_x [i]
        height = grid_y [j+1] - grid_y [j]
        ptchs.append (Rectangle (xy=xy, width=width, height=height, rasterized=True, linewidth=0, linestyle="None"))
    p_nb  = PatchCollection (ptchs, linewidth=0, cmap='Blues', **kwargs)
    p_svm = PatchCollection (ptchs, linewidth=0, cmap='Blues', **kwargs)

    p_nb.set_array (np.array (data_nb))
    p_svm.set_array (np.array (data_svm))

    p_nb.set_clim (vmin, vmax)
    p_svm.set_clim (vmin, vmax)

    fig, axs = plt.subplots (2, sharex = True, sharey = True)
    # fig.set_size_inches (2048/100, 2048/100)

    p = [p_nb, p_svm]
    data = [data_nb, data_svm]
    tmp = file_name.split ('/')[-1].replace ('_', ' ')
    titles = [f'{tmp} \n NB', 'SVM']
    for i in range (2):
        axs [i].set_aspect ("auto")
        # tmp = [0, 5, 10, 15] + list (range (20, 105, 5))
        axs [i].set_xticks (grid_x)
        axs [i].set_yticks (grid_y)
        axs [i].grid ()
        axs [i].set_xlim ([grid_x[0], grid_x[-1]])
        axs [i].set_ylim ([grid_y[0], grid_y[-1]])
        # axs [1].set_ylabel ('number of traces averaged')


        axs [i].set_title (titles [i])
        ret = axs [i].add_collection (p [i])

        count = 0
        for r in ptchs:
            c = round (data [i] [count])
            rx, ry = r.get_xy ()
            cx = rx + r.get_width ()/2.0
            cy = ry + r.get_height ()/2.0

            # if (rx < 15):
            #     axs [i].annotate(c, (cx, cy), color='w', # weight='bold',
            #                      fontsize=2.5, ha='center', va='center', rotation =90)
            # else:
            axs [i].annotate(c, (cx, cy), color='w', weight='bold',
                             fontsize=6, ha='center', va='center')
            count += 1


    # axs [1].set_xlabel ('number of bandwidth')
    fig.supylabel('number of traces averaged')
    fig.supxlabel ('number of bandwidth')

    fig.colorbar (ret, ax = axs, orientation = 'vertical', pad  = 0.05, shrink = 1)
    plt.savefig (f'{file_name}.pdf',
                bbox_inches='tight')

    plt.close ('all')


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

        self.date          = None
        self.hour          = None
        self.path_lists    = None
        self.log_file      = None
        self.model_lda     = None
        self.model_svm     = None
        self.model_nb      = None
        self.means         = None
        self.nb_of_bd      = None
        self.path_acc      = None
        self.time_limit    = None
        self.metric        = None
        self.Q             = None
        self.algo          = None
        self.n_componenets = None
        self.labels        = []
        self.algo_args     = None

        self.DR_duration = None ## DR : Dimension Reduction
        self.NB_duration  = None
        self.SVM_duration = None

        self.means_vect  = None
        self.nb_of_means = None

        # ## to store the results
        # for i in ['DR', 'SVM', 'NB']:
        #     setattr (self, f'{i}_duration', None)

        for i in ['SVM', 'NB']:
            setattr (self, f'{i}_acc', [])

            for j in ['macro', 'weighted']:
                setattr (self, f'{i}_{j}_precision', [])
                setattr (self, f'{i}_{j}_recall', [])
                setattr (self, f'{i}_{j}_f1', [])

    def init_vectors (self):
        ## [2, -2] to remove blank '[' and ']'
        self.means_vect = [int (i) for i in self.means[2:-2].split (',')]
        self.nb_of_means = len (self.means_vect)

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
    reading_means = False

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
                if (tmp_split [0] == 'means'):
                    if ('None' in tmp_split [1] or '[]' in tmp_split [1]):
                        setattr (current_res, 'means',  None)
                        setattr (current_res, 'nb_of_means',  0)
                    else:
                        setattr (current_res, 'means',  tmp_split [1])
                        current_res.init_vectors ()

                elif (tmp_split [0] == 'path_lists' and 'limit=' in  tmp_split [1]):
                    # get the number of traces in the learning set
                    Q = int (tmp_split [1].split ('_limit=')[-1].split ('.')[0])
                    setattr (current_res, 'Q', Q)
                    setattr (current_res, 'path_lists',  tmp_split [1].split ('_limit=')[0])
                elif (tmp_split [0] == 'nb_of_bd'):
                    current_res.nb_of_bd = int (tmp_split [1])
                else :
                     setattr (current_res, tmp_split [0],  tmp_split [1])

        ## reading the data
        elif (reading_data):
            tmp_split = line.split ()

            ## save the times
            # if (len (tmp_split) > 1
            # and tmp_split [0] == 'LDA'
            # and tmp_split [1] == '(compuation):'): # = {DR, NB, SVM} (computation)
            #     setattr (current_res, f'{tmp_split [0]}_duration', float (tmp_split [2]))

            ## Test {NB, SVM}
            # elif
            if (len (tmp_split) > 1 and (tmp_split [0] == 'Test')):
                  current_exp = tmp_split [1]
                  reading_means = False

            # {NB, SVM}{-}{mean}{val:}
            elif (len (tmp_split) > 1 and (tmp_split [0] == 'NB' or tmp_split [0] == 'SVM')):
                  current_exp = tmp_split [0]
                  reading_means = True

            # {NB, SVM}_acc
            elif (len (tmp_split) > 1 and tmp_split [0] == 'accuracy'):
                getattr (current_res, f'{current_exp}_acc').append (float (tmp_split [1]))

            # {NB, SVM}_{macro, weighted}_{precision, recall, f1}
            elif (len (tmp_split) > 1 and (tmp_split [0] == 'macro' or tmp_split [0] == 'weighted')):
                getattr (current_res, f'{current_exp}_{tmp_split [0]}_precision').append (float (tmp_split [2]))
                getattr (current_res, f'{current_exp}_{tmp_split [0]}_recall').append (float (tmp_split [3]))
                getattr (current_res, f'{current_exp}_{tmp_split [0]}_f1').append (float (tmp_split [4]))


            # individual precision and recall f1-score
            elif (len (tmp_split) == 5 and tmp_split [0] != 'compute'):
                if (not reading_means):
                    getattr (current_res, 'labels').append (tmp_split [0])
                    setattr (current_res, f'{current_exp}_{tmp_split [0]}_precision', [])
                    setattr (current_res, f'{current_exp}_{tmp_split [0]}_recall', [])
                    setattr (current_res, f'{current_exp}_{tmp_split [0]}_f1', [])
                    setattr (current_res, f'{current_exp}_{tmp_split [0]}_support', [])

                getattr (current_res, f'{current_exp}_{tmp_split [0]}_precision').append (float (tmp_split [1]))
                getattr (current_res, f'{current_exp}_{tmp_split [0]}_recall').append (float (tmp_split [2]))
                getattr (current_res, f'{current_exp}_{tmp_split [0]}_f1').append (float (tmp_split [3]))
                getattr (current_res, f'{current_exp}_{tmp_split [0]}_support').append (float (tmp_split [4]))

    f.close ()

    ## add the last one
    res.append (current_res)

    ## clean in case means as 1 that will create twice the same results,*
    ## following is a hack in order to always have
    ## means = [1, ...] or None of no means
    for i in range (len (res)):

        ## when meaning has been done with 1: useless so removed

        if (res [i].means is not None):
            if (1 in res [i].means_vect):
                idx = np.where (means_vect == 1)[0][0]

                del res [i].SVM_acc [idx]
                del res [i].NB_acc [idx]

                ## SVM macro
                del res [i].SVM_macro_precision [idx]
                del res [i].SVM_macro_recall [idx]
                del res [i].SVM_macro_f1 [idx]

                ## SVM weighted
                del res [i].SVM_weighted_precision [idx]
                del res [i].SVM_weighted_recall [idx]
                del res [i].SVM_weighted_f1 [idx]

                ## NB macro
                del res [i].NB_macro_precision [idx]
                del res [i].NB_macro_recall [idx]
                del res [i].NB_macro_f1 [idx]

                ## NB weighted
                del res [i].NB_weighted_precision [idx]
                del res [i].NB_weighted_recall [idx]
                del res [i].NB_weighted_f1 [idx]

                ## otherwise
            else:
                res [i].means_vect.insert (0, 1)
                res [i].nb_of_means += 1

    return res

################################################################################
def display_results (exps, output, bin_malware):
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
    bds_idx = {}
    for i in range (len (bds)):
        bds_idx [bds [i]] = i


    count = 0
    if (exps [0].means is not None): ## meaning
        nb_max_of_means = max ([len (i.means_vect) for i in exps])
        means = []
        for i in exps:
            means += i.means_vect

        means = np.sort (np.unique (means))

        means_NB_BA = np.zeros ((nb_max_of_means, len (unique_tags)))
        means_NB_TPR = np.zeros ((nb_max_of_means, len (unique_tags)))
        means_NB_TNR = np.zeros ((nb_max_of_means, len (unique_tags)))

        means_SVM_BA = np.zeros ((nb_max_of_means, len (unique_tags)))
        means_SVM_TPR = np.zeros ((nb_max_of_means, len (unique_tags)))
        means_SVM_TNR = np.zeros ((nb_max_of_means, len (unique_tags)))

        res_matrix_NB  = [np.zeros ((nb_max_of_means, len (bds))) for _ in range (len (unique_tags))]
        res_matrix_SVM = [np.zeros ((nb_max_of_means, len (bds))) for _ in range (len (unique_tags))]


    else: # No means
        means = []

    tags = np.unique (np.array ([i.path_lists.split ('/')[-1] for i in exps]))

    tabular = []

    ## Construct the tabular
    for i in tqdm (range (len (unique_tags))):
        current_label = ''
        oposit_label  = ''

        tmp_name = unique_tags [i]
        tmp_name = tmp_name.replace ('lists_', '')
        tmp_name = tmp_name.replace ('.npy', '')

        current_row = [f'{count}', f'{tmp_name}']
        count += 1
        current_nb = [0, 0]
        current_svm = [0, 0]

        for j in range (len (exps)):
            ## in case we have more than 2 classes (not a detection procedure: clean Vs mal)
            if (unique_tags [i] == exps [j].path_lists and not bin_malware):
                ## for the display of the matrix
                res_matrix_SVM [i][0, bds_idx [exps [j].nb_of_bd]] = exps [j].SVM_acc [0]
                res_matrix_NB [i][0, bds_idx [exps [j].nb_of_bd]] = exps [j].NB_acc [0]
                for k in range (1, len (exps [j].means_vect))   :
                    res_matrix_SVM [i][k, bds_idx [exps [j].nb_of_bd]] = exps [j].SVM_acc [k]
                    res_matrix_NB [i][k, bds_idx [exps [j].nb_of_bd]] = exps [j].NB_acc [k]

                if (len (exps [j].NB_acc)> 0
                    and ((exps [j].NB_acc [0] > current_nb [0]) or (exps [j].NB_acc [0] == current_nb [0]
                                                                    and exps [current_nb [1]].nb_of_bd < exps [j].nb_of_bd))):
                    current_nb [0] = exps [j].NB_acc [0]
                    current_nb [1] = j

                    if (len (means) != 0):
                        means_NB_BA   [0, i] = current_nb [0]

                        for k in range (1, len (exps [j].means_vect)):
                            means_NB_BA  [k, i] = max (means_NB_BA [k, i], exps [j].NB_acc [k])


                if (len (exps [j].SVM_acc) > 0
                    and ((exps [j].SVM_acc [0] > current_svm [0]) or (exps [j].NB_acc [0] == current_svm [0]
                                                                      and exps [current_svm [1]].nb_of_bd < exps [j].nb_of_bd))):
                    current_svm [0] = exps [j].SVM_acc [0]
                    current_svm [1] = j

                    if (len (means) != 0):
                        means_SVM_BA  [0, i] = current_svm [0]

                        for k in range (1, len (exps [j].means_vect)):
                            means_SVM_BA [k, i] = max (means_SVM_BA [k, i], exps [j].SVM_acc [k])




            elif (unique_tags [i] == exps [j].path_lists):
                if (current_label == ''):
                    for k in getattr (exps [j], 'labels'):
                        if ('mal' in k or 'inf' in k or 'rootkit' in k):
                            current_label = k
                        else:
                            oposit_label = k

                tmp_precision_0 = getattr (exps [j], f'NB_{current_label}_recall')
                tmp_precision_1 = getattr (exps [j], f'NB_{oposit_label}_recall')
                metric = (tmp_precision_0 [0] + tmp_precision_1 [0])/2

                ## for the display of the matrix
                res_matrix_NB [i][0, bds_idx [exps [j].nb_of_bd]] = metric
                for k in range (1, len (exps [j].means_vect))   :
                    res_matrix_NB [i][k, bds_idx [exps [j].nb_of_bd]] = (tmp_precision_0 [k] + tmp_precision_1 [k])/2


                if (len (tmp_precision_0)> 0
                    and  (metric > current_nb [0]
                          or (metric == current_nb [0]
                              and exps [current_nb [1]].nb_of_bd > exps [j].nb_of_bd))):
                    current_nb [0] = metric
                    current_nb [1] = j

                    if (len (means) != 0):
                        means_NB_BA   [0, i] = current_nb [0]

                        means_NB_TPR  [0, i] = tmp_precision_0 [0]
                        means_NB_TNR  [0, i] = tmp_precision_1 [0]

                        for k in range (1, len (exps [j].means_vect)):
                            means_NB_BA  [k, i] = 0.5*(tmp_precision_0 [k] +  tmp_precision_1 [k])
                            means_NB_TPR  [k, i] = tmp_precision_0 [k]
                            means_NB_TNR  [k, i] = tmp_precision_1 [k]


                tmp_precision_0 = getattr (exps [j], f'SVM_{current_label}_recall')
                tmp_precision_1 = getattr (exps [j], f'SVM_{oposit_label}_recall')
                metric = (tmp_precision_0 [0] + tmp_precision_1 [0])/2

                ## for the display of the matrix
                res_matrix_SVM [i][0, bds_idx [exps [j].nb_of_bd]] = metric
                for k in range (1, len (exps [j].means_vect))   :
                    res_matrix_SVM [i][k, bds_idx [exps [j].nb_of_bd]] = (tmp_precision_0 [k] + tmp_precision_1 [k])/2


                if (len (tmp_precision_0) > 0
                    and (metric > current_svm [0]
                          or (metric == current_nb [0]
                              and exps [current_nb [1]].nb_of_bd > exps [j].nb_of_bd))):

                    current_svm [0] = metric # exps [j].SVM_acc [0]
                    current_svm [1] = j

                    if (len (means) != 0):
                        means_SVM_BA  [0, i] = current_svm [0]
                        means_SVM_TPR  [0, i] = tmp_precision_0 [0]
                        means_SVM_TNR  [0, i] = tmp_precision_1 [0]

                        for k in range (1, len (exps [j].means_vect)):
                            means_SVM_BA  [k, i] = 0.5*(tmp_precision_0 [k] +  tmp_precision_1 [k])
                            means_SVM_TPR  [k, i] = tmp_precision_0 [k]
                            means_SVM_TNR  [k, i] = tmp_precision_1 [k]

        ## display matrix
        if (output is not None):
            display_matrix_exp (np.array ([0] + list (bds)),
                                np.array ([0] + list (means)),
                                (100*res_matrix_NB [i]).flatten (),
                                (100*res_matrix_SVM [i]).flatten (),
                                output + unique_tags [i].split ('/')[-1].split ('.')[0])

        ## get the algo for projection
        try:
            algos = '{' + ','.join (np.unique ([i.algo.strip () for i in exps])) + '}'
        except:
            algos = 'DR'

        if (not bin_malware):
            current_row += [# NB
                f'{get_color (current_nb [0])}', f'{exps [current_nb [1]].nb_of_bd}',
                f'{get_color (exps [current_nb [1]].NB_macro_precision [0])}/{get_color (exps [current_nb [1]].NB_weighted_precision [0])}',
                f'{get_color (exps [current_nb [1]].NB_macro_recall [0])}/{get_color (exps [current_nb [1]].NB_weighted_recall [0])}',
                f'{get_color (exps [current_nb [1]].NB_macro_f1 [0])}/{get_color (exps [current_nb [1]].NB_weighted_f1 [0])}',
                # SVM
                f'{get_color (current_svm [0])}', f'{exps [current_svm [1]].nb_of_bd}',
                f'{get_color (exps [current_svm [1]].SVM_macro_precision [0])}/{get_color (exps [current_svm [1]].SVM_weighted_precision [0])}',
                f'{get_color (exps [current_svm [1]].SVM_macro_recall [0])}/{get_color (exps [current_svm [1]].SVM_weighted_recall [0])}',
                f'{get_color (exps [current_svm [1]].SVM_macro_f1 [0])}/{get_color (exps [current_svm [1]].SVM_weighted_f1 [0])}']


            head = ['idx', 'exp', f'{algos} + NB', '#bd',
                    'precision (macro/weigh)', 'recall (macro/weigh)', 'f1 (macro/weigh)',
                    f'{algos} + SVM', '#bd',
                    'precision (macro/weigh)', 'recall (macro/weigh)', 'f1 (macro/weigh)']
        else:
            tmp_NB = [getattr (exps [current_nb [1]], f'NB_{current_label}_recall')[0],
                      getattr (exps [current_nb [1]], f'NB_{oposit_label}_recall')[0]]

            tmp_SVM = [getattr (exps [current_svm [1]], f'SVM_{current_label}_recall')[0],
                       getattr (exps [current_svm [1]], f'SVM_{oposit_label}_recall')[0]]

            current_row += [
                # NB
                f'{get_color (exps [current_nb [1]].NB_acc [0])}',
                f'{exps [current_nb [1]].nb_of_bd}',
                f'{get_color (tmp_NB [0])}', f'{get_color (tmp_NB [1])}',
                ' ',
                # SVM
                f'{get_color (exps [current_svm [1]].SVM_acc [0])}',
                f'{exps [current_svm [1]].nb_of_bd}',
                f'{get_color (tmp_SVM [0])}', f'{get_color (tmp_SVM [1])}']

            head = ['idx', 'exp', f'{algos} + NB acc. ', '#bd', 'TPR', 'TNR', '', f'{algos} + SVM acc.', '#bd', 'TPR', 'TNR']

        tabular.append (current_row)

    ## to avoid problem with grep (if grep is used, the last row is ignored)
    tabular.append (['-']*len (tabular [0]))


    res_tabulate = tabulate.tabulate (tabular, headers= head, tablefmt= 'grid')# "latex_longtable")
    print(res_tabulate)


################################################################################
if __name__ == '__main__':
################################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument ('--path',  nargs = '+', default = [],
                        dest='path',
                        help='Absolute path to the log files'
                         +' (could be used multiple times to give more than one list)')

    parser.add_argument ('--plot', action = 'store', default = None,
                         type = str, dest = 'path_to_plot',
                         help = 'Absolute path to the directory where to save the plot')

    parser.add_argument ('--bin_malware', action = 'store_true', default = False,
                         dest = 'bin_malware',
                         help = 'If the results must be display for malware in a binary context (TPR/TNR)')

    args, unknown = parser.parse_known_args ()
    assert len (unknown) == 0, f"[WARNING] Unknown arguments:\n{unknown}\n"

    ## read and parse the log file
    res = []
    for i in range (len (args.path)):
        display_results (read_log (args.path [i]), args.path_to_plot, args.bin_malware)
