__='
   This is the default license template.
   
   File: run_ml_on_selected_bandwidth.sh
   Author: test
   Copyright (c) 2021 test
   
   To edit this license information: Press Ctrl+Shift+P and press 'Create new License Template...'.
'

#!/bin/bash
path_lists=$1
path_models=$2
path_acc=$3

## the following script recompute the classification of the testing dataset
# using the pretrained models, available in the repo
# "pretrained_models". The directorie containing the models and the
# one containing projected (LDA) data must be given as parameters of the
# current script


# list of the scenario
declare -a tagmaps=('executable_classification'
                    'family_classification'
                    'novelty_classification'
                    'obfuscation_classification'
                    'packer_identification'
                    'type_classification'
                    'virtualization_identification')

declare -a NB_bds=('28' '28' '28' '10' '16' '22' '6')
declare -a SVM_bds=('20' '28' '16' '10' '16' '24' '6')

# number of tagmaps
nb_of_tagmaps=${#tagmaps[@]}

for  (( i=0; i<${nb_of_tagmaps}; i++ ));
do
    echo "Computing LDA + NB, tagmap: ${tagmaps[$i]}"
    python3 ml_analysis/NB.py --lists ${path_lists}/files_lists_tagmaps=${tagmaps[$i]}.npy\
	    --log-file ml_analysis/log-evaluation_selected_bandwidth.txt\
	    --acc ${path_acc}\
            --model_nb  ${path_models}/NB/NB_tagmaps=${tagmaps[$i]}_${NB_bds[$i]}bd.jl\
            --model_lda ${path_models}/LDA/LDA_tagmaps=${tagmaps[$i]}_${NB_bds[$i]}bd.jl\
            #--model_lda  ${path_models}/LDA/transformed_traces/transformed_traces_tagmaps=${tagmaps[$i]}_${NB_bds[$i]}bd.npy
    
    # echo "Computing LDA + SVM, tagmap: ${tagmaps[$i]}"
    # python3 ml_analysis/SVM.py --lists ${path_lists}/files_lists_tagmaps=${tagmaps[$i]}.npy\
    #         --log-file ml_analysis/log-evaluation_selected_bandwidth.txt\
    #         --acc ${path_acc}\
    #         --model_svm  ${path_models}/SVM/SVM_tagmaps=${tagmaps[$i]}_${SVM_bds[$i]}bd.jl\
    #         --model_lda ${path_models}/LDA/LDA_tagmaps=${tagmaps[$i]}_${SVM_bds[$i]}bd.jl\
    #         #--model_lda  ${path_models}/LDA/transformed_traces/transformed_traces_tagmaps=${tagmaps[$i]}_${SVM_bds[$i]}bd.npy
done

################################################################################
## display the results
# a figure (pop'up) and a tabular (in the terminal)
python3 ml_analysis/read_logs.py --path ml_analysis/log-evaluation_selected_bandwidth.txt
