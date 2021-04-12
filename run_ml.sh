#!/bin/bash
path_lists=$1
path_models=$2
path_acc=$3

# list of the scenario
declare -a tagmaps=('executable_classification'
                    'novelty_classification'
                    'packer_identification'
                    'virtualization_identification'
                    'family_classification'
                    'obfuscation_classification'
                    'type_classification')

declare -a NB_bds=('24' '28' '6' '16' '26' '24' '22')
declare -a SVM_bds=('20' '28' '6' '16' '26' '24' '24')

# number of tagmaps
nb_of_tagmaps=${#tagmaps[@]}

for  (( i=0; i<${nb_of_tagmaps}; i++ ));
do
    echo "Computing LDA + NB, tagmap: ${tagmaps[$i]}"
    python3 ml_analysis/NB.py --lists ${path_lists}/files_lists_tagmap=${tagmaps[$i]}.npy\
	    --log-file ml_analysis/log-evaluation.txt\
	    --acc ${path_acc}\
	    --model_lda ${path_models}/LDA/LDA_model=${tagmaps[$i]}_${NB_bds[$i]}bd.jl\
            --model_nb  ${path_models}/NB/NB_model=${tagmaps[$i]}_${NB_bds[$i]}bd.jl

    echo "Computing LDA + SVM, tagmap: ${tagmaps[$i]}"
    python3 ml_analysis/SVM.py --lists ${path_lists}/files_lists_tagmap=${tagmaps[$i]}.npy\
	    --log-file ml_analysis/log-evaluation.txt\
	    --acc ${path_acc}\
	    --model_lda ${path_models}/LDA/LDA_model=${tagmaps[$i]}_${SVM_bds[$i]}bd.jl\
	    --model_svm  ${path_models}/SVM/SVM_model=${tagmaps[$i]}_${SVM_bds[$i]}bd.jl
  
done
