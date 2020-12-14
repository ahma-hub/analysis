#!/bin/bash
path_lists=$1
path_models=$2
path_transform_traces=$3

# list of the scenarii
declare -a StringArray=('binary_classification'
                        'novelty_classification'
                        'packer_detection'
                        'virtualization_detection'
                        'family_classification'
                        'obfuscation_classification'
                        'type_classification')


for val in ${StringArray[@]}; do
    echo ${val}

     python3 ml_analysis/evaluate.py --lists ${path_lists}/files_lists_tagmap=${val}.npy\
            --model_lda ${path_models}/LDA/LDA_model=${val}.jl\
            --model_nb  ${path_models}/NB/NB_model=${val}.jl\
            --model_svm ${path_models}/SVM/SVM_model=${val}.jl\
            --transformed_traces ${path_transform_traces}/transformed_traces_model=${val}.npy
    
done
