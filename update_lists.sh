#!/bin/bash
path_lists=$1
path_transform_traces=$2

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

    ## code to change the path of the 'traces_selected_bandwidth'
    ## in the list
    python3 pre-processings/list_manipulation.py\
       --lists ${path_lists}/files_lists_tagmap=${val}.npy\
       --new_dir ${path_transform_traces}
done
