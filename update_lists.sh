#!/bin/bash
path_lists=$1
path_traces=$2

# list of the scenario
declare -a StringArray=('executable_classification'
                        'novelty_classification'
                        'packer_identification'
                        'virtualization_identification'
                        'family_classification'
                        'obfuscation_classification'
                        'type_classification')


for val in ${StringArray[@]}; do
    ## code to change the path of the 'traces_selected_bandwidth'
    ## in the list
    python3 pre-processings/list_manipulation.py\
       --lists ${path_lists}/files_lists_tagmaps=${val}.npy\
       --new_dir ${path_traces}
done
