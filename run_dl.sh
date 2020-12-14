#!/bin/bash
path_lists=$1
path_models=$2

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

     python3 dl_analysis/evaluate.py --list ${path_lists}/files_lists_tagmap=${val}.npy\
            --model ${path_models}/MLP/${val}.h5
    
done




for val in ${StringArray[@]}; do
    echo ${val}

    python3 dl_analysis/evaluate.py --list ${path_lists}/files_lists_tagmap=${val}.npy\
			             --model ${path_models}/CNN/${val}.h5

done
