#!/bin/bash
path_lists=$1
path_models=$2
path_acc=$3

# list of the scenarii
declare -a tagmaps=('executable_classification'
                        'novelty_classification'
                        'packer_identification'
                        'virtualization_identification'
                        'family_classification'
                        'obfuscation_classification'
                        'type_classification')

declare -a MLP_bds=('24' '12' '28' '20' '28' '28' '28')
declare -a CNN_bds=('24' '16' '16' '24' '28' '28' '28')


# number of tagmaps
nb_of_tagmaps=${#tagmaps[@]}

for  (( i=0; i<${nb_of_tagmaps}; i++ ));
do
	echo "Computing MLP, tagmap: ${tagmaps[$i]}"
	python3 dl_analysis/evaluate.py --band ${MLP_bds[$i]} --list ${path_lists}/files_lists_tagmap=${tagmaps[$i]}.npy\
		--acc ${path_acc}\
	       	--model ${path_models}/MLP/${tagmaps[$i]}.h5
done



for  (( i=0; i<${nb_of_tagmaps}; i++ ));
do
    echo "Computing CNN, tagmap: ${tagmaps[$i]}"
    python3 dl_analysis/evaluate.py --band ${CNN_bds[$i]} --list ${path_lists}/files_lists_tagmap=${tagmaps[$i]}.npy\
	    --acc ${path_acc} --model ${path_models}/CNN/${tagmaps[$i]}.h5

done
