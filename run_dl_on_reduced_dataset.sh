#!/bin/bash
path_lists=$1
path_acc=$2
arch=$3
epochs=$4
batch=$5


# list of the scenarii
declare -a tagmaps=('executable_classification'
                        'novelty_classification'
                        'packer_identification'
                        'virtualization_identification'
                        'family_classification'
                        'obfuscation_classification'
                        'type_classification')


# number of tagmaps
nb_of_tagmaps=${#tagmaps[@]}

for  (( i=0; i<${nb_of_tagmaps}; i++ ));
do
	for b in `seq 4 4 28`; do        
		echo "Learning MLP, tagmap: ${tagmaps[$i]}"
        	python3 dl_analysis/training.py --band $b --list ${path_lists}/extracted_bd_files_lists_tagmaps=${tagmaps[$i]}.npy\
                --acc ${path_acc}\
                --epochs ${epochs} --batch ${batch} --arch=mlp\
		--save MLP_${tagmaps[$i]}_band_$b.h5
	done
done

for  (( i=0; i<${nb_of_tagmaps}; i++ ));
do
        for b in `seq 4 4 28`; do
                echo "Learning CNN, tagmap: ${tagmaps[$i]}"
                python3 dl_analysis/training.py --band $b --list ${path_lists}/extracted_bd_files_lists_tagmaps=${tagmaps[$i]}.npy\
                --acc ${path_acc}\
                --epochs ${epochs} --batch ${batch} --arch=cnn\
                --save CNN_${tagmaps[$i]}_band_$b.h5
	done
done
