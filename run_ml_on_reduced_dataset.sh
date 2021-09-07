#!/bin/bash

## the following script compute the end-to-end process on the reduced dataset:
# 1) compute the accumulators to be able to quickly compute NICVs
# 2) extract 40 bandwidth based on the NICVs
# 3) compute the accumulators of the extracted bandwidth
# 4) run machine learning the evaluation (for all scenarii and differents of bandwidth)
# 5) display the results (figure and tabular) 


# list of the scenarii
 declare -a tagmaps=('virtualization_identification'
		    'executable_classification'
		    'novelty_classification'                 
                    'packer_identification'
                    'family_classification'
                    'obfuscation_classification'
                    'type_classification')

################################################################################
## compute the accumulators to be able to compute the nicvs/corr
# creat the directory
mkdir -p acc_raw_reduced_dataset
## compute the accumulators (sum and sum of the square)
python pre-processings/accumulator.py\
       --lists lists_reduced_dataset/files_lists_tagmaps\=executable_classification.npy\
       --output acc_raw_reduced_dataset/\
       --window 8192 --overlap 4096 --core 15 

################################################################################
## extract the bandwidth selected by NICV computed using the accumulators
# creat the directory
mkdir -p traces_40bd_reduced_dataset
# run the computation
python pre-processings/bandwidth_extractor.py --acc acc_raw/\
       --lists lists_reduced_dataset/files_lists_tagmaps=executable_classification.npy\
               lists_reduced_dataset/files_lists_tagmaps=novelty_classification.npy\
               lists_reduced_dataset/files_lists_tagmaps=packer_identification.npy\
               lists_reduced_dataset/files_lists_tagmaps=virtualization_identification.npy\
               lists_reduced_dataset/files_lists_tagmaps=family_classification.npy\
               lists_reduced_dataset/files_lists_tagmaps=obfuscation_classification.npy\
               lists_reduced_dataset/files_lists_tagmaps=type_classification.npy\
               --nb_of_bandwidth 40 --output_traces traces_40bd_reduced_dataset\
               --output_lists lists_reduced_dataset/\
               --window 8192 --overlap 4096 --core 20

################################################################################
## compute the accumulation of the extracted bandwidth
# creat the directory
mkdir -p acc_stft_reduced_dataset
# run tu computation
python pre-processings/accumulator.py\
       --lists lists_reduced_dataset/extracted_bd_files_lists_tagmaps\=executable_classification.npy\
       --output acc_stft_reduced_dataset/ --no_stft --device npy --core 10 

################################################################################
## evaluate: LDA + {NB, SVM}
declare -a bds=('2' '4' '6' '8' '10' '12' '14' '16' '18' '20' '22' '24')
for val in ${tagmaps[@]}; do
    for bd in ${bds[@]}; do
	echo ${val} ${bd}

	python3 ml_analysis/evaluate.py\
                --lists lists_reduced_dataset/extracted_bd_files_lists_tagmaps\=${val}.npy\
		--acc acc_stft_reduced_dataset/ --nb_of_bandwidth ${bd}\
                --log-file ml_analysis/log-evaluation_reduced_dataset.txt --time_limit 1 --metric nicv_max
    done
done

################################################################################
## display the results
# a figure (pop'up) and a tabular (in the terminal)
python3 ml_analysis/read_logs.py --path ml_analysis/log-evaluation_reduced_dataset.txt
