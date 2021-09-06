# Analysis Tools

The current repository contains all the scripts needed to reproduce
the results published in the paper: "*Obfuscation Revealed:
Electromagnetic obfuscated malware classification*".

## Overview
```
.
├── README.md
├── requirements.txt
│── run_dl_on_selected_bandwidth.sh #> script to run the DL for all scenarii on  
|                                   # the testing pre-computed dataset
│── run_ml_on_reduced_dataset.sh    #> script to run the end-to-end analysis on 
|                                   # on a reduced dataset (350 per samples per 
|                                   # executable)
│── run_ml_on_selected_bandwidth.sh #> script to run the ML classification for all
|                                   # for all scenarii on the testing pre-computed 
|                                   # dataset 
│── update_lists.sh                 #> script to update the location of the traces 
│                                   # in the lists 
│
├── ml_analysis
│   │── evaluate.py                            #> code for the LDA + {NB, SVM} on the 
|   |                                          # reduced dataset (raw_data_reduced_dataset)
|   |── NB.py                                  #> Naïve Bayensian with known model 
|   |                                          # (traces_selected_bandwidth)
|   |── SVM.py                                 #> Support vector machine with known model
|   |                                          # (traces_selected_bandwidth)
│   │── log-evaluation_reduced_dataset.txt     #> output log file for the ML evaluation
|   |                                          # on the reduce datasete
│   │── log-evaluation_selected_bandwidth.txt  #> output log file for the ML evaluation
|                                              # using the precomputed models
│ 
│
│
├── dl_analysis
│   │── evaluate.py            #> code to run MLP and CNN on all scenarios
│   │── evaluation_log_DL.txt  #> output log file with stored accuracies
|
│
│
├── list_selected_bandwidth    #> list of the files used for training, 
│   │                          # validating and testing (all in one file)
│   │                          # for each sceanario (but only the testing 
|   |                          # data are available). Lists associated to
|   |                          # the selected bandwidth dataset
│   │── files_lists_tagmap=executable_classification.npy                              
│   │── files_lists_tagmap=novelty_classification.npy   
│   │── files_lists_tagmap=packer_identification.npy
│   │── files_lists_tagmap=virtualization_identification.npy
│   │── files_lists_tagmap=family_classification.npy 
│   │── files_lists_tagmap=obfuscation_classification.npy
│   │── files_lists_tagmap=type_classification.npy   
│
│
├── list_reduced_dataset    #> list of the files used for training, 
│   │                       # validating and testing (all in one file)
│   │                       # for each sceanario. Lists associated to
|   |                       # the reduced dataset
│   │── files_lists_tagmap=executable_classification.npy                              
│   │── files_lists_tagmap=novelty_classification.npy   
│   │── files_lists_tagmap=packer_identification.npy
│   │── files_lists_tagmap=virtualization_identification.npy
│   │── files_lists_tagmap=family_classification.npy 
│   │── files_lists_tagmap=obfuscation_classification.npy
│   │── files_lists_tagmap=type_classification.npy   
│
├── pre-processings            #> codes use to preprocess the raw traces to be 
    │                          # able to run the evaluations
    │── list_manipulation.py   #> split traces in {learning, testing, validating} 
    │                          # sets 
    │── accumulator.py         #> compute the sum and the square of the sum (to 
    │                          # be able to recompute quickly the NICVS)  
    │── nicv.py                #> to compute the NICVs
    │── corr.py                #> to compute Pearson coeff (alternative to the 
    |                          # NICV)
    │── displayer.py           #> use to display NICVs, correlations, traces...
    │── signal_processing.py   #> some signal processings (stft, ...)
    |── bandwidth_extractor.py #> extract bandwidth, based on NICVs results
    |                          # and creat new dataset
    │── tagmaps                #> all tagmaps use for to labelize the data 
        │                      # (use to creat the lists)
    	│── executable_classification.csv  
    	│── family_classification.csv
    	│── novelties_classification.csv  
    	│── obfuscation_classification.csv  
    	│── packer_identification.csv  
    	│── type_classification.csv  
    	│── virtualization_identification.csv
```

## Getting Started
### python 3.6
To be able to run the analysis you need to install python 3.6 and the required 
packages:

```
pip install -r requirements.txt
```

### Data
The testing spectrograms used in the paper can be dowload on the following website:

```
https://zenodo.org/record/5414107
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5414107.svg)](https://doi.org/10.5281/zenodo.5414107)
### File lists
In order to update the location of the data, you previously dowloaded, inside 
the lists you need to run the script ``update_lists.sh``:

```
./update_lists  [directory where are stored the list] [directory where are stored the traces]
```

This must be applyed to directoies ```list_selected_bandwidth``` and ```list_reduced_dataset```
respectively associated to the datasets: ```traces_selected_bandwidth.zip``` and ```raw_data_reduced_dataset.zip```

## Machine Learning (ML)
To run the computation of the all the machine learning experiments, you can use
the scripts ``run_ml_on_reduced_dataset.sh`` and ``run_ml_on_extracted_bandwidth.sh``:

```
./run_ml_on_extracted_bandwidth.sh  [directory where the lists are stored] [directory where the models are stored] [directory where the accumulated data is stored (precomputed in pretrained_models/ACC) ]
```

The results are stored in the file ```ml_analysis/log-evaluation_selected_bandwidth.txt```.

```
./run_ml_on_reduced_dataset.sh  
```

The results are stored in the file ```ml_analysis/log-evaluation_reduced_dataset.txt```.

The directory ``ml_analysis`` contains the code needed for the classification by Machine Learning (ML).

### evaluate.py

``` 
usage: evaluate.py [-h] 
                   [--lists PATH_LISTS]  
                   [--mean_size MEAN_SIZES]  
                   [--log-file LOG_FILE]  
                   [--acc PATH_ACC]  
                   [--nb_of_bandwidth NB_OF_BANDWIDTH]  
                   [--time_limit TIME_LIMIT]  
                   [--metric METRIC]

optional arguments:
  -h, --help                         show this help message and exit
  --lists PATH_LISTS                 Absolute path to a file containing the lists
  --mean_size MEAN_SIZES             Size of each means
  --log-file LOG_FILE                Absolute path to the file to save results
  --acc PATH_ACC                     Absolute path of the accumulators directory
  --nb_of_bandwidth NB_OF_BANDWIDTH  number of bandwidth to extract
  --time_limit TIME_LIMIT            percentage of time to concerve (from the begining)
  --metric METRIC                    Metric to use for select bandwidth: {nicv, corr}_{mean, max}
```

### NB.py

``` 
usage: NB.py [-h] 
             [--lists PATH_LISTS] 
             [--model_lda MODEL_LDA]  
             [--model_nb MODEL_NB]  
             [--mean_size MEAN_SIZES]  
             [--log-file LOG_FILE]  
             [--time_limit TIME_LIMIT]  
             [--acc PATH_ACC]

optional arguments:
  -h, --help               show this help message and exit
  --lists PATH_LISTS       Absolute path to a file containing the lists
  --model_lda MODEL_LDA    Absolute path to the file where the LDA model has been previously saved
  --model_nb MODEL_NB      Absolute path to the file where the NB model has been previously saved
  --mean_size MEAN_SIZES   Size of each means
  --log-file LOG_FILE      Absolute path to the file to save results
  --time_limit TIME_LIMIT  percentage of time to concerve (from the begining)
  --acc PATH_ACC           Absolute path of the accumulators directory

```

### read_logs.py
```
usage: read_logs.py [-h]
				    [--path PATH]
                    [--plot PATH_TO_PLOT]

optional arguments:
  -h, --help           show this help message and exit
  --path PATH          Absolute path to the log file
  --plot PATH_TO_PLOT  Absolute path to save the plot
``` 
### SVM.py

``` 
usage: SVM.py [-h] 
              [--lists PATH_LISTS] 
              [--model_lda MODEL_LDA] 
              [--model_svm MODEL_SVM] 
              [--mean_size MEAN_SIZES] 
              [--log-file LOG_FILE] 
              [--time_limit TIME_LIMIT] 
              [--acc PATH_ACC]

optional arguments:
  -h, --help               show this help message and exit
  --lists PATH_LISTS       Absolute path to a file containing the lists
  --model_lda MODEL_LDA    Absolute path to the file where the LDA model has been previously saved
  --model_svm MODEL_SVM    Absolute path to the file where the SVM model has been previously saved
  --mean_size MEAN_SIZES   Size of each means
  --log-file LOG_FILE      Absolute path to the file to save results
  --time_limit TIME_LIMIT  percentage of time to concerve (from the begining)
  --acc PATH_ACC           Absolute path of the accumulators directory
```

## Deep Learning (DL) 
To run the computation of the all the deep learning experiments, you can use
the script ``run_dl_on_selected_bandwidth.sh``:


```
./run_dl_on_selected_bandwidth.sh  [directory where the lists are stored] [directory where the models are stored] [directory where the accumulated data is stored (precomputed in pretrained_models/ACC) ]
```

## Preprocessings
Once the traces have been aquiered and before beeing able to run the evualuation 
some preprocessings are needed. The needed pre-processings are already written in 
the scripts listed above.

### accumulator.py
```
usage: accumulator.py [-h] 
					  [--lists PATH_LISTS] 
					  [--output OUTPUT_PATH] 
                      [--no_stft] 
                      [--freq FREQ] 
                      [--window WINDOW] 
                      [--overlap OVERLAP] 
                      [--core CORE] 
                      [--duration DURATION] 
                      [--device DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  --lists PATH_LISTS    Absolute path of the lists (cf. list_manipulation.py -- using a main list will help) trace directory
  --output OUTPUT_PATH  Absolute path of the output directory
  --no_stft             If no stft need to be applyed on the listed data
  --freq FREQ           Frequency of the acquisition in Hz
  --window WINDOW       Window size for STFT
  --overlap OVERLAP     Overlap size for STFT
  --core CORE           Number of core to use for multithreading accumulation
  --duration DURATION   to fixe the duration of the input traces (padded if input is short and cut otherwise)
  --device DEVICE       to fixe the duration of the input traces (padded if input is short and cut otherwise)

```
### bandwidth_extractor.py
``` 
usage: bandwidth_extractor.py [-h] 
							  [--acc PATH_ACC] 
							  [--lists LISTS [LISTS ...]]  
							  [--plot PATH_TO_PLOT]  
							  [--nb_of_bandwidth NB_OF_BANDWIDTH]  
							  [--log-level LOG_LEVEL]  
							  [--output_traces PATH_OUTPUT_TRACES]  
							  [--output_lists PATH_OUTPUT_LISTS]
                              [--freq FREQ]  
							  [--window WINDOW]  
							  [--overlap OVERLAP]  
							  [--device DEVICE]  
							  [--metric METRIC]  
							  [--core CORE]  
							  [--duration DURATION]

optional arguments:
  -h, --help                         show this help message and exit
  --acc PATH_ACC                     Absolute path of the accumulators directory
  --lists LISTS [LISTS ...]          Absolute path to all the lists (for each scenario). /!\ The data in the first one must contain all traces.
  --plot PATH_TO_PLOT                Absolute path to a file to save the plot
  --nb_of_bandwidth NB_OF_BANDWIDTH  number of bandwidth to extract (-1 means that all bandwidth will be concerved)
  --log-level LOG_LEVEL              Configure the logging level: DEBUG|INFO|WARNING|ERROR|FATAL
  --output_traces PATH_OUTPUT_TRACES Absolute path to the directory where the traces will be saved
  --output_lists PATH_OUTPUT_LISTS   Absolute path to the files where the new lists will be saved
  --freq FREQ                        Frequency of the acquisition in Hz
  --window WINDOW                    Window size for STFT
  --overlap OVERLAP                  Overlap size for STFT
  --device DEVICE                    Used device under test
  --metric METRIC                    Metric to use for the PoI selection: {nicv, corr}_{mean, max}
  --core CORE                        Number of core to use for multithreading
  --duration DURATION                to fixe the duration of the input traces (padded if input is short and cut otherwise)
```

### corr.py

``` 
usage: corr.py [-h] 
			   [--acc PATH_ACC]  
			   [--lists PATH_LISTS]  
			   [--plot PATH_TO_PLOT]  
			   [--scale SCALE]  
			   [--bandwidth_nb BANDWIDTH_NB]  
			   [--metric METRIC]  
			   [--log-level LOG_LEVEL]

optional arguments:
  -h, --help                  show this help message and exit
  --acc PATH_ACC              Absolute path of the accumulators directory
  --lists PATH_LISTS          Absolute path to a file containing the main lists
  --plot PATH_TO_PLOT         Absolute path to the file where to save the plot (/!\ '.png' expected at the end of the filename)
  --scale SCALE               scale of the plotting: normal|log
  --bandwidth_nb BANDWIDTH_NB display the nb of selected bandwidth, by default no bandwidth selected
  --metric METRIC             Metric used to select bandwidth: {corr}_{mean, max}
  --log-level LOG_LEVEL       Configure the logging level: DEBUG|INFO|WARNING|ERROR|FATAL

``` 

### displayer.py
``` 
usage: displayer.py [-h]
                    [--display_trace PATH_TRACE] 
                    [--display_lists PATH_LISTS] 
                    [--list_idx LIST_IDX] 
                    [--metric METRIC] 
                    [--extension EXTENSION] 
                    [--path_save PATH_SAVE]

optional arguments:
  -h, --help                  show this help message and exit
  --display_trace PATH_TRACE  Absolute path to the trace to display
  --display_lists PATH_LISTS  Absolute path to the list to display
  --list_idx LIST_IDX         which list to display (all = -1, learning: 0, validating: 1, testing: 2)
  --metric METRIC             Applied metric for the display of set (mean, std, means, stds)
  --extension EXTENSION       extensio of the raw traces
  --path_save PATH_SAVE       Absolute path to save the figure (if None, display in pop'up)
``` 

### list_manipulation.py
```
usage: list_manipulation.py [-h] 
                            [--raw PATH_RAW] 
                            [--tagmap PATH_TAGMAP] 
                            [--save PATH_SAVE] 
                            [--main-lists PATH_MAIN_LISTS] 
                            [--extension EXTENSION] 
                            [--log-level LOG_LEVEL] 
                            [--lists PATH_LISTS] 
                            [--new_dir PATH_NEW_DIR]
                            [--nb_of_traces_per_label NB_OF_TRACES_PER_LABEL]

optional arguments:
  -h, --help                                       show this help message and exit
  --raw PATH_RAW                                   Absolute path to the raw data directory
  --tagmap PATH_TAGMAP                             Absolute path to a file containing the tag map
  --save PATH_SAVE                                 Absolute path to a file to save the lists
  --main-lists PATH_MAIN_LISTS                     Absolute path to a file containing the main lists
  --extension EXTENSION                            extensio of the raw traces
  --log-level LOG_LEVEL                            Configure the logging level: DEBUG|INFO|WARNING|ERROR|FATAL
  --lists PATH_LISTS                               Absolute path to a file containing lists
  --new_dir PATH_NEW_DIR                           Absolute path to the raw data, to change in a given file lists
  --nb_of_traces_per_label NB_OF_TRACES_PER_LABEL  number of traces to keep per label

```



### nicv.py
```
usage: nicv.py [-h] 
			   [--acc PATH_ACC]  
			   [--lists PATH_LISTS]  
			   [--plot PATH_TO_PLOT]  
			   [--scale SCALE]  
			   [--time_limit TIME_LIMIT]  
			   [--bandwidth_nb BANDWIDTH_NB]  
			   [--metric METRIC]  
			   [--log-level LOG_LEVEL]

optional arguments:
  -h, --help                   show this help message and exit
  --acc PATH_ACC               Absolute path of the accumulators directory
  --lists PATH_LISTS           Absolute path to a file containing the main lists
  --plot PATH_TO_PLOT          Absolute path to save the plot
  --scale SCALE                scale of the plotting: normal|log
  --time_limit TIME_LIMIT      percentage of time to concerve (from the begining)
  --bandwidth_nb BANDWIDTH_NB  display the nb of selected bandwidth, by default no bandwidth selected
  --metric METRIC              Metric used to select bandwidth: {nicv}_{mean, max}
  --log-level LOG_LEVEL        Configure the logging level: DEBUG|INFO|WARNING|ERROR|FATAL
```

### signal_processing.py
``` 
usage: signal_processing.py [-h] 
                            [--input INPUT] 
                            [--dev DEVICE] 
                            [--output OUTPUT] 
                            [--freq FREQ] [--window WINDOW] [--overlap OVERLAP]

optional arguments:
  -h, --help         show this help message and exit
  --input INPUT      Absolute path to a raw trace
  --dev DEVICE       Type of file as input (pico|hackrf|i)
  --output OUTPUT    Absolute path to file where to save the axis
  --freq FREQ        Frequency of the acquisition in Hz
  --window WINDOW    Window size for STFT
  --overlap OVERLAP  Overlap size for STFT
``` 























