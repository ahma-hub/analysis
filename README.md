# Analysis Tools

The current repository contains all the scripts needed to reproduce the results
published in the paper: "*Out of device malware analysis:leveraging IoT devices 
electromagnetic emanation*".

## Overview
```
.
├── README.md
├── requirements.txt
│── run_dl.sh                  #> script to run the DL for all scenarii
│── run_ml.sh                  #> script to run the ML for all scenarii
│── update_lists.sh            #> script to update the location of the traces 
│                              # in the lists 
│
├── ml_analysis
│   │── evaluation.py          #> code for the LDA, the NB and the SVM
│   │── log-evaluation.txt     #> store the results of the ML
│
│
├── dl_analys
│   │── evaluate.py            #> code to run MLP and CNN on all scenarios
│   │── evaluation_log_DL.txt  #> output log file with stored accuracies
│
│
├── list_selected_bandwidth    #> list of the files used for training, 
│   │                          # validating and testing (all in one file)
│   │                          # for each sceanario  
│   │── files_lists_tagmap=binary_classification.npy                              
│   │── files_lists_tagmap=novelty_classification.npy   
│   │── files_lists_tagmap=packer_detection.npy
│   │── files_lists_tagmap=virtualization_detection.npy
│   │── files_lists_tagmap=family_classification.npy 
│   │── files_lists_tagmap=obfuscation_classification.npy
│   │── files_lists_tagmap=type_classification.npy   
│
│
├── pre-processings            #> codes use to preprocess the raw traces to be 
│   │                          # able to run the evaluations
│   │── list_manipulation.py   #> split traces in {learning, testing, validating} 
│   │						   # sets 
│   │── accumulator.py         #> compute the sum and the square of the sum (to 
│   │                          # be able to recompute quickly the NICVS)  
│   │── nicv.py                #> to compute the NICVs
│   │── poi.py                 #> to extract the features
│   │── signal_processing.py   #> some signal processings (stft, ...)
│   │── tagmaps                #> all tagmaps use for to labelize the data 
│       │                      # (use to creat the lists)
│   	│── binary_classification.csv  
│   	│── family_classification.csv  
│   	│── novelties_classification.csv  
│   	│── obfuscation_classification.csv  
│   	│── packer_detection.csv  
│   	│── type_classification.csv  
│   	│── virtualize_detection.csv
```

## Getting Started
### python 3.6
To be able to run the analysis you need to install python 3.6 and the required 
packages:

```
pip install -r requirements.txt
```

### Data
The data used in the paper can be dowload on the following website:

```
https://zenodo.org/record/4317419
```

### File lists
In order to update the location of the data, you previously dowloaded, inside 
the lists you need to run the script ``update_lists.sh``:

```
./update_lists  [directory where are stored the list] [directory where are stored the traces]
```

## Machine Learning (ML)
To run the computation of the all the machine learning experiments, you can use
the script ``run.sh``:

```
./run_ml.sh  [directory where are stored the list] [directory where are stored the models] [directory where are stored the transformed traces]
```

The results are stored in the file ```log-evaluation```.


## Deep Learning (DL)
To run the computation of the all deep learning experiments, you can use
the script ``run.sh``:

```
./run_dl.sh  [directory where are stored the list] [directory where are stored the models]
```

The results are stored in the file ```evaluation_log_DL.txt```.

## Preprocessings
Once the traces have been aquiered and before beeing able to run the evualuation 
some preprocessings are needed. 
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

optional arguments:
  -h, --help                    show this help message and exit
  --raw PATH_RAW                Absolute path to the raw data directory
  --tagmap PATH_TAGMAP          Absolute path to a file containing the tag map
  --save PATH_SAVE              Absolute path to a file to save the lists
  --main-lists PATH_MAIN_LISTS  Absolute path to a file containing the main 
                                lists
  --extension EXTENSION         extensio of the files in 'datadir'
  --log-level LOG_LEVEL         Configure the logging level: 
                                DEBUG|INFO|WARNING|ERROR|FATAL
  --lists PATH_LISTS            Absolute path to a file containing lists
  --new_dir PATH_NEW_DIR        Absolute path to the raw data, to change in a 
                                given file lists
```

### accumulator.py
```
usage: accumulator.py [-h]
                      [--lists PATH_LISTS] 
                      [--output OUTPUT_PATH] 
                      [--freq FREQ] 
                      [--window WINDOW] 
                      [--overlap OVERLAP] 
                      [--core CORE]

optional arguments:
  -h, --help            show this help message and exit
  --lists PATH_LISTS    Absolute path of the lists 
                        (cf. list_manipulation.py -- using a main list will 
                        help) trace directory
  --output OUTPUT_PATH  Absolute path of the output directory
  --freq FREQ           Frequency of the acquisition in Hz
  --window WINDOW       Window size for STFT
  --overlap OVERLAP     Overlap size for STFT
  --core CORE           Number of core to use for multithreading accumulation

```

### nicv.py
```
usage: nicv.py [-h] 
               [--acc PATH_ACC] 
               [--lists PATH_LISTS] 
               [--save PATH_SAVE] 
               [--plot PATH_TO_PLOT] 
               [--log-level LOG_LEVEL]

optional arguments:
  -h, --help              show this help message and exit
  --acc PATH_ACC          Absolute path of the accumulators directory
  --lists PATH_LISTS      Absolute path to a file containing the main lists
  --save PATH_SAVE        Absolute path to a file to save the NICV
  --plot PATH_TO_PLOT     Absolute path to a previously saved NICV in order 
                          to display it
  --log-level LOG_LEVEL   Configure the logging level: 
                          DEBUG|INFO|WARNING|ERROR|FATAL
```

### poi.py
```
usage: poi.py [-h] 
              [--acc PATH_ACC] 
              [--lists PATH_LISTS]
              [--nb_of_bandwidth NB_OF_BANDWIDTH] 
              [--log-level LOG_LEVEL] 
              [--output_traces PATH_OUTPUT_TRACES] 
              [--output_lists PATH_OUTPUT_LISTS] 
              [--freq FREQ] 
              [--window WINDOW] 
              [--overlap OVERLAP] 
              [--dev DEVICE]

optional arguments:
  -h, --help                          show this help message and exit
  --acc PATH_ACC                      Absolute path of the accumulators 
                                      directory
  --lists PATH_LISTS                  list of the file
  --nb_of_bandwidth NB_OF_BANDWIDTH   number of bandwidth extract
  --log-level LOG_LEVEL               Configure the logging level: 
                                      DEBUG|INFO|WARNING|ERROR|FATAL
  --output_traces PATH_OUTPUT_TRACES  Absolute path to the directory 
                                      where the traces will be saved
  --output_lists PATH_OUTPUT_LISTS    Absolute path to the files where the new 
                                      lists will be saved
  --freq FREQ                         Frequency of the acquisition in Hz
  --window WINDOW                     Window size for STFT
  --overlap OVERLAP                   Overlap size for STFT
  --dev DEVICE                        Used device under test

```
























