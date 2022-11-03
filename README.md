# JLAN: medical code prediction via joint learning attention networks and denoising mechanism
The JLAN: medical code prediction via joint learning attention networks and denoising mechanism (JLAN) is built based on MultiResCNN and CAML. The repo can be used to reproduce the results in the [paper](https://link.springer.com/article/10.1186/s12859-021-04520-x):
## Overview
In this paper, a new joint learning model is proposed to predict medical codes from clinical notes. On the MIMIC-III-50 dataset, 
our model outperforms all the baselines and SOTA models in all quantitative metrics. On the MIMIC-III-full dataset, our model outperforms in the macro-F1, micro-F1, macro-AUC, and precision at eight compared to the most advanced models. In addition, after introducing the denoising mechanism, the convergence speed of the model becomes faster, and the loss of the model is reduced overall.

## Setup
The repo mainly requires the following packages.
+ allennlp 0.9.1
+ nltk 3.3
+ python 3.6.12
+ torch 1.7.0+cu110
+ torchvision 0.8.1
+ scikit-learn 0.20.0

Full packages are listed in requirements.txt.
## 1. Data preprocessing
First, you should go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset (You may need to get the certificate); After that, save the following clinical records into the data folder:
+ D_ICD_DIAGNOSES.csv
+ D_ICD_PROCEDURES.csv
+ NOTEEVENTS.csv
+ DIAGNOSES_ICD.csv
+ PROCEDURES_ICD.csv
+ *_hadm_ids.csv (get from CAML)
## 2. Train and test using full MIMIC-III data
~~~
python main.py -data_path ./data/mimic3/train_full.csv -vocab ./data/mimic3/vocab.csv -Y full -model JLAN -embed_file ./data/mimic3/processed_full.embed -   criterion prec_at_8 -gpu 0 -tune_wordemb
~~~
## 3. Train and test using top-50 MIMIC-III data
~~~
python main.py -data_path ./data/mimic3/train_50.csv -vocab ./data/mimic3/vocab.csv -Y 50 -model JLAN -embed_file ./data/mimic3/processed_full.embed -criterion prec_at_5 -gpu 0 -tune_wordemb
~~~
## Acknowledgement
Many thanks to the open source repositories and libraries to speed up our coding progress.
+ MultiResCNN https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network
+ CAML https://github.com/jamesmullenbach/caml-mimic

