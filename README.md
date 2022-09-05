# Code for probing experiments 
described in the paper

_New or Old? Exploring How Pre-Trained
Language Models Represent Discourse Entities_

## Step 1: Get the ARRAU corpus
The corpus can be acquired through the [LDC](https://catalog.ldc.upenn.edu/LDC2013T22).
The experiments are based on the news sections of the corpus, meaning that `pears` (narratives) and `trains` (spoken dialog) are discarded. Some parts of ARRAU are also available through the [ARRAU corpus GitHub](https://sites.google.com/view/arrau/corpus). We kept the official split used in the [CRAC 2018 shared task](http://anawiki.essex.ac.uk/dali/crac18/crac18_shared_task.html).


## Step 2: Pre-processing 

### Pre-process corpus
Extract data from ARRAU corpus
```
python3 extractor.py PATH/TO/CORPUS/ [--min_words]
```
```PATH/TO/CORPUS/``` needs to contain the annotation directories ```[train|dev|test]/MMAX```.

```--min_words``` only extracts head annotations, otherwise spans are extracted.

Creates output files in directories 
```
data/arrau_processed/[heads|spans]/[train|dev|test]/PATH_TO_CORPUS_filename
```

Output files are in CoNLL format with the columns:
```
ID  TOKEN  POS  POS-baseline-label  BIO-label  entity-ID  BIO-old/new-label
```


### Extract pre-trained representations
Extracts the contextualized hidden representations from a pre-trained model. 

Transformer-XL defines a moses-based tokenizer, GPT-2 uses a BPE tokenizer,
so the data also has to be pre-processed to augment the labels according to the tokenized sequence.
```
python3 extract_pretrained_hidden.py 
--data_in data/arrau_processed/[heads|spans]
--model [transfo-xl-wt103|gpt2] 
--data_out_hidden data/hidden_states/[heads|spans]
--data_out_tokens data/tokenized/[heads|spans]
```

It expects the subdirectories _train_, _dev_ and _test_ to live under the path specified for ```data_in``` and creates analogous ```model``` directories under ```data_out_hidden``` and ```data_out_tokens```.

## Step 3: Probing experiments

### Classification probe
The experiments were run using [comet.ml](comet.ml)

```
python3 entity_classifier.py configs/CONFIG_FILE &> logs/CONFIG_FILE.log
```
Create training and testing configs similar to the ones below:


####Example training config file
```
#comet data
# set ignore comet to false and add credentials for using comet.ml
ignore_comet: true
comet_key: ""
comet_project: ""
comet_workspace: ""

#training params
mode: train
pre_trained_model: transfo-xl-wt103
train_data: ../data/tokenized/arrau_processed/heads/train
val_data: ../data/tokenized/arrau_processed/heads/dev
train_hidden: ../data/hidden_states/arrau_processed/heads/train
val_hidden: ../data/hidden_states/arrau_processed/heads/dev
model: ../entity_classifier/heads/
batch_size: 64
epochs: 20
learning_rate: 0.001
patience: 7
ablation: false
```
* ```mode``` determines whether train/test mode is started
* ```pre_trained_model``` specifies whether to use pre-trained embeddings (transfo-xl-wt103, gpt2, or fasttext) or to use the baseline (null)
* ```train/val_data```  path to the training and validation data, respectively
* ```train/val_hidden```  path to the pre-extracted contextualized representations corresponding to the training and validation data, respectively (if transfo-xl-wt103, else null)
* ```model``` specifies the folder where the model will be stored. The best performing model will be stored as _model_ in this directory
* ```ablation``` specifies whether to use attention over the context (false) or to classify only based on the entity representation (true)

####Example testing config file 
```
#comet data
# set ignore comet to false and add credentials for using comet.ml
ignore_comet: true
comet_key: ""
comet_project: ""
comet_workspace: ""

#test params
mode: test
pre_trained_model: transfo-xl-wt103
test_data: ../data/tokenized/arrau_processed/heads/test
test_hidden: ../data/hidden_states/arrau_processed/heads/test
model: ../entity_classifier/heads/model
```
* ```mode``` determines whether train/test mode is started
* ```pre_trained_model``` specifies whether to use pre-trained embeddings (transfo-xl-wt103 or fasttext) or to use the baseline (null)
* ```test_data```  path to the test data
* ```test_hidden```  path to the pre-extracted contextualized representations corresponding to the test data (if transfo-xl-wt103, else null)
* ```model``` specifies the path to the model to be tested


### Sequence labelling probe
```
python3 run_seq_labelling.py configs/CONFIG_FILE &> logs/CONFIG_FILE.log
```

Create training and testing configs similar to the ones below:

####Example training config file
```
mode: train
seed: 456889
pre_trained: true
contextualized: true
lstm: true
gold_column: -1 
data: /data/tokenized/arrau_pos/heads/
model: /arrau_pos/heads/456889/contextualized/
hidden_vectors: /data/hidden_states/heads/
fixed_vectors: /data/fixed_vectors/
ft_model: cc.en.300.bin
dropout: 0.2
batch_size: 64
hidden: 256
embed_size: 1024 
epochs: 50
lr: 0.01
```
* ```mode``` determines whether train/test mode is started
* ```seed``` fixed value to use as seed
* ```pre_trained``` use true if training with pre_trained Transformer embeddings or fastText; false if training embeddings from scratch
* ```contextualized``` use true if training with Transformer embeddings, otherwise false
* ```lstm``` specifies whether to use an uni-directional lstm layer between the embeddings and the CRF
* ```gold_column``` number of column with gold label
* ```data``` expects the subdirectories _train_ and _dev_ to live under the specified path and contain the respective data
* ```model``` in the training config specifies the folder where the model will be stored. The actual model file for the testing config can be extracted from the script output once the training is done (last line in the .log file)
* ```hidden_vectors``` specifies the file path to previously extracted hidden vectors from a pre-trained model using 
* ```fixed_vectors``` path to static fastText embeddings
* ```ft_model``` specific fastText model to use
* ```dropout``` dropout value
* ```batch_size``` batch size value
* ```hidden``` hidden layer size value
* ```embed_size``` embedding size value: use 1024 for TransformerXL contextualized embeddings; 300 for fastText embeddings.
* ```epochs``` maximum number of epochs (early stopping will be applied in all cases)
* ```lr```: learning rate value
  
####Example testing config file 
```
mode: test
seed: 456889
pre_trained: true
contextualized: false
lstm: true
gold_column: -1 
data: /data/tokenized/arrau_pos/heads/
hidden_vectors: /data/hidden_states/arrau_pos/heads/
fixed_vectors: /data/fixed_vectors/
ft_model: cc.en.300.bin
model: /crf_lstm/arrau_pos/heads/456889/fixed/Epo_6_see_456889_bat_64_hid_256_los_7.1916
log_dir: /crf_lstm/arrau_pos/heads/456889/fixed/
embed_size: 300 
```
* ```mode``` determines whether train/test mode is started
* ```seed``` fixed value to use as seed
* ```pre_trained``` use true if training with pre_trained Transformer embeddings or fastText; false if training embeddings from scratch
* ```contextualized``` use true if training with Transformer embeddings, otherwise false
* ```lstm``` specifies whether to use an uni-directional lstm layer between the embeddings and the CRF
* ```gold_column``` number of column with gold label
* ```data``` expects the subdirectory _test_ to live under the specified path and contain the respective data
* ```model``` specifies the path to the model to be tested
* ```log_dir``` specifies where the file _model_predictions.log_ will be created for checking the model output for the test data. It contains the following,  separated by newlines:
* ```embed_size``` embedding size value: use 1024 for TransformerXL contextualized embeddings; 300 for fastText embeddings. 
```
sequence
gold_labels
model_prediction
```
* ```pre_trained``` specifies whether the model used pre_trained embeddings or to trained them from scratch (has to match training config)
* ```hidden_vectors``` specifies the file path to previously extracted hidden vectors from a pre-trained model using ```extract_pretrained_hidden.py``` if pre_trained is _true_
* ```lstm``` specifies whether to use an additional lstm layer between the embeddings and the CRF (has to match training config)


## Step 4: Error Analysis

#### Manually evaluate output differences
To compare the outputs of different models (e.g. baseline vs. transformer-xl)
```
python3 labelling_analysis.py PATH/TO/model_predictions.log_1 PATH/TO/model_predictions.log_2
```
prints out the samples where the predictions differ (one token per line) for manual inspection



