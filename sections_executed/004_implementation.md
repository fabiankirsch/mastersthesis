

```python

```


\cleardoublepage

# Implemented pipeline layers

In this section the actual implementation of a machine learning pipeline is described. A machine learning pipeline consists of several layers, which can be grouped into extract-transform-load (ETL), pre-processing and modeling (see @sec:composition_pipeline). Each subsection describes a different layer in the pipeline. The order of the subsection reflects the order of the layers in the implementation. Some layers are always required, some are optional and some are mutually exclusive, e.g. if two models that cannot be combined. Some of the layers are specific to the data set used, but most layers are applicable to different data sets. A layer might have various configuration options or in the case of the modeling-layers also different architectures.



## ETL layer: loading and splitting

The raw data is loaded from different files in permanent storage into memory and merged into a single data object. The raw sensory data consists of 60 separate data files representing data from different sessions. The labels are again in a separate file. All these files are loaded and merged into one data object. This data object is then split into a train, test and validation set based on participant ids specified in the config. If the data is not in the data folder specified in the config the data is downloaded first from the UCI machine learning repository and extracted. See @tbl:dimensions_data for the shape of the data after this layer.



## ETL layer: sequencing
The output of this pipeline are discrete classifications, while the inputs are streams of observations sampled at 50Hz representing a continuous signal from different sensors. As each data point represents only a 50th of a second it will not provide enough information for an algorithm to classify single data points. Classifying the entire stream is not possible either, because it contains different behaviors that belong to different classes. It is therefore necessary to slice the stream into separate sequences (or windows), which can then be classified. When sequencing the data, two parameters can be set. The length of the sequence and distance between the starting points of succeeding sequences. The sequence should be long enough to reflect patterns related to the different behaviors to classify. The maxmium length depends on how quickly the behavior changes and which sequence lengths the modeling algorithms can deal with. For example, LSTMs can deal with sequence lengths of up to 300. The stepsize depends on how much overlap of sequences makes sense and how often a classification is needed. See @tbl:dimensions_data for the shape of the data after applying a *sequence length* of *128* and a *sequence stepsize* of *64* (parameters taken from authors of data @reyes-ortiz_transition-aware_2016).






![Sequencing of raw data. Sequences are 128 samples long and overlap by 50%, i.e. every sequence shifts 64 samples. Sequences are displayed in multiple rows so they are visually distinct. The background color indicates if the current data point is labeled as an activity or a transition. If the background is not colored no label is available. Only sequences with a distinct label are kept.](figures/default_etl_raw_labels_sequences.png){#fig:sequences_labels_dropped}



## ETL layer: sequence cleaning {#sec:etl_layer_seq_cleaning}
Next, sequences that contain unlabeled observations and sequences that contain observations with different labels are dropped (sequences with white fill in @fig:sequences_labels_dropped). Now, each sequence has a distinct label. When applying the model to new data this layer is not applied because no labels will exist. However, for training it is important that the sequences have a distinct label. See @tbl:dimensions_data for the shape of the data after this layer.



## ETL layer: separating input and output features


The input and output features need to be passed to the models separately, so the data sets (train, test & validation) containing the sequences need to be split to input features (x) and the output feature (y). The output feature sequences now hold a lot of redundant information as they contain a label for each observation in the sequence. However, all of these labels are the same within a sequence since after the cleaning (@sec:etl_layer_seq_cleaning). Therefore, the length of the output vector within each sequence can be reduced from *128* (sequence length) to *1*. see @tbl:dimensions_data for the shape of the data after this layer.





## ETL layer: label selection {#sec:default_etl_label_selection}


Once all sequences have a unique label, we can filter the labels we want our algorithm to train on, i.e. in this case select particular activities or transitions. See @tbl:activities_and_transitions for an overview of all activities and transitions present in the data. See @tbl:dimensions_data how the data set size changes after keeping only the activities and dropping the transitions.

Type | Label | Name
---|---|---
activity | 1 | walking           
activity | 2 | walking upstairs  
activity | 3 | walking downstairs
activity | 4 | sitting           
activity | 5 | standing          
activity | 6 | laying            
transition | 7  | stand to sit      
transition | 8  | sit to stand      
transition | 9  | sit to lie        
transition | 10 |  lie to sit        
transition | 11 |  stand to lie      
transition | 12 |  lie to stand    

: Overview of activity and transition labels present in the data. {#tbl:activities_and_transitions}



## ETL layer: Recoding output to binary features


The multi-categorical output feature is recoded into binary features - one feature for each category in the original feature. This is called one-hot-encoding and advisable for many algorithms to perform well in multi-classification tasks. The original output feature is dropped. See @tbl:dimensions_data for how the data set size changes after applying this layer.



## Pre-processing layer: noise reduction in input


The raw data contains noise that is removed using a median filter as the authors of the data did [@reyes-ortiz_transition-aware_2016]. See @fig:sequence_before_median_filter and @fig:sequence_after_median_filter.




## Pre-processing layer: separating bodily and gravitational acceleration





The data from the accelerometer holds both body and gravitational components, which are easier to distinguish during modeling if separated before (@fig:sequence_after_separating_body_gravity_acc) [@veltink_detection_1996;@van_hees_separating_2013]. Also this step is performed similar to the authors of the data [@reyes-ortiz_transition-aware_2016]. See @tbl:dimensions_data for the shape of the data after this layer.


![Raw sensory data before applying a median filter](figures/default_etl_sequence_before_median_filter.png){#fig:sequence_before_median_filter}


![Raw sensory data after applying a median filter with *kernel size = 3*.](figures/default_etl_sequence_after_median_filter.png){#fig:sequence_after_median_filter}


![After separating body and gravity components from the acceleration data.](figures/default_etl_sequence_after_separating_body_gravity_acc.png){#fig:sequence_after_separating_body_gravity_acc}




State and dimension labels | train set | test set | validation set
------- | --- | --- | ---
Loaded sets Xy (o,f) | (687249, 10) | (235711, 10) | (199812, 10)
Sequenced sets Xy (s,o,f) | (10684, 128, 7) | (3666, 128, 7) | (3103, 128, 7)
Cleaned sets Xy (s,o,f) | (6350, 128, 7) | (2173, 128, 7) | (1838, 128, 7)
Separate set X (s,o,f) | (6350, 128, 6) | (2173, 128, 6) | (1838, 128, 6)
Separate set y (s,f) | (6350, 1) | (2173, 1) | (1838, 1)
Activity selection X (s,o,f) | (6128, 128, 6) | (2105, 128, 6) | (1784, 128, 6)
Activity selection y (s,f) | (6128, 1) | (2105, 1) | (1784, 1)
Binary encoded y (s,f) | (6350, 6) | (2173, 6) | (1838, 6)
Body & gravity X (s,o,f) | (6128, 128, 9) | (2105, 128, 9) | (1784, 128, 9)

: Number of items within different dimensions of the data sets after passing through various layers in the pipeline: *o=number of observations*, *f=number of features*, *s=number of sequences*. Note, the sequenced data sets are three-dimensional. X denotes the input features, y the output features. The number of features in the one-hot-encoding equals the number of selected activities (*6* in this case). The sequence length and stepsize used for sequencing are *128* and *64*, respectively. {#tbl:dimensions_data}





## Pre-processing layer: normalization




All of the input features are normalized to have a *mean* of *0* and a *standard deviation* of *1*. This layer is still considered to be pre-processing, although actually a mini-model (the mean and standard deviation) is fitted to the train data set and kept to apply it to new data later including the test and validation sets.

## Modeling layer: LSTM
This layer contain a particular kind of recurrent neural network using long-short-term-memory (LSTM) cells [@gers_learning_1999]. Neural networks are capable of representation learning (see @sec:theory_feature_engineering), i.e. they can extract the most relevant patterns in raw data without prior manual feature engineering. Specifically, LSTMs are capable of learning patterns across long time lags, which makes them well suited for classifying the human behaviors present in the sensory data in the HAPT data set. LSTMS can have different architectures, i.e. a combination of one or more layers. Also, various hyperparameters can be set, like loss functions (a proxy perfomance function), activation functions (how input is converted within the network), optimizers (which parameters to try next), dropout (against over-fitting) and regularizers (keeping the parameters within certain limits).

