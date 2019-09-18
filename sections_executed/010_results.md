

```python

```


\cleardoublepage

# Results

In this section various pipeline architectures are described and their performance on classifying the raw data are presented. More than 100 variants were tested during the pipeline development in total. Presenting all of them would neither be feasible nor helpful to the reader. Therefore, only variants that likely provide helpful insights to the reader are presented. The variants are presented in the order they were discovered, so the reader can follow how new insights led to new architectures and configurations. Each subsection presents a particular architecture and different configurations for each architecture are again  presented in sub-subsections. The last subsection of this section validates the best performing variant on the validation set.

The primary performance measure used is accuracy. Accuracy is defined as the proportion of correct classifications compared to all classifications with a score of 1 reflecting a perfect fit of the model. Accuracy can be misleading, if class sizes are very unequal, but this is not the case in the HAPT data set, so accuracy is a save measure. For a more detailed understanding of how well particular classes can be classified a confusion matrix is presented as well. The performance of each variant is tested on the test set. The best performing pipeline will then be tested again on the validation set to understand the pipeline's performance on new data.

The models presented are non-deterministic as they are randomly initiated. To make this thesis entire reproducible a seed is set before running each variant. All variants use the default ETL and pre-processing configurations presented in the first subsection (@sec:default_etl_preprocessing_layers).

## Default ETL and pre-processing layers {#sec:default_etl_preprocessing_layers}

This subsection gives and overview of the default ETL and pre-processing layers applied to every implementation variant (see @tbl:default_etl_preprocessing_pipeline). See @lst:etl_cfg for the default configuration passed to the pipeline.


Type | Layer
--- | -----
ETL | Loading and splitting
ETL | Sequencing
ETL | Sequence cleaning
ETL | Separating input and output features
ETL | Label selection
ETL | Recoding output to binary features
Pre-processing | Noise reduction in input
Pre-processing | Separating bodily and gravitational acceleration
Pre-processing | Normalizing input features

: Default ETL and pre-processing layers used in every variant {#tbl:default_etl_preprocessing_pipeline}





```{#lst:etl_cfg caption='Default configuration for ETL and pre-processing layers in all implemented pipeline variants.' .yaml}

01_etl:
  data_set_dir: 'data/HAPT Data Set'
  download_url: 'https://archive.ics.uci.edu/ml/ machine-learning-databases/00341/HAPT%20Data%20Set.zip'
  data_split:
    train_participant_ids: [20,  6, 22, 18, 26, 27,  3, 11, 13, 30, 19, 12, 10, 17, 21,  4, 14, 24]
    test_participant_ids: [16, 28,  2,  1, 23, 25]
    validation_participant_ids: [ 7,  9, 15, 29,  8,  5]
  selected_labels: [1,2,3,4,5,6]
  channel_names_prior_preprocess: ['gyro-X', 'gyro-Y', 'gyro-Z', 'acc-X', 'acc-Y', 'acc-Z']
  channl_names_post_preprocess: ['gyro-X', 'gyro-Y', 'gyro-Z', 'body-X', 'body-Y', 'body-Z', 'gravity-X', 'gravity-Y', 'gravity-Z']
  sequence_length: 128
  sequence_stepsize: 64
  drop_columns: ['participant_id', 'experiment_id', 'time'] # the columns are loaded initially, because they are needed to sort and group data, but should not be used for modeling  
  group_column: 'experiment_id' # data is sequenced within these groups
  sample_rate: 50
02_preprocessing:
  sample_rate: 50
  median_filter_kernel: 3
  acc_columns_idx: [3,4,5] # indices of columns that contain the acceleration data

```
