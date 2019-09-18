

```python

```


\cleardoublepage

# Discussion

Reviewing and describing essential machine learning concepts as well as  building an exemplary machine learning pipeline was helpful in getting a more profound and comprehensive understanding of machine learning in theory and practice. Integrating the written thesis and the software code into a fully reproducible script (see @sec:reproducable_repo) will hopefully aid others in understanding of how a machine learning pipeline is built. The pipeline was designed to generalize to other data sets in the HCI domain. Therefore a neural network (LSTM) was chosen for the main modeling layer as neural networks are capable of representation learning [@de_jesus_rubio_ainsworth_2015] and can therefore be used on other data sets without the need for intensive manual feature engineering. However, finding a good configuration for the LSTM layer was very time consuming as LSTMs can be implemented in various architectures and many hyper-parameters combinations can be tested. Therefore, exploring other neural networks or combinations of neural networks lies outside the scope of this master's thesis.

The best performing variant of the machine learning pipeline for human activity recognition achieves an accuracy of almost 90% on new data when only activities are included. Including transitions this accuracy drops to about 80% with some of the transitions not being recognized at all. The entire pipeline contains only little pre-processing and no manual feature engineering except for the separation of body and gravity components in the acceleration data. Because of this the pipeline can potentially be fit to other HCI time series data sets with only adapting the configuration file of the LSTM and taking out some of the layers.

LSTMs are stochastic models and the fit will be different every time they are trained - even on the same data. To make the results reproducible seeds where used, which guarantee the same results when retraining. However, using different seeds would lead to different results. Therefore, the presented accuracy measures of the models might be biased and not reflect the average performance of this model. To make the performance measure and the classifications more robust even when retraining the model, an ensemble of models can be built. The same configuration would get trained *n* times and classifications would be mixed from all *n* models.

A potential improvement to the modeling block of the pipeline might be to add a convolutional neural network (CNN) before the LSTM [@sainath_convolutional_2015]. CNNs are good at extracting spatio-temporal features [@yang_deep_2015], which can then be fed to the LSTM, which might improve the performance of the LSTM.

Identifying a well fitting model architecture and hyper-parameters configuration was done manually for the presented pipeline. However, this process is labor intensive (far more than a 100 different architecture-configuration have been manually tested) or requires enough experience to know which variations are worth testing. Alternatively, a gridsearch ^[https://scikit-learn.org/stable/modules/grid_search.html (2019-05-07)] can be conducted on a range of hyperparameter. Here, the computer automatically checks many different hyperparameter configurations. The neural network intelligence (NNI) toolkit ^[https://github.com/Microsoft/nni (2019-05-07)] by Microsoft even automatically tests different neural network architectures. However, both the gridsearch and NNI toolkit require substantial computational resource as a lot of models will be trained and tested during these processes.

