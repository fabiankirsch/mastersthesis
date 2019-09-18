

```python

```


\clearpage


### Variant: LSTM with 1 layer - base config{#sec:WarmSympatheticIntrigued}





This variant uses the LSTM base config for activities and transitions (@lst:lstm_activities_transitions_base_cfg). This variant has an **accuracy of 71.1%** on the test data set. See @fig:WarmSympatheticIntrigued_confusion_matrix for how the model performs in classifying each label. See @fig:WarmSympatheticIntrigued_learning_loss_accuracy for how the accuracy and loss evolved during training. The model struggles to classify the transitions correctly and also performs less well on the activities than when only classifying activities without transitions. A possible reason could be that only very few sequences containing are kept during sequence cleaning and the model does not have enough data to train (see @fig:sequences_labels_dropped).




![Confusion matrix of the predictions made by the model on the test set. The diagonal reflects the correctly classified proportions for each category.](figures/WarmSympatheticIntrigued_confusion_matrix.png){#fig:WarmSympatheticIntrigued_confusion_matrix}


![Accuracy and loss on train and test data sets during training of LSTM on the training data set.](figures/WarmSympatheticIntrigued_learning_loss_accuracy.png){#fig:WarmSympatheticIntrigued_learning_loss_accuracy}
