

```python

```


\clearpage


### Variant: LSTM with 2 layers - less units and regularizer



This variant uses the base configuration for activities and transitions (@lst:lstm_activities_transitions_base_cfg) with 2 LSTM-layers and one output layer. This variant has an **accuracy of 18.6%** on the test data set. See @fig:TiredSurprisedShaky_confusion_matrix for how the model performs in classifying each label. See @fig:TiredSurprisedShaky_learning_loss_accuracy for how the accuracy and loss evolved during training. Again, having 2 LSTM layers seems to be a too complex network - none of the patterns is recognized, the model always predicts the same label.




![Confusion matrix of the predictions made by the model on the test set. The diagonal reflects the correctly classified proportions for each category.](figures/TiredSurprisedShaky_confusion_matrix.png){#fig:TiredSurprisedShaky_confusion_matrix}


![Accuracy and loss on train and test data sets during training of LSTM on the training data set.](figures/TiredSurprisedShaky_learning_loss_accuracy.png){#fig:TiredSurprisedShaky_learning_loss_accuracy}
