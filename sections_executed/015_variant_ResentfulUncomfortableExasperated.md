

```python

```


\clearpage


### Variant: LSTM with 2 layers - base configuration{#sec:ResentfulUncomfortableExasperated}




This variant uses the LSTM base configuration (@lst:lstm_base_cfg) with 2 LSTM-layers and one output layer. This variant has an **accuracy of 27.6%** on the test data set. See @fig:ResentfulUncomfortableExasperated_confusion_matrix for how the model performs in classifying each label. See @fig:ResentfulUncomfortableExasperated_learning_loss_accuracy for how the accuracy and loss evolved during training.

Only labels *sitting* and *standing* are predicted. The model achieves better accuracy in the beginning, but after about 10 epochs the loss increases and stabilizes (@fig:ResentfulUncomfortableExasperated_learning_loss_accuracy). The network's architecture might be too complex for this kind of data.




![Confusion matrix of the predictions made by the model on the test set. The diagonal reflects the correctly classified proportions for each category.](figures/ResentfulUncomfortableExasperated_confusion_matrix.png){#fig:ResentfulUncomfortableExasperated_confusion_matrix}


![Accuracy and loss on train and test data sets during training of LSTM on the training data set.](figures/ResentfulUncomfortableExasperated_learning_loss_accuracy.png){#fig:ResentfulUncomfortableExasperated_learning_loss_accuracy}
