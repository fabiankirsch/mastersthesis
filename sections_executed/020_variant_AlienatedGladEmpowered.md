

```python

```


\clearpage


### Variant: LSTM with 1 layer - base configuration{#sec:AlienatedGladEmpowered}




This variant uses the LSTM base configuration (@lst:lstm_base_cfg) with 1 LSTM-layer and one output layer.This variant has an **accuracy of 19.0%** on the test data set. See @fig:AlienatedGladEmpowered_confusion_matrix for how the model performs in classifying each label. See @fig:AlienatedGladEmpowered_learning_loss_accuracy for how the accuracy and loss evolved during training.

Although less complex than the previous model (@sec:ResentfulUncomfortableExasperated), also in this variant the loss increases suddenly and stabilizes at a high level (@fig:AlienatedGladEmpowered_learning_loss_accuracy). A reason for this might be vanishing or exploding gradients (i.e. the weights of the model stabilizing at *0* or becoming huge, respectively).




![Confusion matrix of the predictions made by the model on the test set. The diagonal reflects the correctly classified proportions for each category.](figures/AlienatedGladEmpowered_confusion_matrix.png){#fig:AlienatedGladEmpowered_confusion_matrix}


![Accuracy and loss on train and test data sets during training of LSTM on the training data set.](figures/AlienatedGladEmpowered_learning_loss_accuracy.png){#fig:AlienatedGladEmpowered_learning_loss_accuracy}
