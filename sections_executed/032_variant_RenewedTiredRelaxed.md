

```python

```


\clearpage


### Variant: LSTM with 1 layer - shorter sequences{#sec:RenewedTiredRelaxed}





This variant uses a *sequence length* of *64* (instead of *128*) and a *sequence stepsize* of *32* (instead of *64*). This variant has an **accuracy of 80.1%** on the test data set. See @fig:RenewedTiredRelaxed_confusion_matrix for how the model performs in classifying each label. See @fig:RenewedTiredRelaxed_learning_loss_accuracy for how the accuracy and loss evolved during training. The shorter sequence length and stepsize have a significant positive effect on the model's performance. However, transitions are still not being recognized well, some of them are not recognized at all.




![Confusion matrix of the predictions made by the model on the test set. The diagonal reflects the correctly classified proportions for each category.](figures/RenewedTiredRelaxed_confusion_matrix.png){#fig:RenewedTiredRelaxed_confusion_matrix}


![Accuracy and loss on train and test data sets during training of LSTM on the training data set.](figures/RenewedTiredRelaxed_learning_loss_accuracy.png){#fig:RenewedTiredRelaxed_learning_loss_accuracy}
