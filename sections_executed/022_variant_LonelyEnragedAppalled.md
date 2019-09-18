

```python

```


\clearpage


### Variant: LSTM with 1 layer - shorter sequences{#sec:LonelyEnragedAppalled}




This variant uses a *sequence length* of *64* (instead of *128*) and a *sequence stepsize* of *32* (instead of *64*). Also, only 50 units in its LSTM layer compared to 100 in the base config (@lst:lstm_base_cfg). It also uses regularizers (0.001 for both l1 and l2 regularizers on LSTM and output layers) instead of dropout (set to 0.0) to control the weights of the nodes in the network (compare @lst:lstm_base_cfg). The neural network has again 1 LSTM-layer and 1 output layer. This variant has an **accuracy of 88.3%** on the test data set. See @fig:LonelyEnragedAppalled_confusion_matrix for how the model performs in classifying each label. See @fig:LonelyEnragedAppalled_learning_loss_accuracy for how the accuracy and loss evolved during training. Using shorter sequence lengths and stepsizes seems to slightly improve the model fit (compare @sec:StimulatedWaryEntranced).





![Confusion matrix of the predictions made by the model on the test set. The diagonal reflects the correctly classified proportions for each category.](figures/LonelyEnragedAppalled_confusion_matrix.png){#fig:LonelyEnragedAppalled_confusion_matrix}


![Accuracy and loss on train and test data sets during training of LSTM on the training data set.](figures/LonelyEnragedAppalled_learning_loss_accuracy.png){#fig:LonelyEnragedAppalled_learning_loss_accuracy}
