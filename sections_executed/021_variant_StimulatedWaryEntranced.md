

```python

```


\clearpage


### Variant: LSTM with 1 layer - regularizers instead of dropout {#sec:StimulatedWaryEntranced}




This variant uses regularizers (0.001 for both l1 and l2 regularizers on LSTM and output layers) instead of dropout (set to 0.0) to control the weights of the nodes in the network (compare @lst:lstm_base_cfg). The neural network has again 1 LSTM-layer and 1 output layer. This variant has an **accuracy of 86.2%** on the test data set. See @fig:StimulatedWaryEntranced_confusion_matrix for how the model performs in classifying each label. See @fig:StimulatedWaryEntranced_learning_loss_accuracy for how the accuracy and loss evolved during training. Apart from a brief peak the loss constantly decrease and the model can classify the activities quite well. Adding regularizers seems like an important step. Only sitting and standing are still difficult to distinguish for the model.





![Confusion matrix of the predictions made by the model on the test set. The diagonal reflects the correctly classified proportions for each category.](figures/StimulatedWaryEntranced_confusion_matrix.png){#fig:StimulatedWaryEntranced_confusion_matrix}


![Accuracy and loss on train and test data sets during training of LSTM on the training data set.](figures/StimulatedWaryEntranced_learning_loss_accuracy.png){#fig:StimulatedWaryEntranced_learning_loss_accuracy}
