

```python

```


\clearpage

### Activities and transitions - performance on validation set




Validating the model performance of the shortened sequence (*length=64*, *stepsize=32*) variant (@sec:RenewedTiredRelaxed) on the validation set. See @lst:best_model_activities_cfg for the configuration of the LSTM. This variant has an **accuracy of 81.7%** on the validation data set. See @fig:RenewedTiredRelaxed_validation_confusion_matrix for how the model performs in classifying each label. This performance reflects how well this variant will perform on new data.




![Confusion matrix of the predictions made by the model on the validation set. The diagonal reflects the correctly classified proportions for each category.](figures/RenewedTiredRelaxed_validation_confusion_matrix.png){#fig:RenewedTiredRelaxed_validation_confusion_matrix}
