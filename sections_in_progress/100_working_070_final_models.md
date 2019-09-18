:::{.cell .markdown}

\clearpage


## Final models

The best performing models on the test data are validated on the validation set to understand their generalizability to entirely new data. Models are checked both for classifying activities only and classifying activities and transitions.

```{#lst:best_model_activities_cfg caption='Configuration of best performing LSTMs on activity and transition classification.' .yaml}

lstm_layer:
  units: 50
  activation: relu
  dropout: 0.0
  kernel_regularizer_l2: 0.001
  activity_regularizer_l1: 0.001
output_layer:
  activation: softmax
  kernel_regularizer_l2: 0.001
  activity_regularizer_l1: 0.001
loss: categorical_crossentropy
optimizer:
  adam:
    lr: 0.001
    beta_1: 0.9
    beta_2: 0.999
    decay: 0.0
metrics: ['accuracy']
epochs: 100
batch_size: 200
```
