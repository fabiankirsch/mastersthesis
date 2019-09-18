:::{.cell .markdown}

\clearpage


## LSTM - activities and transitions

An LSTM network with 1 or more LSTM layers and a Dense layer as the output layer is implemented. Both activities and transitions are classified. See @lst:lstm_activities_transitions_base_cfg for the base configuration used for the LSTM. This base configuration contains parameters that worked well on classifying only activities.

```{#lst:lstm_activities_transitions_base_cfg caption='Base configuration for LSTM' .yaml}

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
