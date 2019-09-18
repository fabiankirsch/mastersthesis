

```python

```


\clearpage


## LSTM - activities only

An LSTM network with 1 or more LSTM layers and a Dense layer as the output layer is implemented. Assuming that activities are easier to model than transitions only sequences that are labeled as activities are considered in this section and transitions are removed. Weights are optimized using the ADAM optimizer [@kingma_adam:_2014], which needs relatively little computational resources. For the loss function categorical cross-entropy is used, which generally performs best for multi-classification problems. See @lst:lstm_base_cfg for the base configuration used for the LSTM. This first configuration was conceived based on experience, but has not been tested on this data.

```{#lst:lstm_base_cfg caption='Base configuration for LSTM' .yaml}

lstm_layer:
units: 100
activation: relu
dropout: 0.2
kernel_regularizer_l2: 0
activity_regularizer_l1: 0
output_layer:
activation: softmax
kernel_regularizer_l2: 0
activity_regularizer_l1: 0
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
