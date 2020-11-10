## Basic Layers

These core layers form the foundation of almost all neural networks.

```@docs
Chain
Dense
```

## Convolution and Pooling Layers

These layers are used to build convolutional neural networks (CNNs).

```@docs
Conv
AdaptiveMaxPool
MaxPool
GlobalMaxPool
AdaptiveMeanPool
MeanPool
GlobalMeanPool
DepthwiseConv
ConvTranspose
CrossCor
SamePad
flatten
Flux.Zeros
Flux.convfilter
Flux.depthwiseconvfilter
```

## Recurrent Layers

Much like the core layers above, but can be used to process sequence data (as well as other kinds of structured data).

```@docs
RNN
LSTM
GRU
Flux.Recur
Flux.reset!
```

## Other General Purpose Layers
These are marginally more obscure than the Basic Layers.
But in contrast to the layers described in the other sections are not readily grouped around a particular purpose (e.g. CNNs or RNNs).

```@docs
Maxout
SkipConnection
```


## Normalisation & Regularisation

These layers don't affect the structure of the network but may improve training times or reduce overfitting.

```@docs
Flux.normalise
BatchNorm
Flux.dropout
Dropout
AlphaDropout
LayerNorm
InstanceNorm
GroupNorm
```

### Testmode

Many normalisation layers behave differently under training and inference (testing). By default, Flux will automatically determine when a layer evaluation is part of training or inference. Still, depending on your use case, it may be helpful to manually specify when these layers should be treated as being trained or not. For this, Flux provides `Flux.testmode!`. When called on a model (e.g. a layer or chain of layers), this function will place the model into the mode specified.

```@docs
Flux.testmode!
trainmode!
```