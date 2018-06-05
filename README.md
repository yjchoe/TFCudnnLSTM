# TFCudnnLSTM

A simple template for TensorFlow's highly efficient `CudnnLSTM` module


## Dependencies

- TensorFlow v1.7+
- CUDA v9.0+
- cuDNN v7.0+

[How to check my CUDA and cuDNN versions](https://medium.com/@changrongko/nv-how-to-check-cuda-and-cudnn-version-e05aa21daf6c)

## Computational Performance

TensorFlow's performance guide includes [a section on RNN performance](https://www.tensorflow.org/performance/performance_guide#rnn_performance),
which states that "the use of `tf.contrib.cudnn_rnn` should always be preferred 
unless you want layer normalization, which it doesn't support."

According to [this benchmark result by RETURNN](http://returnn.readthedocs.io/en/latest/tf_lstm_benchmark.html),
`CudnnLSTM` achieves significant speedups 
compared to TensorFlow's other LSTM implementations
(~2x faster than `LSTMBlockFused` and ~5x faster than `BasicLSTM`).

RETURNN's own version of LSTM also achieves a comparative performance to `CudnnLSTM`,
but we do not test it here.

## Caveats

We did not test the handling of variable-length sequences per batch for `CudnnLSTM`, 
but there seem to be some issues (e.g., see [#6633](https://github.com/tensorflow/tensorflow/issues/6633)). 
[Bucketing](https://www.tensorflow.org/api_guides/python/contrib.training#Bucketing) could be 
a useful (but not perfect) workaround for this problem.

`CudnnLSTM` does not support [layer normalization](https://arxiv.org/pdf/1607.06450.pdf), 
because cuDNN itself does not support it. 

## Authors

[YJ Choe](mailto:yjchoe33@gmail.com)
