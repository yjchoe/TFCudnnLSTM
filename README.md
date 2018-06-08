# TFCudnnLSTM

A simple template for TensorFlow's highly efficient `CudnnLSTM` module


## Dependencies

- TensorFlow v1.8+
- CUDA v9.0+
- cuDNN v7.0+
- scikit-learn
- tqdm

[How to check my CUDA and cuDNN versions](https://medium.com/@changrongko/nv-how-to-check-cuda-and-cudnn-version-e05aa21daf6c)


## Computational Performance

TensorFlow's performance guide includes [a section on RNN performance](https://www.tensorflow.org/performance/performance_guide#rnn_performance),
which states:
> On NVIDIA GPUs, the use of `tf.contrib.cudnn_rnn` should always be preferred 
unless you want layer normalization, which it doesn't support.

According to [this benchmark result by RETURNN](http://returnn.readthedocs.io/en/latest/tf_lstm_benchmark.html),
`CudnnLSTM` achieves significant speedups 
compared to TensorFlow's other LSTM implementations
(~2x faster than `LSTMBlockFused` and ~5x faster than `BasicLSTM`).

### PTB Experiments

We also took [the tutorial code for PTB language modeling](https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py) 
and tried running the three versions of LSTM implemented there: 
`BasicLSTMCell`, `LSTMBlockCell`, and `CudnnLSTM`.
We found that the `CudnnLSTM` example does not run in TF v1.8 due to 
API changes, but after changing we were able to run it on a single GPU.
The benchmark results we found are as follows:

Module          | Average wps* | Speedup w.r.t. `BasicLSTMCell`
----------------|--------------|--------------------------------
`BasicLSTMCell` | **15k**      | 1x
`LSTMBlockCell` | **17k**      | 1.1x
`CudnnLSTM`     | **32k**      | 2.1x

*wps refers to the number of data items (i.e. input word sequence & target word)
processed per second.  

In all three cases, we used a single NVIDIA Tesla P40 GPU, which was utilized 
80-85% (100% memory) during training. 
The tutorial code only supports multi-GPU training using `BasicLSTMCell`, 
and using 2 P40 GPUs we got approximately **25k** wps 
(1.7x speedup w.r.t. single-GPU `BasicLSTMCell`, 
but still 22% slower than a single-GPU `CudnnLSTM`.) 


## Caveats

We did not test the handling of variable-length sequences per batch for `CudnnLSTM`, 
but there seem to be some issues (e.g., see [#6633](https://github.com/tensorflow/tensorflow/issues/6633)). 
[Bucketing](https://www.tensorflow.org/api_guides/python/contrib.training#Bucketing) could be 
a useful (but not perfect) workaround for this problem.

`CudnnLSTM` does not support [layer normalization](https://arxiv.org/pdf/1607.06450.pdf), 
because cuDNN itself does not support it. 


## Comparisons with PyTorch and Keras

**PyTorch**'s built-in `nn.LSTM` module already supports CUDNN integration (!),
as shown [here](https://github.com/pytorch/pytorch/blob/v0.4.0/torch/backends/cudnn/rnn.py) 
and [here](https://github.com/pytorch/pytorch/issues/698).
For one thing, PyTorch's `nn.LSTM` is not a `contrib` module with little documentation. 

While we leave a rigorous comparison between PyTorch's `nn.LSTM` and 
TensorFlow's `cudnn_rnn.CudnnLSTM` as future work, it does appear that 
PyTorch's version is as efficient as but more stable than TensorFlow's counterpart.
When we tried running [PyTorch's own LSTM language model example](https://github.com/pytorch/examples/tree/0.4/word_language_model),
using nearly the same set of parameters 
(2 layers, 1.5k hidden size, 35k vocab size, 20 batch size and 35 timesteps),
we got around **100 milliseconds per batch** on a single P40 GPU (96% utilization). 
For the aforementioned tutorial code from TensorFlow, 
we got around **120 milliseconds per batch** on the same machine.

So, if you're already a PyTorch user and your system is built on PyTorch, 
there's little reason to switch to using TF's `CudnnLSTM` for performance, at least for now. 


**Keras** also has [a similar-looking module](https://keras.io/layers/recurrent/#cudnnlstm) 
that was [introduced last year](https://twitter.com/fchollet/status/918170264608817152?lang=en). 
We did not test it, but it appears to have a nice documentation.


## Authors

[YJ Choe](mailto:yjchoe33@gmail.com)
