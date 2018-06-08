#!/usr/bin/python3

"""
main.py: main training and evaluation loop

Author: YJ Choe (yjchoe33@gmail.com).
"""

import tensorflow as tf

from data import prepare_data
from cudnnlstm import CudnnLSTMModel

# allow global hyperparameters using `tf.app.flags`
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("n", 10000,
                            """Number of synthetic data to be used.""")
tf.app.flags.DEFINE_integer("time_len", 10,
                            """Number of timesteps in synthetic data.""")
tf.app.flags.DEFINE_integer("input_size", 2,
                            """Dimension of inputs. Fix to 2.""")

tf.app.flags.DEFINE_integer("num_layers", 2,
                            """Number of stacked LSTM layers.""")
tf.app.flags.DEFINE_integer("num_units", 64,
                            """Number of units in an LSTM layer.""")
tf.app.flags.DEFINE_string("direction", "unidirectional",
                           """Direction of the LSTM RNN. 
                              Either `unidirectional` or `bidirectional`.""")

tf.app.flags.DEFINE_integer("num_epochs", 50,
                            """Number of epochs for training.""")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            """Batch size per iteration during training.""")
tf.app.flags.DEFINE_float("learning_rate", 0.001,
                          """Learning rate for training using Adam.""")
tf.app.flags.DEFINE_float("dropout", 0.2,
                          """Dropout probability. `0.` means no dropout.""")

tf.app.flags.DEFINE_integer("seed", 0,
                            """Random seed for both numpy and tensorflow.""")


def main(_):

    # generate data
    (inputs_, inputs_valid_, inputs_test_,
     labels_, labels_valid_, labels_test_) = \
        prepare_data(FLAGS.time_len, FLAGS.n, FLAGS.input_size,
                     seed=FLAGS.seed)

    # initialize model & build TF graph
    model = CudnnLSTMModel(FLAGS.input_size,
                           FLAGS.num_layers, FLAGS.num_units, FLAGS.direction,
                           FLAGS.learning_rate, FLAGS.dropout, FLAGS.seed,
                           is_training=True)
    # training
    model.train(inputs_, inputs_valid_, labels_, labels_valid_,
                FLAGS.batch_size, FLAGS.num_epochs)
    # evalutation on test set
    model.eval(inputs_test_, labels_test_)


if __name__ == "__main__":
    tf.app.run(main=main)
