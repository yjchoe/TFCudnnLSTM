#!/usr/bin/python3

"""
cudnnlstm.py:
A simple training and evaluation code using TensorFlow's `CudnnLSTM` module.

Tested on TF v1.8 with CUDA v9.0 and cuDNN v7.0.

Author: YJ Choe (yjchoe33@gmail.com).
"""

import tensorflow as tf
from tqdm import tqdm


class CudnnLSTMModel:
    """TF graph builder for the CudnnLSTM model."""

    def __init__(self, input_size=2,
                 num_layers=2, num_units=64, direction="unidirectional",
                 learning_rate=0.001, dropout=0.2, seed=0, is_training=True):
        """Initialize parameters and the TF computation graph."""

        """
        model parameters
        """
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_units = num_units
        self.direction = direction
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.seed = seed

        self.model_name = "cudnnlstm-{}-{}-{}-{}-{}".format(
            self.num_layers, self.num_units, self.direction,
            self.learning_rate, self.dropout)
        self.save_path = "./outputs/{}.ckpt".format(self.model_name)

        # running TF sessions
        self.is_training = is_training
        self.saver = None
        self.sess = None

        tf.set_random_seed(self.seed)

        """
        TF graph construction
        """
        # [time_len, batch_size, input_size]
        self.inputs = tf.placeholder(tf.float32,
                                     shape=[None, None, self.input_size])
        self.lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
            self.num_layers,
            self.num_units,
            direction=self.direction,
            dropout=self.dropout if is_training else 0.,
            # kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.outputs, self.output_states = self.lstm(
            self.inputs,
            initial_state=None,
            training=True
        )
        # [batch_size, num_units * num_dirs]
        self.mean_outputs = tf.reduce_mean(self.outputs, axis=0)
        # [batch_size]
        self.logits = tf.squeeze(
            tf.layers.dense(
                self.mean_outputs, 1, activation=None,
                # kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
        )
        self.predictions = tf.cast(self.logits >= 0., dtype=tf.float32)

        # necessary for training loops
        if is_training:
            # [batch_size]
            self.labels = tf.placeholder(tf.float32, shape=[None])

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.labels,
                    logits=self.logits,
                )
            )
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.predictions, self.labels),
                        dtype=tf.float32)
            )

            # Training Scheme
            self.global_step = \
                tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            self.gvs = self.optimizer.compute_gradients(self.loss)
            # Gradient Clipping
            self.clipped_gvs = []
            for grad, var in self.gvs:
                if var.name.startswith("dense"):
                    continue
                grad = tf.clip_by_value(grad, -5., 5.)
                grad = tf.clip_by_norm(grad, 100.)
                self.clipped_gvs.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(
                self.clipped_gvs, global_step=self.global_step)

            # Summary
            tf.summary.scalar('global_step', self.global_step)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            for grad, var in self.clipped_gvs:
                tf.summary.histogram('grad-{}'.format(var.name), grad)
            self.summary = tf.summary.merge_all()

        return

    def train(self, inputs_, inputs_valid_, labels_, labels_valid_,
              batch_size, num_epochs):
        """Train a multilayer bidirectional LSTM with dropout using
        `tf.contrib.cudnn_rnn.CudnnLSTM`."""

        assert self.is_training, \
            "train(): model not initialized in training mode"

        self.saver = tf.train.Saver()

        # Create session
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
        )) as sess:

            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter("logdir/{}".format(self.model_name),
                                           graph=tf.get_default_graph())

            print("========Training CudnnLSTM with "
                  "{} layers and {} units=======".format(self.num_layers,
                                                         self.num_units))
            n_train = len(labels_)
            for epoch in range(num_epochs):
                print("Epoch {}:".format(epoch))
                for batch in tqdm(range(n_train // batch_size)):
                    current = batch * batch_size
                    _, summary, gs = sess.run(
                        [self.train_op, self.summary, self.global_step],
                        feed_dict={
                            self.inputs:
                                inputs_[:, current:current+batch_size, :],
                            self.labels:
                                labels_[current:current+batch_size]
                        }
                    )
                    writer.add_summary(summary, gs)
                # monitor per epoch
                train_loss_ = sess.run(
                    self.loss, feed_dict={self.inputs: inputs_,
                                          self.labels: labels_})
                valid_loss_ = sess.run(
                    self.loss, feed_dict={self.inputs: inputs_valid_,
                                          self.labels: labels_valid_})
                print("\ttrain loss: {:.5f}".format(train_loss_))
                print("\tvalid loss: {:.5f}".format(valid_loss_))
                self.saver.save(sess, self.save_path)

            print("========Finished training! "
                  "(Model saved in {})========".format(self.save_path))
        return

    def eval(self, inputs_test_, labels_test_):
        """Evaluate a learned LSTM model on the test data.

        TODO: There's an issue when calling `eval()` from a restored model.
        """

        # assert not self.is_training, \
        #     "eval(): model initialized in training mode"

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
        )) as sess:

            # restore model saved from `train()`
            if self.saver is None:
                self.saver = tf.train.import_meta_graph(self.save_path+".meta")
            self.saver.restore(sess, self.save_path)
            print("========Model restored from {}========".format(
                self.save_path))

            [test_loss_, test_accuracy_] = sess.run(
                [self.loss, self.accuracy],
                feed_dict={self.inputs: inputs_test_,
                           self.labels: labels_test_}
            )
            print("test set cross-entropy loss: {:.5f}".format(test_loss_))
            print("test set accuracy: {:.5f}".format(test_accuracy_))

        with open(self.save_path + ".test.txt", "w") as f:
            f.write("Test set evaluation:\n")
            f.write("\tloss: {:.5f}\n".format(test_loss_))
            f.write("\taccuracy: {:.5f}\n".format(test_accuracy_))

        return test_loss_, test_accuracy_
