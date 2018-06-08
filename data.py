#!/usr/bin/python3

"""
data.py: helper for synthetic data generation

Author: YJ Choe (yjchoe33@gmail.com).
"""

import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data(time_len, n, input_size,
                 valid_size=0.1, test_size=0.1, seed=0):
    """Generates data and returns its train/valid/test split."""
    inputs_ = 2 * np.random.rand(time_len, n, input_size) - 1
    x1s_, x2s_ = [xs.squeeze() for xs in np.split(inputs_, [1], axis=2)]
    values_, labels_ = ground_truth_2d(x1s_, x2s_, seed=seed)

    # train_test_split requires shape[0] to match
    inputs_ = np.transpose(inputs_, axes=(1, 0, 2))
    inputs_, inputs_test_, labels_, labels_test_ = train_test_split(
        inputs_, labels_, test_size=test_size,
        random_state=seed)
    inputs_, inputs_valid_, labels_, labels_valid_ = train_test_split(
        inputs_, labels_, test_size=(1.-test_size) * valid_size,
        random_state=seed)
    inputs_ = np.transpose(inputs_, axes=(1, 0, 2))
    inputs_valid_ = np.transpose(inputs_valid_, axes=(1, 0, 2))
    inputs_test_ = np.transpose(inputs_test_, axes=(1, 0, 2))

    return inputs_, inputs_valid_, inputs_test_, \
           labels_, labels_valid_, labels_test_


def ground_truth_2d(x1s, x2s, noise=0.01, seed=0):
    """Generates some synthetic data with 2-dimensional inputs per timestep
    and a binary output.

        f(x1s, x2s) = mean_t(0.1 * x1s[t]^2 * sin(t)
                             + 0.4 * x2s[t]^3 / (|x1s[t]| + 1))
        y(x1s, x2s) = tanh(f(x1s, x2s) + Gaussian(0., noise^2)) > 0.

    Args:
        x1s, x2s: np.array of shape [time_len, n]

    Returns:
        values, labels: np.array of shape [n]
    """
    assert x1s.shape == x2s.shape
    time_len, n = x1s.shape

    np.random.seed(seed)
    values = np.mean(0.1 * np.power(x1s, 2) *
                     np.sin(np.arange(time_len))[:, np.newaxis]
                     + 0.4 * np.power(x2s, 3) / (np.abs(x1s) + 1.), axis=0)
    labels = np.tanh(values + noise * np.random.randn(n)) > 0.
    return values, np.float32(labels)


def ground_truth_nd(xss, noise=0.01, seed=0):
    """Generates some synthetic data with n-dimensional inputs per timestep
    and a binary output."""
    raise NotImplementedError
