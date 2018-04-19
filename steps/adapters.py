import numpy as np
from scipy import sparse


def identity_inputs(inputs):
    return inputs


def take_first_inputs(inputs):
    return inputs[0]


def to_tuple_inputs(inputs):
    return tuple(inputs)


def to_numpy_label_inputs(inputs):
    return inputs[0].values.reshape(-1)


def sparse_hstack_inputs(inputs):
    return sparse.hstack(inputs)


def hstack_inputs(inputs):
    return np.hstack(inputs)


def vstack_inputs(inputs):
    return np.vstack(inputs)


def stack_inputs(inputs):
    stacked = np.stack(inputs, axis=0)
    return stacked


def sum_inputs(inputs):
    stacked = np.stack(inputs, axis=0)
    return np.sum(stacked, axis=0)


def average_inputs(inputs):
    stacked = np.stack(inputs, axis=0)
    return np.mean(stacked, axis=0)


def squeeze_inputs(inputs):
    return np.squeeze(inputs[0], axis=1)


def exp_transform_inputs(inputs):
    return np.exp(inputs[0])
