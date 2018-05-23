import numpy as np

from scipy import sparse


def identity_inputs(inputs):
    """Performs no operation on the inputs list

    Args:
        inputs (list): list of inputs passed from input_steps and input_data

    Returns:
        list: list of inputs passed from input_steps and input_data
    """
    return inputs


def to_tuple_inputs(inputs):
    """Transforms inputs list to a tuple of inputs

    Args:
        inputs (list): list of inputs passed from input_steps and input_data

    Returns:
        tuple: tuple from the list of inputs
    """
    return tuple(inputs)


def take_first_inputs(inputs):
    """Takes first element from the inputs list

    This function is useful when adapter was taking just one input and no aggregation was needed.

    Args:
        inputs (list): list of inputs passed from input_steps and input_data

    Returns:
        obj: first element from the list of inputs
    """
    return inputs[0]


def to_numpy_label_inputs(inputs):
    """Transforms the first element from the list of inputs into the numpy array of shape (n,)

    This function is useful when working with sklearn estimators that need numpy (n,) arrays as targets.
    It takes first element, casts and reshapes it.

    Args:
        inputs (list): list of inputs passed from input_steps and input_data

    Returns:
        numpy array: numpy array of shape (n,)
    """
    return take_first_inputs(inputs).values.reshape(-1)


def squeeze_inputs(inputs, axis=1):
    """Squeezes the first element of the inputs over desired axis

    Args:
        inputs (list): list of inputs passed from input_steps and input_data
        axis (int): axis over which the first element from inputs list should be squeezed
    Returns:
        numpy array: first element from the inputs list squeezed over the desired axis
    """
    return np.squeeze(take_first_inputs(inputs), axis=axis)


def exp_transform_inputs(inputs):
    """Calculates exponent over the first element from the inputs list

    Args:
        inputs (list): list of inputs passed from input_steps and input_data
    Returns:
        numpy array: exponent of the first element of the inputs list
    """
    return np.exp(take_first_inputs(inputs))


def sparse_hstack_inputs(inputs):
    """Stacks input arrays horizontally in a sparse manner

    It is useful when multiple inputs (for example features extracted via various transformers) need to be
    joined before passing them to the classifier.
    Using sparse version is advised when working with datasets that have a large number of 0-valued features.

    Args:
        inputs (list): list of inputs passed from input_steps and input_data

    Returns:
        sparse array: sparse array of horizontally stacked inputs
    """
    return sparse.hstack(inputs)


def hstack_inputs(inputs):
    """Stacks input arrays horizontally

    It is useful when multiple inputs (for example features extracted via various transformers) need to be
    joined before passing them to the classifier.

    Args:
        inputs (list): list of inputs passed from input_steps and input_data

    Returns:
        numpy array: numpy array of horizontally stacked inputs
    """
    return np.hstack(inputs)


def vstack_inputs(inputs):
    """Stacks input arrays vertically

    Args:
        inputs (list): list of inputs passed from input_steps and input_data

    Returns:
        numpy array: numpy array of vertically stacked inputs
    """
    return np.vstack(inputs)


def stack_inputs(inputs, axis=0):
    """Stacks input arrays over the desired axis

    Args:
        inputs (list): list of inputs passed from input_steps and input_data
        axis (int): axis over which the inputs list should be stacked
    Returns:
        numpy array: numpy array of inputs stacked over the desired axis
    """
    stacked = np.stack(inputs, axis=axis)
    return stacked


def sum_inputs(inputs):
    """Stacks input arrays over the desired axis

    Args:
        inputs (list): list of inputs passed from input_steps and input_data
        axis (int): axis over which the inputs list should be stacked
    Returns:
        numpy array: numpy array of inputs stacked over the desired axis
    """
    stacked = np.stack(inputs, axis=0)
    return np.sum(stacked, axis=0)


def average_inputs(inputs):
    """Calculates average over the inputs

    Args:
        inputs (list): list of inputs passed from input_steps and input_data
    Returns:
        numpy array: averaged output of shape inputs[0].shape
    """
    stacked = np.stack(inputs, axis=0)
    return np.mean(stacked, axis=0)
