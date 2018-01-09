""" Initializer functions. """
import tensorflow as tf


def uniform(name, shape, scale=0.05):
    """ Creates a variable which is randomly initiated between [-scale, +scale]

    :param name: The name of the variable.
    :param shape: The shape of the variable to create.
    :param scale: The minimum and maximum value of the random uniform distribution.
    :return: A variable who is randomly uniformly distributed between [-scale, +scale].
    """
    initial = tf.random_uniform_initializer(minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.get_variable(name=name, shape=shape, initializer=initial)


def glorot(name, shape):
    """ Creates a variable which is initiated using the Glorot & Bengio uniform method.

    :param name: The name of the variable.
    :param shape: The shape of the variable to create.
    :return: A variable who is initiated using Glorot uniform.
    """
    if len(shape) >= 2:
        fan_in, fan_out = shape[-2], shape[-1]
    else:
        fan_in, fan_out = shape[-1], shape[-1]
    init_range = tf.sqrt(6. / (fan_in + fan_out))
    initial = tf.random_uniform_initializer(minval=-1. * init_range, maxval=init_range, dtype=tf.float32)
    return tf.get_variable(name=name, shape=shape, initializer=initial)


def he(name, shape):
    """ Creates a variable which is initiated using the He normal method.

    :param name: The name of the variable.
    :param shape: The shape of the variable to create.
    :return: A variable who is initiated using He normal.
    """
    fan_in = shape[-2] if len(shape) >= 2 else shape[-1]
    init_range = tf.sqrt(2. / fan_in)
    initial = tf.random_normal_initializer(mean=0., stddev=init_range, dtype=tf.float32)
    return tf.get_variable(name=name, shape=shape, initializer=initial)


def zeros(name, shape):
    """ Creates a variable which is initiated to zeros.

    :param name: The name of the variable.
    :param shape: The shape of the variable.
    :return: A zero-initiated variable.
    """
    return constant(name, shape, value=0., dtype=tf.float32)


def ones(name, shape):
    """ Creates a variable which is initiated to ones.

    :param name: The name of the variable.
    :param shape: The shape of the variable.
    :return: A one-initiated variable.
    """
    return constant(name, shape, value=1., dtype=tf.float32)


def constant(name, shape, value, dtype=tf.float32):
    """ Creates a variable which is initiated to a constant value.

    :param name: The name of the variable.
    :param shape: The shape of the variable.
    :param value: The constant value of the tensor.
    :param dtype: The data type.
    :return: A constant-initiated variable.
    """
    initial = tf.constant_initializer(value=value, dtype=dtype)
    return tf.get_variable(name=name, shape=shape, initializer=initial)
