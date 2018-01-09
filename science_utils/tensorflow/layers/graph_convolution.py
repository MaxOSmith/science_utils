""" Graph Convolution.

Reference: https://github.com/tkipf/gcn/
"""
import numpy as np
import tensorflow as tf

from science_utils.tensorflow.layers.initializers import he, zeros


def preprocess_adjacency(adjacency_matrix):
    """ Symmetrically normalize the adjacency matrix for graph convolutions.

    :param adjacency_matrix: A NxN adjacency matrix.
    :return: A normalized NxN adjacency matrix.
    """
    # Computing A^~ = A + I_N
    adj = adjacency_matrix
    adj_tilde = adj + np.eye(adj.shape[0])

    # Calculating the sum of each row
    sum_of_row = np.array(adj_tilde.sum(1))

    # Calculating the D tilde matrix ^ (-1/2)
    d_inv_sqrt = np.power(sum_of_row, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # Calculating the normalized adjacency matrix
    norm_adj = adj_tilde.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return np.array(norm_adj, dtype=np.float32)


class GraphConvolution(object):
    """ Performs a graph convolution (ArXiV 1609.02907) """
    count = 0

    def __init__(self, input_dim, output_dim, norm_adjacency,
                 activation_fn=tf.nn.relu, bias=False, name=None):
        """ Initializes the graph convolutional network.

        :param input_dim: The number of features per node in the input.
        :param output_dim: The number of features per node desired in the output.
        :param norm_adjacency: The sparse normalized adjacency matrix.
            (NxN matrix)
        :param dropout: The dropout rate (dropout will be used if > 0.).
        :param activation_fn: The activation function to use after the
            graph convolution.
        :param bias: Boolean flag that indicates we also want to include a
            bias term.
        :param name: The name of this layer
        """
        self.name = 'graph_convolution_{0:03}'.format(GraphConvolution.count) \
                    if name is None else name
        self.vars = {}
        self.activation_fn = activation_fn
        self.norm_adjacency = norm_adjacency
        self.bias = bias

        # Initializing variables
        with tf.variable_scope(name):
            self.vars['W'] = he('W', [81, input_dim, output_dim])
            if self.bias:
                self.vars['b'] = zeros('b', [output_dim])

        # Increasing count
        GraphConvolution.count += 1

    def __call__(self, inputs):
        """ Actually performs the graph convolution.

        :param inputs: The input feature matrix (b, N, in).
        :return: The activated output matrix (b, N, out).
        """
        norm_adjacency = tf.tile(
            tf.expand_dims(self.norm_adjacency, 0),
            [tf.shape(inputs)[0], 1, 1])

        # Performs the convolution
        pre_act = tf.transpose(inputs, perm=[1, 0, 2])
        pre_act = tf.matmul(pre_act, self.vars['W'])
        pre_act = tf.transpose(pre_act, perm=[1, 0, 2])
        pre_act = tf.matmul(norm_adjacency, pre_act)

        # Adds the bias
        if self.bias:
            pre_act += self.vars['b']

        # Performs activation and returns
        return self.activation_fn(pre_act)
