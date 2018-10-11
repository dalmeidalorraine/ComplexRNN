#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue August 7 2018

@author: ldlameida
"""
"""Module implementing Tunable Efficient Unitary RNN Cell.
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import RNNCell

def modrelu(z, b):
    z_norm = tf.sqrt(tf.square(tf.real(z)) + tf.square(tf.imag(z))) + 0.00001
    step1 = z_norm + b
    step2 = tf.complex(tf.nn.relu(step1), tf.zeros_like(z_norm))
    step3 = z/tf.complex(z_norm, tf.zeros_like(z_norm))       
    return tf.multiply(step3, step2)


def _complexrnn_param(hidden_size, capacity=2):
    """
    Create parameters and do the initial preparations
    """
    theta_phi_initializer = tf.random_uniform_initializer(-np.pi, np.pi)
    capacity_b = capacity//2
    capacity_a = capacity - capacity_b

    hidden_size_a = hidden_size//2
    hidden_size_b = (hidden_size-1)//2

    params_theta_0 = vs.get_variable("theta_0", [capacity_a, hidden_size_a], initializer=theta_phi_initializer)
    cos_theta_0 = tf.reshape(tf.cos(params_theta_0), [capacity_a, -1, 1])
    sin_theta_0 = tf.reshape(tf.sin(params_theta_0), [capacity_a, -1, 1])

    params_theta_1 = vs.get_variable("theta_1", [capacity_b, hidden_size_b], initializer=theta_phi_initializer)
    cos_theta_1 = tf.reshape(tf.cos(params_theta_1), [capacity_b, -1, 1])
    sin_theta_1 = tf.reshape(tf.sin(params_theta_1), [capacity_b, -1, 1])

    params_phi_0 = vs.get_variable("phi_0", [capacity_a, hidden_size_a], initializer=theta_phi_initializer)
    cos_phi_0 = tf.reshape(tf.cos(params_phi_0), [capacity_a, -1, 1])
    sin_phi_0 = tf.reshape(tf.sin(params_phi_0), [capacity_a, -1, 1])

    cos_list_0_re = tf.reshape(tf.concat([cos_theta_0, tf.multiply(cos_theta_0, cos_phi_0)], 2), [capacity_a, -1])
    cos_list_0_im = tf.reshape(tf.concat([tf.zeros_like(cos_theta_0), tf.multiply(cos_theta_0, sin_phi_0)], 2), [capacity_a, -1])
    if hidden_size_a*2 != hidden_size:
        cos_list_0_re = tf.concat([cos_list_0_re, tf.ones([capacity_a, 1])], 1)
        cos_list_0_im = tf.concat([cos_list_0_im, tf.zeros([capacity_a, 1])], 1)
    cos_list_0 = tf.complex(cos_list_0_re, cos_list_0_im)

    sin_list_0_re = tf.reshape(tf.concat([sin_theta_0, - tf.multiply(sin_theta_0, cos_phi_0)], 2), [capacity_a, -1])
    sin_list_0_im = tf.reshape(tf.concat([tf.zeros_like(sin_theta_0), - tf.multiply(sin_theta_0, sin_phi_0)], 2), [capacity_a, -1])
    if hidden_size_a*2 != hidden_size:
        sin_list_0_re = tf.concat([sin_list_0_re, tf.zeros([capacity_a, 1])], 1)
        sin_list_0_im = tf.concat([sin_list_0_im, tf.zeros([capacity_a, 1])], 1)
    sin_list_0 = tf.complex(sin_list_0_re, sin_list_0_im)

    params_phi_1 = vs.get_variable("phi_1", [capacity_b, hidden_size_b], initializer=theta_phi_initializer)
    cos_phi_1 = tf.reshape(tf.cos(params_phi_1), [capacity_b, -1, 1])
    sin_phi_1 = tf.reshape(tf.sin(params_phi_1), [capacity_b, -1, 1])

    cos_list_1_re = tf.reshape(tf.concat([cos_theta_1, tf.multiply(cos_theta_1, cos_phi_1)], 2), [capacity_b, -1])
    cos_list_1_re = tf.concat([tf.ones((capacity_b, 1)), cos_list_1_re], 1)
    cos_list_1_im = tf.reshape(tf.concat([tf.zeros_like(cos_theta_1), tf.multiply(cos_theta_1, sin_phi_1)], 2), [capacity_b, -1])
    cos_list_1_im = tf.concat([tf.zeros((capacity_b, 1)), cos_list_1_im], 1)
    if hidden_size_b*2 != hidden_size-1:
        cos_list_1_re = tf.concat([cos_list_1_re, tf.ones([capacity_b, 1])], 1)
        cos_list_1_im = tf.concat([cos_list_1_im, tf.zeros([capacity_b, 1])], 1)
    cos_list_1 = tf.complex(cos_list_1_re, cos_list_1_im)

    sin_list_1_re = tf.reshape(tf.concat([sin_theta_1, -tf.multiply(sin_theta_1, cos_phi_1)], 2), [capacity_b, -1])
    sin_list_1_re = tf.concat([tf.zeros((capacity_b, 1)), sin_list_1_re], 1)
    sin_list_1_im = tf.reshape(tf.concat([tf.zeros_like(sin_theta_1), -tf.multiply(sin_theta_1, sin_phi_1)], 2), [capacity_b, -1])
    sin_list_1_im = tf.concat([tf.zeros((capacity_b, 1)), sin_list_1_im], 1)
    if hidden_size_b*2 != hidden_size-1:
        sin_list_1_re = tf.concat([sin_list_1_re, tf.zeros([capacity_b, 1])], 1)
        sin_list_1_im = tf.concat([sin_list_1_im, tf.zeros([capacity_b, 1])], 1)
    sin_list_1 = tf.complex(sin_list_1_re, sin_list_1_im)

    if capacity_b != capacity_a:
        cos_list_1 = tf.concat([cos_list_1, tf.complex(tf.zeros([1, hidden_size]), tf.zeros([1, hidden_size]))], 0)
        sin_list_1 = tf.concat([sin_list_1, tf.complex(tf.zeros([1, hidden_size]), tf.zeros([1, hidden_size]))], 0)

    diag_vec = tf.reshape(tf.concat([cos_list_0, cos_list_1], 1), [capacity_a*2, hidden_size])
    off_vec = tf.reshape(tf.concat([sin_list_0, sin_list_1], 1), [capacity_a*2, hidden_size])

    if capacity_b != capacity_a:
        diag_vec = tf.slice(diag_vec, [0, 0], [capacity, hidden_size])
        off_vec = tf.slice(off_vec, [0, 0], [capacity, hidden_size])

    def _toTensorArray(elems):

        elems = tf.convert_to_tensor(elems)
        n = tf.shape(elems)[0]
        elems_ta = tf.TensorArray(dtype=elems.dtype, size=n, dynamic_size=False, infer_shape=True, clear_after_read=False)
        elems_ta = elems_ta.unstack(elems)
        return elems_ta

    diag_vec = _toTensorArray(diag_vec)
    off_vec = _toTensorArray(off_vec)

    omega = vs.get_variable("omega", [hidden_size], initializer=theta_phi_initializer)
    diag = tf.complex(tf.cos(omega), tf.sin(omega))

    return diag_vec, off_vec, diag, capacity


def _complexrnn_loop(state, capacity, diag_vec_list, off_vec_list, diag):
    """
    Complex RNN main loop, applying unitary matrix on input tensor
    """
    i = 0
    def layer_tunable(x, i):

        diag_vec = diag_vec_list.read(i)
        off_vec = off_vec_list.read(i)

        diag = tf.multiply(x, diag_vec)
        off = tf.multiply(x, off_vec)

        def even_input(off, size):

            def even_s(off, size):
                off = tf.reshape(off, [-1, size//2, 2])
                off = tf.reshape(tf.reverse(off, [2]), [-1, size])
                return off

            def odd_s(off, size):
                off, helper = tf.split(off, [size-1, 1], 1)
                size -= 1
                off = even_s(off, size)
                off = tf.concat([off, helper], 1)
                return off

            off = tf.cond(tf.equal(tf.mod(size, 2), 0), lambda: even_s(off, size), lambda: odd_s(off, size))
            return off

        def odd_input(off, size):
            helper, off = tf.split(off, [1, size-1], 1)
            size -= 1
            off = even_input(off, size)
            off = tf.concat([helper, off], 1)
            return off

        size = int(off.get_shape()[1])
        off = tf.cond(tf.equal(tf.mod(i, 2), 0), lambda: even_input(off, size), lambda: odd_input(off, size))

        layer_output = diag + off
        i += 1

        return layer_output, i

    layer_function = layer_tunable
    output, _ = tf.while_loop(lambda state, i: tf.less(i, capacity), layer_function, [state, i])

    if not diag is None:
        output = tf.multiply(output, diag)


    return output

class ComplexRNNCell(RNNCell):
    """Efficient Unitary Network Cell
    The implementation is based on: http://arxiv.org/abs/1612.05231.

    Args:
        hidden_size (Integer): The number of units in the RNN cell, hidden layer size.
        capacity (Integer) (Optional): Only works for tunable style.
    """

    def __init__(self, hidden_size, capacity=2, activation=modrelu):
        super(ComplexRNNCell, self).__init__()
        self._hidden_size = hidden_size
        self._activation = activation
        self._capacity = capacity

        self.diag_vec, self.off_vec, self.diag, self._capacity = _complexrnn_param(hidden_size, capacity)



    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def capacity(self):
        return self._capacity

    def __call__(self, inputs, state, scope=None):
        """The most basic EUNN cell.
        Args:
            inputs (Tensor - batch_sz x num_in): One batch of cell input.
            state (Tensor - batch_sz x num_units): Previous cell state
        Returns:
        A tuple (outputs, state):
            outputs (Tensor - batch_sz x num_units): Cell outputs on the whole batch: COMPLEX
            state (Tensor - batch_sz x num_units): New state of the cell: COMPLEX
        """
        with vs.variable_scope(scope or "complexrnn_cell"):

            state = _complexrnn_loop(state, self._capacity, self.diag_vec, self.off_vec, self.diag)
            V_init_val = np.sqrt(6.)/np.sqrt(10 + 10)
            input_matrix_init = tf.random_uniform_initializer(-V_init_val, V_init_val)
            input_matrix_re = vs.get_variable("U_re", [inputs.get_shape()[-1], self._hidden_size], initializer=input_matrix_init)
            input_matrix_im = vs.get_variable("U_im", [inputs.get_shape()[-1], self._hidden_size], initializer=input_matrix_init)
            inputs_re = tf.matmul(inputs, input_matrix_re)
            inputs_im = tf.matmul(inputs, input_matrix_im)
            inputs = tf.complex(inputs_re, inputs_im)

            bias = vs.get_variable("modReLUBias", [self._hidden_size], initializer=tf.constant_initializer())
            output = self._activation((inputs + state), bias)

        return output, output
