#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue August 7 2018

@author: ldalmeida
"""
"""Module implementing Unitary RNN Cell.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Diagonal unitary matrix
class DiagonalMatrix():
    def __init__(self, name, num_units):
        init_w = tf.random_uniform([num_units], minval=-np.pi, maxval=np.pi)
        self.w = tf.Variable(init_w, name=name)
        self.vec = tf.complex(tf.cos(self.w), tf.sin(self.w))

    # [batch_sz, num_units]
    def mul(self, z): 
        # [num_units] * [batch_sz, num_units] -> [batch_sz, num_units]
        return self.vec * z

def FFT(z):
    return tf.fft(z) 

def IFFT(z):
    return tf.ifft(z) 
	
def normalize(z):
    norm = tf.sqrt(tf.reduce_sum(tf.abs(z)**2))
    factor = (norm + 1e-6)
    return tf.complex(tf.real(z) / factor, tf.imag(z) / factor)

# z: complex[batch_sz, num_units]
# bias: real[num_units]
def modReLU(z, bias): # relu(|z|+b) * (z / |z|)
    norm = tf.abs(z)
    scale = tf.nn.relu(norm + bias) / (norm + 1e-6)
    scaled = tf.complex(tf.real(z)*scale, tf.imag(z)*scale)
    return scaled


class SimpleURNNCell(tf.contrib.rnn.RNNCell):
    """The most basic URNN cell.
    Args:
        num_units (int): The number of units in the LSTM cell, hidden layer size.
        num_in: Input vector size, input layer size.
    """

    def __init__(self, num_units, num_in, reuse=None):
        super(SimpleURNNCell, self).__init__(_reuse=reuse)
        # save class variables
        self._num_in = num_in
        self._num_units = num_units
        self._state_size = num_units*2 
        self._output_size = num_units*2

        # set up input -> hidden connection
        self.w_ih = tf.get_variable("w_ih", shape=[2*num_units, num_in], 
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.b_h = tf.Variable(tf.zeros(num_units), # state size actually
                                    name="b_h")

        # elementary unitary matrices to get the big one
        self.D = DiagonalMatrix("D", num_units)
    # needed properties

    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size # real

    @property
    def output_size(self):
        return self._output_size # real

    def call(self, inputs, state):
        """The most basic URNN cell.
        Args:
            inputs (Tensor - batch_sz x num_in): One batch of cell input.
            state (Tensor - batch_sz x num_units): Previous cell state: COMPLEX
        Returns:
        A tuple (outputs, state):
            outputs (Tensor - batch_sz x num_units*2): Cell outputs on the whole batch.
            state (Tensor - batch_sz x num_units): New state of the cell.
        """

        # prepare input linear combination
        inputs_mul = tf.matmul(inputs, tf.transpose(self.w_ih)) # [batch_sz, 2*num_units]
        inputs_mul_c = tf.complex( inputs_mul[:, :self._num_units], 
                                   inputs_mul[:, self._num_units:] ) 
        
        # prepare state linear combination (always complex!)
        state_c = tf.complex( state[:, :self._num_units], 
                              state[:, self._num_units:] ) 

        state_mul = self.D.mul(state_c)
        
        # calculate preactivation
        preact = inputs_mul_c + state_mul

        new_state_c = modReLU(preact, self.b_h) # [batch_sz, num_units] C

        new_state = tf.concat([tf.real(new_state_c), tf.imag(new_state_c)], 1) # [batch_sz, 2*num_units] R
        # outside network (last dense layer) is ready for 2*num_units -> num_out
        output = new_state
        # print("cell.call output:", output.shape, output.dtype)
        # print("cell.call new_state:", new_state.shape, new_state.dtype)

        return output, new_state
