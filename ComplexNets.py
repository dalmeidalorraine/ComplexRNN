import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.engine.topology import Layer
from keras import regularizers
from keras import initializers
from keras import activations

import numpy as np
import tensorflow as tf

## General Utility Functions
def channels_to_complex(X):
    return tf.complex(X[..., 0], X[..., 1])

def channels_to_complex_np(X):
    return X[..., 0] + 1j * X[..., 1]

def complex_to_channels(Z):
    RE = tf.real(Z)
    IM = tf.imag(Z)

    if Z.get_shape()[-1] == 1:
        RE = tf.squeeze(RE, [-1])
        IM = tf.squeeze(IM, [-1])

    return tf.stack([RE, IM], axis=-1)

def complex_to_channels_np(Z):
    RE = np.real(Z)
    IM = np.imag(Z)

    if Z.shape[-1] == 1:
        RE = np.squeeze(RE, (-1))
        IM = np.squeeze(IM, (-1))

    return np.stack([RE, IM], axis=-1)

def real_to_channels(X):
    # Create complex with zero imaginary part
    X_c = tf.complex(X, 0.)
    return complex_to_channels(X_c)

def real_to_channels_np(X):
    # Create complex with zero imaginary part
    X_c = X + 0.j
    return complex_to_channels_np(X_c)


def polar_to_rect(X):
    return complex_to_channels(tf.complex(X[..., 0], 0.) * tf.exp(1j * tf.complex(0., X[..., 1])))

def rect_to_polar(X):
    Z = channels_to_complex(X)
    R = tf.abs(Z)
    THETA = tf.angle(Z)

    if Z.shape[-1] == 1:
        R = tf.squeeze(R, (-1))
        THETA = tf.squeeze(THETA, (-1))

    return tf.stack([R, THETA], axis=-1)


def polar_to_rect_np(X):
    return complex_to_channels_np(X[..., 0] * np.exp(1j * X[..., 1]))

def rect_to_polar_np(X):
    Z = channels_to_complex_np(X)
    R = np.abs(Z)
    THETA = np.angle(Z)

    if Z.shape[-1] == 1:
        R = np.squeeze(R, (-1))
        THETA = np.squeeze(THETA, (-1))

    return np.stack([R, THETA], axis=-1)

## Sparsity functions
def num_less_than_eps(x, eps):
    return (x < eps).sum()

def num_abs_less_than_eps(x, eps):
    x_c = x[..., 0] + 1j*x[..., 1]
    return num_less_than_eps(abs(x_c), eps)

def zero_elems_less_than_eps(x, eps):
    arr = np.copy(x) # This avoids actually changing the argument x
    arr[np.where(arr < eps)] = 0
    return arr


## Regularizers
class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, W):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AmplitudeRegL1(Regularizer):
    def __init__(self, lamb=0.1):
        self.lamb = K.cast_to_floatx(lamb)

    def __call__(self, W):
        complex_W = channels_to_complex(W)
        return self.lamb * K.sum(tf.abs(complex_W))

    def get_config(self):
        return {'lamb': float(self.lamb),}


class AmplitudeRegL2(Regularizer):
    def __init__(self, lamb=0.1):
        self.lamb = K.cast_to_floatx(lamb)

    def __call__(self, W):
        complex_W = channels_to_complex(W)
        return self.lamb * K.sum(K.square(tf.abs(complex_W)))

    def get_config(self):
        return {'lamb': float(self.lamb),}


class PhaseReg(Regularizer):
    def __init__(self, lamb=0.1):
        self.lamb = K.cast_to_floatx(lamb)

    def __call__(self, W):
        complex_W = channels_to_complex(W)
        return K.sum(self.lamb * tf.abs(tf.angle(complex_W)))

    def get_config(self):
        return {'lamb': float(self.lamb),}


class UnitaryReg(Regularizer):
    def __init__(self, lamb=0.1):
        self.lamb = K.cast_to_floatx(lamb)

    def __call__(self, W):
        complex_W = channels_to_complex(W)
        complex_W_conj_T = K.transpose(tf.conj(complex_W))
        I = tf.eye(tf.shape(complex_W)[0], dtype="complex64")

        return self.lamb * K.sum(K.abs(complex_W @ complex_W_conj_T - I))

    def get_config(self):
        return {'lamb': float(self.lamb),}

class UnitaryReg2(Regularizer):
    def __init__(self, lamb=0.1):
        self.lamb = K.cast_to_floatx(lamb)

    def __call__(self, W):
        complex_W = channels_to_complex(W)
        complex_W_conj_T = K.transpose(tf.conj(complex_W))
        I = tf.eye(tf.shape(complex_W)[1], dtype="complex64")

        return self.lamb * K.sum(K.abs(complex_W_conj_T @ complex_W - I))

    def get_config(self):
        return {'lamb': float(self.lamb),}


class UnitaryAndL2Reg(Regularizer):
    def __init__(self, l_u=0., l_a=0.):
        self.l_u = K.cast_to_floatx(l_u)
        self.l_a = K.cast_to_floatx(l_a)

    def __call__(self, W):
        regularization = 0.
        if self.l_u:
            complex_W = channels_to_complex(W)
            complex_W_conj_T = K.transpose(tf.conj(complex_W))
            I = tf.eye(tf.shape(complex_W)[0], dtype="complex64")
            regularization += K.sum(self.l_u * K.abs(complex_W @ complex_W_conj_T - I))
        if self.l_a:
            regularization += K.sum(self.l_a * K.square(W))
        return regularization

    def get_config(self):
        return {'l_u': float(self.l_u),
                'l_a': float(self.l_a)}


def amplitude_reg_l1(l=0.1):
    return AmplitudeRegL1(lamb=l)

def amplitude_reg_l2(l=0.1):
    return AmplitudeRegL2(lamb=l)

def phase_reg(l=0.1):
    return PhaseReg(lamb=l)

def unitary_reg(l=0.1):
    return UnitaryReg(lamb=l)

def unitary_reg_2(l=0.1):
    return UnitaryReg2(lamb=l)

def unitary_and_l2_reg(l_u=0.1, l_a=0.1):
    return UnitaryAndL2Reg(l_u, l_a)



## Layers
# Learnable Hadamard Product
class Hadamard(Layer):

    def __init__(self, **kwargs):
        super(Hadamard, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,) + input_shape[1:],
                                      initializer='uniform',
                                      trainable=True)
        super(Hadamard, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, X):
        complex_X = channels_to_complex(X)
        complex_W = channels_to_complex(self.kernel)
        complex_res = complex_X @ complex_W    
        output = complex_to_channels(complex_res)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

# Complex Dense Layer
class ComplexDense(Layer):

    def __init__(self, output_dim,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
        super(ComplexDense, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim, 2),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      #constraint=self.kernel_constraint,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.output_dim, 2),
                                        initializer=self.bias_initializer,
                                        #regularizer=self.bias_regularizer,
                                        #constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None
        super(ComplexDense, self).build(input_shape)

    def call(self, X):
        # True Complex Multiplication (by channel combination)
        complex_X = channels_to_complex(X)
        complex_W = channels_to_complex(self.kernel)

        complex_res = complex_X @ complex_W
        
        if self.use_bias:
            complex_b = channels_to_complex(self.bias)
            complex_res = K.bias_add(complex_res, complex_b)
        
        output = complex_to_channels(complex_res)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, 2)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            #'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            #'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            #'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            #'kernel_constraint': constraints.serialize(self.kernel_constraint),
            #'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Amplitude(Layer):

    def __init__(self, **kwargs):
        super(Amplitude, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Amplitude, self).build(input_shape)

    def call(self, X):
        complex_X = channels_to_complex(X)
        output = tf.abs(complex_X)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])


### EXPERIMENTAL SECTION
# Hermitian Layer
class HermitianLayer(Layer):

    def __init__(self, output_dim,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 **kwargs):
        super(HermitianLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim, 2),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        # Called bias but really scalar multiplication of modes
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim, 2),
                                    initializer=self.bias_initializer,
                                    trainable=True)

        super(HermitianLayer, self).build(input_shape)

    def call(self, X):
        # True Complex Multiplication (by channel combination)
        complex_X = channels_to_complex(X)
        complex_W = channels_to_complex(self.kernel)
        complex_V = channels_to_complex(self.bias)
        complex_W_conj_T = tf.transpose(tf.conj(complex_W))

        complex_res = ((complex_X @ complex_W) * complex_V) @ complex_W_conj_T

        return complex_to_channels(complex_res)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        }
        base_config = super(HermitianLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))