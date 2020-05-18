from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys
import os
import math
from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.distributions import Normal

import tensorflow_compression as tfc

from tensorflow import keras as keras

from tensorflow.python import debug as tf_debug

epsilon = 1e-10

BATCH_SIZE = 8

def ema_var(var, ema):
    assert isinstance(ema, tf.train.ExponentialMovingAverage)
    if isinstance(var, list):
        return [ema.average(v) for v in var]
    return ema.average(var)


@tf.custom_gradient
def differentiable_round(x):
    """ customized differentiable round operation"""
    def grad(dy):
        return dy
    return tf.round(x), grad


@tf.custom_gradient
def differentiable_quant(x, n_bins=255):
    """ customized differentiable round operation"""
    def grad(dy):
        return dy
    # x is a float 
    out = tf.cast(tf.round(x * n_bins), tf.float32) / n_bins
    return out, grad


@tf.custom_gradient
def differentiable_cast_to_float(x):
    """ customized differentiable round operation"""
    def grad(dy):
        return dy
    return tf.cast(x, tf.float32), grad


@tf.custom_gradient
def differentiable_cast_to_int(x):
    """ customized differentiable round operation"""
    def grad(dy):
        return dy
    return tf.cast(x, tf.int32), grad


class AnalysisTransform(keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters // 8, (5, 5), name="layer_-1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_-1")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_0")),
            # tfc.SignalConv2D(
            #     self.num_filters, (9, 9), name="layer_0", corr=True, strides_down=4,
            #     padding="same_zeros", use_bias=True,
            #     activation=tfc.GDN(name="gdn_0")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_1")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
                padding="same_zeros", use_bias=False,
                activation=None),
            ]
        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class SynthesisTransform(keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(SynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_0", inverse=True)),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True)),
            # tfc.SignalConv2D(
            #     3, (9, 9), name="layer_2", corr=False, strides_up=4,
            #     padding="same_zeros", use_bias=True,
            #     activation=None),
            tfc.SignalConv2D(
                self.num_filters // 8, (5, 5), name="layer_2", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True)),
            tfc.SignalConv2D(
                3, (5, 5), name="layer_3", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]
        super(SynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class SeqBlock(keras.layers.Layer):
    def __init__(self, num_filters, channel_out, kernel_size=3, residual=False, nin=True, norm="bn", an_ratio=1, *args, **kwargs):
        super(SeqBlock, self).__init__(*args, **kwargs)
        self.residual = residual
        self.nin = nin
        self.norm = norm
        self.conv1 = keras.layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size),
            padding='same', data_format='channels_last', kernel_initializer='glorot_normal')
        if norm == 'bn':
            self.norm1 = keras.layers.BatchNormalization()
            if self.nin:
                self.norm2 = keras.layers.BatchNormalization()
            self.norm3 = keras.layers.BatchNormalization()
            self.lrelu = keras.layers.LeakyReLU(0.2)
        elif norm == 'gdn':
            self.gdn1 = tfc.GDN(name="gdn_1")
            if self.nin:
                self.gdn2 = tfc.GDN(name="gdn_2")
            self.gdn3 = tfc.GDN(name="gdn_3")
        elif norm == 'an':
            # compute_ch_in = lambda ch_in: (int(ch_in / an_ratio), int(ch_in - ch_in / an_ratio))
            self.norm1 = ActNorm()
            if self.nin:
                self.norm2 = ActNorm()
            self.norm3 = ActNorm()
            self.lrelu = keras.layers.LeakyReLU(0.2)

        if self.nin:
            self.conv2 = keras.layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size),
                padding='same', data_format='channels_last', kernel_initializer='glorot_normal')
        self.conv3 = keras.layers.Conv2D(filters=channel_out, kernel_size=(kernel_size, kernel_size),
            padding='same', data_format='channels_last', kernel_initializer='glorot_normal')
        if self.residual:
            self.conv_skip_connection = keras.layers.Conv2D(filters=channel_out, kernel_size=(3, 3),
                padding='same', data_format='channels_last', kernel_initializer='glorot_normal')

    def call(self, x):
        if self.norm == 'bn':
            out = self.lrelu(self.norm1(self.conv1(x)))
            if self.nin:
                out = self.lrelu(self.norm2(self.conv2(out)))
            out = self.norm3(self.conv3(out))
        elif self.norm == 'an':
            out = self.lrelu(self.norm1(self.conv1(x)))
            if self.nin:
                out = self.lrelu(self.norm2(self.conv2(out)))
            out = self.norm3(self.conv3(out))
        elif self.norm == 'gdn':
            out = self.gdn1(self.conv1(x))
            if self.nin:
                out = self.gdn2(self.conv2(out))
            out = self.gdn3(self.conv3(out))
        else:
            out = self.conv1(x)
            if self.nin:
                out = self.conv2(out)
            out = self.conv3(out)
        if self.residual:
            out = self.conv_skip_connection(x) + out
        return out


# class DenseBlock(keras.layers.Layer):
#     def __init__(self, channel_out, gc=8, *args, **kwargs):
#         super(DenseBlock, self).__init__(*args, **kwargs)
#         self.conv1 = keras.layers.Conv2D(filters=gc, kernel_size=(5, 5), 
#             padding='same', data_format='channels_last', kernel_initializer='glorot_normal')
#         self.conv2 = keras.layers.Conv2D(filters=gc, kernel_size=(5, 5),
#             padding='same', data_format='channels_last', kernel_initializer='glorot_normal')
#         self.conv3 = keras.layers.Conv2D(filters=gc, kernel_size=(5, 5),
#             padding='same', data_format='channels_last', kernel_initializer='glorot_normal')
#         self.conv4 = keras.layers.Conv2D(filters=gc, kernel_size=(5, 5),
#             padding='same', data_format='channels_last', kernel_initializer='glorot_normal')
#         self.conv5 = keras.layers.Conv2D(filters=channel_out, kernel_size=(5, 5),
#             padding='same', data_format='channels_last', kernel_initializer='zeros')
#         self.lrelu = keras.layers.LeakyReLU(0.2)

#     def call(self, x):
#         x1 = self.conv1(x)
#         x1 = self.lrelu(x1)
#         # x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(tf.concat([x, x1], -1)))
#         x3 = self.lrelu(self.conv3(tf.concat([x, x1, x2], -1)))
#         x4 = self.lrelu(self.conv4(tf.concat([x, x1, x2, x3], -1)))
#         x5 = self.conv5(tf.concat([x, x1, x2, x3, x4], -1))
#         return x5


class DenseLayer(keras.layers.Layer):
    def __init__(self, channel_out, activation, kernel_size):
        super(DenseLayer, self).__init__()
        self.channel_out = channel_out
        self.activation = activation
        self.kernel_size = kernel_size

    def build(self, input_shape):
        super(DenseLayer, self).build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        last_dim = input_shape[-1]
        self.model = keras.Sequential
        (
            [
                keras.layers.Conv2D(filters=last_dim, kernel_size=(1, 1),
                        padding='same', data_format='channels_last', 
                        kernel_initializer='glorot_normal'), 
                self.activation(), 
                keras.layers.Conv2D(filters=self.channel_out, 
                        kernel_size=(self.kernel_size, self.kernel_size),
                        padding='same', data_format='channels_last', 
                        kernel_initializer='glorot_normal'),
                self.activation()
            ]
        )
        # self.built = True

    def call(self, x):
        h = self.model(x)
        return tf.concat([x, h], axis=-1)

class DenseBlock(keras.layers.Layer):
    def __init__(self, channel_out, depth=12, activation=keras.layers.LeakyReLU, kernel_size=3):
        super(DenseBlock, self).__init__()
        self.channel_out = channel_out
        self.depth = depth
        self.activation = activation
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        super(DenseBlock, self).build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        channel_cur = input_shape[-1]
        future_growth = self.channel_out - channel_cur
        self.model = keras.Sequential()
        for d in range(self.depth):
            growth = future_growth // (self.depth - d)
            self.model.add(DenseLayer(growth, self.activation, 
                            self.kernel_size))
            channel_cur += growth
            future_growth -= growth
        # self.built = True
        
    def call(self, x):
        return self.model(x)


class IntInvBlock(keras.layers.Layer):
    def __init__(self, func, channel_split_ratio, num_filters=128, 
                 clamp=1., kernel_size=3, residual=False, nin=True, 
                 norm='bn', n_ops=3):
        super(IntInvBlock, self).__init__()
        self.func = func
        self.channel_split_ratio = channel_split_ratio
        self.num_filters = num_filters
    
    def build(self, input_shape):
        super(IntInvBlock, self).build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        last_dim = input_shape[-1]
        # print(last_dim)
        self.split_len1 = last_dim // self.channel_split_ratio
        self.split_len2 = last_dim - self.split_len1
        if isinstance(self.func, DenseBlock):
            self.F = DenseBlock(self.split_len1)
            self.G = DenseBlock(self.split_len2)
        else:
            self.F = SeqBlock(self.num_filters, self.split_len1)
            self.G = SeqBlock(self.num_filters, self.split_len2)
        # self.built = True

    def call(self, x, rev=False, quant=True):
        x1 = x[:, :, :, :self.split_len1]
        x2 = x[:, :, :, self.split_len1:]
        if not rev:
            if quant:
                y1 = x1 + differentiable_quant(self.F(x2))
                y2 = x2 + differentiable_quant(self.G(y1))
            else:
                y1 = x1 + self.F(x2)
                y2 = x2 + self.G(y1)
        else:
            if quant:
                y2 = x2 - differentiable_quant(self.G(x1))
                y1 = x1 - differentiable_quant(self.F(y2))
            else:
                y2 = x2 - self.G(x1)
                y1 = x1 - self.F(y2)
        return tf.concat([y1, y2], -1)

    def jacobian(self, x, rev=False):
        return 0.


class InvBlockExp(keras.layers.Layer):
    def __init__(self, func, channel_split_ratio, num_filters=128, 
                 clamp=1., kernel_size=3, residual=False, nin=True, 
                 norm='bn', n_ops=3, depth=12):
        super(InvBlockExp, self).__init__()
        # assert isinstance(func, (DenseBlock, SeqBlock))
        self.func = func
        self.channel_split_ratio = channel_split_ratio
        self.num_filters = num_filters
        self.clamp = clamp
        self.kernel_size = kernel_size
        self.residual = residual
        self.nin = nin
        self.norm = norm
        self.n_ops = n_ops
        self.depth = depth

    def build(self, input_shape):
        super(InvBlockExp, self).build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        last_dim = input_shape[-1]
        self.split_len1 = last_dim // self.channel_split_ratio
        self.split_len2 = last_dim - self.split_len1
        if isinstance(self.func, DenseBlock):
            self.F = DenseBlock(self.split_len1, depth=self.depth, 
                                kernel_size=self.kernel_size)
            self.G = DenseBlock(self.split_len2, depth=self.depth, 
                                kernel_size=self.kernel_size)
            self.H = DenseBlock(self.split_len2, depth=self.depth, 
                                kernel_size=self.kernel_size)
            if self.n_ops == 4:
                self.I = DenseBlock(self.split_len1, kernel_size=self.kernel_size)
        else:
            self.F = SeqBlock(self.num_filters, self.split_len1, self.kernel_size, 
                              residual=self.residual, nin=self.nin, norm=self.norm)
            self.G = SeqBlock(self.num_filters, self.split_len2, self.kernel_size, 
                              residual=self.residual, nin=self.nin, norm=self.norm)
            self.H = SeqBlock(self.num_filters, self.split_len2, self.kernel_size, 
                              residual=self.residual, nin=self.nin, norm=self.norm)
            if self.n_ops == 4:
                self.I = SeqBlock(self.num_filters, self.split_len1, self.kernel_size, 
                                  residual=self.residual, nin=self.nin, norm=self.norm)
        # self.built = True

    def call(self, x, rev=False):
        # x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        x1 = x[:, :, :, :self.split_len1]
        x2 = x[:, :, :, self.split_len1:(self.split_len1 + self.split_len2)]
        if not rev:
            if self.n_ops == 4:
                self.s1 = self.clamp * (keras.activations.sigmoid(self.I(x2)) * 2 - 1) + epsilon
                y1 = tf.math.multiply(x1, tf.math.exp(self.s1)) + self.F(x2)
            else:
                y1 = x1 + self.F(x2)
            self.s2 = self.clamp * (keras.activations.sigmoid(self.H(y1)) * 2 - 1) + epsilon
            y2 = tf.math.multiply(x2, tf.math.exp(self.s2)) + self.G(y1)
        else:
            self.s2 = self.clamp * (keras.activations.sigmoid(self.H(x1)) * 2 - 1) + epsilon
            y2 = tf.math.divide(x2 - self.G(x1), tf.math.exp(self.s2))
            if self.n_ops == 4:
                self.s1 = self.clamp * (keras.activations.sigmoid(self.I(y2)) * 2 - 1) + epsilon
                y1 = tf.math.divide(x1 - self.F(y2), tf.math.exp(self.s1))
            else:
                y1 = x1 - self.F(y2)

        return tf.concat([y1, y2], -1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = tf.reduce_sum(self.s2)
            if self.n_ops == 4:
                jac += tf.reduce_sum(self.s1)
        else:
            jac = -tf.reduce_sum(self.s2)
            if self.n_ops == 4: 
                jac -= tf.reduce_sum(self.s1)

        return jac / 8


class HaarDownsampling(keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super(HaarDownsampling, self).__init__(*args, **kwargs)
        self.last_jac = 0

    def build(self, input_shape):
        super(HaarDownsampling, self).build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        self.channel_in = input_shape[-1]
        # self.haar_weights = np.ones([4, 1, 2, 2])
        self.haar_weights = np.ones([2, 2, 1, 4])
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 1, 0, 1] = -1
        self.haar_weights[0, 1, 0, 2] = -1
        self.haar_weights[1, 1, 0, 2] = -1
        self.haar_weights[0, 1, 0, 3] = -1
        self.haar_weights[1, 0, 0, 3] = -1

        self.haar_weights = tf.constant_initializer(self.haar_weights)
        self.conv = keras.layers.Conv2D(filters=4, kernel_size=2, strides=2, 
                    padding='valid', kernel_initializer=self.haar_weights, 
                    use_bias=False, trainable=False)
        self.conv_transpose = keras.layers.Conv2DTranspose(filters=1, kernel_size=2, 
                    strides=2, padding='valid', kernel_initializer=self.haar_weights, 
                    use_bias=False, trainable=False)
        # self.built = True

    def call(self, x, rev=False):
        self.x_shape = x.get_shape().as_list()[1:]
        # print("x_shape: {}".format(self.x_shape))
        self.elements = np.prod(self.x_shape)
        if not rev:
            self.last_jac = self.elements / 4 * np.log(1/16.)
            # print("x shape: {} and channel in: {}".format(x.get_shape().as_list(), self.channel_in))
            x_s = tf.split(x, self.channel_in, axis=-1, name='split')
            # x_s[0] = print_act_stats(x_s[0], " x_s[0] ")
            out = [self.conv(x_sub) / 4.0 for x_sub in x_s]
            out = tf.concat(out, axis=-1)
            # reshape
            out = tf.concat([out[:, :, :, i::4] for i in range(4)], -1)
            # out = print_act_stats(out, " out ")
            # x = tf.reshape(x, [-1, self.x_shape[0] // 2, 2, self.x_shape[1] // 2, 2, self.channel_in])
            # x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
            # x = tf.reshape(x, [-1, self.x_shape[0] // 2, self.x_shape[1] // 2, self.channel_in * 2 * 2])
            return out 
        else:
            self.last_jac = self.elements / 4 * np.log(16.)
            # reshape
            # x = tf.reshape(x, (-1, self.x_shape[0], self.x_shape[1], self.channel_in, 2, 2))
            # x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
            # x = tf.reshape(x, (-1, self.x_shape[0], self.x_shape[1], self.channel_in * 2**2))
            # x = print_act_stats(x, " x ")
            x = tf.concat([x[:, :, :, i::self.channel_in] for i in range(self.channel_in)], -1)
            x_s = tf.split(x, self.channel_in, axis=-1, name='split')
            out = [self.conv_transpose(x_sub) for x_sub in x_s]
            # out[0] = print_act_stats(out[0], " out[0] ")
            out = tf.concat(out, axis=-1)
            return out

    def jacobian(self, x, rev=False):
        return self.last_jac


class InvConv(keras.layers.Layer):
    def __init__(self, channel_in, init="identity", *args, **kwargs):
        super(InvConv, self).__init__(*args, **kwargs)
        self.channel_in = channel_in

        self.w_shape = [channel_in, channel_in]
        # sample a random orthogonal matrix or identity
        if init == "identity":
            w_init = np.identity(channel_in).astype('float32')
        else:
            w_init = np.linalg.qr(np.random.randn(
                    *self.w_shape))[0].astype('float32')
        w_init = tf.constant_initializer(w_init)
        self.w = self.add_weight(name="W", shape=[channel_in] * 2, 
                initializer=w_init, trainable=True)

    def call(self, x, rev=False):
        if not rev:
            _w = tf.reshape(self.w, [1, 1] + self.w_shape)
            out = tf.nn.conv2d(x, _w, [1, 1, 1, 1], 
                    'SAME', data_format='NHWC')
            return out
        else:
            _w = tf.matrix_inverse(self.w)
            _w = tf.reshape(_w, [1, 1] + self.w_shape)
            out = tf.nn.conv2d(x, _w, [1, 1, 1, 1],
                    'SAME', data_format='NHWC')
            return out
    
    def jacobian(self, x, rev=False):
        H, W = x.get_shape().as_list()[1:3]
        if not rev:
            jac = tf.cast(tf.log(abs(tf.matrix_determinant(
                tf.cast(self.w, 'float64')))), 'float32') * H * W
        else:
            jac = - tf.cast(tf.log(abs(tf.matrix_determinant(
                tf.cast(self.w, 'float64')))), 'float32') * H * W
        return tf.reduce_mean(jac)

class Permute(keras.layers.Layer):
    def __init__(self, n_channels=-1):
        super(Permute, self).__init__()
        self.n_channels = n_channels

    def compute_output_shape(self, input_shape):
        print("custom compute_output_shape called")
        shape = list(input_shape)
        shape[-1] *= 1
        return tuple(shape)
    
    def build(self, input_shape):
        super(Permute, self).build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        # print('Permute input shape: {}'.format(input_shape))
        channel_in = input_shape[-1]
        # assert isinstance(channel_in, int), "channel: {}".format(channel_in) 
        permutation = tf.range(channel_in)
        permutation = tf.random.shuffle(permutation)
        # permutation = np.arange(channel_in, dtype='int')
        # np.random.shuffle(permutation)
        permutation_inv = tf.range(channel_in)
        a = tf.concat([tf.expand_dims(p, -1) for p in [permutation, permutation_inv]], axis=-1)
        b = tf.add(tf.slice(a, [0, 0], [-1, 1]) * 10, tf.slice(a, [0, 1], [-1, 1]))
        reordered = tf.gather(a, tf.nn.top_k(b[:, 0], k=channel_in, sorted=False).indices)
        reordered = tf.reverse(reordered, axis=[0])
        permutation_inv = tf.squeeze(tf.slice(reordered, [0, 1], [-1, -1]))
        # permutation_inv = np.zeros(channel_in, dtype='int')
        # permutation_inv[permutation] = np.arange(channel_in, dtype='int')
        self.permutation = tf.constant(value=permutation, dtype=tf.int32)
        self.permutation_inv = tf.constant(value=permutation_inv, dtype=tf.int32)
        # self.built = True

    def call(self, x, rev=False):
        if not rev:
            x = tf.transpose(x, [3, 0, 1, 2])
            x = tf.gather(x, self.permutation)
            x = tf.transpose(x, [1, 2, 3, 0])
            # x = x[..., self.permutation]
        else:
            x = tf.transpose(x, [3, 0, 1, 2])
            x = tf.gather(x, self.permutation_inv)
            x = tf.transpose(x, [1, 2, 3, 0])
            # x = x[..., self.permutation_inv]
        return x

    def jacobian(self, x, rev=False):
        return 0


class IntDiscreteNet(keras.layers.Layer):
    def __init__(self, blk_type, num_filters, \
                downsample_type, n_levels, n_flows):
        super(IntDiscreteNet, self).__init__()
        self.num_filters = num_filters
        self.func = DenseBlock if blk_type == 'dense' else SeqBlock
        self.channel_split_ratio = 2
        self.func_downsample = HaarDownsampling if downsample_type == 'haar' \
                else SqueezeDownsampling
        self.n_levels = n_levels
        self.n_flows = n_flows
        self.operations = []
        self.num_ops = 0
    
    def build(self, input_shape):
        super(IntDiscreteNet, self).build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        h_in, w_in, channel_in = input_shape[1:]
        self.operations.append(self.func_downsample())
        self.num_ops += 1
        h_in, w_in, channel_in = h_in // 2, w_in // 2, channel_in * 4

        for level in range(self.n_levels):
            for _ in range(self.n_flows):
                # self.operations.append(Permute())
                self.operations.append(IntInvBlock(self.func, 
                                            self.channel_split_ratio, 
                                            num_filters=self.num_filters))
                self.num_ops += 1
            if level < self.n_levels - 1:
                self.operations.append(self.func_downsample())
                self.num_ops += 1
                h_in, w_in, channel_in = h_in // 2, w_in // 2, channel_in * 4
        self.built = True
    
    def call(self, x, rev=False, quant=True):
        jacobian = 0
        if not rev:
            for i in range(self.num_ops):
                if not quant and i == self.num_ops - 1:
                    x = self.operations[i](x, rev, quant)
                else:
                    x = self.operations[i](x, rev)
                # jacobian += self.operations[i].jacobian(x, rev)
        else:
            for i in reversed(range(self.num_ops)):
                if not quant and i == self.num_ops - 1:
                    x = self.operations[i](x, rev, quant)
                else:
                    x = self.operations[i](x, rev)
                # jacobian += self.operations[i].jacobian(x, rev)
        return x, jacobian


class InvCompressionNet(keras.Model):
    def __init__(self, channel_in, channel_out, blk_type, num_filters, \
                kernel_size, residual, nin, norm, n_ops, downsample_type, \
                inv_conv, inv_conv_init='identity', use_norm=False, \
                int_flow=False, depth=12):
        super(InvCompressionNet, self).__init__()
        assert downsample_type == "haar" or downsample_type == "squeeze"
        # self.upscale_log = upscale_log
        self.operations = []
        self.channel_in = channel_in
        self.channel_out = channel_out
        def compute_n_filters(cur_ch):
            ch_in = cur_ch * 2 // 3
            log_ch = math.log(ch_in, 2)
            if log_ch.is_integer():
                n_filters = max(ch_in * 2, num_filters)
            else:
                n_filters = max(2 ** math.ceil(log_ch), num_filters)

            # error checking
            assert cur_ch != 48 or n_filters == num_filters
            assert cur_ch != 192 or n_filters == 256
            assert cur_ch != 768 or n_filters == 1024

            return n_filters

        self.coupling_layer = InvBlockExp if not int_flow else IntInvBlock
        self.func = DenseBlock if blk_type == 'dense' else SeqBlock
        current_channel = self.channel_in
        for _ in range(2):
            if downsample_type == "haar":
                self.operations.append(HaarDownsampling(current_channel))
            else:
                self.operations.append(SqueezeDownsampling())
            current_channel *= 4
        if inv_conv:
            self.operations.append(InvConv(current_channel, inv_conv_init))
            if use_norm:
                self.operations.append(MultiActNorm(split_ratio=3))
        self.operations.append(self.coupling_layer(self.func, 3, 
                        num_filters=compute_n_filters(current_channel), 
                        kernel_size=kernel_size, residual=residual, nin=nin, 
                        norm=norm, n_ops=n_ops, depth=depth))
        if downsample_type == "haar":
            self.operations.append(HaarDownsampling(current_channel))
        else:
            self.operations.append(SqueezeDownsampling())
        current_channel *= 4
        if inv_conv:
            self.operations.append(InvConv(current_channel, inv_conv_init))
            if use_norm:
                self.operations.append(MultiActNorm(split_ratio=3))
        self.operations.append(self.coupling_layer(self.func, 3, 
                        num_filters=compute_n_filters(current_channel), 
                        kernel_size=kernel_size, residual=residual, nin=nin, 
                        norm=norm, n_ops=n_ops, depth=depth))
        if downsample_type == "haar":
            self.operations.append(HaarDownsampling(current_channel))
        else:
            self.operations.append(SqueezeDownsampling())
        current_channel *= 4
        if inv_conv:
            self.operations.append(InvConv(current_channel, inv_conv_init))
            if use_norm:
                self.operations.append(MultiActNorm(split_ratio=3))
        self.operations.append(self.coupling_layer(self.func, 3, 
                        num_filters=compute_n_filters(current_channel), 
                        kernel_size=kernel_size, residual=residual, nin=nin, 
                        norm=norm, n_ops=n_ops, depth=depth))
        
    def call(self, x, rev=False):
        out = []
        jacobian = 0
        if not rev:
            if isinstance(x, list):
                xx = x[-1]
            else:
                xx = x
            for i in range(len(self.operations)):
                xx = self.operations[i](xx, rev)
                jacobian += self.operations[i].jacobian(xx, rev)
            # assert xx.get_shape()[-1] == 768 and xx.get_shape()[-2] == 16, \
                # "x shape is {}\n".format(xx.get_shape())
        else:
            if isinstance(x, list) and x[-2].get_shape()[-1] != 1:
                xx = tf.concat([x[-1], x[-2]], axis=-1)
            else:
                xx = x
            for i in reversed(range(len(self.operations))):
                xx = self.operations[i](xx, rev)
                jacobian += self.operations[i].jacobian(xx, rev)
        out.append(xx)
        return out, jacobian


class InvHSRNet(keras.Model):
    def __init__(self, channel_in, channel_out, block_num, upscale_log, 
            blk_type, num_filters, use_inv_conv, kernel_size, residual):
        super(InvHSRNet, self).__init__()
        self.upscale_log = upscale_log
        self.operations = []
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.block_num = block_num        
        self.use_inv_conv = use_inv_conv
        current_channel = self.channel_in
        for i in range(self.upscale_log):
            b = HaarDownsampling(current_channel)
            self.operations.append(b)
            current_channel *= 4
            # if use_inv_conv:
            #     b = InvConv(current_channel)
            #     self.operations.append(b)
            for _ in range(self.block_num[i]):
                b = InvBlockExp(current_channel, current_channel // 4, blk_type, num_filters, kernel_size=kernel_size, residual=residual)
                self.operations.append(b)
            if use_inv_conv and i == self.upscale_log - 1:
                self.inv_conv = InvConv(current_channel)
            current_channel = self.channel_out[i]

    # if rev, x contains latent z1, z2 ... zn, LR, output contains LR_n-1, ..., LR1, HR
    # else x contains HR, output contains [LR1, z1], [LR2, z2], ... [LR, zn]
    # if multi_reconstruction (rev), x contains z1, z2, ..., zn, [LR1, ..., LR_n-1, LR], output contains LR_n-1, ..., LR1, [HR, HR(by LR_n-1), ...]
    def call(self, x, rev=False, multi_reconstruction=False):
        xx = x[-1]
        out = []
        # xx = print_act_stats(xx, "before all forward")
        if not rev:
            scnt = 0
            for cnt in range(self.upscale_log):
                bcnt = self.block_num[cnt] + 1
                # if self.use_inv_conv:
                #     bcnt += 1
                for i in range(bcnt):
                    # xx = self.operations.get_layer(index=(scnt + i))(xx, rev)
                    xx = self.operations[scnt + i](xx, rev)
                    # xx = print_act_stats(xx, "after forward operation {} at scale {}".format(i, cnt))
                if self.use_inv_conv and cnt == self.upscale_log - 1:
                    xx = self.inv_conv(xx, rev)
                out.append(xx)
                xx = tf.slice(xx, [0, 0, 0, 0], [-1, -1, -1, self.channel_out[cnt]])
                scnt += bcnt
        else:
            if not multi_reconstruction:
                scnt = len(self.operations) - (self.block_num[-1] + 1)
                # if self.use_inv_conv:
                #     scnt -= 1
                for cnt in reversed(range(self.upscale_log)):
                    bcnt = self.block_num[cnt] + 1
                    # if self.use_inv_conv:
                    #     bcnt += 1
                    if x[cnt].get_shape()[-1] != 1:
                        print(x[cnt].get_shape())
                        xx = tf.concat([xx, x[cnt]], -1)
                    if self.use_inv_conv and cnt == self.upscale_log - 1:
                        xx = self.inv_conv(xx, rev)
                    for i in reversed(range(bcnt)):
                        # xx = print_act_stats(xx, "before reverse operation {} at scale {}".format(i, cnt))
                        xx = self.operations[scnt + i](xx, rev)
                    out.append(xx)
                    scnt -= bcnt
                # xx = print_act_stats(xx, "last reverse operation")
                out.append(xx)
            else:
                HR_set = []
                LRs = xx
                sscnt = len(self.operations) - (self.block_num[-1] + 1)
                for LR_start in reversed(range(self.upscale_log)):
                    xx = LRs[LR_start]
                    scnt = sscnt
                    for cnt in reversed(range(LR_start + 1)):
                        bcnt = self.block_num[cnt] + 1
                        xx = tf.concat([xx, x[cnt]], -1)
                        for i in reversed(range(bcnt)):
                            xx = self.operations[scnt + i](xx, rev)
                        if LR_start == self.upscale_log - 1:
                            out.append(xx)
                        scnt -= bcnt
                    HR_set.append(xx)
                    sscnt -= self.block_num[LR_start] + 1
                out.append(HR_set)

        return out


class Conv1x1Gray(keras.layers.Layer):
    def __init__(self, rgb_type):
        super(Conv1x1Gray, self).__init__()

        if rgb_type == 'RGB':
            w_init = [0.299, 0.587, 0.114]
        elif rgb_type == 'BGR':
            w_init = [0.114, 0.587, 0.299]
        else:
            print("Error! Undefined RGB type!")
            exit(1)
        
        w_init = tf.constant_initializer(w_init)
        self.w = self.add_weight(name='W', shape=[3, 1], 
                initializer=w_init, trainable=False)

    def call(self, x):
        _w = tf.reshape(self.w, [1, 1] + [3, 1])
        out = tf.nn.conv2d(x, _w, [1, 1, 1, 1], 
                'SAME', data_format='NHWC')
        return out


class GrayScaleGuidance(keras.layers.Layer):
    """
    Implemented for y's gray scale guidance
    """
    def __init__(self, rgb_type, down_scale):
        super(GrayScaleGuidance, self).__init__()
        self.down_scale = down_scale
        self.conv_gray = Conv1x1Gray(rgb_type)
        self.downsample_ops = []
        current_channel = 1
        for i in range(self.down_scale):
            self.downsample_ops.append(HaarDownsampling(current_channel))
            current_channel *= 4
    
    def call(self, x):
        out = self.conv_gray(x)
        for i in range(self.down_scale):
            out = self.downsample_ops[i](out)
        return out


class SqueezeDownsampling(keras.layers.Layer):
    """
    Squeeze layer for downsampling
    """
    def __init__(self, factor=2):
        super(SqueezeDownsampling, self).__init__()
        assert factor >= 1
        self.factor = factor
    
    def call(self, x, rev=False):
        if self.factor == 1:
            return x
        shape = x.get_shape()
        height = int(shape[1])
        width = int(shape[2])
        n_channels = int(shape[3])

        if not rev:
            assert height % self.factor == 0 and width % self.factor == 0, \
                    "H: {}, W: {}".format(height, width)
            x = tf.reshape(x, [-1, height//self.factor, self.factor,
                            width//self.factor, self.factor, n_channels])
            x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
            out = tf.reshape(x, [-1, height//self.factor, width //
                            self.factor, n_channels*self.factor*self.factor])
        else:
            assert n_channels >= 4 and n_channels % 4 == 0
            x = tf.reshape(
                x, (-1, height, width, int(n_channels/self.factor**2), self.factor, self.factor))
            x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
            out = tf.reshape(x, (-1, int(height*self.factor),
                            int(width*self.factor), int(n_channels/self.factor**2)))
        return out

    def jacobian(self, x, rev=False):
        return 0


class ActNorm(keras.layers.Layer):
    """Actnorm, an affine reversible layer (Prafulla and Kingma, 2018).
    Weights use data-dependent initialization in which outputs have zero mean
    and unit variance per channel (last dimension). The mean/variance statistics
    are computed from the first batch of inputs.
    """

    def __init__(self, 
                 epsilon=keras.backend.epsilon(), 
                 ema=None, 
                 **kwargs):
        super(ActNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.ema = ema

    def build(self, input_shape):
        super(ActNorm, self).build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        last_dim = input_shape[-1]
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `ActNorm` '
                            'should be defined. Found `None`.')
        bias = self.add_weight('bias', [last_dim], dtype=tf.float32)
        log_scale = self.add_weight('log_scale', [last_dim], dtype=tf.float32)
        # Set data-dependent initializers.
        bias = bias.assign(self.bias_initial_value)
        with tf.control_dependencies([bias]):
            self.bias = bias
        log_scale = log_scale.assign(self.log_scale_initial_value)
        with tf.control_dependencies([log_scale]):
            self.log_scale = log_scale
        # self.built = True

    def __call__(self, inputs, *args, **kwargs):
        if not self.built:
            mean, variance = tf.nn.moments(
                inputs, axes=list(range(inputs.shape.ndims - 1)))
            self.bias_initial_value = -mean
            # TODO(trandustin): Optionally, actnorm multiplies log_scale by a fixed
            # log_scale factor (e.g., 3.) and initializes by
            # initial_value / log_scale_factor.
            self.log_scale_initial_value = tf.math.log(
                1. / (tf.sqrt(variance) + self.epsilon)) / 3.

        return super(ActNorm, self).__call__(inputs, *args, **kwargs)
        # if not isinstance(inputs, random_variable.RandomVariable):
        #     return super(ActNorm, self).__call__(inputs, *args, **kwargs)
        # return transformed_random_variable.TransformedRandomVariable(inputs, self)

    def call(self, inputs, rev=False):
        if not rev:
            return (inputs + self.bias) * tf.exp(self.log_scale * 3.)
        return inputs * tf.exp(-self.log_scale) - self.bias
    # def reverse(self, inputs):
    #     return inputs * tf.exp(-self.log_scale) - self.bias

    def jacobian(self, inputs, rev=False):
        """Returns log det | dx / dy | = num_events * sum log | scale |."""
        # del inputs  # unused
        # Number of events is number of all elements excluding the batch and
        # channel dimensions.
        num_events = tf.reduce_prod(tf.shape(inputs)[1:-1])
        log_det_jacobian =  tf.cast(num_events, 'float32') * tf.reduce_sum(self.log_scale)
        return log_det_jacobian


class MultiActNorm(keras.layers.Layer):
    def __init__(self, split_ratio):
        super(MultiActNorm, self).__init__()
        self.actnorm_1 = ActNorm()
        self.actnorm_2 = ActNorm()
        assert split_ratio >= 1
        self.split_ratio = split_ratio
    
    def call(self, x, rev=False):
        ch = x.get_shape().as_list()[-1]
        x1 = x[..., :ch // self.split_ratio]
        x2 = x[..., ch // self.split_ratio:]

        x1 = self.actnorm_1(x1, rev=rev)
        x2 = self.actnorm_2(x2, rev=rev)
        return tf.concat([x1, x2], -1)
    
    def jacobian(self, inputs, rev=False):
        ch = inputs.get_shape().as_list()[-1]
        x1 = inputs[..., :ch // self.split_ratio]
        x2 = inputs[..., ch // self.split_ratio:]
        return self.actnorm_1.jacobian(x1) + \
               self.actnorm_2.jacobian(x2)
