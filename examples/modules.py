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

import tensorflow_compression as tfc

from tensorflow import keras as keras

from tensorflow.python import debug as tf_debug

epsilon = 1e-10

BATCH_SIZE = 8


class AnalysisTransform(keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
                tfc.SignalConv2D(
                    self.num_filters, (9, 9), name="layer_0", corr=True, strides_down=4,
                    padding="same_zeros", use_bias=True,
                    activation=tfc.GDN(name="gdn_0")),
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
            tfc.SignalConv2D(
                3, (9, 9), name="layer_2", corr=False, strides_up=4,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]
        super(SynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class SeqBlock(keras.layers.Layer):
    def __init__(self, num_filters, channel_out, kernel_size, residual, nin, norm="bn", *args, **kwargs):
        super(SeqBlock, self).__init__(*args, **kwargs)
        self.residual = residual
        self.nin = nin
        self.norm = norm
        self.conv1 = keras.layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size),
            padding='same', data_format='channels_last', kernel_initializer='glorot_normal')
        if norm == 'bn':
            self.bn1 = keras.layers.BatchNormalization()
            if self.nin:
                self.bn2 = keras.layers.BatchNormalization()
            self.bn3 = keras.layers.BatchNormalization()
            self.lrelu = keras.layers.LeakyReLU(0.2)
        elif norm == 'gdn':
            self.gdn1 = tfc.GDN(name="gdn_1")
            if self.nin:
                self.gdn2 = tfc.GDN(name="gdn_2")
            self.gdn3 = tfc.GDN(name="gdn_3")

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
            out = self.lrelu(self.bn1(self.conv1(x)))
            if self.nin:
                out = self.lrelu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
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


class DenseBlock(keras.layers.Layer):
    def __init__(self, channel_out, gc=8, *args, **kwargs):
        super(DenseBlock, self).__init__(*args, **kwargs)
        self.conv1 = keras.layers.Conv2D(filters=gc, kernel_size=(5, 5), 
            padding='same', data_format='channels_last', kernel_initializer='glorot_normal')
        self.conv2 = keras.layers.Conv2D(filters=gc, kernel_size=(5, 5),
            padding='same', data_format='channels_last', kernel_initializer='glorot_normal')
        self.conv3 = keras.layers.Conv2D(filters=gc, kernel_size=(5, 5),
            padding='same', data_format='channels_last', kernel_initializer='glorot_normal')
        self.conv4 = keras.layers.Conv2D(filters=gc, kernel_size=(5, 5),
            padding='same', data_format='channels_last', kernel_initializer='glorot_normal')
        self.conv5 = keras.layers.Conv2D(filters=channel_out, kernel_size=(5, 5),
            padding='same', data_format='channels_last', kernel_initializer='zeros')
        self.lrelu = keras.layers.LeakyReLU(0.2)

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.lrelu(x1)
        # x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(tf.concat([x, x1], -1)))
        x3 = self.lrelu(self.conv3(tf.concat([x, x1, x2], -1)))
        x4 = self.lrelu(self.conv4(tf.concat([x, x1, x2, x3], -1)))
        x5 = self.conv5(tf.concat([x, x1, x2, x3, x4], -1))
        return x5


class InvBlockExp(keras.layers.Layer):
    def __init__(self, channel_num, channel_split_num, blk_type='dense', num_filters=128, 
                clamp=1., kernel_size=3, residual=False, nin=True, norm='bn', n_ops=3):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.n_ops = n_ops
        print("split_len1: {} and split_len2: {}".format(self.split_len1, self.split_len2))

        self.clamp = clamp

        assert blk_type == 'dense' or blk_type == 'seq'
        assert n_ops == 3 or n_ops == 4

        if blk_type == 'dense':
            self.F = DenseBlock(self.split_len1)
            self.G = DenseBlock(self.split_len2)
            self.H = DenseBlock(self.split_len2)
            if n_ops == 4:
                self.I = DenseBlock(self.split_len1)
        else:
            self.F = SeqBlock(num_filters, self.split_len1, kernel_size, residual=residual, nin=nin, norm=norm)
            self.G = SeqBlock(num_filters, self.split_len2, kernel_size, residual=residual, nin=nin, norm=norm)
            self.H = SeqBlock(num_filters, self.split_len2, kernel_size, residual=residual, nin=nin, norm=norm)
            if n_ops == 4:
                self.I = SeqBlock(num_filters, self.split_len1, kernel_size, residual=residual, nin=nin, norm=norm)

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
    def __init__(self, channel_in, *args, **kwargs):
        super(HaarDownsampling, self).__init__(*args, **kwargs)
        self.channel_in = channel_in

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
    def __init__(self, channel_in, *args, **kwargs):
        super(InvConv, self).__init__(*args, **kwargs)
        self.channel_in = channel_in

        self.w_shape = [channel_in, channel_in]
        # sample a random orthogonal matrix
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


class InvCompressionNet(keras.Model):
    def __init__(self, channel_in, channel_out, blk_type, num_filters, \
                kernel_size, residual, nin, norm, n_ops, downsample_type, inv_conv):
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

        current_channel = self.channel_in
        for _ in range(2):
            if downsample_type == "haar":
                self.operations.append(HaarDownsampling(current_channel))
            else:
                self.operations.append(SqueezeDownsampling())
            current_channel *= 4
        if inv_conv:
            self.operations.append(InvConv(current_channel))
        self.operations.append(InvBlockExp(current_channel, current_channel // 3, 
                        blk_type, num_filters=compute_n_filters(current_channel), 
                        kernel_size=kernel_size, residual=residual, nin=nin, norm=norm, n_ops=n_ops))

        if downsample_type == "haar":
            self.operations.append(HaarDownsampling(current_channel))
        else:
            self.operations.append(SqueezeDownsampling())
        current_channel *= 4
        if inv_conv:
            self.operations.append(InvConv(current_channel))
        self.operations.append(InvBlockExp(current_channel, current_channel // 3, 
                        blk_type, num_filters=compute_n_filters(current_channel), 
                        kernel_size=kernel_size, residual=residual, nin=nin, norm=norm, n_ops=n_ops))

        if downsample_type == "haar":
            self.operations.append(HaarDownsampling(current_channel))
        else:
            self.operations.append(SqueezeDownsampling())
        current_channel *= 4
        if inv_conv:
            self.operations.append(InvConv(current_channel))
        self.operations.append(InvBlockExp(current_channel, current_channel // 3, 
                        blk_type, num_filters=compute_n_filters(current_channel), 
                        kernel_size=kernel_size, residual=residual, nin=nin, norm=norm, n_ops=n_ops))
        
    def call(self, x, rev=False):
        out = []
        jacobian = 0
        if not rev:
            xx = x[-1]
            for i in range(len(self.operations)):
                xx = self.operations[i](xx, rev)
                jacobian += self.operations[i].jacobian(xx, rev)
            # assert xx.get_shape()[-1] == 768 and xx.get_shape()[-2] == 16, \
                # "x shape is {}\n".format(xx.get_shape())
        else:
            if x[-2].get_shape()[-1] != 1:
                xx = tf.concat([x[-1], x[-2]], axis=-1)
            for i in reversed(range(len(self.operations))):
                xx = self.operations[i](xx, rev)
                jacobian += self.operations[i].jacobian(xx, rev)
        out.append(xx)
        return out, jacobian

    # # if rev, x contains latent z1, z2 ... zn, LR, output contains LR_n-1, ..., LR1, HR
    # # else x contains HR, output contains [LR1, z1], [LR2, z2], ... [LR, zn]
    # # if multi_reconstruction (rev), x contains z1, z2, ..., zn, [LR1, ..., LR_n-1, LR], output contains LR_n-1, ..., LR1, [HR, HR(by LR_n-1), ...]
    # def call(self, x, rev=False, multi_reconstruction=False):
    #     xx = x[-1]
    #     out = []
    #     # xx = print_act_stats(xx, "before all forward")
    #     if not rev:
    #         scnt = 0
    #         for i in range
    #         for cnt in range(self.upscale_log):
    #             bcnt = self.block_num[cnt] + 1
    #             # if self.use_inv_conv:
    #             #     bcnt += 1
    #             for i in range(bcnt):
    #                 # xx = self.operations.get_layer(index=(scnt + i))(xx, rev)
    #                 xx = self.operations[scnt + i](xx, rev)
    #                 # xx = print_act_stats(xx, "after forward operation {} at scale {}".format(i, cnt))
    #             if self.use_inv_conv and cnt == self.upscale_log - 1:
    #                 xx = self.inv_conv(xx, rev)
    #             out.append(xx)
    #             xx = tf.slice(xx, [0, 0, 0, 0], [-1, -1, -1, self.channel_out[cnt]])
    #             scnt += bcnt
    #     else:
    #         if not multi_reconstruction:
    #             scnt = len(self.operations) - (self.block_num[-1] + 1)
    #             # if self.use_inv_conv:
    #             #     scnt -= 1
    #             for cnt in reversed(range(self.upscale_log)):
    #                 bcnt = self.block_num[cnt] + 1
    #                 # if self.use_inv_conv:
    #                 #     bcnt += 1
    #                 if x[cnt].get_shape()[-1] != 1:
    #                     print(x[cnt].get_shape())
    #                     xx = tf.concat([xx, x[cnt]], -1)
    #                 if self.use_inv_conv and cnt == self.upscale_log - 1:
    #                     xx = self.inv_conv(xx, rev)
    #                 for i in reversed(range(bcnt)):
    #                     # xx = print_act_stats(xx, "before reverse operation {} at scale {}".format(i, cnt))
    #                     xx = self.operations[scnt + i](xx, rev)
    #                 out.append(xx)
    #                 scnt -= bcnt
    #             # xx = print_act_stats(xx, "last reverse operation")
    #             out.append(xx)
    #     return out


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
            assert height % self.factor == 0 and width % self.factor == 0
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

