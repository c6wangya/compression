# -*- coding: utf-8 -*-
# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model published in:
J. Ballé, V. Laparra, E.P. Simoncelli (2017):
"End-to-end Optimized Image Compression"
Int. Conf. on Learning Representations (ICLR), 2017
https://arxiv.org/abs/1611.01704

With patches from Victor Xing <victor.t.xing@gmail.com>

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys
import os

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    print(" file name: {}".format(filename))
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


def quantize_image(image):
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def write_png(filename, image):
    """Saves an image to a PNG file."""
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)


class AnalysisTransform(tf.keras.layers.Layer):
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

# class InvTransform(tf.keras.layers.Layer):
#     """The invertible transform"""
#     def __init__(self, num_filters, *args., **kwargs):
#         self.num_filters


# def f(name, h, width, n_out=None):
#     n_out = n_out or int(h.get_shape()[3])
#     with tf.variable_scope(name):
#         h = tf.nn.relu(Z.conv2d("l_1", h, width))
#         h = tf.nn.relu(Z.conv2d("l_2", h, width, filter_size=[1, 1]))
#         h = Z.conv2d_zeros("l_last", h, n_out)
#     return h
# 
# 
# # Simpler, new version
# @add_arg_scope
# def revnet2d_step( name, z, logdet, hps, reverse):
#     with tf.variable_scope(name):
# 
#         shape = Z.int_shape(z)
#         n_z = shape[3]
#         assert n_z % 2 == 0
# 
#         if not reverse:
# 
#             z, logdet = Z.actnorm("actnorm", z, logdet=logdet)
# 
#             if hps.flow_permutation == 0:
#                 z = Z.reverse_features("reverse", z)
#             elif hps.flow_permutation == 1:
#                 z = Z.shuffle_features("shuffle", z)
#             elif hps.flow_permutation == 2:
#                 z, logdet = invertible_1x1_conv("invconv", z, logdet)
#             else:
#                 raise Exception()
# 
#             z1 = z[:, :, :, :n_z // 2]
#             z2 = z[:, :, :, n_z // 2:]
# 
#             if hps.flow_coupling == 0:
#                 z2 += f("f1", z1, hps.width)
#             elif hps.flow_coupling == 1:
#                 h = f("f1", z1, hps.width, n_z)
#                 shift = h[:, :, :, 0::2]
#                 # scale = tf.exp(h[:, :, :, 1::2])
#                 scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
#                 z2 += shift
#                 z2 *= scale
#                 logdet += tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
#             else:
#                 raise Exception()
# 
#             z = tf.concat([z1, z2], 3)
# 
#         else:
# 
#             z1 = z[:, :, :, :n_z // 2]
#             z2 = z[:, :, :, n_z // 2:]
# 
#             if hps.flow_coupling == 0:
#                 z2 -= f("f1", z1, hps.width)
#             elif hps.flow_coupling == 1:
#                 h = f("f1", z1, hps.width, n_z)
#                 shift = h[:, :, :, 0::2]
#                 # scale = tf.exp(h[:, :, :, 1::2])
#                 scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
#                 z2 /= scale
#                 z2 -= shift
#                 logdet -= tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
#             else:
#                 raise Exception()
# 
#             z = tf.concat([z1, z2], 3)
# 
#             if hps.flow_permutation == 0:
#                 z = Z.reverse_features("reverse", z, reverse=True)
#             elif hps.flow_permutation == 1:
#                 z = Z.shuffle_features("shuffle", z, reverse=True)
#             elif hps.flow_permutation == 2:
#                 z, logdet = invertible_1x1_conv(
#                     "invconv", z, logdet, reverse=True)
#             else:
#                 raise Exception()
# 
#             z, logdet = Z.actnorm("actnorm", z, logdet=logdet, reverse=True)
# 
#     return z, logdet
# 
# 
# @add_arg_scope
# def revnet2d(name, z, logdet, hps, reverse=False):
#     with tf.variable_scope(name):
#         if not reverse:
#             for i in range(hps.depth):
#                 z, logdet = checkpoint(z, logdet)
#                 z, logdet = revnet2d_step(str(i), z, logdet, hps, reverse)
#             z, logdet = checkpoint(z, logdet)
#         else:
#             for i in reversed(range(hps.depth)):
#                 z, logdet = revnet2d_step(str(i), z, logdet, hps, reverse)
#     return z, logdet
# 
# 
# def codec(hps):
# 
#     def encoder(z, objective):
#         eps = []
#         for i in range(hps.n_levels):
#             z, objective = revnet2d(str(i), z, objective, hps)
#             if i < hps.n_levels-1:
#                 z, objective, _eps = split2d("pool"+str(i), z, objective=objective)
#                 eps.append(_eps)
#         return z, objective, eps
# 
#     def decoder(z, eps=[None]*hps.n_levels, eps_std=None):
#         for i in reversed(range(hps.n_levels)):
#             if i < hps.n_levels-1:
#                 z = split2d_reverse("pool"+str(i), z, eps=eps[i], eps_std=eps_std)
#             z, _ = revnet2d(str(i), z, 0, hps, reverse=True)
# 
#         return z
# 
#     return encoder, decoder
# 
# 
# def model(sess, hps, train_iterator, test_iterator, data_init):
# 
#     # Only for decoding/init, rest use iterators directly
#     with tf.name_scope('input'):
#         X = tf.placeholder(
#             tf.uint8, [None, hps.image_size, hps.image_size, 3], name='image')
#         Y = tf.placeholder(tf.int32, [None], name='label')
#         lr = tf.placeholder(tf.float32, None, name='learning_rate')
# 
#     encoder, decoder = codec(hps)
#     hps.n_bins = 2. ** hps.n_bits_x
# 
#     def preprocess(x):
#         x = tf.cast(x, 'float32')
#         if hps.n_bits_x < 8:
#             x = tf.floor(x / 2 ** (8 - hps.n_bits_x))
#         x = x / hps.n_bins - .5
#         return x
# 
#     def postprocess(x):
#         return tf.cast(tf.clip_by_value(tf.floor((x + .5)*hps.n_bins)*(256./hps.n_bins), 0, 255), 'uint8')
# 
#     def _f_loss(x, y, is_training, reuse=False):
# 
#         with tf.variable_scope('model', reuse=reuse):
#             y_onehot = tf.cast(tf.one_hot(y, hps.n_y, 1, 0), 'float32')
# 
#             # Discrete -> Continuous
#             objective = tf.zeros_like(x, dtype='float32')[:, 0, 0, 0]
#             z = preprocess(x)
#             z = z + tf.random_uniform(tf.shape(z), 0, 1./hps.n_bins)
#             objective += - np.log(hps.n_bins) * np.prod(Z.int_shape(z)[1:])
# 
#             # Encode
#             z = Z.squeeze2d(z, 2)  # > 16x16x12
#             z, objective, _ = encoder(z, objective)
# 
#             # Prior
#             hps.top_shape = Z.int_shape(z)[1:]
#             logp, _, _ = prior("prior", y_onehot, hps)
#             objective += logp(z)
# 
#             # Generative loss
#             nobj = - objective
#             bits_x = nobj / (np.log(2.) * int(x.get_shape()[1]) * int(
#                 x.get_shape()[2]) * int(x.get_shape()[3]))  # bits per subpixel
# 
#             # Predictive loss
#             if hps.weight_y > 0 and hps.ycond:
# 
#                 # Classification loss
#                 h_y = tf.reduce_mean(z, axis=[1, 2])
#                 y_logits = Z.linear_zeros("classifier", h_y, hps.n_y)
#                 bits_y = tf.nn.softmax_cross_entropy_with_logits_v2(
#                     labels=y_onehot, logits=y_logits) / np.log(2.)
# 
#                 # Classification accuracy
#                 y_predicted = tf.argmax(y_logits, 1, output_type=tf.int32)
#                 classification_error = 1 - \
#                     tf.cast(tf.equal(y_predicted, y), tf.float32)
#             else:
#                 bits_y = tf.zeros_like(bits_x)
#                 classification_error = tf.ones_like(bits_x)
# 
#         return bits_x, bits_y, classification_error
# 
#     def f_loss(iterator, is_training, reuse=False):
#         if hps.direct_iterator and iterator is not None:
#             x, y = iterator.get_next()
#         else:
#             x, y = X, Y
# 
#         bits_x, bits_y, pred_loss = _f_loss(x, y, is_training, reuse)
#         local_loss = bits_x + hps.weight_y * bits_y
#         stats = [local_loss, bits_x, bits_y, pred_loss]
#         global_stats = Z.allreduce_mean(
#             tf.stack([tf.reduce_mean(i) for i in stats]))
# 
#         return tf.reduce_mean(local_loss), global_stats
# 
#     feeds = {'x': X, 'y': Y}
#     m = abstract_model_xy(sess, hps, feeds, train_iterator,
#                           test_iterator, data_init, lr, f_loss)
# 
#     # === Sampling function
#     def f_sample(y, eps_std):
#         with tf.variable_scope('model', reuse=True):
#             y_onehot = tf.cast(tf.one_hot(y, hps.n_y, 1, 0), 'float32')
# 
#             _, sample, _ = prior("prior", y_onehot, hps)
#             z = sample(eps_std=eps_std)
#             z = decoder(z, eps_std=eps_std)
#             z = Z.unsqueeze2d(z, 2)  # 8x8x12 -> 16x16x3
#             x = postprocess(z)
# 
#         return x
# 
#     m.eps_std = tf.placeholder(tf.float32, [None], name='eps_std')
#     x_sampled = f_sample(Y, m.eps_std)
# 
#     def sample(_y, _eps_std):
#         return m.sess.run(x_sampled, {Y: _y, m.eps_std: _eps_std})
#     m.sample = sample
# 
#     if hps.inference:
#         # === Encoder-Decoder functions
#         def f_encode(x, y, reuse=True):
#             with tf.variable_scope('model', reuse=reuse):
#                 y_onehot = tf.cast(tf.one_hot(y, hps.n_y, 1, 0), 'float32')
# 
#                 # Discrete -> Continuous
#                 objective = tf.zeros_like(x, dtype='float32')[:, 0, 0, 0]
#                 z = preprocess(x)
#                 z = z + tf.random_uniform(tf.shape(z), 0, 1. / hps.n_bins)
#                 objective += - np.log(hps.n_bins) * np.prod(Z.int_shape(z)[1:])
# 
#                 # Encode
#                 z = Z.squeeze2d(z, 2)  # > 16x16x12
#                 z, objective, eps = encoder(z, objective)
# 
#                 # Prior
#                 hps.top_shape = Z.int_shape(z)[1:]
#                 logp, _, _eps = prior("prior", y_onehot, hps)
#                 objective += logp(z)
#                 eps.append(_eps(z))
# 
#             return eps
# 
#         def f_decode(y, eps, reuse=True):
#             with tf.variable_scope('model', reuse=reuse):
#                 y_onehot = tf.cast(tf.one_hot(y, hps.n_y, 1, 0), 'float32')
# 
#                 _, sample, _ = prior("prior", y_onehot, hps)
#                 z = sample(eps=eps[-1])
#                 z = decoder(z, eps=eps[:-1])
#                 z = Z.unsqueeze2d(z, 2)  # 8x8x12 -> 16x16x3
#                 x = postprocess(z)
# 
#             return x
# 
#         enc_eps = f_encode(X, Y)
#         dec_eps = []
#         print(enc_eps)
#         for i, _eps in enumerate(enc_eps):
#             print(_eps)
#             dec_eps.append(tf.placeholder(tf.float32, _eps.get_shape().as_list(), name="dec_eps_" + str(i)))
#         dec_x = f_decode(Y, dec_eps)
# 
#         eps_shapes = [_eps.get_shape().as_list()[1:] for _eps in enc_eps]
# 
#         def flatten_eps(eps):
#             # [BS, eps_size]
#             return np.concatenate([np.reshape(e, (e.shape[0], -1)) for e in eps], axis=-1)
# 
#         def unflatten_eps(feps):
#             index = 0
#             eps = []
#             bs = feps.shape[0]
#             for shape in eps_shapes:
#                 eps.append(np.reshape(feps[:, index: index+np.prod(shape)], (bs, *shape)))
#                 index += np.prod(shape)
#             return eps
# 
#         # If model is uncondtional, always pass y = np.zeros([bs], dtype=np.int32)
#         def encode(x, y):
#             return flatten_eps(sess.run(enc_eps, {X: x, Y: y}))
# 
#         def decode(y, feps):
#             eps = unflatten_eps(feps)
#             feed_dict = {Y: y}
#             for i in range(len(dec_eps)):
#                 feed_dict[dec_eps[i]] = eps[i]
#             return sess.run(dec_x, feed_dict)
# 
#         m.encode = encode
#         m.decode = decode
# 
#     return m


class SynthesisTransform(tf.keras.layers.Layer):
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


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
        return average_grads


def inference(x, analysis_transform, entropy_bottleneck, synthesis_transform):
    """Evaluate the model performance by reconstructing an image"""
    x_shape = tf.shape(x)

    # Transform and compress the image.
    y = analysis_transform(x)

    # Transform the quantized image back (if requested).
    y_hat, likelihoods = entropy_bottleneck(y, training=False)
    x_hat = synthesis_transform(y_hat)
    x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]

    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

    # Total number of bits divided by number of pixels.
    eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

    # Bring both images back to 0..255 range.
    x *= 255
    x_hat = tf.clip_by_value(x_hat, 0, 1)
    x_hat = tf.round(x_hat * 255)

    mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
    psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

    return eval_bpp, mse, psnr, msssim, num_pixels


def train(args):
    """Trains the model."""

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_device)

    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    # Create input data pipeline.
    with tf.device("/cpu:0"):
        train_files = glob.glob(args.train_glob)
        if not train_files:
            raise RuntimeError(
                "No training images found with glob '{}'.".format(args.train_glob))
        train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
        train_dataset = train_dataset.shuffle(
            buffer_size=len(train_files)).repeat()
        train_dataset = train_dataset.map(
            read_png, num_parallel_calls=args.preprocess_threads)
        train_dataset = train_dataset.map(
            lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
        train_dataset = train_dataset.batch(args.batchsize)
        train_dataset = train_dataset.prefetch(32)

    num_pixels = args.batchsize * args.patchsize ** 2

    # Get training patch from dataset.
    x = train_dataset.make_one_shot_iterator().get_next()

    # Instantiate model.
    analysis_transform = AnalysisTransform(args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()
    synthesis_transform = SynthesisTransform(args.num_filters)

    # initialize optimizer
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

    """ multi-gpu support """
    # global_step = tf.get_variable(
    #     'global_step', [], initializer=tf.constant_initializer(0), 
    #     dtype=tf.int64, trainable=False)

    # x_split = tf.split(x, args.num_gpus)
    # tower_bpp, tower_mse = [], []
    # tower_main_grads, tower_aux_grads = [], []
    # tower_main_loss = []
    # with tf.variable_scope(tf.get_variable_scope()):
    #     for i in range(args.num_gpus):
    #         with tf.device('/gpu:%d' % i):
    #             with tf.name_scope('%s_%d' % ('tower', i)):
    #                 # Build autoencoder.
    #                 y = analysis_transform(x_split[i])
    #                 y_tilde, likelihoods = entropy_bottleneck(y, training=True)
    #                 x_tilde = synthesis_transform(y_tilde)
                
    #                 # Total number of bits divided by number of pixels.
    #                 _train_bpp = tf.reduce_sum(
    #                     tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    #                 tower_bpp.append(_train_bpp)

    #                 # Mean squared error across pixels.
    #                 _train_mse = tf.reduce_mean(tf.squared_difference(x_split[i], x_tilde))
    #                 # Multiply by 255^2 to correct for rescaling.
    #                 _train_mse *= 255 ** 2
    #                 tower_mse.append(_train_mse)

    #                 # The rate-distortion cost.
    #                 _train_loss = args.lmbda * _train_mse + _train_bpp
    #                 tower_main_loss.append(_train_loss)

    #                 # Compute the gradient
    #                 _main_grads = main_optimizer.compute_gradients(
    #                     loss=_train_loss)
    #                 print("_main_grads: {}".format(_main_grads))
    #                 tower_main_grads.append(_main_grads)
    #                 _aux_grads = aux_optimizer.compute_gradients(
    #                     loss=entropy_bottleneck.losses[0])
    #                 _aux_grads = [(g, v) for (g, v) in _aux_grads if g is not None]
    #                 print("_aux_grads: {}".format(_aux_grads))
    #                 tower_aux_grads.append(_aux_grads)

    #                 tf.get_variable_scope().reuse_variables()

    # # Average bpp
    # train_bpp = tf.stack(axis=0, values=tower_bpp)
    # train_bpp = tf.reduce_mean(train_bpp, 0)
    # # Average mse
    # train_mse = tf.stack(axis=0, values=tower_mse)
    # train_mse = tf.reduce_mean(train_mse, 0)
    # # Average losses
    # train_loss = tf.stack(axis=0, values=tower_main_loss)
    # train_loss = tf.reduce_mean(train_loss, 0)
    # # Average grads
    # main_grads = average_gradients(tower_main_grads)
    # print("global_step = {}".format(global_step))
    # if len(tower_aux_grads) != 0:
    #     aux_grads = average_gradients(tower_aux_grads)

    # # Apply the gradients to adjust the shared variables
    # main_step = main_optimizer.apply_gradients(
    #     main_grads, global_step=global_step)
    # if len(tower_aux_grads) != 0:
    #     aux_step = aux_optimizer.apply_gradients(
    #         aux_grads, global_step=global_step)

    """ 1 gpu """
    # Build autoencoder.
    y = analysis_transform(x)
    y_tilde, likelihoods = entropy_bottleneck(y, training=True)
    x_tilde = synthesis_transform(y_tilde)

    # Total number of bits divided by number of pixels.
    train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

    # Mean squared error across pixels.
    train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
    # Multiply by 255^2 to correct for rescaling.
    train_mse *= 255 ** 2

    # The rate-distortion cost.
    train_loss = args.lmbda * train_mse + train_bpp

    # Minimize loss and auxiliary loss, and execute update op.
    step = tf.train.create_global_step()
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    main_step = main_optimizer.minimize(train_loss, global_step=step)

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
    
    # eval_bpp, mse, psnr, msssim, num_pixels = inference(x, 
    #         analysis_transform, entropy_bottleneck, synthesis_transform)
    # inference_op = tf.group(eval_bpp, mse, psnr, msssim, num_pixels)

    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("bpp", train_bpp)
    tf.summary.scalar("mse", train_mse)
    
    psnr = tf.squeeze(tf.reduce_mean(tf.image.psnr(x_tilde, x, 255)))
    msssim = tf.squeeze(tf.reduce_mean(tf.image.ssim_multiscale(x_tilde, x, 255)))
    tf.summary.scalar("psnr", psnr)
    tf.summary.scalar("msssim", msssim)

    tf.summary.image("original", quantize_image(x))
    tf.summary.image("reconstruction", quantize_image(x_tilde))

    hooks = [
        tf.train.StopAtStepHook(last_step=args.last_step),
        tf.train.NanTensorHook(train_loss),
    ]
    # config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    with tf.train.MonitoredTrainingSession(
            hooks=hooks, checkpoint_dir=args.checkpoint_dir,
            save_checkpoint_secs=300, save_summaries_secs=60) as sess:
        while not sess.should_stop():
            # if (step % 1000 == 0):
            #     sess.run(inference_op)
            #     tf.summary.scalar("inf_bpp", eval_bpp)
            #     tf.summary.scalar("inf_mse", mse)
            #     tf.summary.scalar("inf_psnr", psnr)
            #     tf.summary.scalar("inf_msssim", msssim)
            #     tf.summary.scalar("inf_num_pxl", num_pixels)
            # else:
            sess.run(train_op)


def compress(args):
    """Compresses an image."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_device)

    # Load input image and add batch dimension.
    x = read_png(args.input_file)
    x = tf.expand_dims(x, 0)
    x.set_shape([1, None, None, 3])
    x_shape = tf.shape(x)

    # Instantiate model.
    analysis_transform = AnalysisTransform(args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()
    synthesis_transform = SynthesisTransform(args.num_filters)

    # Transform and compress the image.
    y = analysis_transform(x)
    string = entropy_bottleneck.compress(y)

    # Transform the quantized image back (if requested).
    y_hat, likelihoods = entropy_bottleneck(y, training=False)
    x_hat = synthesis_transform(y_hat)
    x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]

    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

    # Total number of bits divided by number of pixels.
    eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

    # Bring both images back to 0..255 range.
    x *= 255
    x_hat = tf.clip_by_value(x_hat, 0, 1)
    x_hat = tf.round(x_hat * 255)

    mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
    
    luma_x = tf.slice(tf.image.rgb_to_yuv(x), [0, 0, 0, 0], [-1, -1, -1, 1])
    luma_x_hat = tf.slice(tf.image.rgb_to_yuv(x_hat), [0, 0, 0, 0], [-1, -1, -1, 1])
    luma_psnr = tf.squeeze(tf.image.psnr(luma_x_hat, luma_x, 255))
    luma_msssim = tf.squeeze(tf.image.ssim_multiscale(luma_x_hat, luma_x, 255))
    
    chroma_x = tf.slice(tf.image.rgb_to_yuv(x), [0, 0, 0, 1], [-1, -1, -1, 2])
    chroma_x_hat = tf.slice(tf.image.rgb_to_yuv(x_hat), [0, 0, 0, 1], [-1, -1, -1, 2])
    chroma_psnr = tf.squeeze(tf.image.psnr(chroma_x_hat, chroma_x, 255))
    chroma_msssim = tf.squeeze(tf.image.ssim_multiscale(chroma_x_hat, chroma_x, 255))

    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1]]
        arrays = sess.run(tensors)

        # Write a binary file with the shape information and the compressed string.
        packed = tfc.PackedTensors()
        packed.pack(tensors, arrays)
        with open(args.output_file, "wb") as f:
            f.write(packed.string)

        # If requested, transform the quantized image back and measure performance.
        if args.verbose:
            eval_bpp, mse, luma_psnr, luma_msssim, chroma_psnr, chroma_msssim, num_pixels = sess.run(
                [eval_bpp, mse, luma_psnr, luma_msssim, chroma_psnr, chroma_msssim, num_pixels])

            # The actual bits per pixel including overhead.
            bpp = len(packed.string) * 8 / num_pixels

            print("Mean squared error: {:0.4f}".format(mse))
            print("LUMA PSNR (dB): {:0.2f}".format(luma_psnr))
            print("LUMA Multiscale SSIM: {:0.4f}".format(luma_msssim))
            print(
                "LUMA Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - luma_msssim)))
            print("CHROMA PSNR (dB): {:0.2f}".format(chroma_psnr))
            print("CHROMA Multiscale SSIM: {:0.4f}".format(chroma_msssim))
            print(
                "CHROMA Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - chroma_msssim)))
            print("Information content in bpp: {:0.4f}".format(eval_bpp))
            print("Actual bits per pixel: {:0.4f}".format(bpp))


def multi_compress(args):
    ######### Create input data pipeline for compressing multi images ########

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_device)
    with tf.device("/cpu:0"):
        eval_files = glob.glob(args.eval_glob)
        if not eval_files:
            raise RuntimeError(
                "No evaluation images found with glob '{}'.".format(args.eval_glob))
        eval_dataset = tf.data.Dataset.from_tensor_slices(eval_files)
        eval_dataset = eval_dataset.shuffle(
            buffer_size=len(eval_files)).repeat()
        eval_dataset = eval_dataset.map(
            read_png, num_parallel_calls=args.preprocess_threads)
        eval_dataset = eval_dataset.map(
            lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
        eval_dataset = eval_dataset.batch(args.batchsize)
        eval_dataset = eval_dataset.prefetch(32)

    num_pixels = args.batchsize * args.patchsize ** 2

    num_iter = len(eval_files) // args.batchsize
    # iterator = eval_dataset.make_initializable_iterator()
    # next_element = iterator.get_next()
    
    x = eval_dataset.make_one_shot_iterator().get_next()

    ######### create x place holder for eval ########
    # x = tf.placeholder(
    #         tf.float32, [None, args.patchsize, args.patchsize, 3], name='image')
            
    ######### end #########

    # Instantiate model.
    analysis_transform = AnalysisTransform(args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()
    synthesis_transform = SynthesisTransform(args.num_filters)

    # Transform and compress the image.
    y = analysis_transform(x)
    string = entropy_bottleneck.compress(y)

    # Transform the quantized image back (if requested).
    y_hat, likelihoods = entropy_bottleneck(y, training=False)
    x_hat = synthesis_transform(y_hat)
    x_hat = x_hat[:, :args.patchsize, :args.patchsize, :]

    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

    # # Total number of bits divided by number of pixels.
    # eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

    # Bring both images back to 0..255 range.
    x *= 255
    x_hat = tf.clip_by_value(x_hat, 0, 1)
    x_hat = tf.round(x_hat * 255)

    mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
    psnr = tf.reduce_mean(tf.squeeze(tf.image.psnr(x_hat, x, 255)))
    msssim = tf.reduce_mean(tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255)))

    ######### session for compressing multi imgs #########
    
    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        
        # sess.run(iterator.initializer)
        avg_bpp, avg_mse, avg_psnr, avg_msssim = 0., 0., 0., 0.
        for i in range(num_iter):
            # _x = sess.run(next_element)
            # print("_x shape: {}, datatype: {}".format(_x.shape, type(_x[0, 0, 0, 0])))
            print("[mse] {}".format(mse))
            _mse = sess.run(mse)
            _psnr = sess.run(psnr)
            _msssim = sess.run(msssim)
            _num_pixels = sess.run(num_pixels)
            _string = sess.run(string)
            # mse, psnr, msssim, num_pixels, string = sess.run(
            #     [mse, psnr, msssim, num_pixels, string])
                # feed_dict={x: _x})
            # The actual bits per pixel including overhead.
            bpp = len(_string) * 8 / _num_pixels
            avg_bpp += bpp
            avg_mse += _mse
            avg_psnr += _psnr
            avg_msssim += _msssim
        avg_bpp /= num_iter
        avg_mse /= num_iter
        avg_psnr /= num_iter
        avg_msssim /= num_iter
        print("Mean squared error: {:0.4f}".format(avg_mse))
        print("PSNR (dB): {:0.2f}".format(avg_psnr))
        print("Multiscale SSIM: {:0.4f}".format(avg_msssim))
        print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - avg_msssim)))
        print("Actual bits per pixel: {:0.4f}".format(avg_bpp))
    ######### end #########


def decompress(args):
    """Decompresses an image."""

    # Read the shape information and compressed string from the binary file.
    string = tf.placeholder(tf.string, [1])
    x_shape = tf.placeholder(tf.int32, [2])
    y_shape = tf.placeholder(tf.int32, [2])
    with open(args.input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
    tensors = [string, x_shape, y_shape]
    arrays = packed.unpack(tensors)

    # Instantiate model.
    entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
    synthesis_transform = SynthesisTransform(args.num_filters)

    # Decompress and transform the image back.
    y_shape = tf.concat([y_shape, [args.num_filters]], axis=0)
    y_hat = entropy_bottleneck.decompress(
        string, y_shape, channels=args.num_filters)
    x_hat = synthesis_transform(y_hat)

    # Remove batch dimension, and crop away any extraneous padding on the bottom
    # or right boundaries.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

    # Write reconstructed image out as a PNG file.
    op = write_png(args.output_file, x_hat)

    # Load the latest model checkpoint, and perform the above actions.
    with tf.Session() as sess:
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        sess.run(op, feed_dict=dict(zip(tensors, arrays)))


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Report bitrate and distortion when training or compressing.")
    parser.add_argument(
        "--num_filters", type=int, default=128,
        help="Number of filters per layer.")
    
    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: 'train' loads training data and trains (or continues "
             "to train) a new model. 'compress' reads an image file (lossless "
             "PNG format) and writes a compressed binary file. 'decompress' "
             "reads a binary file and reconstructs the image (in PNG format). "
             "input and output filenames need to be provided for the latter "
             "two options. Invoke '<command> -h' for more information.")

    # 'train' subcommand.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model.")
    train_cmd.add_argument(
        "--train_glob", default="images/*.png",
        help="Glob pattern identifying training data. This pattern must expand "
             "to a list of RGB images in PNG format.")
    train_cmd.add_argument(
        "--batchsize", type=int, default=8,
        help="Batch size for training.")
    train_cmd.add_argument(
        "--patchsize", type=int, default=256,
        help="Size of image patches for training.")
    train_cmd.add_argument(
        "--lambda", type=float, default=0.01, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    train_cmd.add_argument(
        "--last_step", type=int, default=1000000,
        help="Train up to this number of steps.")
    train_cmd.add_argument(
        "--preprocess_threads", type=int, default=16,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")
    train_cmd.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of gpus used for training.")
    train_cmd.add_argument(
        "--gpu_device", type=int, default=0,
        help="gpu device to be used.")
    train_cmd.add_argument(
        "--checkpoint_dir", default="train",
        help="Directory where to save/load model checkpoints.")

    # 'compress' subcommand.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes a TFCI file.")

    # 'decompress' subcommand.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a TFCI file, reconstructs the image, and writes back "
                    "a PNG file.")

    # Arguments for both 'compress' and 'decompress'.
    for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
        cmd.add_argument(
            "input_file",
            help="Input filename.")
        cmd.add_argument(
            "output_file", nargs="?",
            help="Output filename (optional). If not provided, appends '{}' to "
                 "the input filename.".format(ext))
        cmd.add_argument(
            "--checkpoint_dir", default="train",
            help="Directory where to save/load model checkpoints.")
        cmd.add_argument(
            "--gpu_device", type=int, default=3,
            help="gpu device to be used.")
    
    # 'compress' subcommand.
    multi_compress_cmd = subparsers.add_parser(
        "multi_compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes a TFCI file.")
    multi_compress_cmd.add_argument(
        "--eval_glob", default="images/*.png",
        help="Glob pattern identifying training data. This pattern must expand "
             "to a list of RGB images in PNG format.")
    multi_compress_cmd.add_argument(
        "--gpu_device", type=int, default=0,
        help="gpu device to be used.")
    multi_compress_cmd.add_argument(
        "--checkpoint_dir", default="train",
        help="Directory where to save/load model checkpoints.")
    multi_compress_cmd.add_argument(
        "--preprocess_threads", type=int, default=16,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")
    multi_compress_cmd.add_argument(
        "--batchsize", type=int, default=8,
        help="Batch size for training.")
    multi_compress_cmd.add_argument(
        "--patchsize", type=int, default=256,
        help="Size of image patches for training.")
    

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand.
    if args.command == "train":
        train(args)
    elif args.command == "compress":
        if not args.output_file:
            args.output_file = args.input_file + ".tfci"
        compress(args)
    elif args.command == "decompress":
        if not args.output_file:
            args.output_file = args.input_file + ".png"
        decompress(args)
    elif args.command == "multi_compress":
        print("multi_compress!")
        multi_compress(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
