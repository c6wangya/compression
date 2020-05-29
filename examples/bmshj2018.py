# -*- coding: utf-8 -*-
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Nonlinear transform coder with hyperprior for RGB images.

This is the image compression model published in:
J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
"Variational Image Compression with a Scale Hyperprior"
Int. Conf. on Learning Representations (ICLR), 2018
https://arxiv.org/abs/1802.01436

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
import modules as m

import tensorflow_compression as tfc


epsilon = 1e-10
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
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

def get_session(sess):
    """ helper function for taking sess out from MonitoredTrainingSession """
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session

def restore_weights(saver, sess, ckpt_dir):
    """ helper function restoring weights for saver """
    assert os.path.exists(ckpt_dir), "the path {} isn't valid".format(ckpt_dir)
    latest = tf.train.latest_checkpoint(checkpoint_dir=ckpt_dir)
    saver.restore(sess, save_path=latest)

def save_weights(saver, sess, ckpt_dir, iters):
    """ helper function saving weights for saver """
    if not os.path.exists(ckpt_dir):
        os.mkdir(path=ckpt_dir)
    saver.save(sess, save_path=ckpt_dir + '/model_{}.ckpt'.format(iters))

class AnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_0")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_1")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_2")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]
        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


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
                self.num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
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


class HyperAnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
                padding="same_zeros", use_bias=False,
                activation=None),
        ]
        super(HyperAnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class HyperSynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(HyperSynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True, kernel_parameterizer=None,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True, kernel_parameterizer=None,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
                padding="same_zeros", use_bias=True, kernel_parameterizer=None,
                activation=None),
        ]
        super(HyperSynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


@tf.custom_gradient
def differentiable_round(x):
    """ customized differentiable round operation"""
    def grad(dy):
        return dy
    return tf.round(x), grad


def lr_schedule(step, mode, warmup_steps=10000, decay=0.999995):
    assert mode == 'constant' or mode == 'scheduled'
    if mode == 'scheduled':
        global curr_lr
        if step < warmup_steps:
            curr_lr = 1 * step / warmup_steps
            return 1 * step / warmup_steps
        elif step > warmup_steps:
            curr_lr *= decay 
            return curr_lr
        return curr_lr
    return 1


def train(args):
    """Trains the model."""
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_device)
    
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    # Create input data pipeline.
    with tf.device("/cpu:0"):
        train_files=glob.glob(args.train_glob)
        if not train_files:
            raise RuntimeError(
                    "No training images found with glob '{}'.".format(args.train_glob))
        
        train_dataset=tf.data.Dataset.from_tensor_slices(train_files)
        train_dataset=train_dataset.shuffle(
                buffer_size=len(train_files)).repeat()
        train_dataset=train_dataset.map(
                read_png, num_parallel_calls=args.preprocess_threads)
        # train_dataset = train_dataset.map(DataPreprocessor())
        # train_dataset = train_dataset.map(
        #         lambda x: tf.image.random_saturation(x, lower=0.7, upper=1))
        # scale = get_normal_random_number(args.scale, 0.1, args.scale - 0.15, args.scale + 0.05)
        scale = 1
        train_dataset=train_dataset.map(
                lambda x: tf.random_crop(x, (int(args.patchsize / scale), int(args.patchsize / scale), 3)))
        # train_dataset = train_dataset.map(
        #         lambda x: tf.image.resize_images(x, (args.patchsize, args.patchsize), method=tf.image.ResizeMethod.BICUBIC))
        train_dataset=train_dataset.batch(args.batchsize)
        train_dataset=train_dataset.prefetch(32)

        if args.val_gap != 0:
            tf.set_random_seed(1234)
            val_files=glob.glob(args.val_glob)
            if not val_files:
                raise RuntimeError(
                        "No validation images found with glob '{}'.".format(args.val_glob))
            
            val_dataset=tf.data.Dataset.from_tensor_slices(val_files)
            val_dataset=val_dataset.repeat()
            val_dataset=val_dataset.map(
                    read_png, num_parallel_calls=args.preprocess_threads)
            val_dataset=val_dataset.map(lambda x: tf.random_crop(x, (512, 512, 3)))
            val_dataset=val_dataset.batch(24)
            val_dataset=val_dataset.prefetch(64)
                
    num_pixels = args.batchsize * args.patchsize ** 2

    # Get training patch from dataset.
    x = train_dataset.make_one_shot_iterator().get_next()

    # Get validation data from dataset
    if args.val_gap != 0:
        x_val = val_dataset.make_one_shot_iterator().get_next()
        print(x_val)

    # Instantiate model.
    hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()
    if args.command == "train":
        analysis_transform = AnalysisTransform(args.num_filters)
        synthesis_transform = SynthesisTransform(args.num_filters)
        hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
        hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
    else:
        # inv train net
        inv_transform = m.InvCompressionNet(channel_in=3, channel_out=args.channel_out, 
                blk_type=args.blk_type, num_filters=args.num_filters,
                kernel_size=args.kernel_size, residual=args.residual, 
                nin=args.nin, norm=args.norm, n_ops=args.n_ops, 
                downsample_type=args.downsample_type, inv_conv=(not args.non1x1), 
                use_norm=args.use_norm)
        if args.guidance_type == "baseline_pretrain":
            analysis_transform = AnalysisTransform(args.channel_out[0])
            synthesis_transform = SynthesisTransform(args.channel_out[0])
            hyper_analysis_transform = HyperAnalysisTransform(args.channel_out[0])
            hyper_synthesis_transform = HyperSynthesisTransform(args.channel_out[0])
        elif args.guidance_type == "baseline":
            analysis_transform = AnalysisTransform(args.channel_out[0])
            hyper_analysis_transform = HyperAnalysisTransform(args.channel_out[0])
            hyper_synthesis_transform = HyperSynthesisTransform(args.channel_out[0])

    # Build autoencoder and hyperprior.
    # Transform Image
    train_flow, train_jac = 0, 0
    if args.command == "train" or args.guidance_type == "baseline_pretrain":
        y = analysis_transform(x)
        z = hyper_analysis_transform(abs(y))
        z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
        sigma = hyper_synthesis_transform(z_tilde)
        scale_table = np.exp(np.linspace(
            np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
        conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
        y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)
        x_tilde = synthesis_transform(y_tilde)
        flow_loss_weight = 0
        # validation 
        if args.val_gap != 0:
            # Transform and compress the image.
            y_val = analysis_transform(x_val)
            y_shape = tf.shape(y_val)
            z_val = hyper_analysis_transform(abs(y_val))
            z_val_hat, _ = entropy_bottleneck(z_val, training=False)
            sigma_val = hyper_synthesis_transform(z_val_hat)
            sigma_val = sigma_val[:, :y_shape[1], :y_shape[2], :]
            val_scale_table = np.exp(np.linspace(
                    np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
            val_conditional_bottleneck = tfc.GaussianConditional(sigma_val, val_scale_table)
            # compute bpp
            side_string = entropy_bottleneck.compress(z_val)
            string = val_conditional_bottleneck.compress(y_val)
            val_num_pixels = tf.cast(tf.reduce_prod(tf.shape(x_val)[:-1]), dtype=tf.float32)
            string_len = tf.reduce_sum(tf.cast(tf.strings.length(string), dtype=tf.float32)) + \
                         tf.reduce_sum(tf.cast(tf.strings.length(side_string), dtype=tf.float32))
            val_bpp = tf.math.divide(string_len * 8, val_num_pixels)
            # Transform the quantized image back (if requested).
            y_val_hat, _ = val_conditional_bottleneck(y_val, training=False)
            # y^ 
            x_val_hat = synthesis_transform(y_val_hat)
            # y
            x_val_hat_reuse_y = synthesis_transform(y_val)
    else:
        if args.guidance_type == "baseline":
            y_base = analysis_transform(x)
            if args.prepos_ste: 
                y_base = differentiable_round(y_base)
        out, train_jac = inv_transform([x])
        zshapes = []
        
        if out[-1].get_shape()[-1] == args.channel_out[-1]:
            flow_z = out[-1][:, :, :, args.channel_out[-1] - 1:]
        else:
            flow_z = out[-1][:, :, :, args.channel_out[-1]:]
        print(flow_z.get_shape())
        zshapes.append(tf.shape(flow_z))
        # mle of flow 
        train_flow += tf.reduce_sum(tf.norm(flow_z + epsilon, ord=2, axis=-1, name="last_norm"))
        if args.train_jacobian:
            train_flow /= -np.log(2) * num_pixels
        
        y = tf.slice(out[-1], [0, 0, 0, 0], [-1, -1, -1, args.channel_out[-1]])
        
        if args.prepos_ste:
            y = differentiable_round(y)
        
        z = hyper_analysis_transform(abs(y))
        z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
        sigma = hyper_synthesis_transform(z_tilde)
        scale_table = np.exp(np.linspace(
            np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
        conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
        y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)

        input_rev = []
        for zshape in zshapes:
            if args.zero_z:
                input_rev.append(tf.zeros(shape=zshape))
            else:
                input_rev.append(tf.random_normal(shape=zshape))
        
        if args.ste or args.prepos_ste:
            y_tilde = differentiable_round(y_tilde)
        input_rev.append(y_tilde)
        x_tilde, _ = inv_transform(input_rev, rev=True)
        x_tilde = x_tilde[-1]
        flow_loss_weight = args.flow_loss_weight

        # validation 
        if args.val_gap != 0:
            out, _ = inv_transform([x_val])
            # z_samples and z_zeros
            zshapes = []
            if out[-1].get_shape()[-1] == args.channel_out[-1]:
                flow_z = out[-1][:, :, :, args.channel_out[-1] - 1:]
            else:
                flow_z = out[-1][:, :, :, args.channel_out[-1]:]
            zshapes.append(tf.shape(flow_z))
            z_samples = [tf.random_normal(shape=zshape) for zshape in zshapes]
            z_zeros = [tf.zeros(shape=zshape) for zshape in zshapes]
            
            # y hat
            y_val = tf.slice(out[-1], [0, 0, 0, 0], [-1, -1, -1, args.channel_out[-1]])
            z_val = hyper_analysis_transform(abs(y_val))
            z_hat, _ = entropy_bottleneck(z_val, training=False)
            sigma = hyper_synthesis_transform(z_hat)
            sigma = sigma[:, :tf.shape(y_val)[1], :tf.shape(y_val)[2], :]
            scale_table = np.exp(np.linspace(
                    np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
            conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
            y_val_hat, _ = conditional_bottleneck(y_val, training=False)

            # compute bpp
            side_string = entropy_bottleneck.compress(z_val)
            string = conditional_bottleneck.compress(y_val)
            val_num_pixels = tf.cast(tf.reduce_prod(tf.shape(x_val)[:-1]), dtype=tf.float32)
            string_len = tf.reduce_sum(tf.cast(tf.strings.length(string), dtype=tf.float32)) + \
                         tf.reduce_sum(tf.cast(tf.strings.length(side_string), dtype=tf.float32))
            val_bpp = tf.math.divide(string_len * 8, val_num_pixels)
            
            # y^, z^
            x_val_y_hat_z_hat, _ = inv_transform(z_samples + [y_val_hat], rev=True)
            # y^, 0
            x_val_y_hat_z_0, _ = inv_transform(z_zeros + [y_val_hat], rev=True)
            # y, z^
            x_val_y_z_hat, _ = inv_transform(z_samples + [y_val], rev=True)
            # y, 0
            x_val_y_z_0, _ = inv_transform(z_zeros + [y_val], rev=True)

    # Total number of bits divided by number of pixels.
    train_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
                 tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

    # Mean squared error across pixels.
    train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
    # Multiply by 255^2 to correct for rescaling.
    train_mse *= 255 ** 2

    if args.guidance_type == "baseline":
        train_y_guidance = tf.reduce_sum(tf.squared_difference(y, tf.stop_gradient(y_base)))
    else:
        train_y_guidance = 0

    if not args.train_jacobian:
            train_jac = 0

    # The rate-distortion cost.
    train_loss = args.lmbda * train_mse + \
                 args.beta * train_bpp + \
                 flow_loss_weight * (train_flow + train_jac) + \
                 args.y_guidance_weight * train_y_guidance
            
    tvars = tf.trainable_variables()
    filtered_vars = [var for var in tvars \
                if not 'haar_downsampling' in var.name]

    # Minimize loss and auxiliary loss, and execute update op.
    main_lr = tf.placeholder(tf.float32, [], 'main_lr')
    aux_lr = tf.placeholder(tf.float32, [], 'aux_lr')
    step = tf.train.create_global_step()
    main_optimizer = tf.train.AdamOptimizer(learning_rate=main_lr)
    main_step = main_optimizer.minimize(train_loss, global_step=step)

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=aux_lr)
    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
    
    if args.val_gap != 0:
        def comp_psnr(img_hat, img):
            img *= 255
            img_hat=tf.clip_by_value(img_hat, 0, 1)
            img_hat=tf.round(img_hat * 255)
            # img = tf.squeeze(img)
            # img_hat = tf.squeeze(img_hat[-1])
            if args.command == "inv_train" and args.guidance_type != "baseline_pretrain":
                img_hat = img_hat[-1]
            rgb_psnr = tf.squeeze(tf.reduce_mean(tf.image.psnr(img_hat, img, 255)))
            luma_img = tf.slice(tf.image.rgb_to_yuv(img), [0, 0, 0, 0], [-1, -1, -1, 1])
            luma_img_hat = tf.slice(tf.image.rgb_to_yuv(img_hat), [0, 0, 0, 0], [-1, -1, -1, 1])
            luma_psnr = tf.squeeze(tf.reduce_mean(tf.image.psnr(luma_img_hat, luma_img, 255)))
            return rgb_psnr, luma_psnr
        
        if args.command == "train" or args.guidance_type == "baseline_pretrain":
            # y^
            val_rgb_psnr, val_luma_psnr = comp_psnr(x_val_hat, x_val)
            # y
            val_reuse_y_rgb_psnr, val_reuse_y_luma_psnr = comp_psnr(x_val_hat_reuse_y, x_val)
            # summary 
            tf.summary.scalar("validation-yhat-rgb-psnr", val_rgb_psnr)
            tf.summary.scalar("validation-yhat-luma-psnr", val_luma_psnr)
            tf.summary.scalar("validation-y-rgb-psnr", val_reuse_y_rgb_psnr)
            tf.summary.scalar("validation-y-luma-psnr", val_reuse_y_luma_psnr)
            tf.summary.scalar("validation-bpp", val_bpp)
            # group operations
            val_op_lst = [val_rgb_psnr, val_luma_psnr, 
                        val_reuse_y_rgb_psnr, val_reuse_y_luma_psnr]
                    #  val_bpp]
            val_bpp_op_list = [val_bpp]
        else:
            # y^, z^
            val_y_hat_z_hat_rgb_psnr, val_y_hat_z_hat_luma_psnr = comp_psnr(x_val_y_hat_z_hat, x_val)
            # y^, 0
            val_y_hat_z_0_rgb_psnr, val_y_hat_z_0_luma_psnr = comp_psnr(x_val_y_hat_z_0, x_val)
            # y, z^
            val_y_z_hat_rgb_psnr, val_y_z_hat_luma_psnr = comp_psnr(x_val_y_z_hat, x_val)
            # y, 0
            val_y_z_0_rgb_psnr, val_y_z_0_luma_psnr = comp_psnr(x_val_y_z_0, x_val)
            # summary
            tf.summary.scalar("validation-yhat-zhat-rgb-psnr", val_y_hat_z_hat_rgb_psnr)
            tf.summary.scalar("validation-yhat-zhat-luma-psnr", val_y_hat_z_hat_luma_psnr)
            tf.summary.scalar("validation-yhat-z0-rgb-psnr", val_y_hat_z_0_rgb_psnr)
            tf.summary.scalar("validation-yhat-z0-luma-psnr", val_y_hat_z_0_luma_psnr)
            tf.summary.scalar("validation-y-zhat-rgb-psnr", val_y_z_hat_rgb_psnr)
            tf.summary.scalar("validation-y-zhat-luma-psnr", val_y_z_hat_luma_psnr)
            tf.summary.scalar("validation-y-z0-rgb-psnr", val_y_z_0_rgb_psnr)
            tf.summary.scalar("validation-y-z0-luma-psnr", val_y_z_0_luma_psnr)
            tf.summary.scalar("validation-bpp", val_bpp)
            # group operations
            val_op_lst = [val_y_hat_z_hat_rgb_psnr, val_y_hat_z_hat_luma_psnr, 
                        val_y_hat_z_0_rgb_psnr, val_y_hat_z_0_luma_psnr, 
                        val_y_z_hat_rgb_psnr, val_y_z_hat_luma_psnr, 
                        val_y_z_0_rgb_psnr, val_y_z_0_luma_psnr]
                    #   val_bpp]
            val_bpp_op_list = [val_bpp]
        val_op = tf.group(*val_op_lst)
        val_bpp_op = tf.group(*val_bpp_op_list)

    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("bpp", train_bpp)
    tf.summary.scalar("mse", train_mse)
    
    psnr = tf.squeeze(tf.reduce_mean(tf.image.psnr(x_tilde, x, 255)))
    # msssim = tf.squeeze(tf.image.ssim_multiscale(x_tilde, x, 255))
    tf.summary.scalar("psnr", psnr)
    # tf.summary.scalar("msssim", msssim)

    tf.summary.image("original", quantize_image(x))
    tf.summary.image("reconstruction", quantize_image(x_tilde))

    hooks = [
        tf.train.StopAtStepHook(last_step=args.last_step),
        tf.train.NanTensorHook(train_loss),
    ]
    # init saver for all the models
    if "baseline" in args.guidance_type:
        analysis_saver = tf.train.Saver(analysis_transform.variables, max_to_keep=1)
        hyper_analysis_saver = tf.train.Saver(hyper_analysis_transform.variables, max_to_keep=1)
        hyper_synthesis_saver = tf.train.Saver(hyper_synthesis_transform.variables, max_to_keep=1)
        entropy_saver = tf.train.Saver(entropy_bottleneck.variables, max_to_keep=1)
    global_iters = 0
    with tf.train.MonitoredTrainingSession(
                hooks=hooks, checkpoint_dir=args.checkpoint_dir,
                save_checkpoint_secs=5000, save_summaries_secs=300) as sess:
        # if "baseline" not in args.guidance_type or args.finetune:
        #     while not sess.should_stop():
        #         lr = lr_schedule(global_iters, 
        #                             args.lr_scheduler, 
        #                             args.lr_warmup_steps, 
        #                             args.lr_decay)
        #         sess.run(train_op, {main_lr: args.main_lr * lr, 
        #                             aux_lr: args.aux_lr * lr})
        #         if args.val_gap != 0 and global_iters % args.val_gap == 0:
        #             sess.run(val_op)
        #             sess.run(val_bpp_op)
        #         global_iters += 1
        # else:
        if not args.finetune and args.guidance_type == "baseline":
            # load analysis and entropybottleneck model
            restore_weights(analysis_saver, get_session(sess), 
                    args.pretrain_checkpoint_dir + "/ana_net")
            restore_weights(entropy_saver, get_session(sess), 
                    args.pretrain_checkpoint_dir + "/entro_net")
            restore_weights(hyper_analysis_saver, get_session(sess), 
                    args.pretrain_checkpoint_dir + "/hyper_ana_net")
            restore_weights(hyper_synthesis_saver, get_session(sess), 
                    args.pretrain_checkpoint_dir + "/hyper_syn_net")
        while not sess.should_stop():
            lr = lr_schedule(global_iters, 
                                args.lr_scheduler, 
                                args.lr_warmup_steps, 
                                args.lr_decay)
            sess.run(train_op, {main_lr: args.main_lr * lr, 
                                aux_lr: args.aux_lr * lr})
            if args.val_gap != 0 and global_iters % args.val_gap == 0:
                sess.run(val_op)
                sess.run(val_bpp_op)
            if global_iters % 5000 == 0 and args.guidance_type == "baseline_pretrain":
                # save analysis, synthesis and entropybottleneck model
                save_weights(analysis_saver, get_session(sess), 
                        args.checkpoint_dir + '/ana_net', global_iters)
                save_weights(entropy_saver, get_session(sess), 
                        args.checkpoint_dir + '/entro_net', global_iters)
                save_weights(hyper_analysis_saver, get_session(sess), 
                        args.checkpoint_dir + '/hyper_ana_net', global_iters)
                save_weights(hyper_synthesis_saver, get_session(sess), 
                        args.checkpoint_dir + '/hyper_syn_net', global_iters)
            global_iters += 1


def compress(args):
    """Compresses an image."""

    # Load input image and add batch dimension.
    x = read_png(args.input_file)
    x = tf.expand_dims(x, 0)
    x.set_shape([1, None, None, 3])
    x_shape = tf.shape(x)

    # Instantiate model.
    analysis_transform = AnalysisTransform(args.num_filters)
    synthesis_transform = SynthesisTransform(args.num_filters)
    hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()

    # Transform and compress the image.
    y = analysis_transform(x)
    y_shape = tf.shape(y)
    z = hyper_analysis_transform(abs(y))
    z_hat, z_likelihoods = entropy_bottleneck(z, training=False)
    sigma = hyper_synthesis_transform(z_hat)
    sigma = sigma[:, :y_shape[1], :y_shape[2], :]
    scale_table = np.exp(np.linspace(
        np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
    conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
    side_string = entropy_bottleneck.compress(z)
    string = conditional_bottleneck.compress(y)

    # Transform the quantized image back (if requested).
    y_hat, y_likelihoods = conditional_bottleneck(y, training=False)
    x_hat = synthesis_transform(y_hat)
    x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]

    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

    # Total number of bits divided by number of pixels.
    eval_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
                tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

    # Bring both images back to 0..255 range.
    x *= 255
    x_hat = tf.clip_by_value(x_hat, 0, 1)
    x_hat = tf.round(x_hat * 255)

    mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
    psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        tensors = [string, side_string,
                   tf.shape(x)[1:-1], tf.shape(y)[1:-1], tf.shape(z)[1:-1]]
        arrays = sess.run(tensors)

        # Write a binary file with the shape information and the compressed string.
        packed = tfc.PackedTensors()
        packed.pack(tensors, arrays)
        with open(args.output_file, "wb") as f:
            f.write(packed.string)

        # If requested, transform the quantized image back and measure performance.
        if args.verbose:
            eval_bpp, mse, psnr, msssim, num_pixels = sess.run(
                [eval_bpp, mse, psnr, msssim, num_pixels])

            # The actual bits per pixel including overhead.
            bpp = len(packed.string) * 8 / num_pixels

            print("Mean squared error: {:0.4f}".format(mse))
            print("PSNR (dB): {:0.2f}".format(psnr))
            print("Multiscale SSIM: {:0.4f}".format(msssim))
            print(
                "Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim)))
            print("Information content in bpp: {:0.4f}".format(eval_bpp))
            print("Actual bits per pixel: {:0.4f}".format(bpp))


def decompress(args):
    """Decompresses an image."""

    # Read the shape information and compressed string from the binary file.
    string = tf.placeholder(tf.string, [1])
    side_string = tf.placeholder(tf.string, [1])
    x_shape = tf.placeholder(tf.int32, [2])
    y_shape = tf.placeholder(tf.int32, [2])
    z_shape = tf.placeholder(tf.int32, [2])
    with open(args.input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
    tensors = [string, side_string, x_shape, y_shape, z_shape]
    arrays = packed.unpack(tensors)

    # Instantiate model.
    synthesis_transform = SynthesisTransform(args.num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)

    # Decompress and transform the image back.
    z_shape = tf.concat([z_shape, [args.num_filters]], axis=0)
    z_hat = entropy_bottleneck.decompress(
        side_string, z_shape, channels=args.num_filters)
    sigma = hyper_synthesis_transform(z_hat)
    sigma = sigma[:, :y_shape[0], :y_shape[1], :]
    scale_table = np.exp(np.linspace(
        np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
    conditional_bottleneck = tfc.GaussianConditional(
        sigma, scale_table, dtype=tf.float32)
    y_hat = conditional_bottleneck.decompress(string)
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
    parser=argparse_flags.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
            "--verbose", "-V", action="store_true",
            help="Report bitrate and distortion when training or compressing.")
    parser.add_argument(
            "--num_filters", type=int, default=128,
            help="Number of filters per layer.")
    parser.add_argument(
            "--debug_mode", action="store_true",
            help="activate the debug mode.")

    subparsers=parser.add_subparsers(
            title="commands", dest="command",
            help="What to do: 'train' loads training data and trains (or continues "
            "to train) a new model. 'compress' reads an image file (lossless "
            "PNG format) and writes a compressed binary file. 'decompress' "
            "reads a binary file and reconstructs the image (in PNG format). "
            "input and output filenames need to be provided for the latter "
            "two options. Invoke '<command> -h' for more information.")

    # 'train' subcommand.
    train_cmd=subparsers.add_parser(
            "train",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Trains (or continues to train) a new model.")
    inv_train_cmd=subparsers.add_parser(
            "inv_train",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Trains (or continues to train) a new model.")
    for cmd in [train_cmd, inv_train_cmd]:
        cmd.add_argument(
                "--train_glob", default="images/*.png",
                help="Glob pattern identifying training data. This pattern must expand "
                "to a list of RGB images in PNG format.")
        cmd.add_argument(
                "--batchsize", type=int, default=8,
                help="Batch size for training.")
        cmd.add_argument(
                "--patchsize", type=int, default=256,
                help="Size of image patches for training.")
        cmd.add_argument(
                "--lambda", type=float, default=0.01, dest="lmbda",
                help="Lambda for rate-distortion tradeoff.")
        cmd.add_argument(
                "--last_step", type=int, default=1000000,
                help="Train up to this number of steps.")
        cmd.add_argument(
                "--preprocess_threads", type=int, default=16,
                help="Number of CPU threads to use for parallel decoding of training "
                "images.")
        cmd.add_argument(
                "--num_gpus", type=int, default=1,
                help="Number of gpus used for training.")
        cmd.add_argument(
                "--gpu_device", type=int, default=0,
                help="gpu device to be used.")
        cmd.add_argument(
                "--checkpoint_dir", default="train",
                help="Directory where to save/load model checkpoints.")
        cmd.add_argument(
                "--noise", action="store_true",
                help="add noise to image patch.")
        cmd.add_argument(
                "--downsample_scale", type=float, default=0.75, dest="scale",
                help="downsample scale for data preprocessing..")
        cmd.add_argument(
                "--adjust_saturation", action="store_true",
                help="adjust saturation of images.")
        cmd.add_argument(
                "--blk_num", type=int, default=4,
                help="num of blocks for flow net")
        cmd.add_argument(
                "--channel_out", nargs='+', type=int, default=[3, 3])
        cmd.add_argument(
                "--upscale_log", type=int, default=2,
                help="upscale times")
        cmd.add_argument(
                "--kernel_size", type=int, default=3,
                help="kernel size of subunit conv")
        cmd.add_argument(
                "--flow_loss_weight", type=float, default=1e-1,
                help="flow loss weight.")
        cmd.add_argument(
                "--y_guidance_weight", type=float, default=0.,
                help="flow loss weight.")
        cmd.add_argument(
                "--blk_type", default="dense",
                help="select which type of block to use")
        cmd.add_argument(
                "--non1x1", action="store_true",
                help="train without 1x1 invertible conv.")
        cmd.add_argument(
                "--main_lr", type=float, default=1e-4,
                help="main learning rate.")
        cmd.add_argument(
                "--aux_lr", type=float, default=1e-3,
                help="aux learning rate.")
        cmd.add_argument(
                "--residual", action="store_true",
                help="use residual block in subnet.")
        cmd.add_argument(
                "--nin", action="store_true",
                help="use 1x1 conv in subnet.")
        cmd.add_argument(
                "--norm", default="bn",
                help="which type of norm to use.")
        cmd.add_argument(
                "--clamp", action="store_true",
                help="Do clamp on y.")
        cmd.add_argument(
                "--grad_clipping", type=float, default=100,
                help="Clipping gradient.")
        cmd.add_argument(
                "--guidance_type", default="none",
                help="guidance type.")
        cmd.add_argument(
                "--quant_grad", action="store_true",
                help="quantize with gradient.")
        cmd.add_argument(
                "--n_ops", type=int, default=3,
                help="number of operations in subnet")
        cmd.add_argument(
                "--downsample_type", default="haar",
                help="type of downsample ('haar' or 'squeeze').")

        cmd.add_argument(
                "--pretrain_checkpoint_dir", default="train",
                help="Directory where to save/load model checkpoints.")
        cmd.add_argument(
                "--num_data", type=int, default=10000,
                help="size of dataset")
        cmd.add_argument(
                "--finetune", action="store_true",
                help="finetune the inv network.")
        cmd.add_argument(
                "--beta", type=float, default=1,
                help="Beta for rate-distortion tradeoff.")
        cmd.add_argument(
                "--freeze_aux", action="store_true",
                help="whether freeze auxiliary net when training.")
        cmd.add_argument(
                "--zero_z", action="store_true",
                help="whether set z to zeros.")
        cmd.add_argument(
                "--no_aux", action="store_true",
                help="whether use aux net to train. \
                    (if not then manually round and pass gradient).")
        cmd.add_argument(
                "--train_jacobian", action="store_true",
                help="whether jacobian loss to train.")
        cmd.add_argument(
                "--val_gap", type=int, default=0,
                help="validation gap, default = 0")
        cmd.add_argument(
                "--val_glob", default="images/*.png",
                help="Glob pattern identifying validation data. This pattern must expand "
                "to a list of RGB images in PNG format.")
        cmd.add_argument(
                "--lr_scheduler", default="constant",
                help="lr scheduler mode, can be either constant or scheduled.")
        cmd.add_argument(
                "--lr_warmup_steps", type=int, default=10000,
                help="warm-up steps")
        cmd.add_argument(
                "--lr_min_ratio", type=float, default=0.1,
                help="minimam ratio of lr")
        cmd.add_argument(
                "--lr_decay", type=float, default=0.999995,
                help="decay ratio of lr")
        cmd.add_argument(
                "--ste", action="store_true",
                help="whether to use ste for recons.")
        cmd.add_argument(
                "--prepos_ste", action="store_true",
                help="whether to use prepositioned ste for recons.")
        cmd.add_argument(
                "--use_norm", action="store_true",
                help="whether to use norm after 1x1 conv.")

    # 'compress' subcommand.
    compress_cmd=subparsers.add_parser(
            "compress",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Reads a PNG file, compresses it, and writes a TFCI file.")

    # 'decompress' subcommand.
    decompress_cmd=subparsers.add_parser(
            "decompress",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Reads a TFCI file, reconstructs the image, and writes back "
            "a PNG file.")

    # 'compress' subcommand.
    evaluation_cmd = subparsers.add_parser(
            "evaluation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Reads a PNG file, evaluate it.")
    
    evaluation_cmd.add_argument(
            "--eval_glob", default="images/*.png",
                help="Glob pattern identifying training data. This pattern must expand "
                "to a list of RGB images in PNG format.")
    evaluation_cmd.add_argument(
            "--preprocess_threads", type=int, default=16,
            help="Number of CPU threads to use for parallel decoding of training "
            "images.")
    evaluation_cmd.add_argument(
            "--batchsize", type=int, default=8,
            help="Batch size for training.")
    evaluation_cmd.add_argument(
            "--patchsize", type=int, default=256,
            help="Size of image patches for training.")

    # Arguments for both 'compress' and 'decompress'.
    for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png"), (evaluation_cmd, ".tfci")):
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
        cmd.add_argument(
            "--n_ops", type=int, default=3,
            help="number of operations in subnet")
        cmd.add_argument(
            "--quant_grad", action="store_true",
            help="quantize with gradient.")
        cmd.add_argument(
            "--downsample_type", default="haar",
            help="type of downsample ('haar' or 'squeeze').")

        cmd.add_argument(
                "--channel_out", nargs='+', type=int, default=[3, 3])
        cmd.add_argument(
                "--upscale_log", type=int, default=2,
                help="upscale times")
        cmd.add_argument(
                "--blk_num", type=int, default=4,
                help="num of blocks for flow net")
        cmd.add_argument(
                "--blk_type", default="dense",
                help="select which type of block to use")
        cmd.add_argument(
                "--invnet", action="store_true",
                help="use inv transform.")
        cmd.add_argument(
                "--non1x1", action="store_true",
                help="train without 1x1 invertible conv.")
        cmd.add_argument(
                "--residual", action="store_true",
                help="use residual block in subnet.")
        cmd.add_argument(
                "--kernel_size", type=int, default=3,
                help="kernel size of subunit conv")
        cmd.add_argument(
                "--nin", action="store_true",
                help="use 1x1 conv in subnet.")
        cmd.add_argument(
                "--norm", default="bn",
                help="which type of norm to use.")
        cmd.add_argument(
                "--reuse_y", action="store_true",
                help="skip quantization and AE&AD.")
        cmd.add_argument(
                "--reuse_z", action="store_true",
                help="skip sampling z.")
        cmd.add_argument(
                "--clamp", action="store_true",
                help="Do clamp on y.")
        cmd.add_argument(
                "--std", type=float, default=1,
                help="std for sampling z.")
        cmd.add_argument(
                "--guidance_type", default="none",
                help="guidance type.")
        cmd.add_argument(
                "--pretrain_checkpoint_dir", default="train",
                help="Directory where to save/load model checkpoints.")
        cmd.add_argument(
                "--zero_z", action="store_true",
                help="whether set z to zeros.")
        cmd.add_argument(
                "--use_y_base", action="store_true",
                help="whether use y_base in reverse process.")
            

        # 'compress' subcommand.
    multi_compress_cmd=subparsers.add_parser(
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
    args=parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand.
    if args.command == "train" or args.command == "inv_train":
        train(args)
    elif args.command == "compress":
        if not args.output_file:
            args.output_file=args.input_file + ".tfci"
        compress(args)
    elif args.command == "decompress":
        if not args.output_file:
            args.output_file=args.input_file + ".png"
        decompress(args)

if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
