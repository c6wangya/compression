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
import modules as m
import flowpp as fpp

from tensorflow.python import debug as tf_debug

import datetime

epsilon = 1e-10

BATCH_SIZE = 8

def print_act_stats(x, _str=""):
    if len(x.get_shape()) == 1:
        x_mean, x_var = tf.nn.moments(x, [0], keep_dims=True)
    if len(x.get_shape()) == 2:
        x_mean, x_var = tf.nn.moments(x, [0], keep_dims=True)
    if len(x.get_shape()) == 4:
        x_mean, x_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
    else:
        return tf.Print(x, [x], "["+_str+"] "+x.name)
    stats = [tf.reduce_min(x_mean), tf.reduce_mean(x_mean), tf.reduce_max(x_mean),
             tf.reduce_min(tf.sqrt(x_var)), tf.reduce_mean(tf.sqrt(x_var)), tf.reduce_max(tf.sqrt(x_var))]
    return tf.Print(x, stats, "["+_str+"] "+x.name)

def read_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    # print(" file name: {}".format(filename))
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    return image

def read_imagenet64(image):
    """Load a imagenet64 file"""
    image = tf.cast(image, tf.float32)
    image /= 255
    return image

def read_int_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    # print(" file name: {}".format(filename))
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
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


def stats_graph(g):
    flops = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


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


def get_normal_random_number(loc, scale, lower=None, upper=None):
    number = np.random.normal(loc=loc, scale=scale)
    if lower != None and upper != None:
        number = upper if number > upper else number
        number = lower if number < lower else number
    return number

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

def lr_schedule(step, mode, warmup_steps=10000, min_ratio=0.1, decay=0.999995):
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

def df_schedule(step, decay_iter, last_iter):
    if step < decay_iter:
        return 1.
    elif step > last_iter:
        return 0.
    return 1 - float((step - decay_iter)) / (last_iter - decay_iter)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def glob_dataset(file_glob, threads, patch_size, batch_size):
    files = glob.glob(file_glob)
    if not files:
        raise RuntimeError(
                "No training images found with glob '{}'.".format(file_glob))
    
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(
            buffer_size=len(files)).repeat()
    dataset = dataset.map(
            read_png, num_parallel_calls=threads)
    dataset = dataset.map(
            lambda x: tf.random_crop(x, (patch_size, patch_size, 3)))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(32)
    return dataset

def glob_imagenet_dataset(file_glob, threads, patch_size, batch_size):
    files = glob.glob(file_glob)
    if not files:
        raise RuntimeError(
                "No training images found with glob '{}'.".format(file_glob))
    xs = []
    for file in files:
        d = unpickle(file)
        x = d['data']
        x = np.dstack((x[:, :patch_size**2], 
                    x[:, patch_size**2:patch_size**2*2], 
                    x[:, patch_size**2*2:]))
        x = x.reshape((x.shape[0], patch_size, patch_size, 3))
        xs.append(x)
    x = np.concatenate(xs, 0)
    x = [x[i, ...] for i in range(x.shape[0])]
    def gen():
        for img in x:
            yield img
    dataset = tf.data.Dataset.from_generator(gen, tf.float32, 
            tf.TensorShape([patch_size, patch_size, 3]))
    dataset = dataset.shuffle(buffer_size=len(x)).repeat()
    dataset = dataset.map(read_imagenet64, num_parallel_calls=threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(32)
    return dataset

def train(args):
    """Trains the model."""

    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu_device)

    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default() as graph:
        # Create input data pipeline.
        with tf.device("/cpu:{}".format(args.gpu_device)):
            tf.set_random_seed(1234)
            glob_func = glob_dataset if args.patchsize == 256 \
                    else glob_imagenet_dataset
            train_dataset = glob_func(args.train_glob,
                                      args.preprocess_threads, 
                                      args.patchsize, 
                                      args.batchsize)
            if args.val_gap != 0:
                val_dataset = glob_func(args.val_glob, 
                                        args.preprocess_threads, 
                                        args.patchsize, 
                                        args.batchsize)

        num_pixels=args.batchsize * args.patchsize ** 2

        # Get training patch from dataset.
        x = train_dataset.make_one_shot_iterator().get_next()

        # Get validation data from dataset
        if args.val_gap != 0:
            x_val = val_dataset.make_one_shot_iterator().get_next()
            print(x_val)

        # add uniform noise
        if args.noise:
            x = tf.add(x, tf.random_uniform(tf.shape(x), 0, 1.))

        # Instantiate model.
        entropy_bottleneck=tfc.EntropyBottleneck()
        if args.command == "train":
            print("training!")
            analysis_transform=m.AnalysisTransform(args.num_filters)
            synthesis_transform=m.SynthesisTransform(args.num_filters)
        else:
            # inv train net
            if args.int_discrete_net:
                inv_transform = m.IntDiscreteNet(blk_type=args.blk_type, 
                        num_filters=args.num_filters, downsample_type=args.downsample_type, 
                        n_levels=args.n_levels, n_flows=args.n_flows)
            else:
                inv_transform = m.InvCompressionNet(channel_in=3, channel_out=args.channel_out, 
                        blk_type=args.blk_type, num_filters=args.num_filters,
                        kernel_size=args.kernel_size, residual=args.residual, 
                        nin=args.nin, norm=args.norm, n_ops=args.n_ops, 
                        downsample_type=args.downsample_type, inv_conv=(not args.non1x1), 
                        inv_conv_init=args.inv_conv_init, use_norm=args.use_norm, int_flow=args.int_flow)
                    
                3, 256, "dense", 256, 2, False, True, "bn", 4, "haar", True

            if args.guidance_type == "baseline_pretrain":
                analysis_transform = m.AnalysisTransform(args.channel_out[0])
                synthesis_transform = m.SynthesisTransform(args.channel_out[0])
            elif args.guidance_type == "baseline":
                analysis_transform = m.AnalysisTransform(args.channel_out[0])

        if args.guidance_type == "grayscale":
            guidance_transform = m.GrayScaleGuidance(rgb_type='RGB', down_scale=4)
    
        """ 1 gpu """
        # Transform Image
        train_flow, train_jac = 0, 0
        if args.command == "train" or args.guidance_type == "baseline_pretrain":
            y = analysis_transform(x)
            if args.guidance_type == "norm": 
                train_y_guidance = tf.reduce_sum(tf.norm(y - 0.5, ord=2, axis=-1, name="guidance_norm"))
            if args.clamp:
                y = tf.clip_by_value(y, 0, 1)
            y_tilde, likelihoods = entropy_bottleneck(y, training=True)
            print("ytilde shape: {}".format(y_tilde.get_shape()))
            x_tilde = synthesis_transform(y_tilde)
            flow_loss_weight = 0
            # validation 
            if args.val_gap != 0:
                y_val = analysis_transform(x_val)
                if args.clamp:
                    y_val = tf.clip_by_value(y_val, 0, 1)
                y_val_hat, _ = entropy_bottleneck(y_val, training=False)
                # compute bpp
                string = entropy_bottleneck.compress(y_val)
                val_num_pixels = args.batchsize * args.patchsize ** 2
                string_len = tf.reduce_sum(tf.cast(tf.strings.length(string), dtype=tf.float32))
                val_bpp = tf.math.divide(string_len * 8, val_num_pixels)
                # y^
                x_val_hat = synthesis_transform(y_val_hat)
                # y
                x_val_hat_reuse_y = synthesis_transform(y_val)
        else:  # For InvCompressionNet
            if args.guidance_type == "baseline":
                y_base = analysis_transform(x)
                if args.prepos_ste: 
                    y_base = m.differentiable_quant(y_base)
            # # place holder for init bool
            # init = tf.placeholder(tf.bool, (), 'init')
            # x = print_act_stats(x, "x")
            out, train_jac = inv_transform(x)
            if not args.int_discrete_net:
                out = out[-1]
            zshapes = []
            
            if out.get_shape()[-1] == args.channel_out[-1]:
                z = out[:, :, :, args.channel_out[-1] - 1:]
            else:
                z = out[:, :, :, args.channel_out[-1]:]
            # z = print_act_stats(z, "z_{}".format(i))
            zshapes.append(tf.shape(z))
            # zshapes.append(z)
            # mle of flow 
            train_flow += tf.reduce_sum(tf.norm(z + epsilon, ord=2, name="last_norm") ** 2)
            if args.train_jacobian:
                train_flow /= -np.log(2) * num_pixels
            
            y = tf.slice(out, [0, 0, 0, 0], [-1, -1, -1, args.channel_out[-1]])
            if args.guidance_type == "norm": 
                train_y_guidance = tf.reduce_sum(tf.norm(y - 0.5, ord=2, axis=-1, name="guidance_norm"))
            if args.clamp:
                y = tf.clip_by_value(y, 0, 1)
            
            if args.prepos_ste:
                y = m.differentiable_quant(y)
            
            if args.no_aux and args.guidance_type == "baseline":
                y_tilde, likelihoods = entropy_bottleneck(tf.stop_gradient(y_base), training=True)
            elif args.no_aux:
                # to compute bpp
                _, likelihoods = entropy_bottleneck(tf.stop_gradient(y), training=True)
            else:
                y_tilde, likelihoods = entropy_bottleneck(y * (255 if args.y_scale_up else \
                                                          1), training=True)

            input_rev = []
            if args.ste or args.prepos_ste:
                y_tilde = m.differentiable_quant(y_tilde)
            
            if args.y_scale_up:
                assert not args.no_aux and args.guidance_type != "baseline"
                y_tilde /= 255
            input_rev.append(y_tilde)
            for zshape in zshapes:
                if args.zero_z:
                    input_rev.append(tf.zeros(shape=zshape))
                else:
                    input_rev.append(tf.random_normal(shape=zshape))
            input_rev = tf.concat(input_rev, axis=-1)
            x_tilde, _ = inv_transform(input_rev, rev=True)
            if not args.int_discrete_net:
                x_tilde = x_tilde[-1]
            flow_loss_weight = args.flow_loss_weight

            # validation 
            if args.val_gap != 0:
                # baseline y
                if args.no_aux and args.guidance_type == "baseline":
                    y_val_base = analysis_transform([x])
                out, _ = inv_transform(x_val)
                if not args.int_discrete_net:
                    out = out[-1]
                
                # z_samples and z_zeros
                zshapes = []
                if out.get_shape()[-1] == args.channel_out[-1]:
                    z = out[:, :, :, args.channel_out[-1] - 1:]
                else:
                    z = out[:, :, :, args.channel_out[-1]:]
                zshapes.append(tf.shape(z))
                z_samples = [tf.random_normal(shape=zshape) for zshape in zshapes]
                z_zeros = [tf.zeros(shape=zshape) for zshape in zshapes]
                
                # y hat
                y_val = tf.slice(out, [0, 0, 0, 0], [-1, -1, -1, args.channel_out[-1]])
                if args.clamp: 
                    y_val = tf.clip_by_value(y_val, 0, 1)
                y_val_hat, _ = entropy_bottleneck(y_val * (255 if args.y_scale_up else 1), training=False)
                if args.y_scale_up:
                    y_val_hat /= 255

                # compute bpp
                string = entropy_bottleneck.compress(y_val * (255 if args.y_scale_up else 1))
                val_num_pixels = args.batchsize * args.patchsize ** 2
                string_len = tf.reduce_sum(tf.cast(tf.strings.length(string), dtype=tf.float32))
                val_bpp = tf.math.divide(string_len * 8, val_num_pixels)
                # y^, z^
                x_val_y_hat_z_hat, _ = inv_transform(tf.concat([y_val_hat] + z_samples, axis=-1), rev=True)
                # y^, 0
                x_val_y_hat_z_0, _ = inv_transform(tf.concat([y_val_hat] + z_zeros, axis=-1), rev=True)
                # y, z^
                x_val_y_z_hat, _ = inv_transform(tf.concat([y_val] + z_samples, axis=-1), rev=True)
                # y, 0
                x_val_y_z_0, _ = inv_transform(tf.concat([y_val] + z_zeros, axis=-1), rev=True)
                # y, z
                x_val_y_z, _ = inv_transform(tf.concat([y_val] + [z], axis=-1), rev=True)

                # baseline y hat & x hat with y base
                if args.no_aux and args.guidance_type == "baseline":
                    y_base_val_hat, _ = entropy_bottleneck(y_val_base, training=False)
                    # y base^, z^
                    x_val_y_base_hat_z_hat, _ = inv_transform(tf.concat([y_base_val_hat] + z_samples, axis=-1), rev=True)
                    # y base^, 0
                    x_val_y_base_hat_z_0, _ = inv_transform(tf.concat([y_base_val_hat] + z_zeros, axis=-1), rev=True)
                    string_base = entropy_bottleneck.compress(y_val_base)
                    string_base_len = tf.cast(tf.strings.length(string_base), dtype=tf.float32)
                    string_base_len = tf.reduce_sum(string_base_len)
                    val_base_bpp = tf.math.divide(string_base_len * 8, val_num_pixels)

        if args.guidance_type == "grayscale":
            y_guidance = guidance_transform(x)
        # Total number of bits divided by number of pixels.
        train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

        # Mean squared error across pixels.
        train_mse=tf.reduce_mean(tf.squared_difference(x, x_tilde))
        # Multiply by 255^2 to correct for rescaling.
        train_mse *= 255 ** 2

        # Compute y's guidance across pixels
        if args.guidance_type == "grayscale":
            train_y_guidance = tf.reduce_sum(tf.squared_difference(y, y_guidance))
        elif args.guidance_type == "norm":
            # train_y_guidance = tf.reduce_sum(tf.norm(y - 0.5, ord=2, axis=-1, name="guidance_norm"))
            pass
        elif args.guidance_type == "likelihood":
            train_y_guidance = entropy_bottleneck.losses[0]
        elif args.guidance_type == "baseline":
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
        # train_loss = print_act_stats(train_loss, "overall train loss")
        
        if args.command == "train" or "baseline" not in args.guidance_type:
            tvars = tf.trainable_variables()
        elif args.guidance_type == "baseline_pretrain":
            tvars = analysis_transform.trainable_variables + \
                    synthesis_transform.trainable_variables + \
                    entropy_bottleneck.trainable_variables
        elif args.freeze_aux:  # freeze auxiliary network
            tvars = inv_transform.trainable_variables
        else:  # not freeze aux net
            tvars = inv_transform.trainable_variables + \
                entropy_bottleneck.trainable_variables

        filtered_vars = [var for var in tvars \
                if not 'haar_downsampling' in var.name \
                and not 'gray_scale_guidance' in var.name]
        if args.debug_mode:
            assert not (len([var for var in tvars if 'haar_downsampling' in var.name]) == 0 \
                and args.downsample_type == "haar"), \
                "there's no variable called haar_downsampling! \n"
            assert not (len([var for var in tvars if 'gray_scale_guidance' in var.name]) == 0 \
                and args.y_guidance_weight != 0), \
                "there's no variable called gray_scale_guidance! \n"
            print("Has variables {} in total, and filtered out {} variables\n".format( \
                    len(tvars), len(tvars) - len(filtered_vars)))
        # Minimize loss and auxiliary loss, and execute update op.
        main_lr = tf.placeholder(tf.float32, [], 'main_lr')
        aux_lr = tf.placeholder(tf.float32, [], 'aux_lr')
        step = tf.train.create_global_step()
        main_optimizer=tf.train.AdamOptimizer(learning_rate=main_lr)
        main_gradients, main_variables = zip(*main_optimizer.compute_gradients(train_loss, filtered_vars))
        main_gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, args.grad_clipping)
            for gradient in main_gradients]
        main_step = main_optimizer.apply_gradients(zip(main_gradients, main_variables), global_step=step)
        
        if not args.freeze_aux:
            aux_optimizer=tf.train.AdamOptimizer(learning_rate=aux_lr)
            aux_gradients, aux_variables = zip(*aux_optimizer.compute_gradients( \
                    entropy_bottleneck.losses[0], filtered_vars))
            aux_gradients = [
                None if gradient is None else tf.clip_by_norm(gradient, args.grad_clipping)
                for gradient in aux_gradients]
            aux_step = aux_optimizer.apply_gradients(zip(aux_gradients, aux_variables))
            # group training operations
            train_op=tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
        else:
            train_op=tf.group(main_step)

        if args.val_gap != 0:
            def comp_psnr(img_hat, img):
                img *= 255
                img_hat=tf.clip_by_value(img_hat, 0, 1)
                img_hat=tf.round(img_hat * 255)
                if isinstance(img_hat, list):
                    img_hat = img_hat[-1]
                img_hat = tf.reshape(img_hat, [-1, int(img.shape[1]), int(img.shape[1]), 3])
                print("image shape: {} and imghat shape: {}".format(img.get_shape(), img_hat.get_shape()))
                rgb_psnr = tf.squeeze(tf.reduce_mean(tf.image.psnr(img_hat, img, 255)))
                luma_img = tf.image.rgb_to_yuv(img)[:, :, :, 0]
                luma_img_hat = tf.image.rgb_to_yuv(img_hat)[:, :, :, 0]
                # luma_img = tf.slice(tf.image.rgb_to_yuv(img), [0, 0, 0, 0], [-1, -1, -1, 1])
                # luma_img_hat = tf.slice(tf.image.rgb_to_yuv(img_hat), [0, 0, 0, 0], [-1, -1, -1, 1])
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
                # y, z
                val_y_z_rgb_psnr, val_y_z_luma_psnr = comp_psnr(x_val_y_z, x_val)
                # summary
                tf.summary.scalar("validation-yhat-zhat-rgb-psnr", val_y_hat_z_hat_rgb_psnr)
                tf.summary.scalar("validation-yhat-zhat-luma-psnr", val_y_hat_z_hat_luma_psnr)
                tf.summary.scalar("validation-yhat-z0-rgb-psnr", val_y_hat_z_0_rgb_psnr)
                tf.summary.scalar("validation-yhat-z0-luma-psnr", val_y_hat_z_0_luma_psnr)
                tf.summary.scalar("validation-y-zhat-rgb-psnr", val_y_z_hat_rgb_psnr)
                tf.summary.scalar("validation-y-zhat-luma-psnr", val_y_z_hat_luma_psnr)
                tf.summary.scalar("validation-y-z0-rgb-psnr", val_y_z_0_rgb_psnr)
                tf.summary.scalar("validation-y-z0-luma-psnr", val_y_z_0_luma_psnr)
                tf.summary.scalar("validation-y-z-rgb-psnr", val_y_z_rgb_psnr)
                tf.summary.scalar("validation-y-z-luma-psnr", val_y_z_luma_psnr)
                tf.summary.scalar("validation-bpp", val_bpp)
                # group operations
                val_op_lst = [val_y_hat_z_hat_rgb_psnr, val_y_hat_z_hat_luma_psnr, 
                          val_y_hat_z_0_rgb_psnr, val_y_hat_z_0_luma_psnr, 
                          val_y_z_hat_rgb_psnr, val_y_z_hat_luma_psnr, 
                          val_y_z_0_rgb_psnr, val_y_z_0_luma_psnr, 
                          val_y_z_rgb_psnr, val_y_z_luma_psnr]
                        #   val_bpp]
                val_bpp_op_list = [val_bpp]
                if args.no_aux and args.guidance_type == "baseline":
                    # y base^, z^
                    val_y_base_hat_z_hat_rgb_psnr, val_y_base_hat_z_hat_luma_psnr = comp_psnr(x_val_y_base_hat_z_hat, x_val)
                    # y base^, 0
                    val_y_base_hat_z_0_rgb_psnr, val_y_base_hat_z_0_luma_psnr = comp_psnr(x_val_y_base_hat_z_0, x_val)
                    # summary 
                    tf.summary.scalar("validation-ybasehat-zhat-rgb-psnr", val_y_base_hat_z_hat_rgb_psnr)
                    tf.summary.scalar("validation-ybasehat-zhat-luma-psnr", val_y_base_hat_z_hat_luma_psnr)
                    tf.summary.scalar("validation-ybasehat-z0-rgb-psnr", val_y_base_hat_z_0_rgb_psnr)
                    tf.summary.scalar("validation-ybasehat-z0-luma-psnr", val_y_base_hat_z_0_luma_psnr)
                    tf.summary.scalar("validation-base-bpp", val_base_bpp)
                    # group operations
                    val_op_lst += [val_y_base_hat_z_hat_rgb_psnr, val_y_base_hat_z_hat_luma_psnr, 
                               val_y_base_hat_z_0_rgb_psnr, val_y_base_hat_z_0_luma_psnr]
                            #    val_base_bpp]
                    val_bpp_op_list.append(val_base_bpp)
            val_op = tf.group(*val_op_lst)
            val_bpp_op = tf.group(*val_bpp_op_list)

        tf.summary.scalar("main-learning-rates", main_lr)

        tf.summary.scalar("loss", train_loss)
        tf.summary.scalar("bpp", train_bpp)
        tf.summary.scalar("mse", train_mse)

        psnr=tf.squeeze(tf.reduce_mean(tf.image.psnr(x_tilde, x, 255)))
        msssim=tf.squeeze(tf.reduce_mean(
            tf.image.ssim_multiscale(x_tilde, x, 255, 
                filter_size=11 if args.patchsize > 64 else 4)))
        tf.summary.scalar("psnr", psnr)
        tf.summary.scalar("msssim", msssim)

        tf.summary.image("original", quantize_image(x))
        tf.summary.image("reconstruction", quantize_image(x_tilde))
        
        # stats_graph(graph)

        # total_parameters = 0
        # for variable in tf.trainable_variables():
        #     # shape is an array of tf.Dimension
        #     shape = variable.get_shape()
        #     # print(shape)
        #     # print(len(shape))
        #     variable_parameters = 1
        #     for dim in shape:
        #         # print(dim)
        #         variable_parameters *= dim.value
        #     # print(variable_parameters)
        #     total_parameters += variable_parameters
        # print("\n[network capacity] total num of parameters -> {}\n\n".format(total_parameters))

        hooks=[
                tf.train.StopAtStepHook(last_step=args.last_step),
                tf.train.NanTensorHook(train_loss),
                ]
        
        # init saver for all the models
        if "baseline" in args.guidance_type:
            analysis_saver = tf.train.Saver(analysis_transform.variables, max_to_keep=1)
            entropy_saver = tf.train.Saver(entropy_bottleneck.variables, max_to_keep=1)
            if args.guidance_type == "baseline_pretrain":
                synthesis_saver = tf.train.Saver(synthesis_transform.variables, max_to_keep=1)
            elif args.guidance_type == "baseline":
                inv_saver = tf.train.Saver(inv_transform.variables, max_to_keep=1)
        
        global_iters = 0
        with tf.train.MonitoredTrainingSession(
                    hooks=hooks, checkpoint_dir=args.checkpoint_dir,
                    save_checkpoint_secs=1000, save_summaries_secs=300) as sess:
            if "baseline" not in args.guidance_type or args.finetune:
                while not sess.should_stop():
                    lr = lr_schedule(global_iters, 
                                     args.lr_scheduler, 
                                     args.lr_warmup_steps, 
                                     args.lr_min_ratio, 
                                     args.lr_decay)
                    sess.run(train_op, {main_lr: args.main_lr * lr, 
                                        aux_lr: args.aux_lr * lr})
                    if args.val_gap != 0 and global_iters % args.val_gap == 0:
                        sess.run(val_op)
                        sess.run(val_bpp_op)
                    global_iters += 1
            else:
                if args.finetune:
                    if args.guidance_type == "baseline_pretrain":
                        # load analysis, synthesis and entropybottleneck model
                        restore_weights(synthesis_saver, get_session(sess), 
                                args.pretrain_checkpoint_dir + "/syn_net")
                        restore_weights(analysis_saver, get_session(sess), 
                                args.pretrain_checkpoint_dir + "/ana_net")
                        restore_weights(entropy_saver, get_session(sess), 
                                args.pretrain_checkpoint_dir + "/entro_net")
                    elif args.guidance_type == "baseline":
                        # load invertible model
                        restore_weights(inv_saver, get_session(sess), 
                                args.pretrain_checkpoint_dir + "/inv_net")
                if args.guidance_type == "baseline":
                    # load analysis and entropybottleneck model
                    restore_weights(analysis_saver, get_session(sess), 
                            args.pretrain_checkpoint_dir + "/ana_net")
                    restore_weights(entropy_saver, get_session(sess), 
                            args.pretrain_checkpoint_dir + "/entro_net")
                while not sess.should_stop():
                    lr = lr_schedule(global_iters, 
                                     args.lr_scheduler, 
                                     args.lr_warmup_steps, 
                                     args.lr_min_ratio, 
                                     args.lr_decay)
                    sess.run(train_op, {main_lr: args.main_lr * lr, 
                                        aux_lr: args.aux_lr * lr})
                    if args.val_gap != 0 and global_iters % args.val_gap == 0:
                        sess.run(val_op)
                        sess.run(val_bpp_op)
                    if global_iters % 5000 == 0:
                        if args.guidance_type == "baseline_pretrain":
                            # save analysis, synthesis and entropybottleneck model
                            save_weights(synthesis_saver, get_session(sess), 
                                    args.checkpoint_dir + '/syn_net', global_iters)
                            save_weights(analysis_saver, get_session(sess), 
                                    args.checkpoint_dir + '/ana_net', global_iters)
                            save_weights(entropy_saver, get_session(sess), 
                                    args.checkpoint_dir + '/entro_net', global_iters)
                        elif args.guidance_type == "baseline":
                            save_weights(inv_saver, get_session(sess), 
                                    args.checkpoint_dir + '/inv_net', global_iters)
                            save_weights(entropy_saver, get_session(sess), 
                                    args.checkpoint_dir + '/entro_net', global_iters)
                    global_iters += 1


def int_train(args):
    """Trains the model."""

    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu_device)

    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default() as graph:
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
            train_dataset=train_dataset.map(
                    lambda x: tf.random_crop(x, (int(args.patchsize), int(args.patchsize), 3)))
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
                val_dataset=val_dataset.batch(1)
                val_dataset=val_dataset.prefetch(32)

        num_pixels = args.batchsize * args.patchsize ** 2

        # Get training patch from dataset.
        x = train_dataset.make_one_shot_iterator().get_next()

        # Get validation data from dataset
        if args.val_gap != 0:
            x_val = val_dataset.make_one_shot_iterator().get_next()
            print(x_val)

        # add uniform noise
        if args.noise:
            x = tf.add(x, tf.random_uniform(tf.shape(x), 0, 1.))

        # Instantiate model.
        entropy_bottleneck=tfc.EntropyBottleneck()
        # inv train net
        inv_transform = m.IntDiscreteNet(blk_type=args.blk_type, 
                num_filters=args.num_filters, downsample_type=args.downsample_type, 
                n_levels=args.n_levels, n_flows=args.n_flows)
        if args.guidance_type == "baseline":
            analysis_transform = m.AnalysisTransform(args.channel_out[0])

        """ 1 gpu """
        # Transform Image
        train_flow, train_jac = 0, 0
        if args.guidance_type == "baseline":
            y_base = analysis_transform(x)
            if args.prepos_ste: 
                y_base = m.differentiable_round(y_base)
        out, train_jac = inv_transform(x)
        z = out[:, :, :, args.channel_out[-1]:]
        # mle of flow 
        train_flow += tf.reduce_sum(tf.norm(z + epsilon, ord=2, axis=-1, name="last_norm"))
        if args.train_jacobian:
            train_flow /= -np.log(2) * num_pixels
        y = tf.slice(out, [0, 0, 0, 0], [-1, -1, -1, args.channel_out[-1]])
        # prepos ste
        # if args.prepos_ste:
        #     y = m.differentiable_quant(y)
        if args.y_scale_up:
            y *= 255
        y_tilde, likelihoods = entropy_bottleneck(y, training=True)
        # if args.ste or args.prepos_ste:
        #     y_tilde = m.differentiable_round(y_tilde)
        if args.y_scale_up:
            y_tilde = y_tilde / 255
        input_rev = [y_tilde if not args.guidance_type == "baseline" else y_base / 255., 
                     tf.zeros(shape=tf.shape(z))]
        input_rev = tf.concat(input_rev, axis=-1)
        x_tilde, _ = inv_transform(input_rev, rev=True)
        flow_loss_weight = args.flow_loss_weight

        # validation 
        if args.val_gap != 0:
            if args.guidance_type == "baseline":
                base_out = analysis_transform(x_val)
                # if args.prepos_ste: 
                #     base_out = m.differentiable_round(base_out)
            out, _ = inv_transform(x_val)
            
            # z_samples and z_zeros
            z = out[:, :, :, args.channel_out[-1]:]
            z_zeros = tf.zeros(shape=tf.shape(z))
            
            # y hat
            y_val = tf.slice(out, [0, 0, 0, 0], [-1, -1, -1, args.channel_out[-1]])
            y_val_hat, _ = entropy_bottleneck(y_val * (255 if args.y_scale_up else 1), training=False)
            if args.y_scale_up:
                y_val_hat /= 255
            # compute bpp
            string = entropy_bottleneck.compress(y_val * (255 if args.y_scale_up else 1))
            val_num_pixels = 1 * 512 ** 2
            string_len = tf.reduce_sum(tf.cast(tf.strings.length(string), dtype=tf.float32))
            val_bpp = tf.math.divide(string_len * 8, val_num_pixels)
            if args.guidance_type == "baseline":
                base_string = entropy_bottleneck.compress(base_out)
                base_string_len = tf.reduce_sum(tf.cast(tf.strings.length(base_string), dtype=tf.float32))
                base_val_bpp = tf.math.divide(base_string_len * 8, val_num_pixels)
            # y^, 0
            x_val_y_hat_z_0, _ = inv_transform(tf.concat([y_val_hat, z_zeros], axis=-1), rev=True)
            # y, 0
            x_val_y_z_0, _ = inv_transform(tf.concat([y_val, z_zeros], axis=-1), rev=True)
            # y, z
            x_val_y_z, _ = inv_transform(tf.concat([y_val, z], axis=-1), rev=True)

        # Total number of bits divided by number of pixels.
        train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

        # Mean squared error across pixels.
        train_mse=tf.reduce_mean(tf.squared_difference(x, x_tilde))
        # Multiply by 255^2 to correct for rescaling.
        train_mse *= 255 ** 2

        # Compute y's guidance across pixels
        if args.guidance_type == "baseline":
            train_y_guidance = tf.reduce_sum(tf.squared_difference(y, tf.stop_gradient(y_base)))
        else:
            train_y_guidance = 0
        if not args.train_jacobian:
            train_jac = 0

        # The rate-distortion cost.
        decay_factor = tf.placeholder("float")
        train_loss = args.lmbda * train_mse + \
                    args.beta * train_bpp + \
                    flow_loss_weight * (train_flow + train_jac) + \
                    args.y_guidance_weight * decay_factor * train_y_guidance
        
        # Minimize loss and auxiliary loss, and execute update op.
        main_lr = tf.placeholder(tf.float32, [], 'main_lr')
        aux_lr = tf.placeholder(tf.float32, [], 'aux_lr')
        step = tf.train.create_global_step()
        if args.optimize_separately:  # train loss won't affect auxiliary net
            main_vars = inv_transform.trainable_variables
            aux_vars = entropy_bottleneck.trainable_variables
            # main optimizer
            main_optimizer = tf.train.AdamOptimizer(learning_rate=main_lr)
            main_gradients, main_variables = zip(*main_optimizer.compute_gradients(train_loss, main_vars))
            main_gradients = [
                None if gradient is None else tf.clip_by_norm(gradient, args.grad_clipping)
                for gradient in main_gradients]
            main_step = main_optimizer.apply_gradients(zip(main_gradients, main_variables), global_step=step)
            if args.train_aux:
                # auxiliary optimizer
                aux_optimizer = tf.train.AdamOptimizer(learning_rate=aux_lr)
                aux_gradients, aux_variables = zip(*aux_optimizer.compute_gradients( \
                        entropy_bottleneck.losses[0], aux_vars))
                aux_gradients = [
                    None if gradient is None else tf.clip_by_norm(gradient, args.grad_clipping)
                    for gradient in aux_gradients]
                aux_step = aux_optimizer.apply_gradients(zip(aux_gradients, aux_variables))
                # group training operations
                train_op=tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
            else:
                train_op=tf.group(main_step)
        else:
            if "baseline" not in args.guidance_type:
                tvars = tf.trainable_variables()
            else:
                tvars = inv_transform.trainable_variables + \
                    entropy_bottleneck.trainable_variables
            filtered_vars = [var for var in tvars \
                    if not 'haar_downsampling' in var.name \
                    and not 'gray_scale_guidance' in var.name]
            # main optimizer
            main_optimizer=tf.train.AdamOptimizer(learning_rate=main_lr)
            main_gradients, main_variables = zip(*main_optimizer.compute_gradients(train_loss, filtered_vars))
            main_gradients = [
                None if gradient is None else tf.clip_by_norm(gradient, args.grad_clipping)
                for gradient in main_gradients]
            main_step = main_optimizer.apply_gradients(zip(main_gradients, main_variables), global_step=step)
            if args.train_aux:
                # auxiliary optimizer
                aux_optimizer = tf.train.AdamOptimizer(learning_rate=aux_lr)
                aux_gradients, aux_variables = zip(*aux_optimizer.compute_gradients( \
                        entropy_bottleneck.losses[0], filtered_vars))
                aux_gradients = [
                    None if gradient is None else tf.clip_by_norm(gradient, args.grad_clipping)
                    for gradient in aux_gradients]
                aux_step = aux_optimizer.apply_gradients(zip(aux_gradients, aux_variables))
                # group training operations
                train_op=tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
            else:
                train_op=tf.group(main_step)

        if args.val_gap != 0:
            def comp_psnr(img_hat, img):
                img *= 255
                img_hat=tf.clip_by_value(img_hat, 0, 1)
                img_hat=tf.round(img_hat * 255)
                if args.command == "inv_train" and \
                        args.guidance_type != "baseline_pretrain" and \
                        not args.int_discrete_net:
                    img_hat = img_hat[-1]
                rgb_psnr = tf.squeeze(tf.reduce_mean(tf.image.psnr(img_hat, img, 255)))
                luma_img = tf.slice(tf.image.rgb_to_yuv(img), [0, 0, 0, 0], [-1, -1, -1, 1])
                luma_img_hat = tf.slice(tf.image.rgb_to_yuv(img_hat), [0, 0, 0, 0], [-1, -1, -1, 1])
                luma_psnr = tf.squeeze(tf.reduce_mean(tf.image.psnr(luma_img_hat, luma_img, 255)))
                return rgb_psnr, luma_psnr
            
            # y^, 0
            val_y_hat_z_0_rgb_psnr, val_y_hat_z_0_luma_psnr = comp_psnr(x_val_y_hat_z_0, x_val)
            # y, 0
            val_y_z_0_rgb_psnr, val_y_z_0_luma_psnr = comp_psnr(x_val_y_z_0, x_val)
            # y, z
            val_y_z_rgb_psnr, val_y_z_luma_psnr = comp_psnr(x_val_y_z, x_val)
            # summary
            tf.summary.scalar("validation-yhat-z0-rgb-psnr", val_y_hat_z_0_rgb_psnr)
            tf.summary.scalar("validation-yhat-z0-luma-psnr", val_y_hat_z_0_luma_psnr)
            tf.summary.scalar("validation-y-z0-rgb-psnr", val_y_z_0_rgb_psnr)
            tf.summary.scalar("validation-y-z0-luma-psnr", val_y_z_0_luma_psnr)
            tf.summary.scalar("validation-y-z-rgb-psnr", val_y_z_rgb_psnr)
            tf.summary.scalar("validation-y-z-luma-psnr", val_y_z_luma_psnr)
            tf.summary.scalar("validation-bpp", val_bpp)
            if args.guidance_type == "baseline":
                tf.summary.scalar("baseline-validation-bpp", base_val_bpp)
            # group operations
            val_op_lst = [val_y_hat_z_0_rgb_psnr, val_y_hat_z_0_luma_psnr, 
                        val_y_z_0_rgb_psnr, val_y_z_0_luma_psnr, 
                        val_y_z_rgb_psnr, val_y_z_luma_psnr]
                    #   val_bpp]
            val_bpp_op_list = [val_bpp]
            if args.guidance_type == "baseline":
                val_bpp_op_list.append(base_val_bpp)
            val_op = tf.group(*val_op_lst)
            val_bpp_op = tf.group(*val_bpp_op_list)

        tf.summary.scalar("main-learning-rates", main_lr)

        tf.summary.scalar("loss", train_loss)
        tf.summary.scalar("bpp", train_bpp)
        tf.summary.scalar("mse", train_mse)

        psnr=tf.squeeze(tf.reduce_mean(tf.image.psnr(x_tilde, x, 255)))
        msssim=tf.squeeze(tf.reduce_mean(
            tf.image.ssim_multiscale(x_tilde, x, 255, 
                filter_size=11 if args.patchsize > 64 else 4)))
        tf.summary.scalar("psnr", psnr)
        tf.summary.scalar("msssim", msssim)

        tf.summary.image("original", quantize_image(x))
        tf.summary.image("reconstruction", quantize_image(x_tilde))
        
        # stats_graph(graph)

        # total_parameters = 0
        # for variable in tf.trainable_variables():
        #     # shape is an array of tf.Dimension
        #     shape = variable.get_shape()
        #     # print(shape)
        #     # print(len(shape))
        #     variable_parameters = 1
        #     for dim in shape:
        #         # print(dim)
        #         variable_parameters *= dim.value
        #     # print(variable_parameters)
        #     total_parameters += variable_parameters
        # print("\n[network capacity] total num of parameters -> {}\n\n".format(total_parameters))

        hooks=[
                tf.train.StopAtStepHook(last_step=args.last_step),
                tf.train.NanTensorHook(train_loss),
                ]
        
        # init saver for all the models
        if "baseline" in args.guidance_type:
            analysis_saver = tf.train.Saver(analysis_transform.variables, max_to_keep=1)
            entropy_saver = tf.train.Saver(entropy_bottleneck.variables, max_to_keep=1)
            if args.guidance_type == "baseline":
                inv_saver = tf.train.Saver(inv_transform.variables, max_to_keep=1)
        
        global_iters = 0
        with tf.train.MonitoredTrainingSession(
                    hooks=hooks, checkpoint_dir=args.checkpoint_dir,
                    save_checkpoint_secs=1000, save_summaries_secs=300) as sess:
            if "baseline" not in args.guidance_type or args.finetune:
                while not sess.should_stop():
                    lr = lr_schedule(global_iters, 
                                     args.lr_scheduler, 
                                     args.lr_warmup_steps, 
                                     args.lr_min_ratio, 
                                     args.lr_decay)
                    df = df_schedule(global_iters, args.df_iter, args.end_iter)
                    sess.run(train_op, {main_lr: args.main_lr * lr, 
                                        aux_lr: args.aux_lr * lr,
                                        decay_factor: df})
                    if args.val_gap != 0 and global_iters % args.val_gap == 0:
                        sess.run(val_op, {decay_factor: df})
                        sess.run(val_bpp_op, {decay_factor: df})
                    global_iters += 1
            else:
                if args.finetune:
                    if args.guidance_type == "baseline":
                        # load invertible model
                        restore_weights(inv_saver, get_session(sess), 
                                args.pretrain_checkpoint_dir + "/inv_net")
                if args.guidance_type == "baseline":
                    # load analysis and entropybottleneck model
                    restore_weights(analysis_saver, get_session(sess), 
                            args.pretrain_checkpoint_dir + "/ana_net")
                    restore_weights(entropy_saver, get_session(sess), 
                            args.pretrain_checkpoint_dir + "/entro_net")
                while not sess.should_stop():
                    lr = lr_schedule(global_iters, 
                                     args.lr_scheduler, 
                                     args.lr_warmup_steps, 
                                     args.lr_min_ratio, 
                                     args.lr_decay)
                    df = df_schedule(global_iters, args.df_iter, args.end_iter)
                    sess.run(train_op, {main_lr: args.main_lr * lr, 
                                        aux_lr: args.aux_lr * lr, 
                                        decay_factor: df})
                    if args.val_gap != 0 and global_iters % args.val_gap == 0:
                        sess.run(val_op, {decay_factor: df})
                        sess.run(val_bpp_op, {decay_factor: df})
                    if global_iters % 5000 == 0:
                        if args.guidance_type == "baseline":
                            save_weights(inv_saver, get_session(sess), 
                                    args.checkpoint_dir + '/inv_net', global_iters)
                            save_weights(entropy_saver, get_session(sess), 
                                    args.checkpoint_dir + '/entro_net', global_iters)
                    global_iters += 1


def compress(args):
    """Compresses an image."""
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu_device)

    with tf.Graph().as_default() as graph:
        tf.set_random_seed(1234)
        # Load input image and add batch dimension.
        x=read_png(args.input_file)
        x=tf.expand_dims(x, 0)
        x.set_shape([1, None, None, 3])
        x_shape=tf.shape(x)
        
        # randomly crop img to 256x256
        x = tf.random_crop(x, (1, 256, 256, 3))

        # Instantiate model.
        entropy_bottleneck=tfc.EntropyBottleneck()
        if not args.invnet:
            analysis_transform=m.AnalysisTransform(args.num_filters)
            synthesis_transform=m.SynthesisTransform(args.num_filters)
        else:
            # inv_transform = m.InvHSRNet(3, channel_out=args.channel_out, 
            #         upscale_log=args.upscale_log, block_num=[args.blk_num, args.blk_num], 
            #         blk_type=args.blk_type, num_filters=args.num_filters, 
            #         use_inv_conv=args.inv_conv, kernel_size=args.kernel_size, residual=args.residual)
            inv_transform = m.InvCompressionNet(channel_in=3, channel_out=args.channel_out, 
                    blk_type=args.blk_type, num_filters=args.num_filters,
                    kernel_size=args.kernel_size, residual=args.residual, 
                    nin=args.nin, norm=args.norm, n_ops=args.n_ops, 
                    downsample_type=args.downsample_type, inv_conv=(not args.non1x1), 
                    use_norm=args.use_norm, int_flow=args.int_flow)
            if "baseline" in args.guidance_type:
                analysis_transform = m.AnalysisTransform(args.channel_out[0])
                synthesis_transform = m.SynthesisTransform(args.channel_out[0])
            elif args.use_y_base:
                analysis_transform = m.AnalysisTransform(args.channel_out[0])

        # Transform and compress the image.
        if not args.invnet:
            y=analysis_transform(x)
            if args.clamp:
                y = tf.clip_by_value(y, 0, 1)
            # Transform the quantized image back (if requested).
            y_hat, likelihoods=entropy_bottleneck(y, training=False)
            x_hat=synthesis_transform(y_hat if not args.reuse_y else y)
            x_hat=x_hat[:, :x_shape[1], :x_shape[2], :]
        else:
            # For InvCompressionNet
            # x = print_act_stats(x, "x")
            out, _ = inv_transform([x])
            zshapes = []
            
            if out[-1].get_shape()[-1] == args.channel_out[-1]:
                z = out[-1][:, :, :, args.channel_out[-1] - 1:]
            else:
                z = out[-1][:, :, :, args.channel_out[-1]:]
            print(z.get_shape())
            # z = print_act_stats(z, "z_{}".format(i))
            if not args.reuse_z:
                zshapes.append(tf.shape(z))
            else:
                zshapes.append(z)
            y = tf.slice(out[-1], [0, 0, 0, 0], [-1, -1, -1, args.channel_out[-1]])
            if args.clamp:
                y = tf.clip_by_value(y, 0, 1)

            if args.use_y_base:
                y_base = analysis_transform(x)
                y_hat, likelihoods = entropy_bottleneck(y_base, training=False)
            else:
                y_hat, likelihoods = entropy_bottleneck(y, training=False)
            # train_flow = print_act_stats(train_flow, "train flow loss")
            # y_tilde = print_act_stats(y_tilde, "y_tilde"
            if not args.reuse_z and args.zero_z:
                input_rev = [tf.zeros(shape=zshape) for zshape in zshapes]
            elif not args.reuse_z:
                input_rev = [tf.random_normal(shape=zshape, stddev=args.std) for zshape in zshapes]
            else:
                input_rev = zshapes
            input_rev.append(y_hat if not args.reuse_y else y)
            # input_rev.append(y)
            x_hat, _ = inv_transform(input_rev, rev=True)
            x_hat = x_hat[-1]
        
        string=entropy_bottleneck.compress(y)
        num_pixels=tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

        # Total number of bits divided by number of pixels.
        eval_bpp=tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

        # Bring both images back to 0..255 range.
        x *= 255
        x_hat=tf.clip_by_value(x_hat, 0, 1)
        x_hat=tf.round(x_hat * 255)

        mse=tf.reduce_mean(tf.squared_difference(x, x_hat))
        rgb_psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
        rgb_msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255, 
            filter_size=11 if args.patchsize > 64 else 4))

        luma_x=tf.slice(tf.image.rgb_to_yuv(x), [0, 0, 0, 0], [-1, -1, -1, 1])
        luma_x_hat=tf.slice(tf.image.rgb_to_yuv(x_hat), [
            0, 0, 0, 0], [-1, -1, -1, 1])
        luma_psnr=tf.squeeze(tf.image.psnr(luma_x_hat, luma_x, 255))
        luma_msssim=tf.squeeze(tf.image.ssim_multiscale(luma_x_hat, luma_x, 255, 
            filter_size=11 if args.patchsize > 64 else 4))

        chroma_x=tf.slice(tf.image.rgb_to_yuv(x), [0, 0, 0, 1], [-1, -1, -1, 2])
        chroma_x_hat=tf.slice(tf.image.rgb_to_yuv(
            x_hat), [0, 0, 0, 1], [-1, -1, -1, 2])
        chroma_psnr=tf.squeeze(tf.image.psnr(chroma_x_hat, chroma_x, 255))
        chroma_msssim=tf.squeeze(
                tf.image.ssim_multiscale(chroma_x_hat, chroma_x, 255, 
                    filter_size=11 if args.patchsize > 64 else 4))

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("\n[network capacity] total num of parameters -> {}\n\n".format(total_parameters))

        # stats_graph(graph)

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Load the latest model checkpoint.
            if args.guidance_type == "baseline":
                sess.run(init_op)
                # init savers
                # analysis_saver = tf.train.Saver(analysis_transform.trainable_variables)
                entropy_saver = tf.train.Saver(entropy_bottleneck.variables)
                inv_saver = tf.train.Saver(inv_transform.variables)
                # restore weights
                restore_weights(inv_saver, sess, 
                        args.checkpoint_dir + "/inv_net")
                # restore_weights(analysis_saver, sess, 
                #         args.pretrain_checkpoint_dir + "/ana_net")
                # restore_weights(entropy_saver, sess, 
                #         args.pretrain_checkpoint_dir + "/entro_net")
            elif args.guidance_type == "baseline_pretrain":
                sess.run(init_op)
                # latest=tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
                # tf.train.Saver().restore(sess, save_path=latest)
                # init savers
                entropy_saver = tf.train.Saver(entropy_bottleneck.variables)
                synthesis_saver = tf.train.Saver(synthesis_transform.variables)
                analysis_saver = tf.train.Saver(analysis_transform.variables)
                # restore weights
                restore_weights(synthesis_saver, sess, 
                        args.checkpoint_dir + "/syn_net")
                restore_weights(analysis_saver, sess, 
                        args.checkpoint_dir + "/ana_net")
                restore_weights(entropy_saver, sess, 
                        args.checkpoint_dir + "/entro_net")
            else:
                latest=tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
                tf.train.Saver().restore(sess, save_path=latest)
                if args.use_y_base:
                    analysis_saver = tf.train.Saver(analysis_transform.variables)
                    restore_weights(analysis_saver, sess, 
                        args.pretrain_checkpoint_dir + "/ana_net")

            # get the compressed string and the tensor shapes.
            tensors=[string, tf.shape(x)[1:-1], tf.shape(y)[1:-1]]
            arrays=sess.run(tensors)

            # Write a binary file with the shape information and the compressed string.
            packed=tfc.PackedTensors()
            packed.pack(tensors, arrays)
            with open(args.output_file, "wb") as f:
                f.write(packed.string)

            # If requested, transform the quantized image back and measure performance.
            if args.verbose:
                eval_bpp, mse, luma_psnr, luma_msssim, chroma_psnr, chroma_msssim, num_pixels, rgb_psnr, rgb_msssim=sess.run(
                        [eval_bpp, mse, luma_psnr, luma_msssim, chroma_psnr, chroma_msssim, num_pixels, rgb_psnr, rgb_msssim])

                # The actual bits per pixel including overhead.
                bpp=len(packed.string) * 8 / num_pixels

                print("Mean squared error: {:0.4f}".format(mse))
                print("RGB Multiscale SSIM: {:0.4f}".format(rgb_msssim))
                print("RGB PSNR (dB): {:0.2f}".format(rgb_psnr))
                print("LUMA Multiscale SSIM: {:0.4f}".format(luma_msssim))
                print("LUMA PSNR (dB): {:0.2f}".format(luma_psnr))
                # print("LUMA Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - luma_msssim)))
                print("CHROMA PSNR (dB): {:0.2f}".format(chroma_psnr))
                # print("CHROMA Multiscale SSIM: {:0.4f}".format(chroma_msssim))
                # print("CHROMA Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - chroma_msssim)))
                print("Information content in bpp: {:0.4f}".format(eval_bpp))
                print("Actual bits per pixel: {:0.4f}".format(bpp))


def idn_compress(args):
    """Compresses an image."""
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu_device)

    with tf.Graph().as_default() as graph:
        tf.set_random_seed(1234)
        # Load input image and add batch dimension.
        x=read_png(args.input_file)
        x=tf.expand_dims(x, 0)
        x.set_shape([1, None, None, 3])
        x_shape=tf.shape(x)
        
        # randomly crop img to 256x256
        x = tf.random_crop(x, (1, 256, 256, 3))

        # Instantiate model.
        entropy_bottleneck = tfc.EntropyBottleneck()
        analysis_transform = m.AnalysisTransform(args.channel_out[0])
        # synthesis_transform = m.SynthesisTransform(args.num_filters)
        inv_transform = m.IntDiscreteNet(blk_type=args.blk_type, 
            num_filters=args.num_filters, downsample_type=args.downsample_type, 
            n_levels=args.n_levels, n_flows=args.n_flows)
        
        # Transform and compress the image.
        # baseline performance
        base_y = analysis_transform(x)
        # base_y_hat, _ = entropy_bottleneck(base_y, training=False)
        # base_x_hat = synthesis_transform(base_y_hat)
        # base_x_hat = base_x_hat[:, :x_shape[1], :x_shape[2], :]

        # idn performance
        out, _ = inv_transform(x)  # k/255
        z = out[:, :, :, args.channel_out[-1]:]
        z_zeros = tf.zeros_like(z)
        y = tf.slice(out, [0, 0, 0, 0], [-1, -1, -1, args.channel_out[-1]])
        scaled_y = y * 255  # k (integer)
        y_hat, likelihoods = entropy_bottleneck(scaled_y, training=False)
        y_hat /= 255
        x_hat, _ = inv_transform(tf.concat([y_hat, z_zeros], axis=-1), rev=True)

        base_string = entropy_bottleneck.compress(base_y)
        string = entropy_bottleneck.compress(scaled_y)
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

        # Total number of bits divided by number of pixels.
        eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

        # Bring both images back to 0..255 range.
        x *= 255
        x_hat = tf.clip_by_value(x_hat, 0, 1)
        x_hat = tf.round(x_hat * 255)
        # base_x_hat = tf.clip_by_value(base_x_hat, 0, 1)
        # base_x_hat = tf.round(base_x_hat * 255)

        # # baseline performance eval
        # mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
        # luma_x = tf.slice(tf.image.rgb_to_yuv(x), [0, 0, 0, 0], [-1, -1, -1, 1])
        # luma_x_hat = tf.slice(tf.image.rgb_to_yuv(x_hat), [
        #     0, 0, 0, 0], [-1, -1, -1, 1])
        # luma_psnr = tf.squeeze(tf.image.psnr(luma_x_hat, luma_x, 255))

        # idn performance eval
        luma_x = tf.slice(tf.image.rgb_to_yuv(x), [0, 0, 0, 0], [-1, -1, -1, 1])
        luma_x_hat = tf.slice(tf.image.rgb_to_yuv(x_hat), [
            0, 0, 0, 0], [-1, -1, -1, 1])
        luma_psnr = tf.squeeze(tf.image.psnr(luma_x_hat, luma_x, 255))

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
            tf.train.Saver().restore(sess, save_path=latest)

            entropy_saver = tf.train.Saver(entropy_bottleneck.variables)
            analysis_saver = tf.train.Saver(analysis_transform.variables)
            # restore weights
            restore_weights(entropy_saver, sess, 
                    args.pretrain_checkpoint_dir + "/entro_net")
            restore_weights(analysis_saver, sess, 
                    args.pretrain_checkpoint_dir + "/ana_net")

            # get the compressed string and the tensor shapes.
            tensors=[string, tf.shape(x)[1:-1], tf.shape(y)[1:-1]]
            arrays=sess.run(tensors)

            # Write a binary file with the shape information and the compressed string.
            packed=tfc.PackedTensors()
            packed.pack(tensors, arrays)
            with open(args.output_file, "wb") as f:
                f.write(packed.string)

            # If requested, transform the quantized image back and measure performance.
            if args.verbose:
                eval_bpp, luma_psnr, num_pixels = sess.run([eval_bpp, luma_psnr, num_pixels])

                # The actual bits per pixel including overhead.
                bpp = len(packed.string) * 8 / num_pixels

                print("LUMA PSNR (dB): {:0.2f}".format(luma_psnr))
                print("Information content in bpp: {:0.4f}".format(eval_bpp))
                print("Actual baseline bits per pixel: {:0.4f}".format(bpp))
                # print("Actual bits per pixel: {:0.4f}".format(bpp))


def multi_compress(args):
    ######### Create input data pipeline for compressing multi images ########

    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu_device)
    with tf.device("/cpu:0"):
        eval_files=glob.glob(args.eval_glob)
        if not eval_files:
            raise RuntimeError(
                    "No evaluation images found with glob '{}'.".format(args.eval_glob))
            eval_dataset=tf.data.Dataset.from_tensor_slices(eval_files)
        eval_dataset=eval_dataset.shuffle(
                buffer_size=len(eval_files)).repeat()
        eval_dataset=eval_dataset.map(
                read_png, num_parallel_calls=args.preprocess_threads)
        eval_dataset=eval_dataset.map(
                lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
        eval_dataset=eval_dataset.batch(args.batchsize)
        eval_dataset=eval_dataset.prefetch(32)

    num_pixels=args.batchsize * args.patchsize ** 2

    num_iter=len(eval_files) // args.batchsize
    # iterator = eval_dataset.make_initializable_iterator()
    # next_element = iterator.get_next()

    x=eval_dataset.make_one_shot_iterator().get_next()

    ######### create x place holder for eval ########
    # x = tf.placeholder(
    #         tf.float32, [None, args.patchsize, args.patchsize, 3], name='image')

    ######### end #########

    # Instantiate model.
    analysis_transform=m.AnalysisTransform(args.num_filters)
    entropy_bottleneck=tfc.EntropyBottleneck()
    synthesis_transform=m.SynthesisTransform(args.num_filters)

    # Transform and compress the image.
    y=analysis_transform(x)
    string=entropy_bottleneck.compress(y)

    # Transform the quantized image back (if requested).
    y_hat, likelihoods=entropy_bottleneck(y, training=False)
    x_hat=synthesis_transform(y_hat)
    x_hat=x_hat[:, :args.patchsize, :args.patchsize, :]

    num_pixels=tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

    # # Total number of bits divided by number of pixels.
    # eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

    # Bring both images back to 0..255 range.
    x *= 255
    x_hat=tf.clip_by_value(x_hat, 0, 1)
    x_hat=tf.round(x_hat * 255)

    mse=tf.reduce_mean(tf.squared_difference(x, x_hat))
    psnr=tf.reduce_mean(tf.squeeze(tf.image.psnr(x_hat, x, 255)))
    msssim=tf.reduce_mean(tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255, 
        filter_size=11 if args.patchsize > 64 else 4)))

    ######### session for compressing multi imgs #########

    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        latest=tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)

        # sess.run(iterator.initializer)
        avg_bpp, avg_mse, avg_psnr, avg_msssim=0., 0., 0., 0.
        for i in range(num_iter):
            # _x = sess.run(next_element)
            # print("_x shape: {}, datatype: {}".format(_x.shape, type(_x[0, 0, 0, 0])))
            print("[mse] {}".format(mse))
            _mse=sess.run(mse)
            _psnr=sess.run(psnr)
            _msssim=sess.run(msssim)
            _num_pixels=sess.run(num_pixels)
            _string=sess.run(string)
            # mse, psnr, msssim, num_pixels, string = sess.run(
            #     [mse, psnr, msssim, num_pixels, string])
                # feed_dict={x: _x})
            # The actual bits per pixel including overhead.
            bpp=len(_string) * 8 / _num_pixels
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
    string=tf.placeholder(tf.string, [1])
    x_shape=tf.placeholder(tf.int32, [2])
    y_shape=tf.placeholder(tf.int32, [2])
    with open(args.input_file, "rb") as f:
        packed=tfc.PackedTensors(f.read())
    tensors=[string, x_shape, y_shape]
    arrays=packed.unpack(tensors)

    # Instantiate model.
    entropy_bottleneck=tfc.EntropyBottleneck(dtype=tf.float32)
    synthesis_transform=m.SynthesisTransform(args.num_filters)

    # Decompress and transform the image back.
    y_shape=tf.concat([y_shape, [args.num_filters]], axis=0)
    y_hat=entropy_bottleneck.decompress(
            string, y_shape, channels=args.num_filters)
    x_hat=synthesis_transform(y_hat)

    # Remove batch dimension, and crop away any extraneous padding on the bottom
    # or right boundaries.
    x_hat=x_hat[0, :x_shape[0], :x_shape[1], :]

    # Write reconstructed image out as a PNG file.
    op=write_png(args.output_file, x_hat)

    # Load the latest model checkpoint, and perform the above actions.
    with tf.Session() as sess:
        latest=tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
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
    int_train_cmd=subparsers.add_parser(
            "int_train",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Trains (or continues to train) a new model.")
    for cmd in [train_cmd, inv_train_cmd, int_train_cmd]:
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
        cmd.add_argument(
                "--int_flow", action="store_true",
                help="whether to use integer discrete flow.")
        cmd.add_argument(
                "--int_discrete_net", action="store_true",
                help="whether to use integer discrete net.")
        cmd.add_argument(
                "--n_levels", type=int, default=4,
                help="num of levels")
        cmd.add_argument(
                "--n_flows", type=int, default=8,
                help="num of flows")
        cmd.add_argument(
                "--y_scale_up", action="store_true",
                help="whether to scale up y before entropy bottleneck.")
        cmd.add_argument(
                "--df_iter", type=int, default=500000,
                help="decay factor starting iter")
        cmd.add_argument(
                "--end_iter", type=int, default=1000000,
                help="decay factor ending iter")
        cmd.add_argument(
                "--train_aux", action="store_true",
                help="whether to train the auxiliary network")
        cmd.add_argument(
                "--optimize_separately", action="store_true",
                help="whether to use optimize main and aux separately.")
        cmd.add_argument(
                "--inv_conv_init", default="ortho",
                help="('ortho' or 'identity').")
                
    # 'compress' subcommand.
    compress_cmd=subparsers.add_parser(
            "compress",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Reads a PNG file, compresses it, and writes a TFCI file.")
    
    # 'idn_compress' subcommand.
    idn_compress_cmd=subparsers.add_parser(
            "idn_compress",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Reads a PNG file, compresses it with idn, and writes a TFCI file.")

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
    for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png"), 
                     (evaluation_cmd, ".tfci"), (idn_compress_cmd, ".tfci")):
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
                "--inv_conv_init", default="ortho",
                help="('ortho' or 'identity').")
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
        cmd.add_argument(
                "--n_levels", type=int, default=4,
                help="num of levels")
        cmd.add_argument(
                "--n_flows", type=int, default=8,
                help="num of flows")
            

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
    # if args.command == "train":
        train(args)
    elif args.command == "int_train":
        int_train(args)
    elif args.command == "compress":
        if not args.output_file:
            args.output_file=args.input_file + ".tfci"
        compress(args)
    elif args.command == "idn_compress":
        if not args.output_file:
            args.output_file=args.input_file + ".tfci"
        idn_compress(args)
    elif args.command == "decompress":
        if not args.output_file:
            args.output_file=args.input_file + ".png"
        decompress(args)
    elif args.command == "multi_compress":
        print("multi_compress!")
        multi_compress(args)

if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
