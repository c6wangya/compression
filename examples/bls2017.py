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

from tensorflow.python import debug as tf_debug

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

# 
# class DataPreprocessor(object):
#     def __call__(self, img):
#         img = tf.cast(img, tf.float32)
#         # randomly adjust saturation in the range of []
#         print(img.get_shape())
#         img = tf.image.random_saturation(img, lower=0.7, upper=1)
#         # randomly add uniform noise
#         print("img shape: {}".format(tf.shape(img)))
#         noise = tf.random_uniform((tf.shape(img), 0, 1)) - 0.5
#         img = tf.add(img, noise)
#         # randomly crop image
#         scale = get_normal_random_number(0.75, 0.1, 0.6, 0.8)
#         img = tf.random_crop(img, (args.patchsize / scale, args.patchsize / scale, 3))
#         size = tf.concat((256, 256), axis=0)
#         img = tf.image.resize_images(img, size, method=tf.image.ResizeMethod.BICUBIC)
#         # # randomly downsample image
#         # scale = get_normal_random_number(0.75, 0.1, 0.6, 0.8)
#         # H, W, _ = img.get_shape()
#         # size = tf.concat((H * scale, W * scale), axis=0)
#         # size = tf.cast(size, tf.int32)
#         # img = tf.image.resize_images(img, size, method=tf.image.ResizeMethod.BICUBIC)
#         return img

def train(args):
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

        num_pixels=args.batchsize * args.patchsize ** 2

        # Get training patch from dataset.
        x=train_dataset.make_one_shot_iterator().get_next()
        
        # add uniform noise
        if args.noise:
            x = tf.add(x, tf.random_uniform(tf.shape(x), 0, 1.))

        # Instantiate model.
        entropy_bottleneck=tfc.EntropyBottleneck(noise=(not args.quant_grad))
        if args.command == "train":
            print("training")
            analysis_transform=m.AnalysisTransform(args.num_filters)
            synthesis_transform=m.SynthesisTransform(args.num_filters)
        else:
            # inv train net
            print("inv training!")
            # inv_transform_1 = m.HaarDownsampling(1)
            # inv_transform_2 = m.HaarDownsampling(3)
            # inv_transform_2 = m.InvBlockExp(12, 3)
            # inv_transform = m.InvHSRNet(channel_in=3, channel_out=args.channel_out, 
            #         upscale_log=args.upscale_log, block_num=[args.blk_num, args.blk_num], 
            #         blk_type=args.blk_type, num_filters=args.num_filters,
            #         use_inv_conv=args.inv_conv, kernel_size=args.kernel_size, residual=args.residual)
            inv_transform = m.InvCompressionNet(channel_in=3, channel_out=args.channel_out, 
                    blk_type=args.blk_type, num_filters=args.num_filters,
                    kernel_size=args.kernel_size, residual=args.residual, 
                    nin=args.nin, gdn=args.gdn, n_ops=args.n_ops)
            # inv_transform = m.InvHSRNet(channel_in=3, channel_out=3, 
            #         upscale_log=2, block_num=[1, 1])
        if args.guidance_type == "grayscale":
            guidance_transform = m.GrayScaleGuidance(rgb_type='RGB', down_scale=4)
    
        """ 1 gpu """
        # Build autoencoder.
        train_flow = 0
        if args.command == "train":
            y=analysis_transform(x)
            if args.guidance_type == "norm": 
                train_y_guidance = tf.reduce_sum(tf.norm(y - 0.5, ord=2, axis=-1, name="guidance_norm"))
            if args.clamp:
                y = tf.clip_by_value(y, 0, 1)
            y_tilde, likelihoods=entropy_bottleneck(y, training=True)
            x_tilde=synthesis_transform(y_tilde)
            flow_loss_weight = 0
        else:
            # Test invertibility 
            # x0 = tf.expand_dims(x[..., 0], -1)
            # x0 = print_act_stats(x0, " initial x[..., 0] ")
            # x1 = x[..., 1:]
            # y = inv_transform_1(x0)
            # # y_tilde, likelihoods = entropy_bottleneck(y, training=True)
            # _x0 = inv_transform_1(y, rev=True)
            # _x0 = print_act_stats(_x0, " inv x[..., 0] ")
            # x = tf.concat([_x0, x1], axis=-1)
            # x = print_act_stats(x, " initial x ")
            # y = inv_transform_2(x)
            # y_tilde, likelihoods = entropy_bottleneck(y, training=True)
            # x_tilde = inv_transform_2(y, rev=True)
            # x_tilde = print_act_stats(x_tilde, " inv x ")
            
            # For InvHSRNet
            # # print("before forward x shape: {}".format(x.get_shape()))
            # # x = print_act_stats(x, "x")
            # out = inv_transform([x])
            # zshapes = []
            # for i in range(args.upscale_log):
            #     xx = out[i]
            #     if xx.get_shape()[-1] == args.channel_out[i]:
            #         z = xx[:, :, :, args.channel_out[i] - 1:]
            #     else:
            #         z = xx[:, :, :, args.channel_out[i]:]
            #     print(z.get_shape())
            #     # z = print_act_stats(z, "z_{}".format(i))
            #     zshapes.append(tf.shape(z))
            #     # zshapes.append(z)
            #     # mle of flow 
            #     train_flow += tf.reduce_sum(tf.norm(z + epsilon, ord=2, axis=-1, name="last_norm_" + str(i)))
            # y = tf.slice(out[-1], [0, 0, 0, 0], [-1, -1, -1, args.channel_out[-1]])
            # y_tilde, likelihoods = entropy_bottleneck(y, training=True)
            # # train_flow = print_act_stats(train_flow, "train flow loss")
            # # y_tilde = print_act_stats(y_tilde, "y_tilde")
            # input_rev = []
            # for zshape in zshapes:
            #     input_rev.append(tf.random_normal(shape=zshape))
            # # input_rev = zshapes
            # input_rev.append(y_tilde)
            # # input_rev.append(y)
            # for i in input_rev:
            #     print(i.get_shape())
            # x_tilde = inv_transform(input_rev, rev=True)[-1]
            # # x_tilde = print_act_stats(x_tilde, "x_tilde")
            # flow_loss_weight = args.flow_loss_weight

            # For InvCompressionNet
            # x = print_act_stats(x, "x")
            out = inv_transform([x])
            zshapes = []
            
            if out[-1].get_shape()[-1] == args.channel_out[-1]:
                z = out[-1][:, :, :, args.channel_out[-1] - 1:]
            else:
                z = out[-1][:, :, :, args.channel_out[-1]:]
            print(z.get_shape())
            # z = print_act_stats(z, "z_{}".format(i))
            zshapes.append(tf.shape(z))
            # zshapes.append(z)
            # mle of flow 
            train_flow += tf.reduce_sum(tf.norm(z + epsilon, ord=2, axis=-1, name="last_norm"))
            
            y = tf.slice(out[-1], [0, 0, 0, 0], [-1, -1, -1, args.channel_out[-1]])
            if args.guidance_type == "norm": 
                train_y_guidance = tf.reduce_sum(tf.norm(y - 0.5, ord=2, axis=-1, name="guidance_norm"))
            if args.clamp:
                y = tf.clip_by_value(y, 0, 1)
            y_tilde, likelihoods = entropy_bottleneck(y, training=True)
            # train_flow = print_act_stats(train_flow, "train flow loss")
            # y_tilde = print_act_stats(y_tilde, "y_tilde")
            input_rev = []
            for zshape in zshapes:
                input_rev.append(tf.random_normal(shape=zshape))
            # input_rev = zshapes
            input_rev.append(y_tilde)
            # input_rev.append(y)
            x_tilde = inv_transform(input_rev, rev=True)[-1]
            # x_tilde = print_act_stats(x_tilde, "x_tilde")
            flow_loss_weight = args.flow_loss_weight
        
        if args.guidance_type == "grayscale":
            y_guidance = guidance_transform(x)
        # Total number of bits divided by number of pixels.
        train_bpp=tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

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
        else:
            train_y_guidance = 0
                

        # The rate-distortion cost.
        train_loss = args.lmbda * train_mse + \
                    train_bpp + \
                    flow_loss_weight * train_flow + \
                    args.y_guidance_weight * train_y_guidance
        # train_loss = print_act_stats(train_loss, "overall train loss")
        
        tvars = tf.trainable_variables()
        filtered_vars = [var for var in tvars \
                if not 'haar_downsampling' in var.name \
                and not 'gray_scale_guidance' in var.name]
        # for v in filtered_vars:
        #     print(v.name + '\n')
        if args.debug_mode:
            assert len([var for var in tvars if 'haar_downsampling' in var.name]) != 0, \
                "there's no variable called haar_downsampling! \n"
            assert len([var for var in tvars if 'gray_scale_guidance' in var.name]) != 0 \
                or args.y_guidance_weight == 0, \
                "there's no variable called gray_scale_guidance! \n"
            print("Has variables {} in total, and filtered out {} variables\n".format( \
                    len(tvars), len(tvars) - len(filtered_vars)))
        # print("all variables: {}".format(filtered_vars))
        # Minimize loss and auxiliary loss, and execute update op.
        step=tf.train.create_global_step()
        # main_step=main_optimizer.minimize(train_loss, global_step=step, var_list=filtered_vars)
        main_optimizer=tf.train.AdamOptimizer(learning_rate=args.main_lr)
        main_gradients, main_variables = zip(*main_optimizer.compute_gradients(train_loss, filtered_vars))
        main_gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, args.grad_clipping)
            for gradient in main_gradients]
        main_step = main_optimizer.apply_gradients(zip(main_gradients, main_variables), global_step=step)
        
        # aux_step=aux_optimizer.minimize(entropy_bottleneck.losses[0], var_list=filtered_vars)
        aux_optimizer=tf.train.AdamOptimizer(learning_rate=args.aux_lr)
        aux_gradients, aux_variables = zip(*aux_optimizer.compute_gradients( \
                entropy_bottleneck.losses[0], filtered_vars))
        aux_gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, args.grad_clipping)
            for gradient in aux_gradients]
        aux_step = aux_optimizer.apply_gradients(zip(aux_gradients, aux_variables))

        train_op=tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

        # eval_bpp, mse, psnr, msssim, num_pixels = inference(x,
        #         analysis_transform, entropy_bottleneck, synthesis_transform)
        # inference_op = tf.group(eval_bpp, mse, psnr, msssim, num_pixels)

        tf.summary.scalar("loss", train_loss)
        tf.summary.scalar("bpp", train_bpp)
        tf.summary.scalar("mse", train_mse)

        psnr=tf.squeeze(tf.reduce_mean(tf.image.psnr(x_tilde, x, 255)))
        msssim=tf.squeeze(tf.reduce_mean(
            tf.image.ssim_multiscale(x_tilde, x, 255)))
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
        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allow_growth = True
        # with tf_debug.LocalCLIDebugWrapperSession(
        with tf.train.MonitoredTrainingSession(
                    hooks=hooks, checkpoint_dir=args.checkpoint_dir,
                    save_checkpoint_secs=1000, save_summaries_secs=300) as sess:
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
        entropy_bottleneck=tfc.EntropyBottleneck(noise=(not args.quant_grad))
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
                    nin=args.nin, gdn=args.gdn, n_ops=args.n_ops)

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
            # For InvHSRNet
            # # x = print_act_stats(x, "x forward")
            # out = inv_transform([x])
            # zshapes = []
            # for i in range(args.upscale_log):
            #     xx = out[i]
            #     if xx.get_shape()[-1] == args.channel_out[i]:
            #         z = xx[:, :, :, args.channel_out[i] - 1:]
            #     else:
            #         z = xx[:, :, :, args.channel_out[i]:]
            #     # zshapes.append(tf.shape(z))
            #     zshapes.append(z)
            # y = tf.slice(out[-1], [0, 0, 0, 0], [-1, -1, -1, args.channel_out[-1]])

            # y_hat, likelihoods = entropy_bottleneck(y, training=False)
            # # input_rev = [tf.random_normal(shape=zshape) for zshape in zshapes]
            # input_rev = zshapes
            # input_rev.append(y_hat)
            # x_hat = inv_transform(input_rev, rev=True)[-1]
            # # x_hat = print_act_stats(x_hat, "x hat")

            # For InvCompressionNet
            # x = print_act_stats(x, "x")
            out = inv_transform([x])
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

            y_hat, likelihoods = entropy_bottleneck(y, training=False)
            # train_flow = print_act_stats(train_flow, "train flow loss")
            # y_tilde = print_act_stats(y_tilde, "y_tilde"
            if not args.reuse_z:
                input_rev = [tf.random_normal(shape=zshape, stddev=args.std) for zshape in zshapes]
            else:
                input_rev = zshapes
            input_rev.append(y_hat if not args.reuse_y else y)
            # input_rev.append(y)
            x_hat = inv_transform(input_rev, rev=True)[-1]
        
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
        rgb_msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

        luma_x=tf.slice(tf.image.rgb_to_yuv(x), [0, 0, 0, 0], [-1, -1, -1, 1])
        luma_x_hat=tf.slice(tf.image.rgb_to_yuv(x_hat), [
            0, 0, 0, 0], [-1, -1, -1, 1])
        luma_psnr=tf.squeeze(tf.image.psnr(luma_x_hat, luma_x, 255))
        luma_msssim=tf.squeeze(tf.image.ssim_multiscale(luma_x_hat, luma_x, 255))

        chroma_x=tf.slice(tf.image.rgb_to_yuv(x), [0, 0, 0, 1], [-1, -1, -1, 2])
        chroma_x_hat=tf.slice(tf.image.rgb_to_yuv(
            x_hat), [0, 0, 0, 1], [-1, -1, -1, 2])
        chroma_psnr=tf.squeeze(tf.image.psnr(chroma_x_hat, chroma_x, 255))
        chroma_msssim=tf.squeeze(
                tf.image.ssim_multiscale(chroma_x_hat, chroma_x, 255))

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

        with tf.Session() as sess:
            # Load the latest model checkpoint, get the compressed string and the tensor
            # shapes.
            latest=tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
            tf.train.Saver().restore(sess, save_path=latest)
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
    msssim=tf.reduce_mean(tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255)))

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
    train_cmd.add_argument(
            "--noise", action="store_true",
            help="add noise to image patch.")
    train_cmd.add_argument(
            "--downsample_scale", type=float, default=0.75, dest="scale",
            help="downsample scale for data preprocessing..")
    train_cmd.add_argument(
            "--adjust_saturation", action="store_true",
            help="adjust saturation of images.")
    train_cmd.add_argument(
            "--clamp", action="store_true",
            help="Do clamp on y.")
    train_cmd.add_argument(
            "--grad_clipping", type=float, default=100,
            help="Clipping gradient.")
    train_cmd.add_argument(
            "--quant_grad", action="store_true",
            help="quantize with gradient.")
    train_cmd.add_argument(
            "--guidance_type", default="none",
            help="guidance type.")
    train_cmd.add_argument(
            "--y_guidance_weight", type=float, default=0.,
            help="flow loss weight.")
    train_cmd.add_argument(
            "--main_lr", type=float, default=1e-4,
            help="main learning rate.")
    train_cmd.add_argument(
            "--aux_lr", type=float, default=1e-3,
            help="aux learning rate.")
    
    # 'inv_train' subcommand
    inv_train_cmd=subparsers.add_parser(
            "inv_train",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Trains (or continues to train) a new model.")
    inv_train_cmd.add_argument(
            "--train_glob", default="images/*.png",
            help="Glob pattern identifying training data. This pattern must expand "
            "to a list of RGB images in PNG format.")
    inv_train_cmd.add_argument(
            "--batchsize", type=int, default=8,
            help="Batch size for training.")
    inv_train_cmd.add_argument(
            "--patchsize", type=int, default=256,
            help="Size of image patches for training.")
    inv_train_cmd.add_argument(
            "--lambda", type=float, default=0.01, dest="lmbda",
            help="Lambda for rate-distortion tradeoff.")
    inv_train_cmd.add_argument(
            "--last_step", type=int, default=1000000,
            help="Train up to this number of steps.")
    inv_train_cmd.add_argument(
            "--preprocess_threads", type=int, default=16,
            help="Number of CPU threads to use for parallel decoding of training "
            "images.")
    inv_train_cmd.add_argument(
            "--num_gpus", type=int, default=1,
            help="Number of gpus used for training.")
    inv_train_cmd.add_argument(
            "--gpu_device", type=int, default=0,
            help="gpu device to be used.")
    inv_train_cmd.add_argument(
            "--checkpoint_dir", default="train",
            help="Directory where to save/load model checkpoints.")
    inv_train_cmd.add_argument(
            "--noise", action="store_true",
            help="add noise to image patch.")
    inv_train_cmd.add_argument(
            "--downsample_scale", type=float, default=0.75, dest="scale",
            help="downsample scale for data preprocessing..")
    inv_train_cmd.add_argument(
            "--adjust_saturation", action="store_true",
            help="adjust saturation of images.")
    inv_train_cmd.add_argument(
            "--blk_num", type=int, default=4,
            help="num of blocks for flow net")
    inv_train_cmd.add_argument(
            "--channel_out", nargs='+', type=int, default=[3, 3])
    inv_train_cmd.add_argument(
            "--upscale_log", type=int, default=2,
            help="upscale times")
    inv_train_cmd.add_argument(
            "--kernel_size", type=int, default=3,
            help="kernel size of subunit conv")
    inv_train_cmd.add_argument(
            "--flow_loss_weight", type=float, default=1e-1,
            help="flow loss weight.")
    inv_train_cmd.add_argument(
            "--y_guidance_weight", type=float, default=0.,
            help="flow loss weight.")
    inv_train_cmd.add_argument(
            "--blk_type", default="dense",
            help="select which type of block to use")
    inv_train_cmd.add_argument(
            "--inv_conv", action="store_true",
            help="use 1x1 invertible conv before last split.")
    inv_train_cmd.add_argument(
            "--main_lr", type=float, default=1e-4,
            help="main learning rate.")
    inv_train_cmd.add_argument(
            "--aux_lr", type=float, default=1e-3,
            help="aux learning rate.")
    inv_train_cmd.add_argument(
            "--residual", action="store_true",
            help="use residual block in subnet.")
    inv_train_cmd.add_argument(
            "--nin", action="store_true",
            help="use 1x1 conv in subnet.")
    inv_train_cmd.add_argument(
            "--gdn", action="store_true",
            help="use GDN in subnet.")
    inv_train_cmd.add_argument(
            "--clamp", action="store_true",
            help="Do clamp on y.")
    inv_train_cmd.add_argument(
            "--grad_clipping", type=float, default=100,
            help="Clipping gradient.")
    inv_train_cmd.add_argument(
            "--guidance_type", default="none",
            help="guidance type.")
    inv_train_cmd.add_argument(
            "--quant_grad", action="store_true",
            help="quantize with gradient.")
    inv_train_cmd.add_argument(
            "--n_ops", type=int, default=3,
            help="number of operations in subnet")

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
        cmd.add_argument(
            "--n_ops", type=int, default=3,
            help="number of operations in subnet")
        cmd.add_argument(
            "--quant_grad", action="store_true",
            help="quantize with gradient.")
    
    compress_cmd.add_argument(
            "--channel_out", nargs='+', type=int, default=[3, 3])
    compress_cmd.add_argument(
            "--upscale_log", type=int, default=2,
            help="upscale times")
    compress_cmd.add_argument(
            "--blk_num", type=int, default=4,
            help="num of blocks for flow net")
    compress_cmd.add_argument(
            "--blk_type", default="dense",
            help="select which type of block to use")
    compress_cmd.add_argument(
            "--invnet", action="store_true",
            help="use inv transform.")
    compress_cmd.add_argument(
            "--inv_conv", action="store_true",
            help="use 1x1 invertible conv before last split.")
    compress_cmd.add_argument(
            "--residual", action="store_true",
            help="use residual block in subnet.")
    compress_cmd.add_argument(
            "--kernel_size", type=int, default=3,
            help="kernel size of subunit conv")
    compress_cmd.add_argument(
            "--nin", action="store_true",
            help="use 1x1 conv in subnet.")
    compress_cmd.add_argument(
            "--gdn", action="store_true",
            help="use GDN in subnet.")
    compress_cmd.add_argument(
            "--reuse_y", action="store_true",
            help="skip quantization and AE&AD.")
    compress_cmd.add_argument(
            "--reuse_z", action="store_true",
            help="skip sampling z.")
    compress_cmd.add_argument(
            "--clamp", action="store_true",
            help="Do clamp on y.")
    compress_cmd.add_argument(
            "--std", type=float, default=1,
            help="std for sampling z.")

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
    elif args.command == "compress":
        if not args.output_file:
            args.output_file=args.input_file + ".tfci"
        compress(args)
    elif args.command == "decompress":
        if not args.output_file:
            args.output_file=args.input_file + ".png"
        decompress(args)
    elif args.command == "multi_compress":
        print("multi_compress!")
        multi_compress(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
