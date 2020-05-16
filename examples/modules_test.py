# coding=utf-8
# Copyright 2020 The Edward2 Authors.
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

"""Tests for normalization layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import modules as m
import numpy as np
import tensorflow.compat.v2 as tf
import os
import tensorflow_compression as tfc


class ModuleTest(tf.test.TestCase):

    # def testActNorm(self):
    #     np.random.seed(83243)
    #     batch_size = 25
    #     length = 15
    #     channels = 4
    #     inputs = 3. + 0.8 * np.random.randn(batch_size, length, channels)
    #     inputs = tf.cast(inputs, tf.float32)
    #     layer = m.ActNorm()
    #     outputs = layer(inputs)
    #     mean, variance = tf.nn.moments(outputs, axes=[0, 1])
    #     self.assertAllClose(mean, np.zeros(channels), atol=1e-3)
    #     self.assertAllClose(variance, np.ones(channels), atol=1e-3)

    #     inputs = 3. + 0.8 * np.random.randn(batch_size, length, channels)
    #     inputs = tf.cast(inputs, tf.float32)
    #     outputs = layer(inputs)
    #     mean, variance = tf.nn.moments(outputs, axes=[0, 1])
    #     self.assertAllClose(mean, np.zeros(channels), atol=0.25)
    #     self.assertAllClose(variance, np.ones(channels), atol=0.25)

    # def testCoupling(self):
    #     np.random.seed(83243)
    #     batch_size = 1
    #     length = 8
    #     channels = 3
    #     inputs = np.random.rand(batch_size, length, length, channels)
    #     inputs = tf.cast(inputs, tf.float32)
    #     layer = m.IntInvBlock(m.SeqBlock, 3)
    #     mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])
    #     outputs = layer(inputs)
    #     recons = layer(layer(inputs), rev=True)
    #     recons_mean, recons_variance = tf.nn.moments(recons, axes=[0, 1, 2])
    #     self.assertAllClose(mean - recons_mean, np.zeros(channels), atol=1e-3)
    #     self.assertAllClose(variance - recons_variance, np.zeros(channels), atol=1e-3)

    #     inputs = 3. + 0.8 * np.random.randn(batch_size, length, length, channels)
    #     inputs = tf.cast(inputs, tf.float32)
    #     mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])
    #     recons = layer(layer(inputs), rev=True)
    #     recons_mean, recons_variance = tf.nn.moments(recons, axes=[0, 1, 2])
    #     self.assertAllClose(mean - recons_mean, np.zeros(channels), atol=0.25)
    #     self.assertAllClose(variance - recons_variance, np.zeros(channels), atol=0.25)

    # def test_diff_round(self):
    #     with self.test_session():
    #         input_tensor = tf.Variable([1.1, 2.3, 5.9, 1.8])
    #         expected_output = [1, 2, 6, 2]
    #         expected_gradients = np.identity(4)
    #         # output_tensor = m.differentiable_round(input_tensor)
    #         with tf.GradientTape() as tape:
    #             tape.watch(input_tensor)
    #             output_tensor = m.differentiable_round(input_tensor)
    #         # grads = tape.gradient(output_tensor, [input_tensor])
    #         theoretical_grad, numerical_grad = tf.test.compute_gradient(tf.identity, [input_tensor])
    #         # theoretical_grad, numerical_grad = tf.test.compute_gradient(m.differentiable_round, [input_tensor])
    #         theoretical_grad = tf.reshape(theoretical_grad, (4, 4))
    #         numerical_grad = tf.reshape(numerical_grad, (4, 4))
    #         # grad_computed = tf.test.compute_gradient(input_tensor, (4,), output_tensor, (4,))
    #         self.assertAllClose(output_tensor, expected_output, atol=1e-3)
    #         self.assertAllClose(theoretical_grad, expected_gradients, atol=1e-3)

    # def test_dense_block(self):
    #     np.random.seed(0)
    #     batch_size = 4
    #     length = 128
    #     channels = 12
    #     low = 0
    #     high = 256
    #     inputs = 3. + 0.8 * np.random.randint(low, high, 
    #                 size=(batch_size, length, length, channels))
    #     inputs = tf.cast(inputs, tf.float32)
        
    #     layer = m.IntInvBlock(m.DenseBlock, 3)
    #     mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])
    #     recons = layer(layer(inputs), rev=True)
    #     recons_mean, recons_variance = tf.nn.moments(recons, axes=[0, 1, 2])
    #     self.assertAllClose(mean - recons_mean, np.zeros(channels), atol=1e-3)
    #     self.assertAllClose(variance - recons_variance, np.zeros(channels), atol=1e-3)

    # def test_squeeze(self):
    #     np.random.seed(83243)
    #     batch_size = 4
    #     length = 256
    #     channels = 3
    #     low = 0
    #     high = 256
    #     inputs = 3. + 0.8 * np.random.randint(low, high, 
    #                 size=(batch_size, length, length, channels))
    #     inputs = tf.cast(inputs, tf.float32)
        
    #     layer = m.SqueezeDownsampling()
    #     outputs = layer(inputs)
    #     recons = layer(outputs, rev=True)
    #     self.assertAllClose(recons - inputs, np.zeros_like(inputs), atol=1e-3)

    # def test_sort(self):
    #     permute = tf.Variable([1, 2, 3, 0])
    #     permute_inv = tf.Variable([0, 1, 2, 3])
    #     a = [tf.expand_dims(p, -1) for p in [permute, permute_inv]]
    #     a = tf.concat(a, axis=-1)
    #     b = tf.add(tf.slice(a, [0, 0], [-1, 1]) * 10, tf.slice(a, [0, 1], [-1, 1]))

    #     reordered = tf.gather(a, tf.nn.top_k(b[:, 0], k=4, sorted=False).indices)
    #     reordered = tf.reverse(reordered, axis=[0])
    #     permute_inv = tf.slice(reordered, [0, 1], [-1, -1])

    # def test_permute(self):
    #     np.random.seed(83243)
    #     batch_size = 25
    #     length = 16
    #     channels = 12
    #     inputs = 3. + 0.8 * np.random.randn(batch_size, length, length, channels)
    #     inputs = tf.cast(inputs, tf.float32)
    #     layer = m.Permute()
    #     mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])
    #     recons = layer(layer(inputs), rev=True)
    #     recons_mean, recons_variance = tf.nn.moments(recons, axes=[0, 1, 2])
    #     self.assertAllClose(inputs - recons, np.zeros_like(recons), atol=1e-3)
    #     self.assertAllClose(mean - recons_mean, np.zeros(channels), atol=1e-3)
    #     self.assertAllClose(variance - recons_variance, np.zeros(channels), atol=1e-3)

    # def test_idn(self):
    #     def comp_psnr(img_hat, img):
    #         img *= 255
    #         img_hat=tf.clip_by_value(img_hat, 0, 1)
    #         img_hat=tf.round(img_hat * 255)
    #         rgb_psnr = tf.squeeze(tf.reduce_mean(tf.image.psnr(img_hat, img, 255)))
    #         luma_img = tf.slice(tf.image.rgb_to_yuv(img), [0, 0, 0, 0], [-1, -1, -1, 1])
    #         luma_img_hat = tf.slice(tf.image.rgb_to_yuv(img_hat), [0, 0, 0, 0], [-1, -1, -1, 1])
    #         luma_psnr = tf.squeeze(tf.reduce_mean(tf.image.psnr(luma_img_hat, luma_img, 255)))
    #         return rgb_psnr, luma_psnr

    #     np.random.seed(83243)
    #     batch_size = 4
    #     length = 256
    #     channels = 3
    #     low = 0
    #     high = 256
    #     inputs = np.random.randint(low, high, 
    #                 size=(batch_size, length, length, channels))
    #     inputs = tf.cast(inputs, tf.float32)
        
    #     # inputs = tf.cast(inputs, tf.float32)
    #     layer = m.IntDiscreteNet('dense', 128, 'squeeze', 4, 8)
    #     mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])
    #     outputs, _ = layer(inputs)
    #     recons, _ = layer(outputs, rev=True)
    #     recons_mean, recons_variance = tf.nn.moments(recons, axes=[0, 1, 2])
    #     psnr = tf.image.psnr(recons, inputs, 255)
    #     self.assertAllClose(mean - recons_mean, np.zeros(channels), atol=1e-3)
    #     self.assertAllClose(variance - recons_variance, np.zeros(channels), atol=1e-3)

    # def test_quant(self):
    #     np.random.seed(83243)
    #     batch_size = 2
    #     length = 4
    #     channels = 3
    #     inputs = np.random.rand(batch_size, length, channels)
    #     inputs = tf.cast(inputs, tf.float32)
    #     inputs = m.differentiable_quant(inputs)
    #     print("")

    def test_imagenet64(self):
        np.random.seed(83243)
        batch_size = 2
        length = 64
        channels = 3
        low = 0
        high = 256
        inputs = 3. + 0.8 * np.random.randint(low, high, 
                    size=(batch_size, length, length, channels))
        inputs = tf.cast(inputs, tf.float32)
        analysis_transform = m.AnalysisTransform(256)
        synthesis_transform = m.SynthesisTransform(256)
        entropy_bottleneck = tfc.EntropyBottleneck()
        y = analysis_transform(inputs)
        # y_tilde, likelihoods = entropy_bottleneck(y, training=True)
        x_tilde = synthesis_transform(y)
        psnr=tf.squeeze(tf.reduce_mean(tf.image.psnr(x_tilde, inputs, 255)))
        msssim=tf.squeeze(tf.reduce_mean(
            tf.image.ssim_multiscale(x_tilde, inputs, 255)))
        
        # self.assertAllClose(recons - inputs, np.zeros_like(inputs), atol=1e-3)
        

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(2)
    tf.enable_v2_behavior()
    tf.test.main()