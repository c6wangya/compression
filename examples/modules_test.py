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


class ModuleTest(tf.test.TestCase):

    def testActNorm(self):
        np.random.seed(83243)
        batch_size = 25
        length = 15
        channels = 4
        inputs = 3. + 0.8 * np.random.randn(batch_size, length, channels)
        inputs = tf.cast(inputs, tf.float32)
        layer = m.ActNorm()
        outputs = layer(inputs)
        mean, variance = tf.nn.moments(outputs, axes=[0, 1])
        self.assertAllClose(mean, np.zeros(channels), atol=1e-3)
        self.assertAllClose(variance, np.ones(channels), atol=1e-3)

        inputs = 3. + 0.8 * np.random.randn(batch_size, length, channels)
        inputs = tf.cast(inputs, tf.float32)
        outputs = layer(inputs)
        mean, variance = tf.nn.moments(outputs, axes=[0, 1])
        self.assertAllClose(mean, np.zeros(channels), atol=0.25)
        self.assertAllClose(variance, np.ones(channels), atol=0.25)

    def testCoupling(self):
        np.random.seed(83243)
        batch_size = 25
        length = 16
        channels = 12
        inputs = 3. + 0.8 * np.random.randn(batch_size, length, length, channels)
        inputs = tf.cast(inputs, tf.float32)
        layer = m.IntInvBlock(m.SeqBlock, 3)
        mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])
        recons = layer(layer(inputs), rev=True)
        recons_mean, recons_variance = tf.nn.moments(recons, axes=[0, 1, 2])
        self.assertAllClose(mean - recons_mean, np.zeros(channels), atol=1e-3)
        self.assertAllClose(variance - recons_variance, np.zeros(channels), atol=1e-3)

        inputs = 3. + 0.8 * np.random.randn(batch_size, length, length, channels)
        inputs = tf.cast(inputs, tf.float32)
        mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])
        recons = layer(layer(inputs), rev=True)
        recons_mean, recons_variance = tf.nn.moments(recons, axes=[0, 1, 2])
        self.assertAllClose(mean - recons_mean, np.zeros(channels), atol=0.25)
        self.assertAllClose(variance - recons_variance, np.zeros(channels), atol=0.25)

    def test_diff_round(self):
        with self.test_session():
            input_tensor = tf.Variable([1.1, 2.3, 5.9, 1.8])
            expected_output = [1, 2, 6, 2]
            expected_gradients = np.identity(4)
            output_tensor = m.differentiable_round(input_tensor)
            theoretical_grad, numerical_grad = tf.test.compute_gradient(m.differentiable_round, [input_tensor])
            theoretical_grad = tf.reshape(theoretical_grad, (4, 4))
            numerical_grad = tf.reshape(numerical_grad, (4, 4))
            # grad_computed = tf.test.compute_gradient(input_tensor, (4,), output_tensor, (4,))
            self.assertAllClose(output_tensor, expected_output, atol=1e-3)
            self.assertAllClose(theoretical_grad, expected_gradients, atol=1e-3)
            self.assertAllClose(numerical_grad, expected_gradients, atol=1e-3)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(2)
    tf.enable_v2_behavior()
    tf.test.main()