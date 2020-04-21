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

def sumflat(x):
    return tf.reduce_sum(tf.reshape(x, [x.shape[0], -1]), axis=1)

def inverse_sigmoid(x):
    return -tf.log(tf.reciprocal(x) - 1.)

def to_default_floatx(x):
    return tf.cast(x, tf.float32)

def gaussian_sample_logp(shape, dtype):
    eps = tf.random_normal(shape)
    logp = Normal(0., 1.).log_prob(eps)
    assert logp.shape == eps.shape
    logp = tf.reduce_sum(tf.layers.flatten(logp), axis=1)
    return tf.cast(eps, dtype=dtype), tf.cast(logp, dtype=dtype)

def get_var(var_name, *, ema, initializer, trainable=True, **kwargs):
    """forced storage dtype"""
    assert 'dtype' not in kwargs
    if isinstance(initializer, np.ndarray):
        initializer = initializer.astype(tf.float32.as_numpy_dtype)
    v = tf.get_variable(var_name, dtype=tf.float32, initializer=initializer, trainable=trainable, **kwargs)
    if ema is not None:
        assert isinstance(ema, tf.train.ExponentialMovingAverage)
        v = ema.average(v)
    return v
    
def dense(x, *, name, num_units, init_scale=1., init=False, ema=None):
    # use weight normalization (Salimans & Kingma, 2016)
    with tf.variable_scope(name):
        assert x.shape.ndims == 2
        _V = get_var('V', shape=[int(x.shape[1]), num_units], initializer=tf.random_normal_initializer(0, 0.05),
                     ema=ema)
        _g = get_var('g', shape=[num_units], initializer=tf.constant_initializer(1.), ema=ema)
        _b = get_var('b', shape=[num_units], initializer=tf.constant_initializer(0.), ema=ema)
        _vinvnorm = tf.rsqrt(tf.reduce_sum(tf.square(_V), [0]))
        V, g, b, vinvnorm = map(to_default_floatx, [_V, _g, _b, _vinvnorm])

        x0 = x = tf.matmul(x, V)
        x = (g * vinvnorm)[None, :] * x + b[None, :]

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            with tf.control_dependencies([
                _g.assign(tf.cast(g * scale_init, dtype=_g.dtype)),
                _b.assign_add(tf.cast(-m_init * scale_init, dtype=_b.dtype))
            ]):
                g, b = map(to_default_floatx, [_g, _b])
                x = (g * vinvnorm)[None, :] * x0 + b[None, :]

        return x

def nin(x, *, name, num_units, init, ema, **kwargs):
    assert 'num_units' not in kwargs
    s = x.shape.as_list()
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = dense(x, name='dense', num_units=num_units, init=init, ema=ema, **kwargs)
    return tf.reshape(x, s[:-1] + [num_units])

def matmul_last_axis(x, w):
    _, out_dim = w.shape
    s = x.shape.as_list()
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = tf.matmul(x, w)
    return tf.reshape(x, s[:-1] + [out_dim])

def concat_elu(x, *, axis=-1):
    return tf.nn.elu(tf.concat([x, -x], axis=axis))

def gate(x, *, axis):
    a, b = tf.split(x, 2, axis=axis)
    return a * tf.sigmoid(b)

def conv2d(x, *, name, num_units, filter_size=(3, 3), stride=(1, 1), pad='SAME', init_scale=1., init, ema):
    # use weight normalization (Salimans & Kingma, 2016)
    with tf.variable_scope(name):
        assert x.shape.ndims == 4
        _V = get_var('V', shape=[*filter_size, int(x.shape[-1]), num_units],
                     initializer=tf.random_normal_initializer(0, 0.05), ema=ema)
        _g = get_var('g', shape=[num_units], initializer=tf.constant_initializer(1.), ema=ema)
        _b = get_var('b', shape=[num_units], initializer=tf.constant_initializer(0.), ema=ema)
        _vnorm = tf.nn.l2_normalize(_V, [0, 1, 2])
        V, g, b, vnorm = map(to_default_floatx, [_V, _g, _b, _vnorm])

        W = g[None, None, None, :] * vnorm

        # calculate convolutional layer output
        input_x = x
        x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1, *stride, 1], pad), b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0, 1, 2])
            scale_init = init_scale * tf.rsqrt(v_init + 1e-8)
            with tf.control_dependencies([
                _g.assign(tf.cast(g * scale_init, dtype=_g.dtype)),
                _b.assign_add(tf.cast(-m_init * scale_init, dtype=_b.dtype))
            ]):
                g, b = map(to_default_floatx, [_g, _b])
                W = g[None, None, None, :] * vnorm
                x = tf.nn.bias_add(tf.nn.conv2d(input_x, W, [1, *stride, 1], pad), b)

        return x

def gated_resnet(x, *, name, a, nonlinearity=concat_elu, conv=conv2d, use_nin, init, ema, dropout_p):
    with tf.variable_scope(name):
        num_filters = int(x.shape[-1])

        c1 = conv(nonlinearity(x), name='c1', num_units=num_filters, init=init, ema=ema)
        if a is not None:  # add short-cut connection if auxiliary input 'a' is given
            c1 += nin(nonlinearity(a), name='a_proj', num_units=num_filters, init=init, ema=ema)
        c1 = nonlinearity(c1)
        if dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)

        c2 = (nin if use_nin else conv)(c1, name='c2', num_units=num_filters * 2, init_scale=0.1, init=init, ema=ema)
        return x + gate(c2, axis=3)

def _norm(x, *, axis, g, b, e=1e-5):
    assert x.shape.ndims == g.shape.ndims == b.shape.ndims
    u = tf.reduce_mean(x, axis=axis, keepdims=True)
    s = tf.reduce_mean(tf.squared_difference(x, u), axis=axis, keepdims=True)
    x = (x - u) * tf.rsqrt(s + e)
    return x * g + b

def norm(x, *, name, ema):
    """Layer norm over last axis"""
    with tf.variable_scope(name):
        dim = int(x.shape[-1])
        _g = get_var('g', ema=ema, shape=[dim], initializer=tf.constant_initializer(1))
        _b = get_var('b', ema=ema, shape=[dim], initializer=tf.constant_initializer(0))
        g, b = map(to_default_floatx, [_g, _b])
        bcast_shape = [1] * (x.shape.ndims - 1) + [dim]
        return _norm(x, g=tf.reshape(g, bcast_shape), b=tf.reshape(b, bcast_shape), axis=-1)

def attn(x, *, name, pos_emb, heads, init, ema, dropout_p):
    with tf.variable_scope(name):
        bs, height, width, ch = x.shape.as_list()
        assert pos_emb.shape == [height, width, ch]
        assert ch % heads == 0
        timesteps = height * width
        dim = ch // heads
        # Position embeddings
        c = x + pos_emb[None, :, :, :]
        # b, h, t, d == batch, num heads, num timesteps, per-head dim (C // heads)
        c = nin(c, name='proj1', num_units=3 * ch, init=init, ema=ema)
        assert c.shape == [bs, height, width, 3 * ch]
        # Split into heads / Q / K / V
        c = tf.reshape(c, [bs, timesteps, 3, heads, dim])  # b, t, 3, h, d
        c = tf.transpose(c, [2, 0, 3, 1, 4])  # 3, b, h, t, d
        q_bhtd, k_bhtd, v_bhtd = tf.unstack(c, axis=0)
        assert q_bhtd.shape == k_bhtd.shape == v_bhtd.shape == [bs, heads, timesteps, dim]
        # Attention
        w_bhtt = tf.matmul(q_bhtd, k_bhtd, transpose_b=True) / np.sqrt(float(dim))
        w_bhtt = tf.cast(tf.nn.softmax(at_least_float32(w_bhtt)), dtype=x.dtype)
        assert w_bhtt.shape == [bs, heads, timesteps, timesteps]
        a_bhtd = tf.matmul(w_bhtt, v_bhtd)
        # Merge heads
        a_bthd = tf.transpose(a_bhtd, [0, 2, 1, 3])
        assert a_bthd.shape == [bs, timesteps, heads, dim]
        a_btc = tf.reshape(a_bthd, [bs, timesteps, ch])
        # Project
        c1 = tf.reshape(a_btc, [bs, height, width, ch])
        if dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
        c2 = nin(c1, name='proj2', num_units=ch * 2, init_scale=0.1, init=init, ema=ema)
        return x + gate(c2, axis=3)


class Flow:
    def forward(self, x, **kwargs):
        raise NotImplementedError
    def backward(self, y, **kwargs):
        raise NotImplementedError


class CheckerboardSplit(Flow):
    def forward(self, x, **kwargs):
        assert isinstance(x, tf.Tensor)
        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H, W // 2, 2, C])
        a, b = tf.unstack(x, axis=3)
        assert a.shape == b.shape == [B, H, W // 2, C]
        return (a, b), None

    def inverse(self, y, **kwargs):
        assert isinstance(y, tuple)
        a, b = y
        assert a.shape == b.shape
        B, H, W_half, C = a.shape
        x = tf.stack([a, b], axis=3)
        assert x.shape == [B, H, W_half, 2, C]
        return tf.reshape(x, [B, H, W_half * 2, C]), None


class Sigmoid(Flow):
    def forward(self, x, **kwargs):
        y = tf.sigmoid(x)
        logd = -tf.nn.softplus(x) - tf.nn.softplus(-x)
        return y, sumflat(logd)
    def inverse(self, y, **kwargs):
        x = inverse_sigmoid(y)
        logd = -tf.log(y) - tf.log(1. - y)
        return x, sumflat(logd)


class Dequantizer(Flow):
    def __init__(self, dequant_flow):
        super().__init__()
        assert isinstance(dequant_flow, Flow)
        self.dequant_flow = dequant_flow


        def deep_processor(x, *, init, ema, dropout_p): 
            (this, that), _  = CheckerboardSplit().forward(x)  # split in half
            processed_context = conv2d(tf.concat([this, that], 3), name='proj', num_units=32, init=init, ema=ema)
            B, H, W, C = processed_context.shape.as_list()

            pos_emb = to_default_floatx(get_var(
                'pos_emb_dq', ema=ema, shape=[H, W, C], initializer=tf.random_normal_initializer(stddev=0.01),
            ))

            for i in range(8):
                processed_context = gated_resnet(
                    processed_context, name=f'c{i}',
                    a=None, dropout_p=dropout_p, ema=ema, init=init,
                    use_nin=False
                )
                processed_context = norm(processed_context, name=f'dqln{i}', ema=ema)
                processed_context = attn(processed_context, name=f'dqattn{i}', pos_emb=pos_emb, heads=4, init=init, ema=ema, dropout_p=dropout_p)
                processed_context = norm(processed_context, name=f'ln{i}', ema=ema)

            return processed_context

        self.context_proc = tf.make_template("context_proc", deep_processor)

    def forward(self, x, init=False, ema=None, dropout_p=0., verbose=True, **kwargs):
        eps, eps_logli = gaussian_sample_logp(x.shape, dtype=tf.float32)
        unbound_xd, logd = self.dequant_flow.forward(
            eps,
            context=self.context_proc(x / 256.0 - 0.5, init=init, ema=ema, dropout_p=dropout_p),
            init=init, ema=ema, dropout_p=dropout_p, verbose=verbose
        )
        xd, sigmoid_logd = Sigmoid().forward(unbound_xd)
        assert x.shape == xd.shape and logd.shape == sigmoid_logd.shape == eps_logli.shape
        return x + xd, logd + sigmoid_logd - eps_logli