"""
Created on Sun Jan 09 2022

tf units

@author: Ray
"""
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
from spatial_transformer_network.stn import spatial_transformer_network as transformer


# Normalization layer
def normalize_layer(inputs, k_initilizer, type='batch'):

    if type == 'batch':
        h = tf.layers.batch_normalization(inputs, epsilon=1e-5, gamma_initializer=k_initilizer)
    else:
        h = tfa.layers.InstanceNormalization(epsilon=1e-5, center=False, scale=False)(inputs)
    return h

# Convolution layer
def conv_layer(inputs, kernel_sz, filters, strides, pad, init, reuse):

    h = tf.layers.conv2d(inputs, kernel_size=kernel_sz,
        filters=filters,
        strides=strides,
        padding=pad,
        kernel_initializer=init,
        reuse=reuse)
    return h


# Activation function
def activation(inputs, act='lkrelu'):
    if act == 'lkrelu':
        return tf.nn.leaky_relu(inputs, alpha=0.2)
    elif act == 'sigmoid':
        return tf.nn.sigmoid(inputs)
    elif act == 'relu':
        return tf.nn.relu(inputs)
    elif act == 'tanh':
        return tf.nn.tanh(inputs)
    elif act == None:
        return inputs


# Downconv block
def downconv_block(inputs, filters, k_initilizer, kernel_sz=4, strides=2, pad='same', do_norm=False, act='relu', reuse=None, name='down_conv'):

    with tf.variable_scope(name, reuse=reuse):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h = conv_layer(inputs, kernel_sz, filters, strides, pad, k_initilizer, reuse)
        if do_norm:
            h = normalize_layer(h, k_initilizer)
        h = activation(h, act=act)
        return h



def LocalNet(inputs, k_initilizer, reuse=None, name='STN'):

    with tf.variable_scope(name, reuse=reuse):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
        h = downconv_block(inputs, 64, k_initilizer, reuse=reuse, name='down_conv_1')
        h = downconv_block(h, 128, k_initilizer, reuse=reuse, name='down_conv_2')
        h = downconv_block(h, 256, k_initilizer, reuse=reuse, name='down_conv_3')
        h = downconv_block(h, 512, k_initilizer, reuse=reuse, name='down_conv_4')
        h = downconv_block(h, 1024, k_initilizer, reuse=reuse, name='down_conv_5')

        h_latent = tf.layers.flatten(h)
        h_latent = tf.layers.dense(h_latent, 1024, kernel_initializer=k_initilizer)
        h_latent = activation(h_latent)

        return h_latent


# Discriminator unit
def d_block(inputs, filters, k_initilizer, kernel_sz=4, strides=1, reuse=False, act='lkrelu', do_norm=True, name='discriminator_block'):

    with tf.variable_scope(name, reuse=reuse):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h = conv_layer(inputs, kernel_sz, filters, strides, 'same', k_initilizer, reuse)
        
        if do_norm:
            h = normalize_layer(h, k_initilizer)

        h = activation(inputs, act=act)

        return h