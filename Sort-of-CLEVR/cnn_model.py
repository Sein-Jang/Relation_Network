from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, relu=True, init_zero=False,
                    data_format='channels_last'):
    """Performs a batch normalization followed by a ReLU."""
    if init_zero:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()

    if data_format == 'channels_first':
        axis = 1
    else:
        axis = 3

    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training,
        fused=True,
        gamma_initializer=gamma_initializer)

    if relu:
        inputs = tf.nn.relu(inputs)
    return inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format='channels_last'):
    """Strided 2-D convolution with explicit padding."""
    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding='SAME', bias_initializer=tf.constant_initializer(0.0),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        data_format=data_format)


def resnet_v1_generator(data_format='channels_first'):
    """Generator for ResNet v1 models."""

    def model(inputs, is_training):
        logging.info('ResNet')
        """Creation of the model graph."""
        with tf.device('/gpu:0'):
            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=24, kernel_size=5, strides=2, data_format='channels_last')
            inputs = batch_norm_relu(inputs, is_training, data_format='channels_last')
            inputs = tf.identity(inputs, 'conv_1')

            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=24, kernel_size=5, strides=2, data_format='channels_last')
            inputs = batch_norm_relu(inputs, is_training, data_format='channels_last')
            inputs = tf.identity(inputs, 'conv_2')

            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=24, kernel_size=5, strides=3, data_format='channels_last')
            inputs = batch_norm_relu(inputs, is_training, data_format='channels_last')
            inputs = tf.identity(inputs, 'conv_3')

            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=24, kernel_size=5, strides=3, data_format='channels_last')
            inputs = batch_norm_relu(inputs, is_training, data_format='channels_last')
            inputs = tf.identity(inputs, 'conv_4')

        return inputs

    model.default_image_size = 128
    return model


def resnet_v1(resnet_depth, data_format='channels_first'):
    return resnet_v1_generator(data_format)
