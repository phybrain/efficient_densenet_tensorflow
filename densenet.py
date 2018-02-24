#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 11:53


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gc
import numpy as np
import pickle
from  io import  BytesIO
import dill

slim = tf.contrib.slim


@slim.add_arg_scope
def _conv(inputs, num_filters, kernel_size, stride=1, dropout_rate=None,
          scope=None, outputs_collections=None):
  with tf.variable_scope(scope, 'xx', [inputs]) as sc:
    net = slim.batch_norm(inputs)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, num_filters, kernel_size)

    if dropout_rate:
      net = tf.nn.dropout(net)

    #net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net


@slim.add_arg_scope
def _conv_block(num_filters, share,scope=None, outputs_collections=None):
  share.seek(0)
  inputs=dill.load(share)
  with tf.variable_scope(scope, 'conv_blockx', [inputs]) as sc:
    net=inputs
    net = _conv(net, num_filters*4, 1, scope='x1')
    net = _conv(net, num_filters, 3, scope='x2')
    net = tf.concat([inputs, net], axis=3)
    # inputlist=np.insert(inputlist, inputlist.shape[3], values=net.eval(session=tf.InteractiveSession), axis=3)
    share.seek(0)
    dill.dump(net,share,0)
#   slim.utils.collect_named_outputs(outputs_collections, sc.name, tf.convert_to_tensor(inputlist))
    del net
    gc.collect()
#  return tf.convert_to_tensor(inputlist)


@slim.add_arg_scope
def _dense_block(inputs, num_layers, num_filters, growth_rate,
                 grow_num_filters=True, scope=None, outputs_collections=None,inputlist=None):

  with tf.variable_scope(scope, 'dense_blockx', [inputs]) as sc:
    net = inputs
    # inputlist=inputs.eval(session=tf.InteractiveSession)
    share = BytesIO()
    dill.dump(net, share, 0)

    for i in range(num_layers):
      branch = i + 1

      _conv_block(growth_rate,share, scope='conv_block'+str(branch))

      if grow_num_filters:
        num_filters += growth_rate
    share.seek(0)
    outputs=dill.load(share)
    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, outputs)
    share.close()
    del share,outputs
    gc.collect()
  return net, num_filters


@slim.add_arg_scope
def _transition_block(inputs, num_filters, compression=1.0,
                      scope=None, outputs_collections=None):

  num_filters = int(num_filters * compression)
  with tf.variable_scope(scope, 'transition_blockx', [inputs]) as sc:
    net = inputs
    net = _conv(net, num_filters, 1, scope='blk')

    net = slim.avg_pool2d(net, 2)

    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net, num_filters


def densenet(inputs,
             num_classes=1000,
             reduction=None,
             growth_rate=None,
             num_filters=None,
             num_layers=None,
             dropout_rate=None,
             is_training=True,
             reuse=None,
             scope=None):
  assert reduction is not None
  assert growth_rate is not None
  assert num_filters is not None
  assert num_layers is not None

  compression = 1.0 - reduction
  num_dense_blocks = len(num_layers)

  with tf.variable_scope(scope, 'densenetxxx', [inputs, num_classes],
                         reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                         is_training=is_training), \
         slim.arg_scope([slim.conv2d, _conv, _conv_block,
                         _dense_block, _transition_block],
                         outputs_collections=end_points_collection), \
         slim.arg_scope([_conv], dropout_rate=dropout_rate):
      net = inputs

      # initial convolution
      net = slim.conv2d(net, num_filters, 7, stride=2, scope='conv1')
      net = slim.batch_norm(net)
      net = tf.nn.relu(net)
      net = slim.max_pool2d(net, 3, stride=2, padding='SAME')

      # blocks
      for i in range(num_dense_blocks - 1):
        # dense blocks
        net, num_filters = _dense_block(net, num_layers[i], num_filters,
                                        growth_rate,
                                        scope='dense_block' + str(i+1))

        # Add transition_block
        net, num_filters = _transition_block(net, num_filters,
                                             compression=compression,
                                             scope='transition_block' + str(i+1))

      net, num_filters = _dense_block(
              net, num_layers[-1], num_filters,
              growth_rate,
              scope='dense_block' + str(num_dense_blocks))

      # final blocks
      with tf.variable_scope('final_block', [inputs]):
        net = slim.batch_norm(net)
        net = tf.nn.relu(net)
        net = tf.reduce_mean(net, [1,2], name='global_avg_pool', keep_dims=True)

      net = slim.conv2d(net, num_classes, 1,
                        biases_initializer=tf.zeros_initializer(),
                        scope='logits')

      end_points = slim.utils.convert_collection_to_dict(
          end_points_collection)

      if num_classes is not None:
        end_points['predictions'] = slim.softmax(net, scope='predictions')

      return net, end_points


def densenet121(inputs, num_classes=1000, is_training=True, reuse=None):
  return densenet(inputs,
                  num_classes=num_classes,
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,12,24,16],
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet121')
densenet121.default_image_size = 224


def densenet161(inputs, num_classes=1000, is_training=True, reuse=None):
  return densenet(inputs,
                  num_classes=num_classes,
                  reduction=0.5,
                  growth_rate=48,
                  num_filters=96,
                  num_layers=[6,12,36,24],
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet161')
densenet161.default_image_size = 224


def densenet169(inputs, num_classes=1000, is_training=True, reuse=None):
  return densenet(inputs,
                  num_classes=num_classes,
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,12,32,32],
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet169')
densenet169.default_image_size = 224


def densenet_arg_scope(weight_decay=1e-4,
                       batch_norm_decay=0.99,
                       batch_norm_epsilon=1.1e-5):
  with slim.arg_scope([slim.conv2d],
                       weights_regularizer=slim.l2_regularizer(weight_decay),
                       activation_fn=None,
                       biases_initializer=None):
    with slim.arg_scope([slim.batch_norm],
                        scale=True,
                        decay=batch_norm_decay,
                        epsilon=batch_norm_epsilon) as scope:
      return scope