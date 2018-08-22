#coding:utf-8
#Bin GAO

import tensorflow as tf
import numpy as np
import os
import yaml

USE_FUSED_BN = True
BN_EPSILON = 9.999999747378752e-06
BN_MOMENTUM = 0.99

input_depth = [128,256,512,1024]
BACKBONE = 'resnet101'

with open('cfg.yml') as file:
    cfg = yaml.load(file)


def conv_layer(input,filter,kernel,stride,name,padding='same'):
    net = tf.layers.conv2d(inputs=input,filters=filter,kernel_size=kernel,strides=stride,padding=padding,name=name)

    return net


def upsampling_2d(tensor,name,size=(2,2)):
    h_,w_,c_ = tensor.get_shape().as_list()[1:]
    h_multi,w_multi = size
    h = h_multi * h_
    w = w_multi * w_
    target = tf.image.resize_nearest_neighbor(tensor,size=(h,w),name='upsample_{}'.format(name))

    return target



def upsampling_concat(input_A,input_B,name):
    upsampling = upsampling_2d(input_A,name=name,size=(2,2))
    up_concat = tf.concat([upsampling,input_B],axis=-1,name='up_concat_{}'.format(name))

    return up_concat



def Global_Average_Pooling(input,name,stride=1):
    width = np.shape(input)[1]
    height = np.shape(input)[2]
    pool_size = [width,height]

    return tf.layers.average_pooling2d(inputs=input,pool_size=pool_size,strides=stride,padding='SAME',name=name)


def Batch_Normalization(input,training,data_format,name):

    return tf.layers.batch_normalization(
        inputs=input,axis=1 if data_format== 'channels_first' else 3,
        momentum=BN_MOMENTUM,name=name,epsilon=BN_EPSILON,center=True,
        scale=True,training=training,reuse=None,fused=USE_FUSED_BN)


def Max_pooling(x, pool_size=(3,3), stride=2, padding='VALID'):

    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Relu(input,name):

    return tf.nn.relu(input,name=name)


def first_layer(input,data_format,is_training):
    net = conv_layer(input=input,filter=64,kernel=(7,7),stride=2,name='conv1/7x7_s2')
    net_bn = Batch_Normalization(net,training=is_training,data_format=data_format,name='conv1/7x7_s2/bn')
    net_relu = Relu(net_bn,name='conv1/7x7_s2/relu')

    net_pooled = Max_pooling(net_relu)

    return net_pooled


def proj_block(input,filter,name_prefix,is_training,data_format='channels_first',
                   need_reduce=True,is_root=True):

    bn_axis = -1 if data_format == 'channels_last' else 1
    if need_reduce:
        stride_to_use = 1 if is_root else 2

        proj_mapping = conv_layer(input=input,name=name_prefix + '_1x1_identity_block',filter=filter*2,
                                  kernel=(1,1),stride=stride_to_use,padding='same')

        net = tf.layers.batch_normalization(proj_mapping, momentum=BN_MOMENTUM,
                                                  name=name_prefix + '_1x1_identity_block/bn', axis=bn_axis,
                                                  epsilon=BN_EPSILON, training=is_training, reuse=None,
                                                  fused=USE_FUSED_BN)

        return net

    else:
        return input


def residual_block(input,filter,name_prefix,is_training,data_format='channels_first',reduce_scale=16,is_root=False):
    stride_to_use = 1 if is_root else 2


    reduced_net = conv_layer(input=input,name=name_prefix+'_1x1_reduce',filter=filter/2,kernel=(1,1),
                             stride=1,padding='valid')
    reduced_net_bn = Batch_Normalization(input=reduced_net,training=is_training,data_format=data_format,
                                         name=name_prefix+'_1x1_reduce/bn')
    reduced_net_relu = Relu(reduced_net_bn,name=name_prefix+'_1x1_reduce/relu')


    conv_net = conv_layer(input=reduced_net_relu,name=name_prefix+'_3x3',filter=filter/2,kernel=(3,3),
                      stride=stride_to_use,padding='same')
    conv_net_bn = Batch_Normalization(input=conv_net,training=is_training,data_format=data_format,
                                  name=name_prefix+'_3x3/bn')
    conv_net_relu = Relu(conv_net_bn,name=name_prefix+'_3x3/relu')


    increase_net = conv_layer(input=conv_net_relu,name=name_prefix+'_1x1_increase',filter=filter*2,
                              kernel=(1,1),stride=1,padding='valid')
    increase_net_bn = Batch_Normalization(input=increase_net,training=is_training,data_format=data_format,
                                          name=name_prefix+'_1x1_increase/bn')


    gap_net = Global_Average_Pooling(input=increase_net_bn,stride=1,name=name_prefix+'_GAP')


    down_net = conv_layer(input=gap_net,name=name_prefix+'_1x1_down',filter=(filter*2)//reduce_scale,
                        kernel=(1,1),stride=1,padding='valid')
    down_net_relu = Relu(down_net,name=name_prefix+'_1x1_down/relu')


    up_net = conv_layer(input=down_net_relu,name=name_prefix+'_1x1_up',filter=filter*2,kernel=(1,1),
                        stride=1,padding='valid')


    prob_output = tf.nn.sigmoid(up_net,name=name_prefix+'_prob')
    rescaled_feat = tf.multiply(prob_output,increase_net_bn,name=name_prefix+'_mul')


    return rescaled_feat




def se_bottleneck_block(input,filter,name_prefix,is_training,data_format='channels_last',
                        need_reduce=True,is_root=False):

    rediduals_net = proj_block(input,filter,name_prefix,is_training,data_format=data_format,
                   need_reduce=need_reduce,is_root=is_root)

    rescaled_net = residual_block(input,filter,name_prefix,is_training,data_format=data_format,is_root=is_root)

    pre_act = tf.add(rediduals_net,rescaled_net,name=name_prefix+'_add')

    return tf.nn.relu(pre_act,name=name_prefix+'/relu')



def SE_resnet(input,is_training=False,data_format='channels_first',net_depth=BACKBONE):
    #convert RGB to BGR
    if data_format == 'channels_first':
        #矩阵分解
        image_channels = tf.unstack(input,axis=-1)
        swaped_image = tf.stack([image_channels[2],image_channels[1],image_channels[0]],axis=-1)

    else:
        image_channels = tf.unstack(input, axis=1)
        swaped_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=1)


    if net_depth not in ['resnet50','resnet101','resnet152']:
        raise TypeError('Only ResNet 50, ResNet 101 or ResNet152 is supported now.')

    if net_depth == 'resnet50':
        num_units = [3,4,6,3]

    if net_depth == 'resnet101':
        num_units = [3,4,23,3]

    if net_depth == 'resnet152':
        num_units = [3,8,36,3]

    else:
        print('Only ResNet50, ResNet101 or ResNet152 is supprted now.')


    block_name_prefix = ['conv2_{}', 'conv3_{}', 'conv4_{}', 'conv5_{}']


    #stage 1
    C1 = x = first_layer(input=swaped_image,data_format=data_format,is_training=is_training)


    #stage 2
    need_reduce = True
    is_root = True
    for unit_index in range(1, num_units[0] + 1):
        x = se_bottleneck_block(input=x, filter=input_depth[0],
                                name_prefix=block_name_prefix[0].format(unit_index),
                                is_training=is_training, data_format=data_format,
                                need_reduce=need_reduce, is_root=is_root)
        need_reduce = False
    C2 = x

    # stage 3
    need_reduce = True
    is_root = False
    for unit_index in range(1, num_units[1] + 1):
        x = se_bottleneck_block(input=x, filter=input_depth[1],
                                name_prefix=block_name_prefix[1].format(unit_index),
                                is_training=is_training, data_format=data_format,
                                need_reduce=need_reduce, is_root=is_root)
        is_root = True
        need_reduce = False
    C3 = x

    # stage 4
    need_reduce = True
    is_root = False
    for unit_index in range(1, num_units[2] + 1):
        x = se_bottleneck_block(input=x, filter=input_depth[2],
                                name_prefix=block_name_prefix[2].format(unit_index),
                                is_training=is_training, data_format=data_format,
                                need_reduce=need_reduce, is_root=is_root)
        is_root = True
        need_reduce = False
    C4 = x

    # stage 5
    need_reduce = True
    is_root = False
    for unit_index in range(1, num_units[3] + 1):
        x = se_bottleneck_block(input=x, filter=input_depth[3],
                                name_prefix=block_name_prefix[3].format(unit_index),
                                is_training=is_training, data_format=data_format,
                                need_reduce=need_reduce, is_root=is_root)
        is_root = True
        need_reduce = False
    C5 = x

    return C1,C2,C3,C4,C5


def inference(input,is_training):
    #if callable(config.BACKBONE):
    C1, C2, C3, C4, C5 = SE_resnet(input=input,is_training=is_training)

    pooled_inputs = tf.layers.flatten(C5)
    logits_output = tf.layers.dense(pooled_inputs,cfg['num_classes'],kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=False)

    return C1,C2,C3,C4,C5,logits_output,tf.nn.softmax(logits_output, name='prob')


def loss_CE(y_pred,y_true):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    return cross_entropy_mean



def make_train_op(loss,learning_rate):
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    return optimizer.minimize(loss,global_step=global_step)


if __name__ == '__main__':
    with open('cfg.yml') as file:
        cfg = yaml.load(file)






















