import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import os
import logging


def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool_4x4(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name=name)


def conv_layer(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        filt = get_conv_filter(name, w_shape)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name, b_shape)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu


def pre_fc_layer(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = pre_get_fc_weight(name, w_shape)
        biases = pre_get_fcbias(name, b_shape)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc


def pre_get_fcbias(name, shape):
    init = vgg_para[name][1]
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(init),trainable=True)


def pre_get_fc_weight(name, shape):
    init = vgg_para[name][0]
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(init),trainable=True)


def fc_layer(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_fc_weight(name, w_shape)
        biases = get_fcbias(name, b_shape)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc


def get_fcbias(name, shape):
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(0))


def get_fc_weight(name, shape):
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.01))


def get_conv_filter(name, shape):
    init = vgg_para[name][0]
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(init), trainable=True)


def get_bias(name, shape):
    init = vgg_para[name][1]
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(init), trainable=True)

# 读入train 或者 test 数据集
def read_and_decode(filename,flag): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列
    options_zlib = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    options_gzip = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           'width':tf.FixedLenFeature([],tf.int64),
                                           'height':tf.FixedLenFeature([],tf.int64),
                                          # 'channel':tf.FixedLenFeature([],tf.int64)
                                            })#将image数据和label取出来
    width = tf.cast(features['width'],tf.int32)
    height = tf.cast(features['height'],tf.int32)
    #channel = tf.cast(features['channel'],tf.int32)
    if flag:

        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img,tf.stack([height, width, 3]))  #reshape为448*448的3通道图片

        img = tf.random_crop(img, [448, 448,3])
        img = tf.image.random_flip_left_right(img)
        #img = tf.image.random_contrast(img,lower = 0.5,upper = 1.5)
        #img = tf.image.random_hue(img, max_delta=0.2)
        #img = tf.image.random_brightness(img, max_delta=32. / 255.)
        #img = tf.image.random_saturation(img,lower = 0.5,upper = 1.5)


        img = tf.cast(img, tf.float32) * (1. / 255)  #在流中抛出img张量
    else:
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img,tf.stack( [height, width,3]))  # reshape为448*448的3通道图片
        img=tf.image.resize_image_with_crop_or_pad(img,448,448)
        img = tf.cast(img, tf.float32) * (1. / 255)

    label = tf.cast(features['label'], tf.int64) #在流中抛出label张量
    return img, label


# 生成一个batch的数据集，返回
def generate_batch(example, batch_size, shuffle):
    """
    Arg:
        list of tensors.
    """
    num_preprocess_threads = 1

    if shuffle:
        ret = tf.train.shuffle_batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=20 * batch_size,
            min_after_dequeue=10*batch_size)
    else:
        ret = tf.train.batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            allow_smaller_final_batch=False,
            capacity=10 * batch_size)
    return ret



def train_log(sess_outputs,i):
    nll_xents_val, at_xents_val, loss_val, reward_val, baseline_val, logllratio_val, _ = sess_outputs

    logger.info(
        'step {}: reward = {:3.4f}\tloss = {:3.4f}\tnll_xents = {:3.4f}\tat_xents = {:3.4f}'.format(
            i, reward_val, loss_val, nll_xents_val,at_xents_val))

    logger.info('step {}: llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
        i, logllratio_val, baseline_val, ))

def eval_log(acc,i):
    logger.info('step {}: valid accuracy = {}'.format(i, acc))
    '''
    logger.info('step {} prob: {}'.format(i, np.array_str(cal_prob_extract_val[0], 200)))
    coord_arr_stack = np.stack(coordinate_arr_val, axis=0)
    coord_arr_stack = np.transpose(coord_arr_stack, [1, 2, 0])
    
    logger.info('step {} zoom_size: {}'.format(i, np.array_str(coord_arr_stack[0, -1, :], 200)))
    '''


