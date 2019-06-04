import tensorflow as tf
import numpy as np
import globalvar as Glovar
from utils import *
import logging

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 32, "batch size for training the model")
flags.DEFINE_float("learning_rate1", 1e-3, "Learning rate for first only fc layer")
flags.DEFINE_float("learning_rate2", 1e-3, "Learning rate for first train step of full layers")
flags.DEFINE_float("learning_rate3", 1e-4, "Learning rate for second train step  of full layers")
flags.DEFINE_float("learning_rate4", 1e-5, "Learning rate for third train step of full layers")
flags.DEFINE_integer("total_step", 50000, "batch size for training the model")
flags.DEFINE_string("checkpoint_dir", 'models/','path to save model parameters')
flags.DEFINE_string("data_dir", './dataset/','path to save tfrecords data')
tf.app.flags.DEFINE_integer('seed', 2, "initial random seed")

FLAGS = flags.FLAGS

#set placeholder
phase = tf.placeholder(tf.bool)
isTrainable = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)


# load pretrained model
if Glovar.get_para() is None:
    Glovar.set_para()
vgg_para = Glovar.get_para()

# log info
def set_log_info():
    logger = logging.getLogger('vgg')
    logger.setLevel(logging.INFO)
    # True to log file False to print
    logging_file = True
    if logging_file == True:
        hdlr = logging.FileHandler('vgg.log')
    else:
        hdlr = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger


logger = set_log_info()




# inputs
def input(dataname, batchsize, isShuffel, flag):
    image, label = read_and_decode(dataname, flag=flag)
    images, labels = generate_batch([image, label], batchsize, isShuffel)
    return images, labels


"""
load variable from npy to build the VGG
:param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
"""


def Vgg19(rgb, y):
    win_size = 224
    image = tf.image.resize_images(rgb, [win_size, win_size])
    #image = rgb
    bgr = image * 255.0
    VGG_MEAN = [103.939, 116.779, 123.68]
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=bgr)


    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(axis=3, values=[
       blue - VGG_MEAN[0],
       green - VGG_MEAN[1],
       red - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
    conv1_1 = conv_layer(bgr, "conv1_1", w_shape=[3, 3, 3, 64], b_shape=[64, ],)
    conv1_2 = conv_layer(conv1_1, "conv1_2", w_shape=[3, 3, 64, 64], b_shape=[64, ])
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 = conv_layer(pool1, "conv2_1", w_shape=[3, 3, 64, 128], b_shape=[128, ])
    conv2_2 = conv_layer(conv2_1, "conv2_2", w_shape=[3, 3, 128, 128], b_shape=[128, ])
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 = conv_layer(pool2, "conv3_1", w_shape=[3, 3, 128, 256], b_shape=[256, ],)
    conv3_2 = conv_layer(conv3_1, "conv3_2", w_shape=[3, 3, 256, 256], b_shape=[256, ])
    conv3_3 = conv_layer(conv3_2, "conv3_3", w_shape=[3, 3, 256, 256], b_shape=[256, ])
    conv3_4 = conv_layer(conv3_3, "conv3_4", w_shape=[3, 3, 256, 256], b_shape=[256, ])
    pool3 = max_pool(conv3_4, 'pool3')

    conv4_1 = conv_layer(pool3, "conv4_1", w_shape=[3, 3, 256, 512], b_shape=[512, ])
    conv4_2 = conv_layer(conv4_1, "conv4_2", w_shape=[3, 3, 512, 512], b_shape=[512, ])
    conv4_3 = conv_layer(conv4_2, "conv4_3", w_shape=[3, 3, 512, 512], b_shape=[512, ])
    conv4_4 = conv_layer(conv4_3, "conv4_4", w_shape=[3, 3, 512, 512], b_shape=[512, ])
    pool4 = max_pool(conv4_4, 'pool4')

    conv5_1 = conv_layer(pool4, "conv5_1", w_shape=[3, 3, 512, 512], b_shape=[512, ])
    conv5_2 = conv_layer(conv5_1, "conv5_2", w_shape=[3, 3, 512, 512], b_shape=[512, ])
    conv5_3 = conv_layer(conv5_2, "conv5_3", w_shape=[3, 3, 512, 512], b_shape=[512, ])
    conv5_4 = conv_layer(conv5_3, "conv5_4", w_shape=[3, 3, 512, 512], b_shape=[512, ])
    #pool5 = max_pool_4x4(conv5_4, 'pool5') # [batch,2,2,512]
    pool5 = max_pool(conv5_4, 'pool5')


    fc6 = pre_fc_layer(pool5, "fc6", w_shape=[25088, 4096], b_shape=[4096])
    assert fc6.get_shape().as_list()[1:] == [4096]
    relu6 = tf.nn.relu(fc6)

    relu6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

    fc7 = pre_fc_layer(relu6, "fc7", w_shape=[4096, 4096], b_shape=[4096])
    relu7 = tf.nn.relu(fc7)
    relu7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

    # pool5=tf.reshape(pool5,[-1,512])
    logit = fc_layer(relu7, "fc8", w_shape=[4096, 200], b_shape=[200])
    softmax = tf.nn.softmax(logit, name='prob')
    y_oh = tf.one_hot(y, depth=200)
    # loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_oh,logit))
    correct_prediction = tf.equal(tf.cast(tf.argmax(softmax, 1), tf.int64), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    var_list = tf.trainable_variables()
    global_step = tf.Variable(0, trainable=False)


    # compute loss
    lossXent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_oh, logits=logit))
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in var_list
                       if 'bias' not in v.name]) * 5e-4
    loss = lossXent + lossL2


    var_list_fc8 = [var for var in var_list
                         if 'fc8' in var.name]
    #var_list_else = [var for var in var_list
    #                    if 'fc8' not in var.name]

    fc8_train_op = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate1).minimize(loss,var_list= var_list_fc8,global_step=global_step)
    full_train_op1 = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate2).minimize(loss,global_step=global_step)
    full_train_op2 = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate3).minimize(loss,global_step=global_step)
    full_train_op3 = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate4).minimize(loss,global_step=global_step)
    #train_op = tf.group(train_op_fc8,train_op_fc67)


    # tensorboard visualization
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('lossXent', lossXent)
    tf.summary.scalar('lossL2', lossL2)

    merged = tf.summary.merge_all()
    return global_step,prob_extract, loss,lossXent,lossL2, softmax, fc8_train_op,full_train_op1,full_train_op2,full_train_op3,accuracy, merged,var_list,



#save checkpoint and restore
def save_checkpoint(sess,step,saver):
    checkpoint_dir = FLAGS.save_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    saver.save(sess=sess,
               save_path=checkpoint_dir+'model.ckpt',
               global_step=step)
    print('step:%d | save model success'%step)

def load_checkpoint(sess,saver):
    checkpoint_dir = FLAGS.save_dir
    checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoints and checkpoints.model_checkpoint_path:
        checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
        #saver.restore(sess, os.path.join(checkpoint_dir,checkpoints_name))
        #saver.restore(sess,checkpoints.model_checkpoint_path)
        saver.restore(sess,checkpoint_dir+"model.ckpt-"+str(10001))
        print('load model success,contuinue training...')
    else:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        print('None checkpoint file found,initialize model... ')



train_images, train_labels = input(FLAGS.data_dir + 'train.tfrecords', batchsize=FLAGS.batch_size, isShuffel=True,
                                   flag=True)
test_images, test_labels = input(FLAGS.data_dir + 'test.tfrecords', batchsize=FLAGS.batch_size, isShuffel=True,
                                 flag=False)

X_input = tf.cond(phase, lambda: train_images, lambda: test_images)
Y_input = tf.cond(phase, lambda: train_labels, lambda: test_labels)

global_step,prob_extract, loss,lossXent,lossL2, softmax, fc8_train_op,full_train_op1,full_train_op2,full_train_op3,acc, merge, var_list = Vgg19(X_input, Y_input)


with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)

    # load model
    saver = tf.train.Saver(max_to_keep=5)
    load_checkpoint(sess,saver)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # threads = tf.train.start_queue_runners(coord=coord)
    # writer = tf.summary.FileWriter("logs/", sess.graph)

    np.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(np.random.randint(1234))




    for i in range(FLAGS.total_step):

        if i <=5000:
            result = sess.run([global_step,prob_extract, loss,lossXent,lossL2, softmax, fc8_train_op, acc],
                          feed_dict={phase: True,keep_prob:1.0})
        elif i <= 15000:
            result = sess.run([global_step,prob_extract, loss, lossXent, lossL2, softmax, full_train_op1, acc],
                          feed_dict={phase: True,keep_prob: 0.5})
        elif i <= 20000:
            result = sess.run([global_step, prob_extract, loss, lossXent, lossL2, softmax, full_train_op2, acc],
                              feed_dict={phase: True, keep_prob: 0.5})
        else:
            result = sess.run([global_step, prob_extract, loss, lossXent, lossL2, softmax, full_train_op3, acc],
                              feed_dict={phase: True, keep_prob: 0.5})

        if i and i % 100 == 0:
            # labels = sess.run([train_op],feed_dict={ph:True})
            # print(i, labels)
            # logger.info(
            #    'step {}: loss = {:3.4f}'.format(i, 0))

            logger.info(
                'step {}: loss = {:3.4f}\tlossXent = {:3.4f}\tlossL2 = {:3.4f}'.format(result[0], result[2],result[3],result[4]))
            # , np.array_str(result[0])))

            logger.info('step {}: train_acc = {}'.format(
                result[0], result[-1], ))

            summary = sess.run(merge, feed_dict={phase: True,keep_prob:0.5})
            writer.add_summary(summary, result[0])

        # evalresult
        if i and i % 1000 == 0:

            # cal acc for one batch
            # prediction = np.argmax(softmax_val, 1)
            # equality = np.equal(prediction, label_eval)
            # accuracy = np.mean(tf.cast(equality, tf.float32))
            # print('eval_acc:', result[-1])
            eval_num = 6000 // FLAGS.batch_size
            total_accuracy = 0
            for k in range(eval_num):
                result = sess.run([global_step,prob_extract, loss, softmax, acc], feed_dict={phase: False,keep_prob:1.0})
                total_accuracy += result[-1]

            accuracy = total_accuracy / eval_num
            # print('eval_acc',accuracy)
            # print(result[-1])
            logger.info(
                'step {}: Test_acc = {:3.4f}\t acc_batch = {}'.format(
                    result[0], accuracy, result[-1]))

        if i and i % 5000 == 0:
            save_checkpoint(sess,result[0],saver)

    save_checkpoint(sess,total_step,saver)


    coord.request_stop()
    coord.join(threads)
