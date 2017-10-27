import os
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


def define_parameters():
    params_dict = dict()
    params_dict['batch_size'] = 1
    params_dict['height'] = int(32*1.)
    params_dict['width'] = int(48*1.)
    params_dict['lambda'] = 1.
    params_dict['epoch'] = 100
    params_dict['learning_rate'] = 0.00005

    params_dict['policy'] = 'enhence&&jaccard'
    params_dict['model_path'] = '../model_epoch'
    params_dict['data_path'] = '../data'
    params_dict['model_ckpt'] = '../model_records/model_ckpts'
    params_dict['model_performance'] = '../model_records/model_performances'
    params_dict['training_image'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Training_Data')
    params_dict['training_segment'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Training_Part1_GroundTruth')
    params_dict['validation_image'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Validation_Data')
    params_dict['validation_segment'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Validation_Part1_GroundTruth')

    return params_dict


def hybrid_cnn(cnn_in, scope):
    with tf.name_scope(scope):
        conv11 = slim.conv2d(cnn_in, num_outputs=128, kernel_size=1)
        conv12 = slim.conv2d(conv11, num_outputs=96, kernel_size=3, padding='SAME')

        conv21 = slim.conv2d(cnn_in, num_outputs=64, kernel_size=1)
        conv22 = slim.conv2d(conv21, num_outputs=32, kernel_size=5, padding='SAME')

        dilated_filter1 = slim.variable(name=scope+'_filter1', shape=[3, 3, 128, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.01))
        dilated_filter2 = slim.variable(name=scope+'_filter2', shape=[3, 3, 64, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.01))

        conv31 = tf.nn.atrous_conv2d(cnn_in, dilated_filter1, rate=2, padding='SAME')
        conv32 = tf.nn.atrous_conv2d(conv31, dilated_filter2, rate=3, padding='SAME')

        conv41 = slim.conv2d(cnn_in, num_outputs=32, kernel_size=1)
        conv42 = slim.max_pool2d(conv41, kernel_size=2, stride=1, padding='SAME')

        concat = tf.concat([conv12, conv22, conv32, conv42], axis=3)

    return concat


def model(inputs):

    with tf.name_scope('block1'):
        with tf.name_scope('conv1_1'):
            conv1_11 = slim.conv2d(inputs, num_outputs=32, kernel_size=3, padding='SAME')
            conv1_12 = slim.conv2d(conv1_11, num_outputs=48, kernel_size=3, padding='SAME')
            conv1_13 = slim.conv2d(conv1_12, num_outputs=64, kernel_size=3, padding='SAME')
            conv1_13_scale2 = slim.conv2d_transpose(conv1_13, num_outputs=64, stride=2, kernel_size=2)

        with tf.name_scope('conv1_2'):
            conv1_21 = slim.conv2d(conv1_13, num_outputs=64, kernel_size=3, padding='SAME')
            conv1_22 = slim.conv2d(conv1_21, num_outputs=96, kernel_size=3, padding='SAME')
            conv1_23 = slim.conv2d(conv1_22, num_outputs=128, kernel_size=3, padding='SAME')

        block1 = hybrid_cnn(conv1_23, 'hybrid1')

    with tf.name_scope('upsample2'):
        upsample2 = slim.conv2d_transpose(block1, num_outputs=256, stride=2, kernel_size=2)

    with tf.name_scope('scale2'):
        scale2 = slim.conv2d_transpose(block1, num_outputs=2, stride=2, kernel_size=2)

    with tf.name_scope('block2'):
        with tf.name_scope('conv2_1'):
            conv2_11 = slim.conv2d(upsample2, num_outputs=32, kernel_size=3)
            conv2_12 = slim.conv2d(conv2_11, num_outputs=48, kernel_size=3)
            conv2_13 = slim.conv2d(conv2_12, num_outputs=64, kernel_size=3)
            conv2_13_scale2 = slim.conv2d_transpose(conv2_13, num_outputs=64, stride=2, kernel_size=2)
            conv2_13_concat = tf.add(tf.multiply(0.1, conv1_13_scale2), conv2_13)

        with tf.name_scope('conv2_2'):
            conv2_21 = slim.conv2d(conv2_13_concat, num_outputs=64, kernel_size=3)
            conv2_22 = slim.conv2d(conv2_21, num_outputs=96, kernel_size=3)
            conv2_23 = slim.conv2d(conv2_22, num_outputs=128, kernel_size=3)

        block2 = hybrid_cnn(conv2_23, 'hybrid2')

    with tf.name_scope('upsample4'):
        upsample4 = slim.conv2d_transpose(block2, num_outputs=256, stride=2, kernel_size=2)

    with tf.name_scope('scale4'):
        scale4 = slim.conv2d_transpose(block2, num_outputs=2, stride=2, kernel_size=2)

    with tf.name_scope('block3'):
        with tf.name_scope('conv3_1'):
            conv3_11 = slim.conv2d(upsample4, num_outputs=32, kernel_size=3)
            conv3_12 = slim.conv2d(conv3_11, num_outputs=48, kernel_size=3)
            conv3_13 = slim.conv2d(conv3_12, num_outputs=64, kernel_size=3)
            conv3_13_concat = tf.add(tf.multiply(0.1, conv2_13_scale2), conv3_13)

        with tf.name_scope('conv3_2'):
            conv3_21 = slim.conv2d(conv3_13_concat, num_outputs=64, kernel_size=3)
            conv3_22 = slim.conv2d(conv3_21, num_outputs=96, kernel_size=3)
            conv3_23 = slim.conv2d(conv3_22, num_outputs=128, kernel_size=3)

        block3 = hybrid_cnn(conv3_23, 'hybrid3')

    with tf.name_scope('scale8'):
        scale8 = slim.conv2d_transpose(block3, num_outputs=2, stride=2, kernel_size=2)

    return [scale2, scale4, scale8]


def to_graph():

    params = define_parameters()

    session = tf.InteractiveSession()

    input_node = tf.placeholder(tf.float32, (params['batch_size'],
                                             params['height'], params['width'], 3), name='input_node')

    [scale_2, scale_4, scale_8] = model(input_node)

    output = tf.nn.softmax(scale_8, name="output")

    restorer = tf.train.Saver()
    epoch_to_load = 120
    restorer.restore(session, params['model_ckpt']+"/scale_segnet_{}_{}.ckpt".format(params['policy'], epoch_to_load))

    output_graph_def = graph_util.convert_variables_to_constants(session, tf.get_default_graph().as_graph_def(), ['output'])
    with gfile.FastGFile(params['model_ckpt']+"/scale_segnet_{}_{}.pb".format(params['policy'], epoch_to_load), "wb") as f:
        f.write(output_graph_def.SerializeToString())

if __name__ == "__main__":
    to_graph()