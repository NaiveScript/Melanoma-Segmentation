import os
import numpy as np
import pandas as pd
from skimage import io, img_as_float, img_as_bool, img_as_ubyte, exposure, util
from PIL import Image
from skimage.transform import resize

from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.contrib import slim
from crfrnn_layer import CrfRnnLayer


# from tensorflow.contrib import crf


def hybrid_cnn(cnn_in, scope):
    # with slim.arg_scope([slim.conv2d, slim.fully_connected],
    #                     normalizer_fn=slim.batch_norm,
    #                     weights_regularizer=slim.l2_regularizer(0.0005)):

    with tf.name_scope(scope):
        conv11 = slim.conv2d(cnn_in, num_outputs=128, kernel_size=1)
        conv12 = slim.conv2d(conv11, num_outputs=96, kernel_size=3, padding='SAME')

        conv21 = slim.conv2d(cnn_in, num_outputs=64, kernel_size=1)
        conv22 = slim.conv2d(conv21, num_outputs=32, kernel_size=5, padding='SAME')

        dilated_filter1 = slim.variable(name=scope + '_filter1', shape=[3, 3, 128, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.01))
        dilated_filter2 = slim.variable(name=scope + '_filter2', shape=[3, 3, 64, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.01))

        conv31 = tf.nn.atrous_conv2d(cnn_in, dilated_filter1, rate=2, padding='SAME')
        conv32 = tf.nn.atrous_conv2d(conv31, dilated_filter2, rate=3, padding='SAME')

        conv41 = slim.conv2d(cnn_in, num_outputs=32, kernel_size=1)
        conv42 = slim.max_pool2d(conv41, kernel_size=2, stride=1, padding='SAME')

        concat = tf.concat([conv12, conv22, conv32, conv42], axis=3)

    return concat


def model(inputs):
    # with slim.arg_scope([slim.conv2d, slim.fully_connected],
    #                     normalizer_fn=slim.batch_norm,
    #                     weights_regularizer=slim.l2_regularizer(0.0005)):

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


def softmax_loss(labels, logits):
    sm_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return sm_loss


def total_varational_loss(logits):
    pass


# # position and probability sensitive
def jaccard_approximate_loss(params_j, labels, logits):
    # #featch probability of semantic area
    logits_prob = tf.squeeze(tf.slice(
        tf.nn.softmax(logits), [0, 0, 0, 1],
        [params_j['batch_size'], params_j['height'] * 8, params_j['width'] * 8, 1]))

    logits_activation = tf.multiply(1.0 / 0.4, tf.subtract(tf.maximum(0., tf.subtract(logits_prob, 0.3)),
                                                           tf.maximum(0., tf.subtract(logits_prob, 0.7))))
    # logits_count = tf.count_nonzero(tf.greater(logits_activation, 0.5), axis=[1, 2])
    logits_count = tf.reduce_sum(tf.pow(logits_activation, 2))
    labels_count = tf.count_nonzero(labels, axis=[1, 2])
    labels = tf.cast(labels, tf.float32)

    substraction = tf.subtract(labels, logits_activation)
    # substraction_count = tf.count_nonzero(substraction, axis=[1, 2])

    labels_count = tf.cast(labels_count, tf.float32)
    # logits_count = tf.cast(logits_count, tf.float32)
    # substraction_count = tf.cast(substraction_count, tf.float32)

    # l2_loss = tf.sqrt(tf.reduce_sum(tf.pow(substraction, 2), axis=[1, 2]))
    l2_loss = tf.reduce_sum(tf.pow(substraction, 2), axis=[1, 2])

    # # smooth L2
    return tf.reduce_mean(l2_loss / (tf.multiply(2., labels_count)))


def image_augmentation(image, segment, policy):
    # # TODO melanoma area mask or replacement
    # # 30% room in
    if 'enhence' in policy and np.random.rand() > 0.7:
        # print("room in")
        # # find object border
        h, w, _ = image.shape
        locy, locx = np.where(segment == 255)
        objl, objr, obju, objd = min(locx), max(locx), min(locy), max(locy)

        # #        (objl + objr) / 2, (obju + objd) / 2, r - l, d - u
        cropl_max = (3 * objl + objr) / 4.
        cropr_min = (3 * objr + objl) / 4.
        cropu_max = (3 * obju + objd) / 4.
        cropd_min = (3 * objd + obju) / 4.

        mu_l, sigma_l = 2 * cropl_max / 3., cropl_max / 6.
        mu_r, sigma_r = (2 * cropr_min + w) / 3., (w - cropr_min) / 6.
        mu_u, sigma_u = 2 * cropu_max / 3., cropu_max / 8.
        mu_d, sigma_d = (2 * cropd_min + h) / 3., (h - cropd_min) / 8.

        cropl = np.random.normal(mu_l, sigma_l)
        cropr = np.random.normal(mu_r, sigma_r)
        cropu = np.random.normal(mu_u, sigma_u)
        cropd = np.random.normal(mu_d, sigma_d)

        cropl = np.clip(cropl, 0, cropl_max).astype(int)
        cropr = np.clip(cropr, cropr_min, w).astype(int)
        cropu = np.clip(cropu, 0, cropu_max).astype(int)
        cropd = np.clip(cropd, cropd_min, h).astype(int)

        # print("image", h, w, "obj", objl, objr, obju, objd, 'crop_m', cropl_max, cropr_min, cropu_max, cropd_min, "crop", cropl, cropr, cropu, cropd)

        image = image[cropu:cropd, cropl:cropr]
        # io.imshow(image)
        # io.show()
        segment = segment[cropu:cropd, cropl:cropr]

    # print(image.shape, segment.shape)

    # # 30% exposure
    if np.random.rand() > 0.7:
        # print("exposure")
        # # Brightness adjust
        image = exposure.adjust_gamma(image, np.random.normal(1.0, 0.1), np.random.normal(1.0, 0.1))
        image = exposure.adjust_log(image, np.random.normal(1.0, 0.1))
        image = exposure.adjust_sigmoid(image, np.random.normal(.25, 0.1), 10 * np.random.normal(1.0, 0.1))
        image = img_as_ubyte(image)

    # # 30% rotation
    if np.random.rand() > 0.7:
        # print("rotation")
        image = Image.fromarray(image)
        image.rotate(60 * np.random.normal(1.0, 0.2))
        image = np.asarray(image)

    # # 30% random noise
    if np.random.rand() > 0.7:
        # print("noise")
        image = util.random_noise(image, mean=0., var=5e-4 * np.random.normal(1.0, 0.1))
        image = img_as_ubyte(image)

    # # 30% padding
    if np.random.rand() > 0.7:
        # print("padding")
        p = 20
        th, tw, _ = image.shape
        tmp_img = np.zeros((th + 2 * p, tw + 2 * p, 3))
        tmp_img[p:th + p, p:tw + p] = image
        tmp_segment = np.zeros((th + 2 * p, tw + 2 * p))
        tmp_segment[p:th + p, p:tw + p] = segment
    else:
        tmp_img = image
        tmp_segment = segment

    # io.imshow(tmp_img)
    # io.show()
    # io.imshow(tmp_segment)
    # io.show()
    return tmp_img, tmp_segment


def filelist_generator(params, metric='training'):
    segmentfile_list = os.listdir(params[metric + '_segment'])
    imagefile_list = []
    for segmentation in segmentfile_list:
        imagefile_list.append(os.path.join(params[metric + '_image'], segmentation[:12] + '.jpg'))

    return imagefile_list, [os.path.join(params[metric + '_segment'], segmentfile) for segmentfile in segmentfile_list]


def gen_batchs(imagefile_list, segmentfile_list, params):
    imagefile_list, segmentfile_list = \
        shuffle(imagefile_list, segmentfile_list, random_state=np.random.randint(2 * params['epoch']))

    for i in range(len(imagefile_list) // params['batch_size']):
        raw_batch = dict()
        raw_batch['image_batch'], raw_batch['image8_batch'], raw_batch['segment2_batch'], \
        raw_batch['segment4_batch'], raw_batch['segment8_batch'] = [], [], [], [], []
        for j in range(params['batch_size']):
            index = i * params['batch_size'] + j
            image = io.imread(imagefile_list[index])
            segment = io.imread(segmentfile_list[index])

            image, segment = image_augmentation(image, segment, params['policy'])
            # io.imshow(image)
            # io.show()
            # io.imshow(segment)
            # io.show()

            image8 = 2 * resize(image, (params['height'] * 8, params['width'] * 8)) - 1.0
            image = 2 * resize(image, (params['height'], params['width'])) - 1.0

            segment8 = img_as_bool(resize(segment, (params['height'] * 8, params['width'] * 8))).astype(float)

            raw_batch['image_batch'].append(image)
            raw_batch['image8_batch'].append(image8)
            # raw_batch['segment2_batch'].append(segment2)
            # raw_batch['segment4_batch'].append(segment4)
            raw_batch['segment8_batch'].append(segment8)

        yield raw_batch


def gen_epochs(imagefile_list, segmentfile_list, params):
    for i in range(params['epoch']):
        yield gen_batchs(imagefile_list, segmentfile_list, params)


def define_parameters():
    params_dict = dict()
    params_dict['batch_size'] = 4
    params_dict['height'] = int(32*1.25)
    params_dict['width'] = int(48*1.25)
    params_dict['lambda'] = 0.0
    params_dict['epoch'] = 20
    params_dict['learning_rate'] = 0.00001

    params_dict['policy'] = 'enhence&&jaccard'
    params_dict['data_path'] = '../data'
    params_dict['model_ckpt'] = '../model_records/model_ckpts'
    params_dict['model_performance'] = '../model_records/model_performances'
    params_dict['training_image'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Training_Data')
    params_dict['training_segment'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Training_Part1_GroundTruth')
    params_dict['validation_image'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Validation_Data')
    params_dict['validation_segment'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Validation_Part1_GroundTruth')

    return params_dict


def define_recorder():
    recorder_dict = dict()
    recorder_dict['softmax_loss_2'] = []
    recorder_dict['softmax_loss_4'] = []
    recorder_dict['softmax_loss_8'] = []
    recorder_dict['softmax_loss'] = []
    recorder_dict['jaccard_approximate_loss'] = []
    recorder_dict['loss'] = []

    return recorder_dict


# def feed_fun(input_node, input8_node, output2_node, output4_node, output8_node, raw):
#
#     feed_dict = {input_node: raw['image_batch'], input8_node: raw['image8_batch'], output2_node: raw['segment2_batch'],
#                  output4_node: raw['segment4_batch'], output8_node: raw['segment8_batch']}
#
#     return feed_dict


def feed_fun(input_node, output8_node, raw):
    feed_dict = {input_node: raw['image_batch'], output8_node: raw['segment8_batch']}

    return feed_dict


if __name__ == '__main__':

    params = define_parameters()
    recorder = define_recorder()

    session = tf.InteractiveSession()

    input_node = tf.placeholder(tf.float32, (params['batch_size'],
                                             params['height'], params['width'], 3), name='input_node')

    input8_node = tf.placeholder(tf.float32, (params['batch_size'],
                                              params['height'] * 8, params['width'] * 8, 3), name='input8_node')

    output2_node = tf.placeholder(tf.int32, (params['batch_size'], params['height'] * 2, params['width'] * 2),
                                  name='output2_node')
    output4_node = tf.placeholder(tf.int32, (params['batch_size'], params['height'] * 4, params['width'] * 4),
                                  name='output4_node')
    output8_node = tf.placeholder(tf.int32, (params['batch_size'], params['height'] * 8, params['width'] * 8),
                                  name='output8_node')

    [scale_2, scale_4, scale_8] = model(input_node)

    # scale_8_crf = CrfRnnLayer(image_dims=(params['height']*8, params['width']*8),
    #                           num_classes=2,
    #                           theta_alpha=160.,
    #                           theta_beta=3.,
    #                           theta_gamma=3.,
    #                           num_iterations=10,
    #                           name='crfrnn')([tf.nn.softmax(scale_8), input8_node])
    #
    # print(scale_8_crf)

    # # define loss
    softmax_loss_2 = softmax_loss(output2_node, scale_2)
    softmax_loss_4 = softmax_loss(output4_node, scale_4)
    softmax_loss_8 = softmax_loss(output8_node, scale_8)

    # softmax_loss_sum = 0.1*softmax_loss_2 + 0.2*softmax_loss_4 + 0.7*softmax_loss_8
    softmax_loss_sum = softmax_loss_8

    jaccard_approximate_loss = jaccard_approximate_loss(params, output8_node, scale_8)

    loss = softmax_loss_sum + params['lambda'] * jaccard_approximate_loss

    # # define optimizer
    optimizer = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    # # initialize network
    session.run(tf.global_variables_initializer())

    # # generate related file paths

    training_images, training_segments = filelist_generator(params)
    validation_images, validation_segments = filelist_generator(params, metric='validation')

    # # start training

    restorer = tf.train.Saver()
    restorer_idx = 100
    restorer.restore(session, os.path.join(params['model_ckpt'], 'scale_segnet_{}_{}.ckpt'.format( params['policy'], restorer_idx)))
    saver = tf.train.Saver()
    for idx, feed_epoch in enumerate(gen_epochs(training_images, training_segments, params)):
        for step, raw in enumerate(feed_epoch):
            # print raw
            softmax_loss_sum_step, jaccard_approximate_loss_step, loss_step = \
                session.run([softmax_loss_sum, jaccard_approximate_loss, loss],
                            feed_dict=feed_fun(input_node, output8_node, raw))

            # softmax_loss_2_step, softmax_loss_4_step, softmax_loss_8_step, \
            # softmax_loss_sum_step, jaccard_approximate_loss_step, loss_step = \
            #     session.run([softmax_loss_2, softmax_loss_4, softmax_loss_8, softmax_loss_sum, jaccard_approximate_loss, loss],
            #     feed_dict=feed_fun(input_node, input8_node, output2_node, output4_node, output8_node, raw))

            # recorder['softmax_loss_2'].append(softmax_loss_2_step)
            # recorder['softmax_loss_4'].append(softmax_loss_4_step)
            # recorder['softmax_loss_8'].append(softmax_loss_8_step)
            recorder['softmax_loss'].append(softmax_loss_sum_step)
            recorder['jaccard_approximate_loss'].append(jaccard_approximate_loss_step)
            recorder['loss'].append(loss_step)

            print('step:%d, softmax_loss:%f, jaccard_approximate_loss:%f'
                  % (step, softmax_loss_sum_step, jaccard_approximate_loss_step))

            session.run(optimizer, feed_dict=feed_fun(input_node, output8_node, raw))

        print('epoch:%d, softmax_loss:%f, jaccard_approximate_loss:%f'
              % (idx, pd.Series(recorder['softmax_loss'][idx * step:(idx + 1) * step + 1]).mean(),
                 pd.Series(recorder['jaccard_approximate_loss'][idx * step:(idx + 1) * step + 1]).mean()))

        if idx > 0 and (idx + 1) % 10 == 0:
            saver.save(session, os.path.join(
                params['model_ckpt'], 'scale_segnet_{}_{}.ckpt'.format(params['policy'], restorer_idx + idx + 1)))

        df_recorder = pd.DataFrame()
        df_recorder['softmax_loss'] = recorder['softmax_loss']
        df_recorder['jaccard_approximate_loss'] = recorder['jaccard_approximate_loss']
        df_recorder['loss'] = recorder['loss']
        df_recorder.to_csv(os.path.join(params['model_performance'], params['policy'] + '.csv'), index=None)