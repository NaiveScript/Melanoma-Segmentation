import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, compute_unary
from tensorflow.python.platform import gfile
from skimage import io, transform, measure, img_as_float, img_as_ubyte, img_as_uint
from sklearn.metrics import jaccard_similarity_score


def define_parameters():
    params_dict = dict()
    params_dict['batch_size'] = 8
    params_dict['height'] = int(32*1.)
    params_dict['width'] = int(48*1.)
    params_dict['lambda'] = 0.1
    params_dict['epoch'] = 100
    params_dict['learning_rate'] = 0.00005

    params_dict['policy'] = 'enhence'
    params_dict['model_path'] = '../model_epoch'
    params_dict['data_path'] = '../data'
    params_dict['model_ckpt'] = '../model_records/model_ckpts'
    params_dict['model_performance'] = '../model_records/model_performances'
    params_dict['training_image'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Training_Data')
    params_dict['training_segment'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Training_Part1_GroundTruth')
    params_dict['validation_image'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Validation_Data')
    params_dict['validation_segment'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Validation_Part1_GroundTruth')
    params_dict['test_image'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Test_v2_Data')
    params_dict['test_segment'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Test_v2_Part1_GroundTruth')

    ## NOTE:set proper thresh hold to seperate segmentaion
    params_dict['thresh'] = 0.2
    params_dict['gfile_path'] = params_dict['model_path']
    params_dict['validation_segment_model'] = os.path.join(params_dict['data_path'], 'ISIC-2017_Validation_Part1_Model')
    params_dict['test_segment_model'] = os.path.join(params_dict['data_path'],
                                                     'ISIC-2017_Test_v2_Part1_Model_{}_{}'.format(
                                                         params_dict['policy'], params_dict['thresh']))

    return params_dict


def process_segment(segimg, params):
    # #scale
    # min_prob, max_prob = np.min(segimg), np.max(segimg)
    # segimg = (segimg - min_prob) / (max_prob - min_prob)

    segimg[segimg > params['thresh']] = 1
    segimg[segimg <= params['thresh']] = 0

    num_pixel = np.count_nonzero(segimg)
    for region in measure.regionprops(img_as_ubyte(segimg)):
        if region.area <= num_pixel / 3.:
            # minr, minc, maxr, maxc = region.bbox
            # segimg[minr: maxr, minc: maxc] = 0
            segimg[region.area] = 0

    segimg = img_as_float(segimg)

    contours = measure.find_contours(segimg, 0.8)

    return segimg, contours


def crf_segment(segment, testimage, w, h):

    # # transform to the desired size
    segment_bg = np.reshape(transform.resize(segment[0, :, :, 0], (h, w)), (h, w, 1))
    segment_obj = np.reshape(transform.resize(segment[0, :, :, 1], (h, w)), (h, w, 1))
    segment_ob = np.append(segment_bg, segment_obj, axis=2)

    segment_unary = unary_from_softmax(segment_ob.transpose(2, 0, 1))

    # # define CRF
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(segment_unary)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(1, 1), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(15, 15), srgb=(3, 3, 3), rgbim=testimage,
                           compat=3,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(3)

    segment_crf = np.argmax(Q, axis=0).reshape(h, w)

    num_pixel = np.count_nonzero(segment_crf)
    for region in measure.regionprops(img_as_ubyte(segment_crf)):
        if region.area <= num_pixel / 4.:
            segment_crf[region.area] = 0
    contours_crf = measure.find_contours(segment_crf, 0.8)

    del Q, d, segment, segment_unary, testimage

    return segment_crf, contours_crf


def writetest():
    method = 'test'
    params = define_parameters()
    # config = tf.ConfigProto(device_count={'CPU': 0})
    # sess = tf.InteractiveSession(config=config)
    with tf.Session() as sess:
        with tf.device('/cpu:0'):

            # gfilepath = os.path.join(params['model_path'], "scale_segnet_207.pb")
            gfilepath = os.path.join(params['model_ckpt'], "scale_segnet_{}_{}.pb".format(params['policy'], 100))

            with gfile.FastGFile(gfilepath, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                input, output = \
                    (tf.import_graph_def(graph_def, name='',
                                         return_elements=['input_node:0', 'output:0']))

                testfiles = os.listdir(params['{}_image'.format(method)])
                start_time = time.time()
                if not os.path.exists(params['{}_segment_model'.format(method)]):
                    os.makedirs(params['{}_segment_model'.format(method)])
                imagenames, jaccards, jaccards_crf = [], [], []
                cnt = 1
                for testfile in testfiles:
                    if '.jpg' in testfile:
                        imagenames.append(testfile)
                        batch = []
                        testimage = io.imread(os.path.join(params['{}_image'.format(method)], testfile))

                        h, w, _ = testimage.shape

                        testimagedown = 2 * transform.resize(testimage,
                                                             (params['height'], params['width'])) - 1.0

                        batch.append(testimagedown)

                        segment = sess.run(output, feed_dict={input: batch})
                        segment_model = segment[0, :, :, 1]

                        segment_model, contours_model = process_segment(
                            transform.resize(segment_model, (h, w)), params)

                        segment_crf, contours_crf = crf_segment(segment, testimage, w, h)

                        # io.imshow(segment_crf)
                        # io.show()
                        # io.imsave(os.path.join(
                        #     params['{}_segment_model'.format(method)], testfile.split('.')[0] + '_segmentation.png'), segment_model)

                        segment_label = img_as_float(io.imread(os.path.join(
                            params['{}_segment'.format(method)], testfile.split('.')[0] + '_segmentation.png')))

                        segment_label[segment_label > params['thresh']] = 1
                        segment_label[segment_label <= params['thresh']] = 0

                        contours_label = measure.find_contours(segment_label, 0.8)

                        fig = plt.figure(frameon=False)
                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig.add_axes(ax)

                        ax.imshow(testimage, interpolation='nearest', cmap=plt.cm.gray)

                        for n, contour in enumerate(contours_model):
                            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=(0.5, 0., 0.))

                        for n, contour in enumerate(contours_crf):
                            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=(0., 0.5, 0.))

                        for n, contour in enumerate(contours_label):
                            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=(0., 0., .5))

                        ax.axis('image')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # plt.imshow()
                        # plt.show()

                        # plt.savefig(os.path.join(
                        #     params['{}_segment_model'.format(method)], testfile.split('.')[0] + '_segmentation.eps'))

                        fig.savefig(os.path.join(
                            params['{}_segment_model'.format(method)], testfile.split('.')[0] + '_segmentation.eps')
                            , bbox_inches='tight', pad_inches=0)
                        plt.close()
                        jaccard = jaccard_similarity_score(segment_label, segment_crf)
                        # jaccard_crf = jaccard_similarity_score(segment_label, segment_crf)
                        # print(jaccard_similarity_score(segment_label, map))
                        print(cnt, jaccard, pd.Series(jaccards).mean())
                        cnt += 1
                        jaccards.append(jaccard)
                        # jaccards_crf.append(jaccard_crf)
                df_records = pd.DataFrame()
                df_records['images'] = imagenames
                df_records['jaccards'] = jaccards
                # df_records['jaccards_crf'] = jaccards_crf
                score_overall = df_records['jaccards'].mean()
                # score_overall_crf = df_records['jaccards_crf'].mean()
                print(score_overall)
                df_records.to_csv(os.path.join(
                    params['data_path'], '{}_records_{}_{}.csv'.format(method, params['policy'], score_overall)), index=None)

                print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    writetest()