#!/usr/bin/env python
# %% Imports
from __future__ import print_function
import fertilized

from matplotlib.pyplot import sca

from hough_preferences import scales, ratios, root_dir, min_prob_threshold, n_feature_channels, \
    patch_size, application_step_size, use_reduced_grid, forest_name, n_detec, interactive, eval_image_dir, \
    eval_annot_dir, fertilized_sys_path, extension, iou, feature_type

from PIL import Image, ImageDraw
import skimage.color
from skimage.transform import rescale
from skimage.filters import gaussian
from skimage.draw import line_aa
import numpy as np
import os
import sys
import time
# import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pickle
import sys
import sklearn.metrics
import threading
import scipy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from scipy.stats import gamma
import glob
import pandas as pd
import ast
import statistics
import time
import csv


sys.path.insert(0, fertilized_sys_path)
# sys.path.insert(0, deep_features_path)
#import vgg_19_conv_feat
#from sklearn import preprocessing
# BB class


class BB():
    def __init__(self):
        self.luc = -1
        self.rlc = -1
        self.score = -1
        self.label = ''

# Plots


def plotDetections(luc, rlc, img, c):
    draw = ImageDraw.Draw(img)
    draw.line(((luc[0], luc[1]), (rlc[0], luc[1])), fill=c, width=6)
    draw.line(((rlc[0], luc[1]), (rlc[0], rlc[1])), fill=c, width=6)
    draw.line(((rlc[0], rlc[1]), (luc[0], rlc[1])), fill=c, width=6)
    draw.line(((luc[0], rlc[1]), (luc[0], luc[1])), fill=c, width=6)
    plt.imshow(img)


def plotBBs(bb, draw, orig_img, img, c):
    bb[1] = max(0, min(bb[1], orig_img.shape[0] - 1))
    bb[0] = max(0, min(bb[0], orig_img.shape[1] - 1))
    bb[3] = max(0, min(bb[3], orig_img.shape[0] - 1))
    bb[2] = max(0, min(bb[2], orig_img.shape[1] - 1))

    draw.line(((int(bb[0]), int(bb[1])),
               (int(bb[0]), int(bb[3]))), fill=c, width=6)
    draw.line(((int(bb[0]), int(bb[3])),
               (int(bb[2]), int(bb[3]))), fill=c, width=6)
    draw.line(((int(bb[2]), int(bb[3])),
               (int(bb[2]), int(bb[1]))), fill=c, width=6)
    draw.line(((int(bb[2]), int(bb[1])),
               (int(bb[0]), int(bb[1]))), fill=c, width=6)
    plt.imshow(img)


def init():
    init_start_time = time.time()
    widths = []
    heights = []
    images = {}

    # %% Load the forest.
#    print(root_dir+''+forest_name)
    with open(root_dir+''+forest_name, 'rb') as df:
        forest = pickle.load(df)
    soil = fertilized.Soil()

    image_path = os.path.join(root_dir, eval_image_dir, '*'+extension)
    annot_path = os.path.join(root_dir, eval_annot_dir, "via_region_data.csv")
#    print(image_path)
    for filename in glob.glob(image_path):
        #        print(filename)
        img_name = filename.replace(root_dir + '' + eval_image_dir + '/', "")
        images[img_name] = np.array(Image.open(filename))
        assert not images[img_name] is None
        if images[img_name].ndim == 2:
            images[img_name] = skimage.color.gray2rgb(images[img_name])
        else:
            images[img_name] = np.ascontiguousarray(images[img_name][:, :, :3])
        reader = pd.read_csv(annot_path)
        files = reader['#filename']
        bbs = reader['region_shape_attributes']
        indices = [i for i, x in enumerate(files) if x == img_name]
        for i in indices:
            w = (ast.literal_eval(bbs[i]).get('width'))
            h = (ast.literal_eval(bbs[i]).get('height'))
            if (w != None and h != None):
                widths.append(w)
                heights.append(h)
    # %% Determine mean box size of training set.
    box_width = statistics.median(widths)
    box_height = statistics.median(heights)
    # print('wh',box_width,box_height)
    print("--- %s seconds for init ---" % (time.time() - init_start_time))
    return forest, soil, box_height, box_width, images


def evaluate(name, im, neg_image_store_flag, forest, soil, box_height, box_width):
    true_pos_bb = []
    false_pos_bb = []
    false_neg_bb = []

    pos_correct = 0
    bbs = []

    # Get vote array
    vprobmap = np.ones((im.shape[0], im.shape[1], len(scales)))

    for idx, scale in enumerate(scales):
        scaled_image = np.ascontiguousarray(
            (rescale(im, scale) * 255.).astype('uint8'))
        print("SCALE IMAGE", scaled_image.shape)
        scaled_image = np.transpose(scaled_image, (2, 0, 1))
        print("SCALE IMAGE AFTER RESHAPE:", scaled_image.shape)
        if(scaled_image.shape[0] < patch_size[0] or scaled_image.shape[1] < patch_size[1]):
            # print(scaled_image.shape,patch_size)
            # print("the test scaled image is smaller than patch size")
            continue
        else:
            for ratio in ratios:
                if feature_type == 1:
                    feat_image = np.repeat(np.ascontiguousarray(np.rollaxis(
                        scaled_image, 2, 0).astype(np.uint8))[:3, :, :], 5, 0)
                    # feat_image = np.repeat(soil.extract_hough_forest_features(scaled_image, (n_feature_channels == 32))[:3, :, :],5,1)

                if feature_type == 2:
                    feat_image = soil.extract_hough_forest_features(
                        scaled_image, (n_feature_channels == 32))

                # if feature_type == 3:
                #     max_abs_scaler = preprocessing.MaxAbsScaler()
                #     feat = vgg_19_conv_feat.getDeepFeatures(scaled_image)
                #     feat_dim = np.zeros((feat.shape[2], feat.shape[0], feat.shape[1]))
                #     for ch in range(feat.shape[2]):
                #         scaled_feat = max_abs_scaler.fit_transform(feat[:, :, ch])
                #         scaled_feat = scaled_feat * 255
                #         feat_dim[ch, :, :] = scaled_feat
                #     feat_image = np.ascontiguousarray(feat_dim[:15, :, :].astype(np.uint8))

                probmap = forest.predict_image(feat_image,
                                               application_step_size,
                                               use_reduced_grid,
                                               ratio,
                                               min_prob_threshold)
                # print('idx: {} max_score:{}'.format(idx, probmap.max()))
                probmap = scipy.misc.imresize(probmap, im.shape, mode='F')
                probmap = scipy.ndimage.gaussian_filter(probmap, sigma=2)
                vprobmap[:, :, idx] = probmap

    for bbidx in range(n_detec):
        max_score = vprobmap.max()
        # print(max_score)
        # if max_score > 0.32:
        max_loc = np.array(np.unravel_index(
            np.argmax(vprobmap), vprobmap.shape)[:2])
        max_sidx = np.array(np.unravel_index(
            np.argmax(vprobmap), vprobmap.shape)[2])
        max_ratio = 1  # TODO: works only for aspect ratio=1
        # print('bbidx: {} max_score:{}'.format(bbidx, max_score))
        # calculate the bounding box
        bb = BB()
        bbw = box_width / scales[max_sidx]
        bbh = box_height / scales[max_sidx]
        bb.luc = np.array([max_loc[1] - max_ratio * 0.5 *
                           bbw, max_loc[0] - 0.5 * bbh])
        bb.rlc = np.array([max_loc[1] + max_ratio * 0.5 *
                           bbw, max_loc[0] + 0.5 * bbh])
        bb.score = max_score
        # im_scores.append(max_score)
        bbs.append(bb)
        # non maximal suppression
        vprobmap[max(0, int(max_loc[0]) - int(bbw / 2)):int(max_loc[0]) + int(bbw / 2),
                 max(0, int(max_loc[1]) - int(bbh / 2)):int(max_loc[1]) + int(bbh / 2), :] = 0
    annot_path = os.path.join(root_dir, eval_annot_dir, "via_region_data.csv")
    im = Image.fromarray(im)
    reader = pd.read_csv(annot_path)
    files = reader['#filename']
    bbs_attr = reader['region_shape_attributes']
    gt_bbs = []
    # if im_idx < n_pos:
    indices = [i for i, x in enumerate(files) if x == name]
    for i in indices:
        l = (ast.literal_eval(bbs_attr[i]).get('x'))
        t = (ast.literal_eval(bbs_attr[i]).get('y'))
        w = (ast.literal_eval(bbs_attr[i]).get('width'))
        h = (ast.literal_eval(bbs_attr[i]).get('height'))
        try:
            r = l + w
            b = t + h
            gt_bbs.append([l, t, r, b])
        except:
            print('no annotation entry')
            continue
    for idx, bb in enumerate(bbs):
        min_iou = 100
        gt_box_ele = None
        d_box_ele = None
        for gt in gt_bbs:
            l = gt[0]
            t = gt[1]
            r = gt[2]
            b = gt[3]
            area_gt = float((r - l) * (b - t))
            try:
                assert area_gt > 0.
                intersect_l = max(l, min(r, bb.luc[0]))
                intersect_r = min(r, max(l, bb.rlc[0]))
                if (intersect_r < intersect_l):
                    area_overlap = 0
                else:
                    intersect_t = max(t, min(b, bb.luc[1]))
                    intersect_b = min(b, max(t, bb.rlc[1]))
                    area_overlap = float(
                        (intersect_r - intersect_l) * (intersect_b - intersect_t))
                    assert area_overlap <= area_gt
                    area_det = float(
                        (bb.rlc[0] - bb.luc[0]) * (bb.rlc[1] - bb.luc[1]))
                    assert area_overlap <= area_det, '%f, %f' % (
                        area_overlap, area_det)
                total_overlap = area_overlap / \
                    (area_det + area_gt - area_overlap)
                if (total_overlap >= iou and min_iou > total_overlap):
                    pos_correct += 1
                    min_iou = total_overlap
                    gt_box_ele = [l, t, r, b]
                    d_box_ele = [bb.luc[0], bb.luc[1],
                                 bb.rlc[0], bb.rlc[1], bb.score]
            except:
                pass
        if (d_box_ele != None):
            true_pos_bb.append(d_box_ele)
            # im_truth.append(1)
            bbs[idx].label = "tp"
        if min_iou != 100:  # if bb overlaps
            gt_bbs.remove(gt_box_ele)
        else:
            false_pos_bb.append(
                [bb.luc[0], bb.luc[1], bb.rlc[0], bb.rlc[1], bb.score])
            bbs[idx].label = "fp"
            # im_truth.append(0)

    for bb in gt_bbs:
        # im_truth.append(1)
        # im_scores.append(0.)
        false_neg_bb.append(bb)
        fn_bb = BB()
        fn_bb.luc = [bb[0], bb[1]]
        fn_bb.rlc = [bb[2], bb[3]]
        fn_bb.score = 0
        fn_bb.label = "fn"
        bbs.append(fn_bb)

    # len_fn = len_fn +len(false_neg_bb)
    # len_fp = len_fp + len(false_pos_bb)

    orig_img = np.asarray(im)
    if orig_img.ndim == 2:
        orig_img = skimage.color.gray2rgb(orig_img)
    else:
        orig_img = np.ascontiguousarray(orig_img[:, :, :3])
    img = Image.fromarray(orig_img)

    # print("rand number" ,rand_idx,false_pos_bb[rand_idx[0]])
    # for bb in false_pos_bb:
#    try:
#        if neg_image_store_flag == 1:
#            #print("saving neg image")
#            #print(root_dir)
#            print( train_image_dir)
#            rand_idx = np.random.randint(0, len(false_pos_bb), 1)
#            bb = false_pos_bb[rand_idx[0]]
#            temp_img = img.copy()
#            crop_img = temp_img.crop((bb[0], bb[1], bb[2], bb[3]))
#            #plt.imshow(crop_img)
#
#            crop_img.save(os.path.join(root_dir, train_image_dir, time.strftime("%Y%m%d-%H%M%S") + "_auto_neg" + extension))
#    except:
#        print("No false positives to store back as negative images!")
#        pass

    if interactive:
        img_tp = img.copy()
        draw = ImageDraw.Draw(img_tp)
        for bb in true_pos_bb:
            plotBBs(bb, draw, orig_img, img_tp, c="red")
        for bb in false_pos_bb:
            temp_img = img.copy()
            crop_img = temp_img.crop((bb[0], bb[1], bb[2], bb[3]))
#            crop_img.save(os.path.join(root_dir, eval_image_dir, time.strftime("%Y%m%d-%H%M%S")+"_auto_neg"+extension))
            plotBBs(bb, draw, orig_img, img_tp, c="blue")
        for bb in false_neg_bb:
            if (len(bb) != 0):
                plotDetections(
                    (bb[0], bb[1]), (bb[2], bb[3]), img_tp, c="cyan")
        plt.axis('off')
        plt.savefig(time.strftime("%Y%m%d-%H%M%S") + extension)
        plt.title("Detections")
        plt.show()
    #
    return bbs


def precision_recall_curve(vvbb):
    prcurve_start_time = time.time()
    # get max and min conf
    max_thrs = -1
    min_thrs = 1e10
    for vbb in vvbb:
        for bb in vbb:
            if(bb.score > max_thrs):
                max_thrs = bb.score
            if(bb.score < min_thrs):
                min_thrs = bb.score

    # get total gt positives
    total_pos = 0
    for vbb in vvbb:
        for bb in vbb:
            if(bb.label == "tp" or bb.label == "fn"):
                total_pos += 1

    # init variables
    thrs = np.linspace(min_thrs, max_thrs, 100)
    precision = np.zeros(thrs.shape)
    recall = np.zeros(thrs.shape)

    # populate precision and recall
    for idx, thr in enumerate(thrs):
        tp = 0.0
        fp = 0.0
        for vbb in vvbb:
            for bb in vbb:
                if(bb.score >= thr):
                    if(bb.label == "tp"):
                        tp += 1
                    if (bb.label == "fp"):
                        fp += 1
        fn = total_pos - tp
        recall[idx] = tp / (tp + fn + 1e-10)
        precision[idx] = tp / (tp + fp + 1e-10)

    # sort arrays as per decreasing recall
    indices = [i[0] for i in sorted(
        enumerate(recall), key=lambda x: x[1], reverse=True)]
    recall = recall[indices]
    precision = precision[indices]
    thrs = thrs[indices]

    # post process precision
    # print(recall)
    # print(precision)
    for idx in range(1, len(precision)):
        if(precision[idx] < precision[idx-1]):
            precision[idx] = precision[idx - 1]

    auc = 0
    for idx in range(1, len(precision)):
        auc += 0.5 * \
            np.abs((precision[idx-1]+precision[idx])
                   * (recall[idx-1]-recall[idx]))
    # print("mAP : ", auc)
    #print("--- %s seconds for pr curve ---" % (time.time() - prcurve_start_time))
    return precision, recall, thrs, auc


def runEvaluate():
    runeval_start_time = time.time()
    all_bbs = []
    forest, soil, box_height, box_width, images = init()
    eval_start_time = time.time()
    for idx, (name, im) in enumerate(images.items()):
        neg_image_store_flag = 1
#        if idx % 20 == 0:
#            neg_image_store_flag =1
#        else:
#            neg_image_store_flag = 0
        bbs = evaluate(name, im, neg_image_store_flag,
                       forest, soil, box_height, box_width)

        all_bbs.append(bbs)
        # im_idx = im_idx + 1
    print("--- %s seconds for eval ---" % (time.time() - eval_start_time))
    precision, recall, thrs, auc = precision_recall_curve(all_bbs)
#    print("precision",precision)
#    print("recall",recall)
    return auc
