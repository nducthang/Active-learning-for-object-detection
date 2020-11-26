#!/usr/bin/env python
# %% Imports
from __future__ import print_function

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
import shutil
import time

from hough_preferences import scales, ratios, root_dir, test_image_dir, test_annot_dir, \
    test_offset, n_pos, n_threads, min_prob_threshold, n_feature_channels, \
    patch_size, application_step_size, use_reduced_grid, dataset_flag, forest_name, feat_name, n_detec, clear_area, \
    num_samples, eps, interactive,fertilized_sys_path,train_image_dir,extension,deep_features_path,feature_type

sys.path.insert(0, fertilized_sys_path)
import fertilized
#sys.path.insert(0, deep_features_path)
#import vgg_19_conv_feat
from sklearn import preprocessing
############BB class
class BB():
    def __init__(self):
        self.luc = -1
        self.rlc = -1
        self.score = -1
        self.scale = -1

def plotgammaPDF(ax, data, pdf_gamma, label, clr):
    plt.plot(data, pdf_gamma, clr, lw=5, alpha=0.6, label=label)
    plt.plot(data, pdf_gamma, 'k-')
    ax.set_xlabel('Detection score ', size=15)
    ax.legend(loc='best', frameon=False)

def plotDetections(luc, rlc, img,c):
    draw = ImageDraw.Draw(img)
    draw.line(((luc[0], luc[1]), (rlc[0], luc[1])), fill=c, width=int(7))
    draw.line(((rlc[0], luc[1]), (rlc[0], rlc[1])), fill=c, width=int(7))
    draw.line(((rlc[0], rlc[1]), (luc[0], rlc[1])), fill=c, width=int(7))
    draw.line(((luc[0], rlc[1]), (luc[0], luc[1])), fill=c, width=int(7))
    plt.imshow(img)

###############Calculate parameters of gamma distribution
def calculateGammaParam(tp_na, fp_na):
    calculateGammaParam_start_time = time.time()
    print('num tps',len(tp_na),'num fps',len(fp_na))
    if len(tp_na) == 0:
        max_tp = 0
    else:
        max_tp = max(tp_na)
    if len(fp_na) == 0:
        max_fp =0
    else:
        max_fp = max(fp_na)
    score_max = max([max_tp, max_fp])
    # print("score_max",score_max)
    tp_data = np.linspace(0, score_max, num_samples)
    if len(tp_na) <=1:
        tp_shape = eps
        tp_loc = eps
        tp_scale = eps
    else:
        tp_shape, tp_loc, tp_scale = gamma.fit(tp_na,floc=0)
    tp_pdf_gamma = gamma.pdf(tp_data, tp_shape, tp_loc, tp_scale)
    fp_data = np.linspace(0, score_max, num_samples)
    if len(fp_na) <=1:
        fp_shape = eps
        fp_loc = eps
        fp_scale = eps
    else:
        fp_shape, fp_loc, fp_scale = gamma.fit(fp_na,floc=0)
    fp_pdf_gamma = gamma.pdf(fp_data, fp_shape, fp_loc, fp_scale)
    if interactive:
        fig, ax = plt.subplots(figsize=(7, 5))
        #print("tp_pdf_gamma",tp_pdf_gamma)
        #print("fp_pdf_gamma",fp_pdf_gamma)
        plotgammaPDF(ax, tp_data, tp_pdf_gamma, label="true pos pdf", clr='r-')
        plotgammaPDF(ax, fp_data, fp_pdf_gamma, label="false pos pdf", clr='b-')
    # print("--- %s seconds for calculateGammaParam ---" % (time.time() - calculateGammaParam_start_time))
    return tp_shape, tp_loc, tp_scale, fp_shape, fp_loc, fp_scale


def init():
    init_start_time = time.time()
    widths = []
    heights = []
    images ={}
    # %% Load the forest.
    with open(root_dir+''+forest_name, 'rb') as df:
        forest = pickle.load(df)
    soil = fertilized.Soil()
    image_path = os.path.join(root_dir, test_image_dir, '*'+extension)
    annot_path = os.path.join(root_dir, test_annot_dir, "via_region_data.csv")
    # print(image_path)
    # print(annot_path)
    for filename in glob.glob(image_path):
        img_name = filename.replace(root_dir + '' + test_image_dir + '/', "")
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
            if (w!= None and h!= None):
                widths.append(w)
                heights.append(h)
    box_width = statistics.median(widths)
    box_height = statistics.median(heights)
    # print('wh',box_width,box_height)
    # print("--- %s seconds for init ---" % (time.time() - init_start_time))
    return forest,soil,box_height,box_width,images


def detection(im,forest, soil, box_height, box_width):
    # Detections
    bbs = []
    #Get vote array
    vprobmap = np.ones((im.shape[0],im.shape[1],len(scales)))
    init_start_time = time.time()
    for idx,scale in enumerate(scales):
        scaled_image = np.ascontiguousarray((rescale(im, scale) * 255.).astype('uint8'))
        print("TEST FUNCTION")
        print("SCALE IMAGE", scaled_image.shape)
        scaled_image = np.transpose(scaled_image, (2, 0, 1))
        print("SCALE IMAGE AFTER RESHAPE:", scaled_image.shape)
        if (scaled_image.shape[0] < patch_size[0] or scaled_image.shape[1] < patch_size[1]):
            print(scaled_image.shape, patch_size)
            print("the test scaled image is smaller than patch size")
            continue
        else:
            for ratio in ratios:
                if feature_type == 1:
                    feat_image = np.repeat(np.ascontiguousarray(np.rollaxis(scaled_image, 2, 0).astype(np.uint8))[:3,:,:],5,0)
                    # feat_image = np.repeat(soil.extract_hough_forest_features(scaled_image, (n_feature_channels == 32))[:3, :, :], 5, 1)

                if feature_type == 2:
                    feat_image = soil.extract_hough_forest_features(scaled_image, (n_feature_channels == 32))

#                if feature_type == 3:
#                    max_abs_scaler = preprocessing.MaxAbsScaler()
#                    feat = vgg_19_conv_feat.getDeepFeatures(scaled_image)
#                    feat_dim = np.zeros((feat.shape[2], feat.shape[0], feat.shape[1]))
#                    for ch in range(feat.shape[2]):
#                        scaled_feat = max_abs_scaler.fit_transform(feat[:, :, ch])
#                        scaled_feat = scaled_feat * 255
#                        feat_dim[ch, :, :] = scaled_feat
#                    feat_image = np.ascontiguousarray(feat_dim[:15, :, :].astype(np.uint8))

                probmap = forest.predict_image(feat_image,
                                               application_step_size,
                                               use_reduced_grid,
                                               ratio,
                                               min_prob_threshold)
                # print('idx: {} max_score:{}'.format(idx, probmap.max()))
                probmap = scipy.misc.imresize(probmap, im.shape, mode='F')
                probmap = scipy.ndimage.gaussian_filter(probmap, sigma=2)
                vprobmap[:,:,idx] = probmap
    # print("--- %s seconds for diff scale pred ---" % (time.time() - init_start_time))
    for bbidx in range(n_detec):
        max_score = vprobmap.max()
        # if  max_score > 0.12:
        max_loc = np.array(np.unravel_index(np.argmax(vprobmap), vprobmap.shape)[:2])
        max_sidx = np.array(np.unravel_index(np.argmax(vprobmap), vprobmap.shape)[2])
        max_ratio = 1 #TODO: works only for aspect ratio=1
        # print('bbidx: {} max_score:{}'.format(bbidx, max_score))
        # calculate the bounding box
        bb = BB()
        bbw = box_width / scales[max_sidx]
        bbh = box_height / scales[max_sidx]
        bb.luc = np.array([max_loc[1] - max_ratio * 0.5 * bbw, max_loc[0] - 0.5 * bbh])
        bb.rlc = np.array([max_loc[1] + max_ratio * 0.5 * bbw, max_loc[0] + 0.5 * bbh])
        bb.score = max_score
        bbs.append(bb)
        # non maximal suppression
        vprobmap[max(0, int(max_loc[0]) - int(clear_area / 2)):int(max_loc[0]) + int(clear_area / 2),
        max(0, int(max_loc[1]) - int(clear_area / 2)):int(max_loc[1]) + int(clear_area / 2), :] = 0
    #print('im_scores',im_scores)
    return bbs


######################Thresh calculation
def getThreshold(epoch):
    getThreshold_start_time = time.time()
    # im_scores = []
    tp_scores = []
    fp_scores = []
    bbsDict = {}

    forest, soil, box_height, box_width,images = init()
    detection_start_time = time.time()
    for name, im in images.items():
        # print(name)
        bbs = detection(im,forest, soil, box_height, box_width)
        bbsDict[name] = bbs
    # ("--- %s seconds for detection ---" % (time.time() - detection_start_time))
    # print("len bbs", len(bbsDict))
    scores = []
    # first epoch opt thresh is median of the scores
    if epoch == 0:
        for name, bbs in bbsDict.items():
            for bb in bbs:
                scores.append(bb.score)
        opt_tao = statistics.median(scores)
        np.save("opt_thresh_obj.npy", opt_tao)
    else:
        opt_tao = np.load("opt_thresh_obj.npy")
    # print("opt tao",opt_tao)
    # divide detections based on old opt threshold
    for name, bbs in bbsDict.items():
        for bb in bbs:
            # print("score",bb.score)
            if bb.score >= opt_tao:
                tp_scores.append(bb.score)
            else:
                fp_scores.append(bb.score)
    # Find new opt thresold
    #print("length of tp and fp scores, len of all bbs",len(tp_scores),len(tp_scores),len(bbsDict))
    p_pos = float(len(tp_scores)) / (float((len(tp_scores)) + float(len(fp_scores)))+eps)
    p_neg = 1 - p_pos
    if p_pos == 0:
        p_pos = eps
    if p_neg == 0:
        p_neg = eps
    print("p_neg",p_neg,"p_pos",p_pos)
    tp_shape, tp_loc, tp_scale, fp_shape, fp_loc, fp_scale = calculateGammaParam(tp_scores, fp_scores)
    if interactive:
        plt.show()
    #print("parametes",tp_shape, tp_loc, tp_scale, fp_shape, fp_loc, fp_scale)
    # Here S is list of all scores
    S = tp_scores + fp_scores
    min_cost = eps
    min_tao = min(S)
    max_tao = max(S)
    list_tao = np.linspace(min_tao, max_tao, 100)
    for t in list_tao:
        tp_cdf = (gamma.cdf((t), tp_shape, tp_loc, tp_scale))
        fn_cdf = (1 - gamma.cdf((t), fp_shape, fp_loc, fp_scale))
        cost = p_pos * tp_cdf + p_neg * fn_cdf
        if cost < min_cost:
            opt_tao = t
            min_cost = cost
    np.save("opt_thresh_obj.npy", opt_tao)
    opt_tao = np.load("opt_thresh_obj.npy")
    tp_cdf = (gamma.cdf((opt_tao), tp_shape, tp_loc, tp_scale))
    fn_cdf = (1 - gamma.cdf((opt_tao), fp_shape, fp_loc, fp_scale))
    max_annot_cost = 0
    max_annot_cost_img_name = ''
    for name, bbs in bbsDict.items():
        # Here S is list of per image tp and fp scores
        S = []
        annot_cost = 0
        for bb in bbs:
            S.append(bb.score)
        for s in S:
            p_s_pos_s = gamma.cdf(s + eps, tp_shape, loc=tp_loc, scale=tp_scale) - \
                            gamma.cdf(s - eps, tp_shape, loc=tp_loc, scale=tp_scale)
            p_s_neg_s = gamma.cdf(s + eps, fp_shape, loc=fp_loc, scale=fp_scale) - \
                        gamma.cdf(s - eps, fp_shape, loc=fp_loc, scale=fp_scale)
            #print("p_s_pos_s",p_s_pos_s,"p_s_neg_s",p_s_neg_s)
            # print("s,fp_shape,fp_loc, fp_scale",s,fp_shape,fp_loc, fp_scale)
            # print("p_s_neg_s,s,fp_shape,fp_loc, fp_scale",p_s_neg_s,s,fp_shape,fp_loc, fp_scale)
            if s >= opt_tao:
                p_s_fp_tao = float(p_s_neg_s * p_neg) / float(tp_cdf + eps)
                p_s_fn_tao = 0
            else:
                p_s_fp_tao = 0
                p_s_fn_tao = float(p_s_pos_s * p_pos) / float(fn_cdf + eps)
            #print("p_s_fp_tao",p_s_fp_tao ,"p_s_fn_tao",p_s_fn_tao)
            p_fp_s_tao = float(p_s_fp_tao * tp_cdf) / float((p_s_neg_s * p_neg) + (p_s_pos_s * p_pos) + eps)
            p_fn_s_tao = float(p_s_fn_tao * fn_cdf) / float((p_s_neg_s * p_neg) + (p_s_pos_s * p_pos) + eps)
            #print("fp s tao ",p_fp_s_tao ,"fn s tao ", p_fn_s_tao)
            annot_cost += (p_fp_s_tao + p_fn_s_tao)
            # print(annot_cost,)
        #print('annot_cost',annot_cost)
        if annot_cost >= max_annot_cost:
            max_annot_cost = annot_cost
            # print("max_annot_cost",max_annot_cost)
            # print(name)
            max_annot_cost_img_name = name

    src_image_path = os.path.join(root_dir, test_image_dir, max_annot_cost_img_name)
    print(os.path.join(root_dir, test_image_dir, max_annot_cost_img_name))
    copied_img = Image.open(os.path.join(root_dir, test_image_dir, max_annot_cost_img_name))
    if(interactive):
        plt.title('Image with high annotation cost')
        plt.imshow(copied_img)
        plt.show('Image with high annotation cost')
    dst_image_path = os.path.join(root_dir, train_image_dir, max_annot_cost_img_name[:-4] + '_pos'+extension)
    # print('max annot cost', max_annot_cost, 'image name', max_annot_cost_img_name)
    print()
    shutil.copy(src_image_path, dst_image_path)
    os.remove(src_image_path)
    # print("--- %s seconds for getThreshold ---" % (time.time() - getThreshold_start_time))
    return opt_tao,max_annot_cost,src_image_path

def detect(epoch):
    opt_tao, max_annot_cost,src_image_path = getThreshold(epoch)
    return opt_tao,max_annot_cost,src_image_path

# epoch = 0
# while (epoch == 0):
#      detect(epoch)
#      epoch = epoch + 1
