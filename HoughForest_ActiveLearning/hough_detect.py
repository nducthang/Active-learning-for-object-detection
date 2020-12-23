from __future__ import print_function
from numpy.core.records import array
from scipy.stats.stats import PearsonRConstantInputWarning
from sklearn import preprocessing
import fertilized
from PIL import Image, ImageDraw
from matplotlib.pyplot import sca
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
from skimage.transform import resize

from hough_preferences import scales, ratios, root_dir, test_image_dir, test_annot_dir, min_prob_threshold, n_feature_channels, \
    patch_size, application_step_size, use_reduced_grid, forest_name, n_detec, clear_area, \
    num_samples, eps, interactive, fertilized_sys_path, extension, feature_type, train_image_dir

import cv2
import matplotlib.patches as patches
sys.path.insert(0, fertilized_sys_path)
#sys.path.insert(0, deep_features_path)
#import vgg_19_conv_feat


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

count_image = 0
def plotBoundingBox(im, bbs):
    global count_image
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    for bb in bbs:
        rect = patches.Rectangle((bb.luc[0], bb.luc[1]), bb.rlc[0]-bb.luc[0], bb.rlc[1]-bb.luc[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.savefig(str(count_image)+'.png')
    count_image += 1

# Calculate parameters of gamma distribution


def calculateGammaParam(tp_na, fp_na):
    # calculateGammaParam_start_time = time.time()
    print('num tps', len(tp_na), 'num fps', len(fp_na))
    if len(tp_na) == 0:
        max_tp = 0
    else:
        max_tp = max(tp_na)
    if len(fp_na) == 0:
        max_fp = 0
    else:
        max_fp = max(fp_na)
    score_max = max([max_tp, max_fp])
    # print("score_max",score_max)
    tp_data = np.linspace(0, score_max, num_samples)
    if len(tp_na) <= 1:
        tp_shape = eps
        tp_loc = eps
        tp_scale = eps
    else:
        tp_shape, tp_loc, tp_scale = gamma.fit(tp_na, floc=0)
    tp_pdf_gamma = gamma.pdf(tp_data, tp_shape, tp_loc, tp_scale)
    fp_data = np.linspace(0, score_max, num_samples)
    if len(fp_na) <= 1:
        fp_shape = eps
        fp_loc = eps
        fp_scale = eps
    else:
        fp_shape, fp_loc, fp_scale = gamma.fit(fp_na, floc=0)
    fp_pdf_gamma = gamma.pdf(fp_data, fp_shape, fp_loc, fp_scale)
    if interactive:
        fig, ax = plt.subplots(figsize=(7, 5))
        # print("tp_pdf_gamma",tp_pdf_gamma)
        # print("fp_pdf_gamma",fp_pdf_gamma)
        plotgammaPDF(ax, tp_data, tp_pdf_gamma, label="true pos pdf", clr='r-')
        plotgammaPDF(ax, fp_data, fp_pdf_gamma,
                     label="false pos pdf", clr='b-')
    # print("--- %s seconds for calculateGammaParam ---" % (time.time() - calculateGammaParam_start_time))
    return tp_shape, tp_loc, tp_scale, fp_shape, fp_loc, fp_scale


def init():
    """ Hàm khởi tạo """
    init_start_time = time.time()
    widths = []
    heights = []
    images = {}
    # %% Load the forest.
    with open(root_dir+''+forest_name, 'rb') as df:
        forest = pickle.load(df)
    soil = fertilized.Soil()
    # root_dir = './leaf/'
    # test_image_dir = 'test_images'
    # test_annot_dir = 'test_boundingboxes'
    image_path = os.path.join(root_dir, test_image_dir, '*'+extension)
    annot_path = os.path.join(root_dir, test_annot_dir, "via_region_data.csv")

    # Duyệt tất cả các file trong thư mục test_images
    for filename in glob.glob(image_path):
        # Chỉ trích xuất ra tên ảnh (Không trích xuất các thư mục cha)
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
            if (w != None and h != None):
                widths.append(w)
                heights.append(h)
    # Lấy box_width và box_height là median của các box
    box_width = statistics.median(widths)
    box_height = statistics.median(heights)
    print("--- %s seconds for init ---" % (time.time() - init_start_time))
    return forest, soil, box_height, box_width, images


def detection(im, forest, soil, box_height, box_width):
    """ Thực hiện detection với mỗi ảnh """
    # chuyển đổi ảnh về định dạng BGR theo oepncv để tương thích với thư viện
    if im.ndim == 2:
        im = np.ascontiguousarray(skimage.color.gray2rgb(im))
    else:
        im = np.ascontiguousarray(im[:, :, :3])
    # Shape = (530, 500, 3)
    # Detections
    bbs = []
    # Get vote array
    # scales = (0.75, 1.0)
    # shape = (530, 500, 3, 2)
    vprobmap = np.ones((im.shape[0], im.shape[1], len(scales)))
    # init_start_time = time.time()
    # Duyệt từng scale được cài đặt
    for idx, scale in enumerate(scales):
        # scaled_image = np.ascontiguousarray((rescale(im, scale) * 255.).astype('uint8'))
        new_height, new_width = int(
            im.shape[0] * scale), int(im.shape[1] * scale)
        b = im[:, :, 0]
        g = im[:, :, 1]
        r = im[:, :, 2]
        scaled_b = cv2.resize(src=b, dsize=(new_height, new_width))
        scaled_g = cv2.resize(src=g, dsize=(new_height, new_width))
        scaled_r = cv2.resize(src=r, dsize=(new_height, new_width))
        scaled_image = np.array([scaled_b, scaled_g, scaled_r])

        # shape = (3, 375, 397)
        scaled_image = np.transpose(scaled_image, (2, 1, 0))
        # Chuyển về (397, 375, 3)

        # Chuyển về định dạng tương thích để trích xuất đặc trưng
        if scaled_image.ndim == 2:
            scaled_image = np.ascontiguousarray(
                skimage.color.gray2rgb(scaled_image))
        else:
            scaled_image = np.ascontiguousarray(scaled_image[:, :, :3])

        # Nếu kích cỡ ảnh sau khi scale mà nhỏ hơn kích cỡ patch thì bỏ qua
        if (scaled_image.shape[0] < patch_size[0] or scaled_image.shape[1] < patch_size[1]):
            print(scaled_image.shape, patch_size)
            print("the test scaled image is smaller than patch size")
            continue
        else:
            # Ngược lại thì thực hiện thuật toán
            # Duyệt từng scale (tỷ lệ) của ratios (tỷ lệ cho mỗi ảnh)
            for ratio in ratios:
                feat_image = None
                if feature_type == 1:
                    # Trích xuất đặc trưng RGB
                    feat_image = np.repeat(np.ascontiguousarray(np.rollaxis(
                        scaled_image, 2, 0).astype(np.uint8))[:3, :, :], 5, 0)

                if feature_type == 2:
                    # Trích xuất đặc trưng bằng hough forest
                    feat_image = soil.extract_hough_forest_features(
                        scaled_image, (n_feature_channels == 32))

#                if feature_type == 3:
#                    max_abs_scaler = preprocessing.MaxAbsScaler()
#                    feat = vgg_19_conv_feat.getDeepFeatures(scaled_image)
#                    feat_dim = np.zeros((feat.shape[2], feat.shape[0], feat.shape[1]))
#                    for ch in range(feat.shape[2]):
#                        scaled_feat = max_abs_scaler.fit_transform(feat[:, :, ch])
#                        scaled_feat = scaled_feat * 255
#                        feat_dim[ch, :, :] = scaled_feat
#                    feat_image = np.ascontiguousarray(feat_dim[:15, :, :].astype(np.uint8))

                # feat_image shape = (15, 397, 375)
                # Đưa đặc trưng vào và tạo ra ma trận xác suất
                # probmap = (397, 375)
                probmap = forest.predict_image(feat_image,
                                               application_step_size,
                                               use_reduced_grid,
                                               ratio,
                                               min_prob_threshold)

                # probmap = scipy.misc.imresize(probmap, im.shape, mode='F')
                # Đưa ảnh map xác suất về kích cỡ ban đầu của ảnh
                # probmap = (530, 500)
                probmap = resize(probmap, output_shape=(
                    im.shape[0], im.shape[1]))
                # Lọc gaussian
                probmap = scipy.ndimage.gaussian_filter(probmap, sigma=2)
                # Lưu ma trận xác suất lại
                # vprobmap shape = (530, 500, 2)
                vprobmap[:, :, idx] = probmap

    # print("--- %s seconds for diff scale pred ---" % (time.time() - init_start_time))

    # Detect n_dectec đối tượng
    for bbidx in range(n_detec):
        max_score = vprobmap.max()
        # Chuyển đổi chỉ số mảng 1D tương ứng vể chỉ số 2D
        # Ví dụ 6 trong 1D tương ứng với vị trí (1,2) trong mảng 2D chiều 3x4
        # np.argmax trả về index 1D mà tại đó value lớn nhất
        # return [235, 172, 0]
        max_loc = np.array(np.unravel_index(
            np.argmax(vprobmap), vprobmap.shape)[:2])  # [235, 172]
        # Lấy thông tin scale mà tại đó thu được xác suất lớn nhất
        max_sidx = np.array(np.unravel_index(
            np.argmax(vprobmap), vprobmap.shape)[2])  # [0]
        max_ratio = 1  # TODO: works only for aspect ratio=1

        # calculate the bounding box
        bb = BB()
        # Tính toán chiều rộng và chiều cao cho bouding box
        bbw = box_width / scales[max_sidx]  # scales = (0.75 1.0)
        bbh = box_height / scales[max_sidx]
        # Lấy ra tọa độ góc trái và góc phải của box
        bb.luc = np.array([max_loc[1] - max_ratio * 0.5 *
                           bbw, max_loc[0] - 0.5 * bbh])
        bb.rlc = np.array([max_loc[1] + max_ratio * 0.5 *
                           bbw, max_loc[0] + 0.5 * bbh])
        bb.score = max_score
        bbs.append(bb)

        # non maximal suppression
        # clear_area = 200
        # Đưa các pixel xung quanh về 0 để lần chọn sau không chọn box chồng lấn lại
        vprobmap[max(0, int(max_loc[0]) - int(clear_area / 2)):int(max_loc[0]) + int(clear_area / 2),
                 max(0, int(max_loc[1]) - int(clear_area / 2)):int(max_loc[1]) + int(clear_area / 2), :] = 0

    # print('im_scores',im_scores)
    # print(bbs)
    return bbs


# Thresh calculation
def getThreshold(epoch):
    getThreshold_start_time = time.time()
    # im_scores = []
    tp_scores = []  # true postive
    fp_scores = []  # false positive
    bbsDict = {}  # bounding box dict
    # forest và soil là của thư viện
    # box_height là median height của các box đã được gán
    # box_width là median width của các box đã được gán
    # images là dictionary {"tên ảnh": ma trận ảnh}
    forest, soil, box_height, box_width, images = init()
    detection_start_time = time.time()
    for name, im in images.items():
        bbs = detection(im, forest, soil, box_height, box_width)
        bbsDict[name] = bbs
        plotBoundingBox(im, bbs) # Thử vẽ các bounding box

    ("--- %s seconds for detection ---" % (time.time() - detection_start_time))
    # print("len bbs", len(bbsDict))
    scores = []
    # first epoch opt thresh is median of the scores
    if epoch == 0:
        for name, bbs in bbsDict.items():
            for bb in bbs:
                scores.append(bb.score)
        # Lấy median các score và lưu lại tại epoch 0
        opt_tao = statistics.median(scores)
        np.save("opt_thresh_obj.npy", opt_tao)
    else:
        # với các epoch khác thì sẽ load score
        opt_tao = np.load("opt_thresh_obj.npy")

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
    p_pos = float(len(tp_scores)) / \
        (float((len(tp_scores)) + float(len(fp_scores)))+eps)
    p_neg = 1 - p_pos
    if p_pos == 0:
        p_pos = eps
    if p_neg == 0:
        p_neg = eps

    print("p_neg", p_neg, "p_pos", p_pos)
    tp_shape, tp_loc, tp_scale, fp_shape, fp_loc, fp_scale = calculateGammaParam(
        tp_scores, fp_scores)

    # Hiển thị đồ thị nếu interative = True
    if interactive:
        plt.show()

    print("parametes", tp_shape, tp_loc, tp_scale, fp_shape, fp_loc, fp_scale)

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

            if s >= opt_tao:
                p_s_fp_tao = float(p_s_neg_s * p_neg) / float(tp_cdf + eps)
                p_s_fn_tao = 0
            else:
                p_s_fp_tao = 0
                p_s_fn_tao = float(p_s_pos_s * p_pos) / float(fn_cdf + eps)

            p_fp_s_tao = float(p_s_fp_tao * tp_cdf) / \
                float((p_s_neg_s * p_neg) + (p_s_pos_s * p_pos) + eps)
            p_fn_s_tao = float(p_s_fn_tao * fn_cdf) / \
                float((p_s_neg_s * p_neg) + (p_s_pos_s * p_pos) + eps)

            annot_cost += (p_fp_s_tao + p_fn_s_tao)

        if annot_cost >= max_annot_cost:
            max_annot_cost = annot_cost
            max_annot_cost_img_name = name

    # Lấy đường dẫn hình ành đáng test
    src_image_path = os.path.join(
        root_dir, test_image_dir, max_annot_cost_img_name)
    print(os.path.join(root_dir, test_image_dir, max_annot_cost_img_name))

    copied_img = Image.open(os.path.join(
        root_dir, test_image_dir, max_annot_cost_img_name))

    if(interactive):
        plt.title('Image with high annotation cost')
        plt.imshow(copied_img)
        # plt.show('Image with high annotation cost')
        plt.show()

    dst_image_path = os.path.join(
        root_dir, train_image_dir, max_annot_cost_img_name[:-4] + '_pos'+extension)

    print('max annot cost', max_annot_cost,
          'image name', max_annot_cost_img_name)
    print()
    # copy ảnh ở thư mục test vào thư mục train
    shutil.copy(src_image_path, dst_image_path)
    # Xóa ảnh ở thư mục test đi
    os.remove(src_image_path)
    print("--- %s seconds for getThreshold ---" %
          (time.time() - getThreshold_start_time))
    return opt_tao, max_annot_cost, src_image_path


def detect(epoch):
    opt_tao, max_annot_cost, src_image_path = getThreshold(epoch)
    return opt_tao, max_annot_cost, src_image_path

# epoch = 0
# while (epoch == 0):
#      detect(epoch)
#      epoch = epoch + 1


if __name__ == '__main__':
    opt_tao, max_annot_cost, src_image_path = getThreshold(0)
    print(opt_tao, max_annot_cost, src_image_path)
