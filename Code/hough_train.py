import time
from numpy.lib import index_tricks
from skimage import transform
from sklearn import preprocessing
from skimage import img_as_uint
import fertilized
from hough_preferences import patch_size, n_trees, min_samples_per_split, \
    max_depth, n_splits, n_thresholds, n_samples_pos, n_samples_neg, \
    n_feature_channels, max_patches_per_node, n_threads, root_dir, \
    train_image_dir, train_annot_dir, fertilized_sys_path, extension, feat_name, forest_name, feature_type

from PIL import Image
import skimage.color
import numpy as np
import os
import sys
import pickle
import glob
import pandas as pd
import ast
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, fertilized_sys_path)
np.set_printoptions(threshold=sys.maxsize)


def init():
    # Lấy danh sách tất cả các ảnh positive và negative
    # root_dir = './leaf/'
    # train_image_dỉr = '/train_images'
    pos_image_path = os.path.join(root_dir, train_image_dir, '*_pos'+extension)
    neg_image_path = os.path.join(root_dir, train_image_dir, '*_neg'+extension)
    # Lấy đường dẫn file annotation
    # train_annot_dỉr = '/train_boundingboxes'
    annot_path = os.path.join(root_dir, train_annot_dir, "via_region_data.csv")

    # Hiển thị thông tin số lượng ảnh positive và negative
    train_pos_ids = len(glob.glob(pos_image_path))
    train_neg_ids = len(glob.glob(neg_image_path))
    print("Number of positive images", train_pos_ids)
    print("Number of negative images", train_neg_ids)

    # Trả về: Đường dẫn ảnh positive, đường dẫn ảnh negative, đường dẫn file annotation, số ảnh positve, số ảnh negative
    return pos_image_path, neg_image_path, annot_path, train_pos_ids, train_neg_ids


def trainHoughForests():
    # %% Positive images.
    pos_image_path, neg_image_path, annot_path, train_pos_ids, train_neg_ids = init()
    # Patch descriptions
    # n_samples_pos = 25, n_samples_neg = 25
    # train_pos_ids = 2 (số ảnh positive), train_neg_ids = 3 (số ảnh negative)
    # annotations shape = (125, 5)
    annotations = np.zeros(
        (n_samples_pos * train_pos_ids + n_samples_neg * train_neg_ids, 5), dtype='int16')
    images = []
    pos_number = 0
    # Duyệt từng file positive
    for filename in glob.glob(pos_image_path):
        img_name = filename.replace(root_dir + '' + train_image_dir + '/', "")
        reader = pd.read_csv(annot_path)
        files = reader['#filename']
        # Các thuộc tính của bbs: name, x, y, width, height
        bbs = reader['region_shape_attributes']
        # Các chỉ số hàng tương ứng với file đang xét
        indices = [i for i, x in enumerate(files) if x == img_name]

        images.append(np.array(Image.open(filename)))
        for i in indices:
            # Lấy thông tin của từng bounding box
            # ast.literal_eval để kiểm tra chặt hơn (bbs[i] phải là một kiểu của python)
            # left, top, width, height
            l = (ast.literal_eval(bbs[i]).get('x'))
            t = (ast.literal_eval(bbs[i]).get('y'))
            w = (ast.literal_eval(bbs[i]).get('width'))
            h = (ast.literal_eval(bbs[i]).get('height'))
            #print(l, t, w, h)
            if (l != None and t != None and w != None and h != None):
                # right, bottom
                r = l + w
                b = t + h
                try:
                    # Kiểm tra xem bounding box có đủ lớn để trích xuất hay không
                    # (.,., 0) đánh dấu pos_number, đánh ID ảnh
                    assert t + patch_size[0] // 2 <= b - patch_size[0] // 2
                    annotations[pos_number *
                                n_samples_pos: (pos_number + 1) * n_samples_pos, 0] = pos_number
                    # print(pos_number * n_samples_pos,'to',(pos_number + 1) * n_samples_pos, 0,'is',pos_number)
                    # Result:
                    # 0 to 25 0 is 0
                    # 0 to 25 0 is 0
                    # 25 to 50 0 is 1
                    # 25 to 50 0 is 1
                    # 25 to 50 0 is 1
                    try:
                        # Lấy ngẫu nhiên các vị trí x positive trong box
                        # Theo setting là 25 vị trí = n_samples_pos
                        # (.,., 1) Lưu vị trí x được lấy
                        annotations[pos_number * n_samples_pos: (pos_number + 1) * n_samples_pos, 1] = np.random.randint(
                            l + patch_size[1] // 2, r - patch_size[1] // 2, size=(n_samples_pos,))  # get rand x positions in obj
                    except:
                        print(
                            'Please check annotations, object size too small in x axis')
                        continue

                    # Kiểm tra lại tọa độ x xem thỏa mãn không
                    assert np.all(annotations[pos_number * n_samples_pos: (pos_number + 1) *
                                              n_samples_pos, 1] - l >= patch_size[1] // 2)  # check x position in obj x min

                    assert np.all(r - annotations[pos_number * n_samples_pos: (
                        pos_number + 1) * n_samples_pos, 1] >= patch_size[1] // 2)  # check x position in obj x max
                    try:
                        # Lấy ngẫu nhiên các vị trí y positive trong box
                        # (,. ,. , 2) Lưu vị trí y được lấy
                        annotations[pos_number * n_samples_pos: (pos_number + 1) * n_samples_pos, 2] = \
                            np.random.randint(t + patch_size[0] // 2,
                                              b - patch_size[0] // 2,
                                              size=(n_samples_pos,))  # get rand y positions in obj

                    except:
                        print(
                            'Please check annotations, object size too small in y axis')
                        continue

                    # Kiểm tra lại tọa độ y xem thỏa mãn không
                    assert np.all(
                        annotations[pos_number * n_samples_pos: (pos_number + 1) * n_samples_pos, 2] - t >= patch_size[
                            0] // 2)  # check y position in obj y min
                    assert np.all(
                        b - annotations[pos_number * n_samples_pos: (pos_number + 1) * n_samples_pos, 2] >= patch_size[
                            0] // 2)  # check y position in obj y max

                    # (.,., 3) lưu khoảng cách từ trung tâm trục x đến các vị trí được chọn ngẫu nhiên của trục x
                    annotations[pos_number * n_samples_pos: (pos_number + 1) * n_samples_pos, 3] = \
                        int(float(l + r) / 2.) - annotations[
                        pos_number * n_samples_pos: (pos_number + 1) * n_samples_pos, 1]  # Distance to centre x

                    # (.,.,, 4) lưu khoảng cách từ trung tâm trục y đến các vị trí được chọn ngẫu nhiên của trục y
                    annotations[pos_number * n_samples_pos: (pos_number + 1) * n_samples_pos, 4] = \
                        int(float(t + b) / 2.) - annotations[
                        pos_number * n_samples_pos: (pos_number + 1) * n_samples_pos, 2]  # Distance to centre y
                except:
                    print('Please check annotations')
                    continue
            else:
                print("annotation not present in csv file")
        pos_number = pos_number + 1

    # pos_offset = 25 * 2 -  số patches positive lấy ra từ ảnh
    pos_offset = n_samples_pos * train_pos_ids

    # %% Negative images.
    neg_number = 0
    # Duyệt từng file ảnh negative
    for filename in glob.glob(neg_image_path):
        images.append(np.array(Image.open(filename)))
        img_name = filename.replace(root_dir + '' + train_image_dir + '/', "")
        assert not images[-1] is None
        # Đánh dấu ID ảnh
        annotations[pos_offset + neg_number * n_samples_neg: pos_offset + (neg_number + 1) * n_samples_neg,
                    0] = neg_number + train_pos_ids
        # Lấy ngẫu nhiên vị trí x cho negative
        annotations[pos_offset + neg_number * n_samples_neg: pos_offset + (neg_number + 1) * n_samples_neg, 1] = np.random.randint(
            patch_size[1] // 2, images[-1].shape[1] - patch_size[1] // 2, size=(n_samples_neg,))
        # Lấy ngẫu nhiên vị trí y cho negative
        annotations[pos_offset + neg_number * n_samples_neg: pos_offset + (neg_number + 1) * n_samples_neg, 2] = np.random.randint(
            patch_size[0] // 2, images[-1].shape[0] - patch_size[0] // 2, size=(n_samples_neg,))
        neg_number = neg_number + 1

    # %% Feature extraction. - Khởi tạo thư viện hough forest
    soil = fertilized.Soil('uint8', 'int16', 'int16',
                           fertilized.Result_Types.hough_map)

    # Chuyển ảnh sang định dạng của openCv để chạy được thư viện Hough forest
    # Ảnh đầu vào phải là định dạng BGR
    cvimages = []
    for im in images:
        if im.ndim == 2:
            cvimages.append(np.ascontiguousarray(skimage.color.gray2rgb(im)))
        else:
            cvimages.append(np.ascontiguousarray(im[:, :, :3]))

    feat_images = None
    if feature_type == 1:
        print('RGB')
        feat_images = [np.repeat(np.ascontiguousarray(np.rollaxis(
            im, 2, 0).astype(np.uint8))[:3, :, :], 5, 0) for im in cvimages]
    if feature_type == 2:
        print('HOG')
        # Extract the Hough forest features. If `full` is set, uses the
        # 32 feature channels used by Juergen Gall in his original publications,
        # else use 15 feature channels as used by Matthias Dantone.
        # The image must be in OpenCV (BGR) channel format!
        feat_images = [soil.extract_hough_forest_features(
            im, full=(n_feature_channels == 32)) for im in cvimages]
        # print("shape", feat_images[0].shape)
        # for i in range(15):
        #    plt.imshow(feat_images[0][i,:,:])
        #    plt.axis('off')
        #    plt.savefig(str(i)+'.png')
        #    plt.show()

#    if feature_type == 3:
#        ## deep features
#        print('Deep')
#        max_abs_scaler = preprocessing.MaxAbsScaler()
#        feat = [vgg_19_conv_feat.getDeepFeatures(im) for im in cvimages]
#        feat_images = []
#        for f in feat:
#            feat_dim = np.zeros((f.shape[2], f.shape[0], f.shape[1]))
#            for ch in range(f.shape[2]):
#                scaled_feat = max_abs_scaler.fit_transform(f[:, :, ch])
#                scaled_feat = scaled_feat * 255
# plt.imshow(scaled_feat)
#                sizes = np.shape(scaled_feat)
#                height = float(sizes[0])
#                width = float(sizes[1])
#
#                fig = plt.figure()
#                fig.set_size_inches(width/height, 1, forward=False)
#                ax = plt.Axes(fig, [0., 0., 1., 1.])
#                ax.set_axis_off()
#                fig.add_axes(ax)
#
#                ax.imshow(scaled_feat)
#                plt.savefig(time.strftime("%Y%m%d-%H%M%S")+'.png', dpi = height)
#                plt.close()
#
# plt.show()
##                plt.savefig(time.strftime("%Y%m%d-%H%M%S")+'.png', dpi = 300, bbox_inches='tight')
#                feat_dim[ch, :, :] = scaled_feat
#            app_feat = feat_dim[:, :, :]
#
#            feat_images.append(np.ascontiguousarray(app_feat).astype(np.uint8))
#    print(feat_images[0].shape)

#    plt.imshow(feat_images[0][:,:,:3])
#    plt.show()
#    plt.imshow(feat_images[0][:,:,2:6])
#    plt.show()
#    plt.imshow(feat_images[0][:,:,5:9])
#    plt.show()
#    plt.imshow(feat_images[0][:,:,8:12])
#    plt.show()
#    plt.imshow(feat_images[0][:,:,11:15])
#    plt.show()

    # Lưu đặc trưng được trích xuất
    # feat_name = 'feat_images.pkl'
    with open(root_dir + '' + feat_name, 'wb') as f:
        pickle.dump(feat_images, f)

    # %% Forest construction.
    # Xây dựng Forest
    random_init = 1
    trees = []  # danh sách cây
    for tree_idx in range(n_trees):
        print('Constructing and training tree %d.' % (tree_idx))
        random_seed = tree_idx * 2 + 1 + random_init * n_trees
        # uses the patch locations to map all feature requests by the trees or forests to the correct positions in the images.
        # Sử dụng các vị trí patch để ánh xạ đến tất cả các đặc trưng yêu cầu bởi cây hoặc rừng tới vị trí đúng trên ảnh
        sman = soil.NoCopyPatchSampleManager(
            feat_images,
            annotations,
            n_samples_pos * train_pos_ids,
            n_feature_channels,
            patch_size[0],
            patch_size[1],
            False)
        # a special dataprovider
        dprov = soil.SubsamplingDataProvider(max_patches_per_node,
                                             sman,
                                             random_seed)
        tree = soil.StandardHoughTree((patch_size[1], patch_size[0], n_feature_channels),
                                      n_thresholds,
                                      n_splits,
                                      max_depth,
                                      (1, min_samples_per_split),
                                      random_seed,
                                      (0., 0.),
                                      patch_annot_luc=False,
                                      allow_redraw=True,
                                      num_threads=n_threads,
                                      entropy_names=['shannon', 'shannon'])
        tree.fit_dprov(dprov, True)
        trees.append(tree)
    forest = soil.CombineTrees(trees)

    # Lưu lại mô hình được huấn luyện
    with open(root_dir+''+forest_name, 'wb') as df:
        pickle.dump(forest, df,  protocol=2)


def train():
    trainHoughForests()
