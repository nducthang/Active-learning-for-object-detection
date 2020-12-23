
import hough_preferences
import glob
import shutil

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random

# import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

import time
import sys

from hough_preferences import all_image_dir, root_dir, extension, train_image_dir, test_image_dir, eval_image_dir, interactive,experiments,exp_dir
import hough_detect
import hough_train
import hough_eval

#experiments = 10
num_init_images = 2
all_init_images=[]

import warnings
warnings.filterwarnings("ignore")


def train_detec_active():
    # epoch = 1
    annot_cost = 100
    global mAP, tao_l, ac_l
    al_mAP = []
    tao_l = []
    ac_l = []
    hough_train.train()
    al_mAP.append(hough_eval.runEvaluate())
    i=0
    image_path = os.path.join(root_dir, test_image_dir, '*' + extension)
    max_test_images = len(glob.glob(image_path))

    while annot_cost > 0 and max_test_images > 0 and i < 16:
        # print('max_test_images', max_test_images)
        files = glob.glob(os.path.join(root_dir, exp_dir + '/_tp*'))
        for f in files:
            os.remove(f)
        files = glob.glob(os.path.join(root_dir, exp_dir + '/_fn'))
        for f in files:
            os.remove(f)
        print('')
        print(image_path)
        print("*********** Detection ***********")
        detect_start_time = time.time()
        opt_tao, annot_cost,_ = hough_detect.detect(i)
        print("--- %s seconds for detection ---" % (time.time() - detect_start_time))
        tao_l.append(opt_tao)
        ac_l.append(annot_cost)
        print('')
        print("*********** Training ***********")
        train_start_time = time.time() 
        hough_train.train()
        print("--- %s seconds for training ---" % (time.time() - train_start_time))
        print('')
        print("*********** Evaluation ***********")
        eval_start_time = time.time()
        al_mAP.append(hough_eval.runEvaluate())
        print("--- %s seconds for evaluation ---" % (time.time() - eval_start_time))
        print("mAP ",al_mAP)
        print("tao ",tao_l)
        print("annotation cost ",ac_l)
        i=i+1
        image_path = os.path.join(root_dir, test_image_dir, '*' + extension)
        max_test_images = len(glob.glob(image_path))


#    if fpfn:
#        plt.show()
    print("fs mAP", fs_mAP)
    print("Active learning mAP", al_mAP_ll)
    print("passive learning mAP", pl_mAP_ll)
    return al_mAP,tao_l,ac_l

def train_detec_passive(epoch):
    global pl_mAP
    pl_mAP = []
    for i in range(epoch):
        print(root_dir)
        print('Passive learning, epoch', i, '***********')
        # print("epoch",i)
        print('')
        print("*********** Training ***********")
        hough_train.train()
        print('')
        print("*********** Evaluation ***********")
        pl_mAP.append(hough_eval.runEvaluate())

        train_flag = '_pos'
        filename = random.choice(os.listdir(os.path.join(root_dir, test_image_dir)))
        src_image_path = os.path.join(root_dir, test_image_dir, filename)
        dst_img_path = os.path.join(root_dir, train_image_dir, filename[:-4] + train_flag + extension)
        shutil.copy(src_image_path, dst_img_path)
        print("fs mAP", fs_mAP)
        print("Active learning mAP", al_mAP_ll)
        print("passive learning mAP", pl_mAP_ll)
    return pl_mAP

def train_detec_fs():
    epoch = experiments
    global fs_mAP
    fs_mAP =[]
    # hough_train.train()
    # _ = hough_eval.runEvaluate()
    print(root_dir)
    for i in range(epoch):
        print('***********Fully supervised learning, Experiment number', i, '***********')
        # opt_tao, max_annot_cost = hough_detect.detect(i)
        # tao_l.append(opt_tao)
        # ac_l.append(max_annot_cost)
        hough_train.train()
        print("***********Training done***********")
        fs_mAP.append(hough_eval.runEvaluate())
        print("***********Evaluation done***********")
        #print(mAP)
        # print(tao_l)
        # print(ac_l)
        print("fs mAP", fs_mAP)
        print("Active learning mAP", al_mAP_ll)
        print("passive learning mAP", pl_mAP_ll)

    return fs_mAP

if __name__ == "__main__":
    # hough_preferences.dataset_flag = int(sys.argv[1])
    # from hough_preferences import dataset_flag
    # print(dataset_flag)

    global mAP_ll, tao_ll,ac_ll,fs_mAP,fs_ll
    al_mAP_ll = []
    pl_mAP_ll = []
    tao_ll = []
    ac_ll = []
    images = []
    fs_mAP =[]
    fs_ll = []
    dataset = sys.argv

    # remove earlier copied files eval
    files = glob.glob(os.path.join(root_dir, eval_image_dir + '/*'))
    for f in files:
        os.remove(f)

    # remove earlier copied files test
    files = glob.glob(os.path.join(root_dir, test_image_dir + '/*'))
    for f in files:
        os.remove(f)

    # remove earlier copied files neg train
    files = glob.glob(os.path.join(root_dir, train_image_dir + '/*_auto_pos*'))
    for f in files:
        os.remove(f)
    files = glob.glob(os.path.join(root_dir, train_image_dir + '/*_auto_neg*'))
    for f in files:
        os.remove(f)
    files = glob.glob(os.path.join(root_dir, train_image_dir + '/*_pos*'))
    for f in files:
        os.remove(f)

    # Divide images into train and validation set
    image_path = os.path.join(root_dir, all_image_dir, '*' + extension)
    print(image_path)
    for filename in glob.glob(image_path):
        img_name = filename.replace(root_dir + '' + all_image_dir + '/', "")
        images.append(img_name)
    x, x_test, _, _ = train_test_split(images, images, test_size=0.20, train_size=0.80)
    print(len(x))
    print(len(x_test))


    for idx,img_name in enumerate(x):
        train_flag = ''
        src_image_path = os.path.join(root_dir, all_image_dir,img_name)
        dst_img_path = os.path.join(root_dir,test_image_dir,img_name[:-4]+train_flag+extension)
        shutil.copy(src_image_path, dst_img_path)


    for img_name in x_test:
        src_image_path = os.path.join(root_dir, all_image_dir,img_name)
        dst_img_path = os.path.join(root_dir,eval_image_dir,img_name)
        shutil.copy(src_image_path, dst_img_path)

    # list of random images for training
    for i in range(experiments):
        init_images = []
        for j in range(num_init_images):
            subdirs = os.listdir(root_dir + test_image_dir)
            rand_img = random.choice(subdirs)
            init_images.append(rand_img)
        all_init_images.append(init_images)

    # randomly copy two images for training
    # Active learning
    for idx in range(len(all_init_images)):
        print('')
        print('')
        print('***********Active learning, Experiment number',idx,'***********' )
        for img in glob.glob(root_dir + train_image_dir + "/*_pos" + extension):
            src = img
            skip_len = len(root_dir) + len(train_image_dir)
            dst = root_dir + test_image_dir + img[skip_len:]
            shutil.copy(img, dst.replace('_pos' + extension, extension))
            os.remove(img)

        init_images = all_init_images[idx]
        for im in init_images:
            src = root_dir + test_image_dir + '/' + im
            dst = root_dir + train_image_dir + '/' + im[:-4] + '_pos' + extension
            shutil.copy(src, dst)
            os.remove(src)
        mAP, tao_l, ac_l = train_detec_active()
        print("tao_l",tao_l)
        print("ac_l",ac_l)
        # remove earlier copied files neg train
#        files = glob.glob(os.path.join(root_dir, train_image_dir + '/*_auto_neg*'))
#        for f in files:
#            os.remove(f)
#        files = glob.glob(os.path.join(root_dir, train_image_dir + '/*_auto_pos*'))
#        for f in files:
#            os.remove(f)
        al_mAP_ll.append(mAP)
        tao_ll.append(tao_l)
        ac_ll.append(ac_l)
        print(al_mAP_ll)

    epoch = np.mean([len(el) for el in al_mAP_ll])
    print('mean mAP size',epoch)
    epoch = int(epoch)
    # randomly copy two images for training
    # Passive learning
    for idx in range(len(all_init_images)):
        print('')
        print('')
        print('Passive learning, Experiment number',idx,'***********'  )
        for img in glob.glob(root_dir + train_image_dir + "/*_pos" + extension):
            src = img
            skip_len = len(root_dir) + len(train_image_dir)
            dst = root_dir + test_image_dir + img[skip_len:]
            shutil.copy(img, dst.replace('_pos' + extension, extension))
            os.remove(img)

        init_images = all_init_images[idx]
        for im in init_images:
            src = root_dir + test_image_dir + '/' + im
            dst = root_dir + train_image_dir + '/' + im[:-4] + '_pos' + extension
            shutil.copy(src, dst)
            os.remove(src)
        mAP = train_detec_passive(epoch)
        pl_mAP_ll.append(mAP)
        print(pl_mAP_ll)

    print("Active learning mAP", al_mAP_ll)
    print("passive learning mAP",pl_mAP_ll)

    for img_name in glob.glob(os.path.join(root_dir,test_image_dir,'*'+extension)):
        dst_dir = train_image_dir
        train_flag = '_pos'
        src_image_path = img_name
        dst_img_path = os.path.join(root_dir,dst_dir,img_name[len(root_dir)+len(dst_dir):-4]+train_flag+extension)
        shutil.copy(src_image_path, dst_img_path)

    fs_ll.append(train_detec_fs())
    print("fs mAP",fs_ll)

#
#
