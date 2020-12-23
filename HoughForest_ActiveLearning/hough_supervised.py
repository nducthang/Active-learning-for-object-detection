# import hough_detect

from matplotlib import pyplot as plt

from hough_preferences import dataset_flag
import glob
import shutil
import os
import sys

from hough_preferences import all_image_dir, root_dir, extension, train_image_dir, test_image_dir, eval_image_dir,dataset_flag
import hough_train
import hough_eval
from sklearn.cross_validation import train_test_split
import warnings
warnings.filterwarnings("ignore")

def train_detec():
    epoch = 10
    global mAP
    for i in range(epoch):
        hough_train.train()
        print("***********Training done***********")
        mAP.append(hough_eval.runEvaluate())
        print("***********Evaluation done***********")
        print(mAP)
        # print(tao_l)
        # print(ac_l)
if __name__ == "__main__":
    # hough_preferences.dataset_flag = int(sys.argv[1])
    # print("dataset_flag", hough_preferences.dataset_flag)
    images = []
    image_path = os.path.join(root_dir, all_image_dir, '*' + extension)
    print(image_path)
    for filename in glob.glob(image_path):
        img_name = filename.replace(root_dir + '' + all_image_dir + '/', "")
        images.append(img_name)
    x, x_test, _, _ = train_test_split(images, images, test_size=0.20, train_size=0.80)
    print(len(x))
    print(len(x_test))
    
    
    for img_name in x_test:
        src_image_path = os.path.join(root_dir, all_image_dir,img_name)
        dst_img_path = os.path.join(root_dir,eval_image_dir,img_name)
        shutil.copy(src_image_path, dst_img_path)
    
    for idx,img_name in enumerate(x):
        train_flag = '_pos'
        src_image_path = os.path.join(root_dir, all_image_dir,img_name)
        dst_img_path = os.path.join(root_dir,train_image_dir,img_name[:-4]+train_flag+extension)
        shutil.copy(src_image_path, dst_img_path)


#    for img_name in glob.glob(root_dir+'/'+test_image_dir+'/'+'*'+extension):
#        dst_dir = train_image_dir
#        train_flag = '_pos'
#        src_image_path = img_name
#        dst_img_path = os.path.join(root_dir,dst_dir,img_name[len(root_dir)+len(dst_dir):-4]+train_flag+extension)
#        shutil.copy(src_image_path, dst_img_path)
    global mAP, tao_l,ac_l
    mAP=[]
    tao_l =[]
    ac_l =[]
    train_detec()
    print("mAP",mAP)
    plt.plot(mAP,"b-")
    plt.title("mAP")
    plt.plot()

