#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:17:58 2018

@author: priyanka
"""
import hough_detect
import hough_train
import hough_eval
from hough_preferences import all_image_dir,root_dir,extension,train_image_dir,test_image_dir,eval_image_dir,interactive,extension,train_annot_dir
import glob
import shutil
from PIL import Image
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os
import numpy as np
from flask import Flask, render_template, jsonify,request,send_file,make_response
import os
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO,BytesIO
import random
import time
import csv
import io

PEOPLE_FOLDER = os.path.join('static', 'dataset')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
csvfile = open(root_dir+''+train_annot_dir+'/via_region_data.csv','w')
writer = csv.writer(csvfile)
header = ["#filename","file_size","	file_attributes","region_count","region_id","region_shape_attributes","region_attributes"]
writer.writerow(header)
csvfile.close()
mAP = [0,0]
global epoch
epoch =0

@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def show_index():
    print("index")
    return render_template("via_active_learning.html")


@app.route('/start', methods=['GET'])
def start():
    full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], test_img1)
    full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], test_img2)
    train_full_filename = [full_filename1,full_filename2]
    return jsonify({ 'train_full_filename': train_full_filename})

@app.route('/train', methods=['POST'])
def train():
    global epoch
    print('downloading annotation')
    csvfile = open(root_dir+''+train_annot_dir+'/via_region_data.csv', 'a')
    writer = csv.writer(csvfile)
    csvEntries = request.form['csvFile']
    for idx, csvEntry in enumerate(csvEntries.split('\n')):
        if idx != 0 and csvEntry.split(',') != '{}':
            entry = csvEntry.split(',')
            writeList = [entry[0][:-4] + '_pos' + extension, entry[1], entry[2].strip('"'), entry[3], entry[4],
                         entry[5][1:].replace('""', '"') + ',' + entry[6].replace('""', '"') + ',' + entry[7].replace(
                             '""', '"') + ',' + entry[8].replace('""', '"') + ',' + entry[9][:-1].replace('""', '"'),
                         entry[10].strip('"')]
            # print(writeList)
            writer.writerow(writeList)
    print('csv file written')
    csvfile.close()
    hough_train.train()
    mAP.append(hough_eval.runEvaluate())
    epoch = epoch + 1
    return ('', 204)

@app.route("/test", methods=['GET'])
def test():
    global epoch
    #    code to active learning, returns name of image with highest annotation cost
    opt_tao, max_annot_cost,annot_file_name = hough_detect.detect(epoch)
    # epoch = epoch+1
    print("filename",annot_file_name)
    filename = annot_file_name.split('/')[9]
    
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print("filename", full_filename)
    full_filename = [full_filename]
    #    urllib.request.urlretrieve("http://127.0.0.1:5000/3240f7fe-dd7f-46e1-9179-d63d373f002e")
    return jsonify({ 'full_filename': full_filename})

@app.route("/addPlot",methods=['GET'])
def addPlot():
    fig = plt.figure(figsize=(50, 2))
    plt.grid()
    plt.xlabel("iterarions")
    plt.ylabel("mAP")
    ax = fig.add_subplot(111)
    fs_map = np.mean([0.38071525577022231, 0.45602836879153175, 0.51538973539924582, 0.40478723403977762, 0.46076023883397021])

    ax.plot(mAP,'ro-',label="active learning mAP")
    pl = [[0.15100906977892917, 0.17199103423212836, 0.2612462006066375, 0.28867448878153579, 0.48427281937571864, 0.37634905951073583], [0.31117442010523333, 0.61384843833496505, 0.23539886010039983, 0.45705528249907634, 0.28382642129656992, 0.2662856683050399], [0.42395964508467598, 0.22487262661700255, 0.21762415349571556, 0.29270213715411642, 0.31061131667892244, 0.31633715779422428], [0.33156858275285167, 0.19878939027760315, 0.17073201620952255, 0.33226235059308551, 0.19013069133113314, 0.37324581258256173], [0.3443003398089895, 0.25971845106772062, 0.20460168233425713, 0.30602082390091601, 0.29614822234345034, 0.34320196647152679]]

    for idx, _ in enumerate(pl):
        pl[idx].insert(0, 0)
        pl[idx].insert(1, 0)
    pl_mean = np.mean(pl, axis=0)
    pl_std = np.std(pl, axis=0)
    ax.plot(range(7), [fs_map] * (len(pl) + 2), 'g-', label='fully supervised mAP')
    ax.fill_between(range(len(pl[0])), pl_mean - pl_std, pl_mean + pl_std, alpha=0.1, color="b")
    ax.plot(range(len(pl[0])), pl_mean, 'o-', color="b", label="passive learning mAP")
    img = io.BytesIO()
    response=make_response(img.getvalue())
    response.headers['Content-Type'] = 'image/png'
#    img = Image.open(response)
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    # remove earlier copied files eval
    images = []
    num_init_images = 2
    files = glob.glob(os.path.join(root_dir, eval_image_dir + '/*'))
    for f in files:
        os.remove(f)

        # remove earlier copied files test
    files = glob.glob(os.path.join(root_dir, test_image_dir + '/*'))
    for f in files:
        os.remove(f)

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

    init_images = []
    for j in range(num_init_images):
        subdirs = os.listdir(root_dir + test_image_dir)
        print(subdirs)
        rand_img = random.choice(subdirs)
        init_images.append(rand_img)

    all_init_images = init_images

    for img in glob.glob(root_dir + train_image_dir + "/*_pos" + extension):
        src = img
        skip_len = len(root_dir) + len(train_image_dir)
        dst = root_dir + test_image_dir + img[skip_len:]
        shutil.copy(img, dst.replace('_pos' + extension, extension))
        os.remove(img)

    for im in init_images:
        src = root_dir + test_image_dir + '/' + im
        dst = root_dir + train_image_dir + '/' + im[:-4] + '_pos' + extension
        shutil.copy(src, dst)
        os.remove(src)

    global test_img1, test_img2
    test_img1 = init_images[0]
    test_img2 = init_images[1]

#    app.run(host='0.0.0.0')
    app.run()




