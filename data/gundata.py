import glob
import math
import os
import numpy as np

files = []
for ext in ["*.png", "*.jpeg", "*.jpg"]:
    image_files = glob.glob(os.path.join("/media/thang/New Volume/Active-learning-for-object-detection/data/gun/", ext))
    files += image_files

nb_val = math.floor(len(files)*0.2)
# Lấy ngẫu nhiên các số từ 0 đến len(files)-1
# Lấy nb_val phần tử
rand_idx = np.random.randint(0, len(files), nb_val)

# Tao file train.txt
with open("./data/train.txt", "w") as f:
    for idx in np.arange(len(files)):
        if (os.path.exists(files[idx][:-3]+"txt")):
            f.write(files[idx]+"\n")

# Tao file val.txt
with open("./data/val.txt", "w") as f:
    for idx in np.arange(len(files)):
        if (idx in rand_idx) and (os.path.exists(files[idx][:-3]+"txt")):
            f.write(files[idx]+"\n")