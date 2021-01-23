"""
THUẬT TOÁN ACTIVE LEARNING
Require:
    - Tập các mẫu đã gán nhãn L
    - Tạp các mẫu chưa gán nhãn U
    - Model khởi tạo f0
    - Active learning metric v
Algorithm:
1. Chia U thành các batches
2. f <- f0
3. Nếu U vẫn trống hoặc chưa đạt điều kiện dừng thì:
    - Tính scores cho tất cả batches của U sử dụng f
    - U_best <- Batches điểm cao nhất trong U theo v
    - Y_best <- Gán nhãn cho U_best (người)
    - Train f sử dụng L và (U_best, Y_best)
    - U = U- U_best
    - L = L + (U_best, y_best)

"""

from torch.cuda.memory import reset_accumulated_memory_stats
from AL_yolov5 import Yolov5
import AL_config as config
import glob
import os
from shutil import copyfile, move
import io
import copy
import random

def RandomSelect(num_select, result):
    return random.sample(result.keys(), num_select)

def UncertaintySamplingBinary(num_select, result, typ):
    """
    result = 
        {"<link ảnh>": 
            [
                {"class": cls.item(), "box": [x,y,w,h], "conf": conf.item(),
                ...
            ],
        ...
        }
    """
    probas = {}
    if typ == 'sum':
        for item, lst_dic in result.items():
            conf = 0
            for dic in lst_dic:
                conf += (1.0 - dic["conf"])
            probas[item] = conf
    elif typ == 'avg':
        for item, lst_dic in result.items():
            conf = 0
            for dic in lst_dic:
                conf += (1.0 - dic["conf"])
            probas[item] = conf/len(lst_dic)
    elif typ == 'max':
        for item, lst_dic in result.items():
            conf = 0
            for dic in lst_dic:
                conf = max(conf, 1.0 - dic["conf"])
            probas[item] = conf
    return sorted(probas, key=probas.get, reverse=True)[:num_select]


class ActiveLearning(object):
    def __init__(self, model):
        self.model = model
        self.num_select = config.num_select
        self.type = 'sum' # 'avg' , 'max', 'sum'

    def run(self):
        # số truy vấn
        queried = 33
        ep = 66
        # nếu chưa đủ số truy vấn thì tiếp tục truy vấn tiếp
        while queried < config.max_queried:
            print("TRUY VẤN THỨ: ", queried)
            # Dự đoán các ảnh trong tập unlabeled
            result = self.model.detect()

            # Chon ra k ảnh có score cao nhất
            # Sử dụng lấy mẫu không chắc chắn
            if len(result) >= self.num_select:
                U_best = RandomSelect(self.num_select, result)
                # U_best = UncertaintySamplingBinary(self.num_select, result, 'sum')
                print(U_best)
                
                # Gán nhãn cho các file trong samples (Người tương tác)

                # Duyệt tất cả các file được chọn
                for f in U_best:
                    # Chuyển file ảnh vào thự mục labeled
                    move(f, f.replace("unlabeled", "labeled"))
                    # Tạo file nhãn vào thư mục labeled
                    type_file = f.split('.')[-1]
                    copyfile(f.replace("unlabeled","gun").replace(type_file, 'txt'), f.replace("unlabeled", "labeled").replace(type_file, 'txt'))

                # thêm danh sách file đã gán nhãn vào dữ liệu train
                with open('data/train.txt', "a") as f:
                    for file_name in U_best:
                        f.write(file_name.replace("unlabeled","labeled") + '\n')

                # Train model
                self.model.train(ep)

                ####################### LOADING ########################
                # Xoá file weight cũ
                if os.path.exists(config.weight):
                    os.remove(config.weight)

                # Cập nhật weight mới
                copyfile(os.path.join(config.project_train, config.name, 'weights', 'best.pt'), config.weight)

                queried+=1
                ep += config.epochs
            else:
                print("Số lượng file chưa gán nhãn không đủ {} files".format(self.num_select))
                break




if __name__ == '__main__':
    bot = ActiveLearning(model=Yolov5())
    bot.run()