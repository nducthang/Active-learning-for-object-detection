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

from AL_yolov5 import Yolov5
import activelearning.select_function as select
import activelearning.config as config
import glob
import os
from shutil import copyfile, move
import io
import copy

class ActiveLearning(object):
    def __init__(self, model, select_function):
        self.model = model
        self.select_function = select_function
        self.num_select = config.num_select

    def run(self):
        self.queried = 0
        # chưa thỏa mãn điều kiện dừng thì tiếp tục lặp lại active learning
        while self.queried < config.max_queried:
            # Xoá file dự đoán cũ
            if os.path.exists(config.info_predict_path):
                os.remove(config.info_predict_path)
            # Dự đoán các ảnh trong tập unlabeled
            self.model.detect()
            # Kết quả sau khi dự đoán được lưu ở config.info_predict_path
            # Tổng hợp kết quả
            probas = {str(file.split('/')[-1]): 0.0  for file in glob.glob(config.source + '/*')}
            num_object = probas.copy()

            with open(config.info_predict_path, 'r') as f:
                for line in f.readlines():
                    *cxywh, prob, file_name = line.split(',')
                    file_name = file_name[:-1]
                    probas[file_name] += 1.0 - float(prob)
                    num_object[file_name] += 1
            
            if config.mode_active == 'mean':
                for key, value in probas.items():
                    probas[key] = value/num_object[key]
            
            # Chọn ra k samples
            if len(probas) >= self.num_select:
                U_best = self.select_function.select(self.num_select, probas)
            
                # Gán nhãn cho U_best
                """ GIẢ SỬ ĐOẠN NÀY LÀ NGƯỜI GÁN """
                # Tạo ra các file label cho U_best vào thư mục labeled
                for f in U_best:
                    # Chuyển file ảnh vào thự mục labeled
                    move(os.path.join(config.unlabeled, f), os.path.join(config.labeled, f))
                    # Tạo file nhãn vào thư mục labeled
                    file_name = '.'.join(f.split('.')[:-1])
                    # print(f)
                    # source = open(os.path.join('data/gun/',f),"r",encoding='utf-8').readlines()
                    # dect = open(os.path.join(config.labeled, file_name + '.txt', "w", encoding='utf-8').write(copy.copy(source)))
                    copyfile(os.path.join('data/gun/', file_name+'.txt'), os.path.join(config.labeled, file_name + '.txt'))

                # Cập nhật file train.txt
                with open('data/train.txt', 'w') as f:
                    for fn in U_best:
                        f.write(config.project + config.labeled + "/" + fn + "\n")
                
                # Train model
                self.model.train()

                # Xoá file weight cũ
                if os.path.exists(config.weight):
                    os.remove(config.weight)

                # Cập nhật weight mới
                copyfile(os.path.join(config.project_train, config.name, 'weights', 'best.pt'), config.weight)
            else:
                print("Số lượng file chưa gán nhãn không đủ {} files".format(self.num_select))
                break




if __name__ == '__main__':
    bot = ActiveLearning(model=Yolov5(), select_function=select.RandomSelect())
    bot.run()
    # model = Yolov5()
    # model.train()
