# Config for active learning
max_queried = 2
unlabeled = 'data/unlabeled'
labeled = 'data/labeled'
num_select = 8
info_predict_path = 'activelearning/predict.txt'
mode_active = 'sum' # 'sum or mean

# Config general
project = "/media/thang/New Volume/Active-learning-for-object-detection/"
weight = 'activelearning/yolov5s.pt'
device = '0' # cpu or 0,1,...
name = 'gun'
exist_ok = 1

# Config for train model
config_model = 'models/yolo_gun.yaml'
config_data = 'data/gun.yaml'
batch_size = 1
epochs = 10
adam = 0
project_train = 'runs/train'

# Config for detection
source = 'data/unlabeled' # '0' for webcam
conf_thres = 0.25
iou_thres = 0.45
project_detect = 'runs/detect'
save_conf = 0