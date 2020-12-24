from activelearning.base_model import BaseModel
from detect import detect
import argparse
import logging
import os
import random
import time
from pathlib import Path
from train import train
from warnings import warn
import numpy as np
import torch.distributed as dist
import torch.utils.data
import yaml
from torch.utils.tensorboard import SummaryWriter
from utils.general import increment_path, fitness, get_latest_run, check_file, check_git_status, print_mutation, set_logging, strip_optimizer
from utils.plots import plot_evolution
from utils.torch_utils import select_device
import activelearning.config as config

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    logger.info(
        "Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


class Yolov5(BaseModel):
    model_type = 'Yolo version 5'

    def train(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default=config.weight, help='initial weights path')
        parser.add_argument('--cfg', type=str, default=config.config_model, help='models/yolov5s.yaml path')
        parser.add_argument('--data', type=str, default=config.config_data, help='data.yaml path')
        parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=config.epochs)
        parser.add_argument('--batch-size', type=int, default=config.batch_size, help='total batch size for all GPUs')
        parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--notest', action='store_true', help='only test final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--device', default=config.device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # using GPU 0
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        # parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        parser.add_argument('--adam', type=int, default=config.adam, help='use torch.optim.Adam() optimizer') # using Adam optimizer
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
        parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
        parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
        parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
        parser.add_argument('--project', default=config.project_train, help='save to project/name')
        parser.add_argument('--name', default=config.name, help='save to project/name')
        # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--exist-ok', type=int, default=config.exist_ok, help='existing project/name ok, do not increment')
        opt = parser.parse_args()

        # Set DDP variables
        opt.total_batch_size = opt.batch_size
        opt.world_size = int(
            os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        opt.global_rank = int(
            os.environ['RANK']) if 'RANK' in os.environ else -1
        set_logging(opt.global_rank)
        if opt.global_rank in [-1, 0]:
            check_git_status()

        # Resume
        if opt.resume:  # resume an interrupted run
            # specified or most recent path
            ckpt = opt.resume if isinstance(
                opt.resume, str) else get_latest_run()
            assert os.path.isfile(
                ckpt), 'ERROR: --resume checkpoint does not exist'
            with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
                opt = argparse.Namespace(
                    **yaml.load(f, Loader=yaml.FullLoader))  # replace
            opt.cfg, opt.weights, opt.resume = '', ckpt, True
            logger.info('Resuming training from %s' % ckpt)
        else:
            # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
            opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(
                opt.cfg), check_file(opt.hyp)  # check files
            assert len(opt.cfg) or len(
                opt.weights), 'either --cfg or --weights must be specified'
            # extend to 2 sizes (train, test)
            opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
            opt.name = 'evolve' if opt.evolve else opt.name
            opt.save_dir = increment_path(Path(
                opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

        # DDP mode
        device = select_device(opt.device, batch_size=opt.batch_size)
        if opt.local_rank != -1:
            assert torch.cuda.device_count() > opt.local_rank
            torch.cuda.set_device(opt.local_rank)
            device = torch.device('cuda', opt.local_rank)
            # distributed backend
            dist.init_process_group(backend='nccl', init_method='env://')
            assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
            opt.batch_size = opt.total_batch_size // opt.world_size

        # Hyperparameters
        with open(opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
            if 'box' not in hyp:
                warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                     (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
                hyp['box'] = hyp.pop('giou')

        # Train
        logger.info(opt)
        if not opt.evolve:
            tb_writer = None  # init loggers
            if opt.global_rank in [-1, 0]:
                logger.info(
                    f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
                tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
            train(hyp, opt, device, tb_writer, wandb)

        # Evolve hyperparameters (optional)
        else:
            # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
            meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                    # final OneCycleLR learning rate (lr0 * lrf)
                    'lrf': (1, 0.01, 1.0),
                    'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                    'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                    # warmup epochs (fractions ok)
                    'warmup_epochs': (1, 0.0, 5.0),
                    # warmup initial momentum
                    'warmup_momentum': (1, 0.0, 0.95),
                    'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                    'box': (1, 0.02, 0.2),  # box loss gain
                    'cls': (1, 0.2, 4.0),  # cls loss gain
                    'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                    'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                    'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                    'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                    'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                    # anchors per output grid (0 to ignore)
                    'anchors': (2, 2.0, 10.0),
                    # focal loss gamma (efficientDet default gamma=1.5)
                    'fl_gamma': (0, 0.0, 2.0),
                    # image HSV-Hue augmentation (fraction)
                    'hsv_h': (1, 0.0, 0.1),
                    # image HSV-Saturation augmentation (fraction)
                    'hsv_s': (1, 0.0, 0.9),
                    # image HSV-Value augmentation (fraction)
                    'hsv_v': (1, 0.0, 0.9),
                    'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                    # image translation (+/- fraction)
                    'translate': (1, 0.0, 0.9),
                    'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                    'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                    # image perspective (+/- fraction), range 0-0.001
                    'perspective': (0, 0.0, 0.001),
                    # image flip up-down (probability)
                    'flipud': (1, 0.0, 1.0),
                    # image flip left-right (probability)
                    'fliplr': (0, 0.0, 1.0),
                    'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                    'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

            assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
            opt.notest, opt.nosave = True, True  # only test/save final epoch
            # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
            yaml_file = Path(opt.save_dir) / \
                'hyp_evolved.yaml'  # save best result here
            if opt.bucket:
                os.system('gsutil cp gs://%s/evolve.txt .' %
                          opt.bucket)  # download evolve.txt if exists

            for _ in range(300):  # generations to evolve
                if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                    # Select parent(s)
                    parent = 'single'  # parent selection method: 'single' or 'weighted'
                    x = np.loadtxt('evolve.txt', ndmin=2)
                    # number of previous results to consider
                    n = min(5, len(x))
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min()  # weights
                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # random selection
                        x = x[random.choices(range(n), weights=w)[
                            0]]  # weighted selection
                    elif parent == 'weighted':
                        x = (x * w.reshape(n, 1)).sum(0) / \
                            w.sum()  # weighted combination

                    # Mutate
                    mp, s = 0.8, 0.2  # mutation probability, sigma
                    npr = np.random
                    npr.seed(int(time.time()))
                    g = np.array([x[0] for x in meta.values()])  # gains 0-1
                    ng = len(meta)
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng)
                             * npr.random() * s + 1).clip(0.3, 3.0)
                    for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                        hyp[k] = float(x[i + 7] * v[i])  # mutate

                # Constrain to limits
                for k, v in meta.items():
                    hyp[k] = max(hyp[k], v[1])  # lower limit
                    hyp[k] = min(hyp[k], v[2])  # upper limit
                    hyp[k] = round(hyp[k], 5)  # significant digits

                # Train mutation
                results = train(hyp.copy(), opt, device, wandb=wandb)

                # Write mutation results
                print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

            # Plot results
            plot_evolution(yaml_file)
            print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
                  f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')

    def detect(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=config.weight, help='model.pt path(s)')
        parser.add_argument('--source', type=str, default=config.source, help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=config.conf_thres, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=config.iou_thres, help='IOU threshold for NMS')
        parser.add_argument('--device', default=config.device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-txt', type=int, default=1, help='save results to *.txt')
        # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-conf', type=int, default=config.save_conf, help='save confidences in --save-txt labels')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=config.project_detect, help='save results to project/name')
        parser.add_argument('--name', default=config.name, help='save results to project/name')
        # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--exist-ok', type=int, default=config.exist_ok, help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        print(opt)

        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    detect(opt)
                    strip_optimizer(opt.weights)
            else:
                detect(opt)

if __name__ == '__main__':
    model = Yolov5()
    model.train()
