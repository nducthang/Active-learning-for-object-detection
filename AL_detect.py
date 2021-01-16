import time
import torch
from models.experimental import attempt_load
from utils.datasets import  LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized
import AL_config as config
import time

def AL_detect(opt):
    # Detect ảnh 
    # File weight model, nguồn dữ liệu detect, Kích cỡ ảnh sử dụng
    weights, source, imgsz = opt.weights, opt.source, opt.img_size

    # khởi tạo
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights=weights, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    result = {}
    # duyệt tất cả các ảnh
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # chuuẩn hóa hình ảnh
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        result[path] = []

        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string
                
                # Lưu thông tin về box vào 1 file
                for *xyxy, conf, cls in reversed(det):
                    with open(config.info_predict_path, 'a') as f:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh)
                        x,y,w,h = xywh
                        data = {"class": cls.item(), "box": [x,y,w,h], "conf": conf.item()}
                        result[path].append(data)
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
    print(f'Done. ({time.time() - t0:.3f}s)')
    return result


