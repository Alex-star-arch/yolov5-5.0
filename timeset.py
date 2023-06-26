import argparse
import datetime
import sys
import threading
import time
from pathlib import Path
import schedule
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import pymysql
import os
import shutil
def getStreams(offLineList):
    connection = pymysql.connect(host="localhost", user="root", password="2277", database="video",
                    charset='utf8', use_unicode=True, max_allowed_packet=24 * 1024 * 1024 * 1024)
    cursor = connection.cursor()
    try:
        with open("streams.txt", encoding="utf-8", mode="a") as file:
            file.truncate(0)
        sql = """select id,stream from streams"""
        cursor.execute(sql)  # 执行SQL语句
        streams = cursor.fetchall()
        streams = list(streams)
        streamlist = []
        for stream in streams:
            if str(stream[0]) in offLineList:
                continue
            streamUrl = ''.join(stream[1])
            streamlist.append(streamUrl)
        with open("streams.txt", encoding="utf-8",mode="a") as file:
            for (i,stream)in enumerate(streamlist):
                if i!=0:
                    file.write("\n" + stream)
                else:
                    file.write(stream)
        return streamlist
    finally:
        connection.close()

def getStreamsId():
    connection = pymysql.connect(host="localhost", user="root", password="2277", database="video",
                    charset='utf8', use_unicode=True, max_allowed_packet=24 * 1024 * 1024 * 1024)
    cursor = connection.cursor()
    try:
        # sql = """select id from streams"""
        # cursor.execute(sql)  # 执行SQL语句
        # res = cursor.fetchall()
        # ids = list(res)
        idList = []
        # for streamid in ids:
        #     # 查询streams表中的stream列
        cursor.execute("SELECT stream FROM streams")
        streams = cursor.fetchall()
        with open('streams.txt', 'r') as file:
            lines = file.readlines()
        for line in lines:
            # 移除行末尾的换行符
            line = line.strip()
            # 检查当前行的内容是否在streams中
            if (line,) in streams:
                # 执行匹配时的操作，例如存入数组
                cursor.execute("SELECT id FROM streams WHERE stream = %s", line)
                matching_id = cursor.fetchone()
                idList.append(matching_id[0])

            # if str(stream[0]) in offLineList:
            #     continue
            # streamid = str(streamid).strip('(,)')
            # idList.append(streamid)
        return idList
    finally:
        connection.close()

def detect():
    #getStreams()
    Idlist = getStreamsId()
    print(Idlist)
    # 在此处编写YOLOv5检测算法的代码
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp/weights/best.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='C:/Users/Administrator/Desktop/mmm.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='streams.txt', help='source')  # 单网络多线程 实时检测
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')#置信度阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')#做nms的iou阈值
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')#是否展示预测之后的图片/视频，默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')#是否将预测的框坐标以txt文件形式保存，默认False
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')#设置只保留某一部分类别，形如0或者0 2 3
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')#进行nms是否也去除不同类别之间的框，默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')#推理的时候进行多尺度，翻转等操作(TTA)推理
    parser.add_argument('--update', action='store_true', help='update all models')#如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    parser.add_argument('--project', default='D:/weed/system/Video-Viewer_backend/app/static/images', help='save results to project/name')
    parser.add_argument('--name', default='picture', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save_to_system',
                        default='D:/weed/system/Video-Viewer_backend/app/static/images/picture',
                        help='save results to system')
    opt = parser.parse_args()

    ## 获取输出文件夹，输入源，权重，参数等参数
    #save_to_system = opt.save_to_system
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    #save_to_system = Path(save_to_system)
    # Initialize
    # 获取设备
    set_logging()
    device = select_device(opt.device)
    # 如果设备为gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # 加载Float32模型，确保用户设定的输入图片分辨率能整除32(如不能则调整为能整除并返回)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # 设置第二次分类，默认不使用
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        # 如果检测视频的时候想显示出来，可以在这里加一行view_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        view_img = False
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # 设置画框的颜色
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors = [(255, 0, 0) for _ in names]
    # Run inference
    # 进行一次前向推理,测试程序是否正常
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    num = 0;
    """
    path 图片/视频路径
    img 进行resize+pad之后的图片
    img0 原size图片
    cap 当读取图片时为None，读取视频时为视频源
    """

    start_time = time.time()  # 记录时间，准备执行10秒后结束检测
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        # 进行nms
        """
        pred:前向传播的输出
        conf_thres:置信度阈值
        iou_thres:iou阈值
        classes:是否只保留特定的类别
        agnostic:进行nms是否也去除不同类别之间的框
        经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
        pred是一个列表list[torch.tensor]，长度为batch_size
        每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
        """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # 设置保存图片/视频的路径
            save_path = str(save_dir / p.name)  # img.jpg
            # 设置保存框坐标txt文件的路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # 设置打印信息(图片长宽)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                # 此时坐标格式为xyxy
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                res = ""
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    res += f"{n}{names[int(c)]}{'s' * (n > 1)},"
                # Write results

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                # **********************************************
                a0 = (int(xyxy[0].item()) + int(xyxy[2].item())) / 2
                if a0 != 0:
                    cv2.imwrite(save_path + f'{num}+{Idlist[i]}+{res}.jpg', im0)
                    num = num + 1
                else:
                    im1 = cv2.imread('no.jpg', 1)
                    cv2.imwrite(save_path + f'{num}+{Idlist[i]}+{res}.jpg', im1)
                    num += 1
                # *************************************************
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        # 结束检测（设定5s结束检测）
        if time.time() - start_time >= 2:
            break
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    print("YOLOv5检测算法已执行完毕")

# 设定每天固定时间执行检测算法的函数
def set_detect_time(hour, minute):
    schedule.clear()
    # 获取当前日期时间
    now = datetime.datetime.now()
    # 构造目标日期时间对象
    target = datetime.datetime(now.year, now.month, now.day, hour, minute)
    # 如果目标日期时间已经过去，则加一天
    if target < now:
        target += datetime.timedelta(days=1)
    # 使用date触发器安排任务，在目标日期时间执行一次检测函数
    schedule.every().day.at(target.strftime("%H:%M")).do(detect).tag("detect")
    #schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(detect)

    # 设定默认执行时间为每天的12:00
detect_hour = 12
detect_minute = 00
set_detect_time(detect_hour, detect_minute)

    # 启动定时器
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)