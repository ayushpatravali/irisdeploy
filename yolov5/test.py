import argparse
import torch
from pathlib import Path
from val import run  # YOLOv5's val.py

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path')
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset.yaml path')
    parser.add_argument('--img', type=int, default=640, help='input image size')
    parser.add_argument('--task', type=str, default='test', help='task type: test')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IoU threshold for NMS')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--device', type=str, default='', help='device: cpu or cuda')
    parser.add_argument('--name', type=str, default='test_results', help='save results to project/name')
    opt = parser.parse_args()
    return opt

def main(opt):
    print("Running custom YOLOv5 test script...")
    run(
        weights=opt.weights,
        data=opt.data,
        imgsz=opt.img,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        batch_size=opt.batch_size,
        device=opt.device,
        task=opt.task,
        name=opt.name,
    )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
