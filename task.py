import argparse
from client import Client

import my_test
from utils.datasets import create_dataloader, my_create_dataloader
import yaml
from utils.loss import ComputeLoss
from models.yolo import Model
import torch

import numpy as np
import random
import os
import torch.backends.cudnn as cudnn#

import warnings
warnings.filterwarnings('ignore')

# 新增设置固定随机种子的函数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class Tester:
    def __init__(self, val_path='data/dogs/client_0/test_image.txt',
                 cfg='models/yolov3.yaml',
                 batch_size=8, conf_thres=0.001):
        self.val_path = val_path
        self.batch_size = batch_size
        hyp_path = 'data/hyp.scratch.yaml'
        device = torch.device('cuda')
        nc = 1
        self.conf_thres=conf_thres
        with open(hyp_path) as f:
            hyp = yaml.safe_load(f)  # load hyps
        self.model = Model(cfg, ch=3, nc=nc).to(device)  # create无初始参数
        self.testloader = my_create_dataloader(val_path, 640, batch_size, 32, None, hyp=hyp, workers=0)[0]

    def test(self, state_dict):
        self.model.load_state_dict(state_dict)
        results, maps, times = my_test.my_test(val_path=self.val_path,
                                               batch_size=self.batch_size,
                                               imgsz=640,
                                               model=self.model,
                                               dataloader=self.testloader,
                                               save_dir='.',
                                               conf_thres=self.conf_thres)

    def load_test(self, pt_path):
        self.model.load_state_dict(torch.load(pt_path))
        results, maps, times = my_test.my_test(val_path=self.val_path,
                                               batch_size=self.batch_size,
                                               imgsz=640,
                                               model=self.model,
                                               dataloader=self.testloader,
                                               save_dir='.',
                                               conf_thres=self.conf_thres)


def test_vehicle():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val_path', type=str, default='data/vehicle_4_600/base/test_image.txt')
    parser.add_argument('--weights', type=str, default='./last.pt')
    parser.add_argument('--conf_thres', type=float, default=0.001)
    parser.add_argument('--iou_thres', type=float, default=0.6)

    opt = parser.parse_args()

    tester = Tester(val_path=opt.val_path,
                    cfg='models/yolov3.yaml',
                    batch_size=opt.batch_size, conf_thres=opt.conf_thres)
    tester.load_test(opt.weights)


def task_vehicle_4_600_all():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all_vehicle_4_600')#vehicle
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('--epochs', type=int, default=530)#
    parser.add_argument('--batch-size', type=int, default=32)#64
    parser.add_argument('--weights', type=str, default='')#随机初始化！！！
    parser.add_argument('--conf_thres', type=float, default=0.001)
    parser.add_argument('--iou_thres', type=float, default=0.6)
    opt = parser.parse_args()

    # write_vehicle_data()

    c = Client(task=opt.task, name=opt.name)
    c.set_all(epochs=opt.epochs,
              batch_size=opt.batch_size,
              weights=opt.weights,
              lr=0.01)
    c.workers = 8
    c.use_decay = True
    c.decay_ratio = 0.997
    c.train()

    c.use_decay = False
    c.lr = 0.002
    c.epochs = 100
    c.train()


def task_vehicle_base():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vehicle_4_600')#vehicle
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('--epochs', type=int, default=530)#
    parser.add_argument('--batch-size', type=int, default=32)#64
    parser.add_argument('--weights', type=str, default='')#随机初始化！！！
    parser.add_argument('--conf_thres', type=float, default=0.001)
    parser.add_argument('--iou_thres', type=float, default=0.6)
    opt = parser.parse_args()

    # write_vehicle_data()

    c = Client(task=opt.task, name=opt.name)
    c.set_all(epochs=opt.epochs,
              batch_size=opt.batch_size,
              weights=opt.weights,
              lr=0.01)
    c.workers = 8
    c.use_decay = True
    c.decay_ratio = 0.997
    c.train()

    c.use_decay = False
    c.lr = 0.002
    c.epochs = 100
    c.train()


if __name__ == '__main__':
    setup_seed(2021)

    # 预训练
    task_vehicle_base()

    # 全量数据训练 天花板
    task_vehicle_4_600_all()

    # 测试
    test_vehicle()

