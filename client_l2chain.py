import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# import test  # import test.py to get mAP after each epoch
import my_test  #
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader, my_create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

from split_dog_data_few import write_data
from split_human_data import write_data as write_human_data

# seed   seed的设置方式需要改一下
init_seeds(1)

logger = logging.getLogger(__name__)

DOG_CLASSNAMES = ['dog']
ONE_LIVING_CLASSNAMES = ['one_living']
LIVING_CLASSNAMES = ['dog', 'turtle', 'lizard', 'snake', 'spider', 'grouse', 'parrot',
                     'crab', 'salamander', 'wolf', 'fox', 'cat', 'bear', 'beetle', 'butterfly', 'ape', 'monkey']
HUMAN_CLASSNAMES = ['human']


class Client:
    def __init__(self, task='dog', name='base'):
        print(f'task:{task}  client_name:{name}  data_path:data/{task}/{name}')
        if task not in ['dog', 'all_dog', 'one_living', 'all_one_living', 'living', 'all_living', 'human', 'all_huamn']:
            print('!!!!!task is invalid    valid:dog, all_dog, living, all_living, human, all_huamn')
        if name not in ['base', 'client_0', 'client_1', 'client_2', 'client_3', 'client_4']:
            print('!!!!!name is invalid    valid:base, client_0, client_1, client_2, client_3, client_4')

        self.task = task
        self.name = name

        if self.task in ['dog', 'all_dog']:
            self.CLASSNAMES = DOG_CLASSNAMES
        elif self.task in ['one_living', 'all_one_living']:
            self.CLASSNAMES = ONE_LIVING_CLASSNAMES
        elif self.task in ['living', 'all_living']:
            self.CLASSNAMES = LIVING_CLASSNAMES
        elif self.task in ['human', 'all_human']:
            self.CLASSNAMES = HUMAN_CLASSNAMES
        else:
            self.CLASSNAMES = DOG_CLASSNAMES

        # data/dogs/client_0/train_image.txt
        self.train_path = f'data/{self.task}/{self.name}/train_image.txt'
        self.val_path = f'data/{self.task}/{self.name}/test_image.txt'
        self.nc = 1
        self.data_dict = {'train': self.train_path, 'val': self.val_path, 'nc': self.nc, 'names': self.CLASSNAMES}

        self.weights = ''
        self.cfg = 'models/yolov3.yaml'
        self.device = torch.device('cuda')
        self.hyp_path = 'data/hyp.scratch.yaml'  # hyp.scratch.yaml  hyp.scratch_lrfzero.yaml

        self.epochs = 1
        self.batch_size = 16
        self.imgsz = 640
        self.imgsz_test = 640
        self.cache_images = False
        self.rect = False
        self.rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
        self.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        self.workers = 4  # 加载数据集时的线程数0个即可  self.workers = 8
        self.image_weights = False
        self.quad = False
        self.notest = False  # True:不做测试
        self.noautoanchor = False
        self.resume = False
        self.label_smoothing = 0.0

        self.cuda = self.device.type != 'cpu'

        self.last_path = 'last.pt'
        self.best_path = 'best.pt'

        self.lr = 0.01  # 学习率初始值
        self.decay_ratio = 0.99  # 学习率衰减倍率
        self.use_decay = True  # 是否使用学习率衰减
        self.weight_decay_start = 0.0005  # 默认为0.0005  设为0也能让收敛快

        self.state_dict = None  # 不为None就自动使用
        self.state_dict_bn = None  # 用来存之前的bn，不为none就使用其bn参数
        self.use_bn = False

        # test  测试函数用到的参数
        self.conf_thres = 0.001  # 计算mAP时的置信度阈值   0.001能让训练时mAP震荡小？
        self.plots = False  # 是否要画图

        # 只加载一次dataloader  不为None就使用
        self.dataset = None
        self.dataloader = None
        self.testloader = None
        # 全存下来试试
        self.model = None
        self.optimizer = None
        self.use_adam = False
        self.compute_loss = None
        self.scaler = None
        self.accumulate = 0


        # 是否每轮
        self.save_model = True  # 是否每轮存模型
        self.save_mAP = False  # 是否每轮存mAP


        # 测试模型
        self.test = True

        self.pretrained_model = None 
        self.alpha=10            # l2范数在损失中占的比率

    def train(self):
        print(
            f'-- {self.name} train  data_path:data/{self.task}/{self.name}  lr:{self.lr}  use_decay:{self.use_decay} --')

        with open(self.hyp_path) as f:
            hyp = yaml.safe_load(f)  # load hyps

        # Trainloader -------------------------------
        if self.dataloader is None:
            self.dataloader, self.dataset = my_create_dataloader(self.train_path, self.imgsz, self.batch_size, 32,
                                                                 None,
                                                                 hyp=hyp, augment=True, cache=self.cache_images,
                                                                 rect=self.rect, rank=self.rank,
                                                                 world_size=self.world_size, workers=self.workers,
                                                                 image_weights=self.image_weights, quad=self.quad,
                                                                 prefix=colorstr('train: '))
            mlc = np.concatenate(self.dataset.labels, 0)[:, 0].max()  # max label class

            assert mlc < self.nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (
                mlc, self.nc, self.data_dict, self.nc - 1)
        nb = len(self.dataloader)  # number of batches

        # Process 0
        if self.testloader is None:
            self.testloader = my_create_dataloader(self.val_path, self.imgsz_test, self.batch_size, 32, None,
                                                   hyp=hyp, cache=self.cache_images and not self.notest, rect=True,
                                                   rank=-1,
                                                   world_size=self.world_size, workers=self.workers,
                                                   pad=0.5, prefix=colorstr('val: '))[0]
        # loader -------------------------------end
        # --------------------------------------model
        if self.model is None:
            pretrained = self.weights.endswith('.pt')
            if pretrained:
                ckpt = torch.load(self.weights, map_location=self.device)  # load checkpoint
                self.model = Model(self.cfg, ch=3, nc=self.nc, anchors=hyp.get('anchors')).to(
                    self.device)  # create无初始参数
                self.model.load_state_dict(ckpt)
            else:
                self.model = Model(self.cfg, ch=3, nc=self.nc, anchors=hyp.get('anchors')).to(
                    self.device)  # create无初始参数

            # optimizer
            nbs = 64  # nominal batch size
            self.accumulate = max(round(nbs / self.batch_size), 1)  # accumulate loss before optimizing
            hyp['weight_decay'] *= self.batch_size * self.accumulate / nbs  # scale weight_decay

            # Image sizes
            gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
            nl = self.model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])

            # Model parameters
            hyp['box'] *= 3. / nl  # scale to layers
            hyp['cls'] *= self.nc / 80. * 3. / nl  # scale to classes and layers
            hyp['obj'] *= (self.imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
            hyp['label_smoothing'] = self.label_smoothing
            self.model.nc = self.nc  # attach number of classes to model
            self.model.hyp = hyp  # attach hyperparameters to model
            self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
            self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(
                self.device) * self.nc  # attach class weights
            self.model.names = self.CLASSNAMES

        if self.state_dict is not None:  # 是否使用self存的dict state_dict不为None就会直接用
            # 沿用之前的本地模型bn-------------------------------------------------------
            if self.use_bn:
                if self.state_dict_bn is not None:
                    for key in self.state_dict_bn.keys():
                        if 'bn' in key:
                            # print(key, '上一次的', '传进来的')
                            # print(torch.max(self.state_dict_bn[key]), torch.max(self.state_dict[key]))
                            self.state_dict[key] = self.state_dict_bn[key]
            # --------------------------------------------------------------------------

            self.model.load_state_dict(self.state_dict)

            # if self.pretrained_model == None:
            if 1:
                self.pretrained_model=deepcopy(self.model)
                self.pretrained_model.load_state_dict(self.state_dict)
        # --------------------------------------model end

        # --------------------------------------optimizer
        if self.optimizer is None:
            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
            for k, v in self.model.named_modules():
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d):
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            if self.use_adam:
                self.optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
            else:
                self.optimizer = optim.SGD(pg0, lr=self.lr, momentum=hyp['momentum'], nesterov=True)
            self.optimizer.add_param_group(
                {'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
            self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
            del pg0, pg1, pg2
        # --------------------------------------optimizer end

        if self.scaler is None:
            self.scaler = amp.GradScaler(enabled=self.cuda)
        if self.compute_loss is None:
            self.compute_loss = ComputeLoss(self.model)  # init loss class
        # Resume
        start_epoch, best_fitness = 0, 0.0
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

        # initial lr
        # weight_decay同步
        self.optimizer.param_groups[1]['weight_decay'] = self.weight_decay_start * self.lr / 0.01  # 等比例  只给pg1设置
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

        for epoch in range(start_epoch, self.epochs):  # epoch ------------------------------------------------------
            # lr衰减
            if self.use_decay:
                self.lr *= self.decay_ratio
                # weight_decay同步    等比例  只给pg1设置
                self.optimizer.param_groups[1]['weight_decay'] = self.weight_decay_start * self.lr / 0.01
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr

            self.model.train()

            mloss = torch.zeros(4, device=self.device)  # mean losses

            pbar = enumerate(self.dataloader)
            # =================
            # logger.info(('\n' + '%7s' + '%10s' * 8) % ('lr', 'Epoch', 'gpu_mem', 'box', 'obj', 'cls',
            #                                            'total', 'labels', 'img_size'))

            pbar = tqdm(pbar, total=nb, ncols=120)  # progress bar
            self.optimizer.zero_grad()

            for i, (imgs, targets, paths, _) in pbar:  # batch --------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                # Forward
                with amp.autocast(enabled=self.cuda):
                    pred = self.model(imgs)  # forward
                    loss, loss_items = self.compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size
                    loss_l2=self.compute_l2_loss(self.model,self.pretrained_model)
                    if self.rank != -1:
                        loss *= self.world_size  # gradient averaged between devices in DDP mode
                    if self.quad:
                        loss *= 4.
                loss_final=loss+self.alpha*loss_l2 
                # Backward
                self.scaler.scale(loss_final).backward()
                # self.scaler.scale(loss).backward()
                  
                # Optimize
                if ni % self.accumulate == 0:
                    self.scaler.step(self.optimizer)  # optimizer.step
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # Print
                if True:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s   ' + '%10.7f' + '%10s' + '%10.4g' * 4) % (
                        '%g/%g' % (epoch, self.epochs - 1), self.lr, mem, *mloss)

                    pbar.set_description(s)
                # end batch -----------------------------------------------------------------------------------------
            # end epoch ---------------------------------------------------------------------------------------------

            # DDP process 0 or single-GPU

            if self.test:
                if self.rank in [-1, 0]:
                    # mAP
                    final_epoch = epoch + 1 == self.epochs
                    if not self.notest or final_epoch:  # Calculate mAP
                        results, maps, times = my_test.my_test(self.data_dict,
                                                               batch_size=self.batch_size * 2,
                                                               imgsz=self.imgsz_test,
                                                               model=self.model,
                                                               dataloader=self.testloader,
                                                               save_dir='.',
                                                               compute_loss=self.compute_loss,
                                                               conf_thres=self.conf_thres,
                                                               plots=self.plots)

                    # Update best mAP
                    fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                    if fi > best_fitness:
                        best_fitness = fi

                    # Save model
                    if self.save_model:  # if save
                        self.model.train()  # =====
                        ckpt = self.model.state_dict()

                        # Save last, best and delete
                        torch.save(ckpt, self.last_path)
                        if best_fitness == fi:
                            torch.save(ckpt, self.best_path)


        self.model.train()  # =====
        self.state_dict = self.model.state_dict()
        # 复制一份存bn -------------------------------------------
        if self.use_bn:
            self.state_dict_bn = self.state_dict
        
    def set_all(self,
                weights='',
                epochs=2,
                batch_size=16,
                lr=0.01):
        self.weights = weights
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def compute_l2_loss(self,
                        model,
                        pretrained_model):
        if pretrained_model is None:
            return 0
        else:
            loss=0
            for m,p in zip(model.parameters(),pretrained_model.parameters()):
                loss=loss+((m-p)**2).sum()
            return loss
if __name__ == '__main__':
    # 数据集
    write_human_data()

    #
    c = Client(task='human', name='client_0')
    c.set_all(epochs=3, weights='',
              batch_size=8)
    c.train()

    pt = c.state_dict

    c.lr = 0.005
    c.weights = 'best.pt'
    c.epochs = 1
    c.train()

    c.lr = 0.002
    c.state_dict = pt
    c.epochs = 1
    c.train()
