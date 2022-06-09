import torch
import argparse
from client import Client
from client_ewc import Client as Client_ewc
import my_test
from utils.datasets import create_dataloader, my_create_dataloader
import yaml
from utils.loss import ComputeLoss
from models.yolo import Model

import numpy as np
import random
import os

from utils.metrics import fitness

import warnings

warnings.filterwarnings('ignore')


#  lg新增设置固定随机种子的函数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False  # 新增！！！
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)  # 新增！！！


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def avg(ws: list):
    k = len(ws)
    w_avg = {}

    for i, w in enumerate(ws):
        if i == 0:
            for key, values in w.items():
                w_avg[key] = values  # 3## pra
        else:
            for key, values in w.items():
                w_avg[key] = w_avg[key] + values

    for key, values in w_avg.items():
        w_avg[key] = torch.div(w_avg[key], k)

    return w_avg


class Tester:
    def __init__(self, val_path='data/dogs/client_0/test_image.txt', batch_size=8,
                 cfg='models/yolov3.yaml', show_detail=True):
        self.val_path = val_path
        self.batch_size = batch_size
        hyp_path = 'data/hyp.scratch.yaml'
        device = torch.device('cuda')
        nc = 1
        with open(hyp_path) as f:
            hyp = yaml.safe_load(f)  # load hyps
        self.model = Model(cfg, ch=3, nc=nc).to(device)  # create无初始参数
        self.testloader = my_create_dataloader(val_path, 640, batch_size, 32, None, hyp=hyp, workers=0)[0]

        self.conf_thres = 0.001
        self.show_detail = show_detail

    def test(self, state_dict):
        self.model.load_state_dict(state_dict)
        results, maps, times = my_test.my_test(val_path=self.val_path,
                                               batch_size=self.batch_size,
                                               imgsz=640,
                                               model=self.model,
                                               dataloader=self.testloader,
                                               save_dir='.',
                                               conf_thres=self.conf_thres,
                                               show_detail=self.show_detail)
        mp, mr, map50, map, loss1, loss2, loss3 = results
        return results


def save_model(state_dict, file_name='w.pt'):
    torch.save(state_dict, file_name)


def load_model(file_name='w.pt'):
    return torch.load(file_name)


def save_file(file, file_name='precision_matrices/base.pr'):
    torch.save(file, file_name)


def load_file(file_name='precision_matrices/base.pr'):
    return torch.load(file_name)


def avg_list(a):
    max_ = torch.max(a)
    min_ = torch.min(a)
    mean_ = torch.mean(a)

    # mean_ = mean_ * 4   # 调整重要度的数量

    a = torch.where(a[...] < mean_, a[...] * 0, a[...])

    # a = a / max_
    max_ = max_ + 0.00001
    a = torch.div(a, max_)

    return a, max_, min_, mean_


def fid_dif_list(a, b):
    count = 0
    c1 = 0
    c2 = 0
    dif = 0

    i = len(a)
    j = len(a[0])
    k = len(a[0][0])
    l = len(a[0][0][0])
    for ii in range(i):
        for jj in range(j):
            for kk in range(k):
                # print('a, b', a[ii][jj][kk], b[ii][jj][kk])

                for ll in range(l):
                    count += 1
                    if a[ii][jj][kk][ll] != 0:
                        c1 += 1
                    if b[ii][jj][kk][ll] != 0:
                        c2 += 1
                    if a[ii][jj][kk][ll] != 0 and b[ii][jj][kk][ll] != 0:
                        dif += 1
    return count, c1, c2, dif


def change_dif_list(a, b):
    # 两两计算重叠部分 or 计算一个和其它所有的重叠部分   每4个都加起来，再和其它的比较

    count = 0
    c1 = 0
    c2 = 0
    dif = 0

    i = len(a)
    j = len(a[0])
    k = len(a[0][0])
    l = len(a[0][0][0])
    for ii in range(i):
        for jj in range(j):
            for kk in range(k):
                for ll in range(l):
                    count += 1
                    if a[ii][jj][kk][ll] != 0:
                        c1 += 1
                    if b[ii][jj][kk][ll] != 0:
                        c2 += 1
                    if a[ii][jj][kk][ll] != 0 and b[ii][jj][kk][ll] != 0:
                        dif += 1
    return count, c1, c2, dif


def find_dif_dict(p1, p2):
    for key, values in p1.items():
        # if 'bn' not in key and 'bias' not in key:
        if 'conv' in key:
            count, c1, c2, dif = fid_dif_list(p1[key], p2[key])
            print(key, 'count, c1, c2, dif: ', count, c1, c2, dif)


def cal_precision_matrices(w='./../task_vehicle_base/last.pt',
                           save_file_name='precision_matrices/base.pr',
                           task='vehicle', name=f'base',
                           cfg='models/yolov3-tiny.yaml',
                           lr=0.002,
                           use_decay=False,
                           decay_ratio=0.98,
                           epoch_train_before=0):  # 先训几轮再算

    weights = torch.load(w)

    # ewc -----------------
    c_temp = Client_ewc(task=task, name=name)
    # 先跑几轮
    c_temp.set_all(epochs=epoch_train_before, weights='',
                   batch_size=16,
                   lr=lr)

    c_temp.cfg = cfg

    c_temp.use_decay = use_decay
    c_temp.decay_ratio = decay_ratio
    c_temp.save_model = False
    c_temp.workers = 4
    c_temp.use_ewc = False
    c_temp.state_dict = weights

    c_temp.train()  # 先正常训5轮

    c_temp.epochs = 1

    c_temp.calewc()  # 用于获得ewc相关参数
    w_start = c_temp.state_dict
    _means = c_temp._means
    precision_matrices = c_temp.precision_matrices

    for key, values in precision_matrices.items():
        if 'conv' in key:
            print('key', key)
            print(len(precision_matrices[key]))
            print(precision_matrices[key].shape)
            print(precision_matrices[key][0][0][0])

            # 先存下原始的
            # max_, min_, avg_ = avg_list(precision_matrices[key])
            # print(precision_matrices[key])
            # print('max_, min_, avg_', max_, min_, avg_)

    save_file(precision_matrices, file_name=save_file_name)


def all_pre(client_num,
            task,
            epoch_train_before=100,
            cfg='models/yolov3.yaml',
            w='./../task_vehicle_base_2/last.pt',
            save_root='precision_matrices'):
    cal_precision_matrices(w=w,
                           save_file_name=f'{save_root}/base.pr',
                           task=task, name=f'base',
                           cfg=cfg,
                           epoch_train_before=0,
                           lr=0.002)

    for i in range(client_num):
        cal_precision_matrices(w=w,
                               save_file_name=f'{save_root}/client_{i}.pr',
                               task=task, name=f'client_{i}',
                               cfg=cfg,
                               epoch_train_before=epoch_train_before,
                               lr=0.02,
                               use_decay=True,
                               decay_ratio=0.98)


def write_res(epoch, res_list, file='res/res_0.txt'):
    mkdir('res')

    with open(file, "a") as f:
        f.write(str(epoch) + ' ')
        for res in res_list:
            for info in res:
                f.write(str(info))
                f.write(' ')
        f.write('\n')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def avg_ori(client_num,
            root='./precision_matrices_ori',
            save_root='./precision_matrices_avg'):
    mkdir(save_root)

    base_ori = load_file(file_name=f'{root}/base.pr')
    clients_ori = []
    for i in range(5):
        clients_ori.append(load_file(file_name=f'{root}/client_{i}.pr'))

    for key, values in base_ori.items():
        if 'conv' in key:
            print('key', key)
            print(len(base_ori[key]))
            print(base_ori[key].shape)
            print(base_ori[key][0][0][0])

            # 先存下原始的
            base_ori[key], max_, min_, avg_ = avg_list(base_ori[key])
            # print(base_ori[key])
            print('max_, min_, avg_', max_, min_, avg_)

    save_file(base_ori, file_name=f'{save_root}/base.pr')

    for i in range(client_num):
        for key, values in clients_ori[i].items():
            if 'conv' in key:
                print('key', key)
                print(len(clients_ori[i][key]))
                print(clients_ori[i][key].shape)
                print(clients_ori[i][key][0][0][0])

                # 先存下原始的
                max_, min_, avg_ = avg_list(clients_ori[i][key])
                # print(clients_ori[i][key])
                print('max_, min_, avg_', max_, min_, avg_)

        save_file(clients_ori[i], file_name=f'{save_root}/client_{i}.pr')


def chuli_2():
    base_pre = load_file(file_name='./../../save/precision_matrices_avg/base.pr')
    clients_pre = []
    for i in range(5):
        clients_pre.append(load_file(file_name=f'./../../save/precision_matrices_avg/client_{i}.pr'))

    for i in range(5):
        print(f'----       base and client_{i}             ----')
        find_dif_dict(base_pre, clients_pre[i])


# 算出重合部分
def overlap_avg(client_num,
                root='./precision_matrices_avg',
                save_root='./precision_matrices_lap'):
    mkdir(save_root)

    base_pre = load_file(file_name=f'{root}/base.pr')
    clients_pre = []
    for i in range(client_num):
        clients_pre.append(load_file(file_name=f'{root}/client_{i}.pr'))

    # 计算重复的矩阵，不为0即为有值，为0则无值

    for i in range(client_num):
        # 复制一份
        temp = {}
        first = True
        for j in range(client_num):
            if i == j:
                continue

            if first:
                first = False
                for key, values in clients_pre[j].items():
                    if 'conv' in key:
                        temp[key] = values.clone()
                        print('key', key, 'value', temp[key][0][0][0])
            else:
                for key, values in clients_pre[j].items():
                    if 'conv' in key:
                        temp[key] = temp[key] + values.clone()
                        print('key', key, 'value', temp[key][0][0][0])
            # for key, values in temp.items():
            #     print(temp[key])
        # 存下来
        save_file(temp, file_name=f'{save_root}/overlap_{i}.pr')


def displace(a, b, op=0):
    # op = 1 b不为0的位置复制到a  op=0  b不为0的位置 将a对应位置为0
    if op == 1:
        a = torch.where(b[...] != 0, b[...], a[...])
    if op == 0:
        a = torch.where(b[...] != 0, a[...] * 0, a[...])
    return a


def overlap_all(client_num,
                avg_root='./precision_matrices_avg',
                lap_root='./precision_matrices_lap',
                save_root='./precision_matrices_lap_all'):
    mkdir(save_root)

    base_avg = load_file(file_name=f'{avg_root}/base.pr')
    client_avg = []
    client_lap = []

    for i in range(client_num):
        client_avg.append(load_file(file_name=f'{avg_root}/client_{i}.pr'))

    for i in range(client_num):
        client_lap.append(load_file(file_name=f'{lap_root}/overlap_{i}.pr'))

    # 先放入overlap_pre client_pre全置0(对自己重要的参数不做限制)  再替换为base_pre

    for i in range(client_num):
        temp = {}
        for key, values in client_lap[i].items():
            if 'conv' in key:
                temp[key] = values.clone()

        for key, values in client_avg[i].items():
            if 'conv' in key:
                temp[key] = displace(temp[key], client_avg[i][key], op=0)

        for key, values in base_avg.items():
            if 'conv' in key:
                temp[key] = displace(temp[key], base_avg[key], op=1)

        # 存下来
        save_file(temp, file_name=f'{save_root}/overlap_all_{i}.pr')


def test(f1='./../../save/precision_matrices_shuaijian_avg/clients_0.pr',
         f2='./precision_matrices/overlap_0.pr'):
    client_0 = load_file(file_name=f1)
    overlap_0 = load_file(file_name=f2)

    find_dif_dict(client_0, overlap_0)


def cal_ewc(task, client_num=3, cfg='models/yolov3.yaml', w='w.pt'):
    cal_precision_matrices(w=w,
                           save_file_name=f'precision_matrices/base.pr',
                           task=task, name=f'base',
                           cfg=cfg,
                           epoch_train_before=0,
                           lr=0.002)

    # 原本的模型
    save_file(torch.load(w), file_name='./precision_matrices/mean.pr')

    # 归一化 avg_list
    base_ori = load_file(file_name=w)
    mkdir('precision_matrices_avg')

    for key, values in base_ori.items():
        if 'conv' in key:
            print('key', key)
            print(len(base_ori[key]))
            print(base_ori[key].shape)
            print(base_ori[key][0][0][0])

            # 先存下原始的
            base_ori[key], max_, min_, avg_ = avg_list(base_ori[key])
            # print(base_ori[key])
            print('max_, min_, avg_, torch.max(base_ori[key])', max_, min_, avg_, torch.max(base_ori[key]))

    save_file(base_ori, file_name=f'./precision_matrices_avg/base.pr')


def cal_overlap_all(client_num,
                    task,
                    epoch_train_before=100,
                    cfg='models/yolov3.yaml', w='w.pt'):
    """
    在当前轮计算
    :return:
    """

    # all_pre  用当前的聚合模型算   原始的
    all_pre(client_num=client_num,
            task=task,
            epoch_train_before=epoch_train_before,
            w=w,
            save_root='precision_matrices')

    # 原本的模型
    save_file(torch.load(w), file_name='./precision_matrices/mean.pr')

    # 归一化 avg_list
    avg_ori(client_num=client_num,
            root='./precision_matrices',
            save_root='./precision_matrices_avg')

    # 其它4个重叠起来
    overlap_avg(client_num=client_num,
                root='./precision_matrices_avg',
                save_root='./precision_matrices_lap')

    overlap_all(client_num=client_num,
                avg_root='./precision_matrices_avg',
                lap_root='./precision_matrices_lap',
                save_root='./precision_matrices_lap_all')


def fed_vehicle_ewc(client_num,
                      epochs=500,
                      local_epochs=5,
                      lr=1e-4,
                      test_per_epochs=1,
                      lambda_ewc=0.1,
                      ewc_epoch_train_before_cal_pri=100):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--startweights', type=str, default='./../final_vehicle_base_model/last.pt')
    parser.add_argument('--startepoch', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cfg', type=str, default='big', help='tiny->tiny, other->yolov3')
    parser.add_argument('--ewccal', type=int, default=10000, help='ewc cal per epoch')
    parser.add_argument('--task', type=str, default='vehicle_4_600')
    opt = parser.parse_args()

    batch_size = opt.batch_size

    cfg = 'models/yolov3.yaml'
    if opt.cfg == 'tiny':
        cfg = 'models/yolov3-tiny.yaml'

    # tester
    base_tester = Tester(val_path=f'data/{opt.task}/base/test_image.txt',
                         cfg=cfg)
    client_testers = []
    for i in range(client_num):
        client_testers.append(Tester(val_path=f'data/{opt.task}/client_{i}/test_image.txt',
                                     cfg=cfg))

    clients = []
    for i in range(client_num):
        clients.append(Client_ewc(task=opt.task, name=f'client_{i}'))

    for i in range(client_num):
        clients[i].set_all(epochs=local_epochs, weights='',
                           batch_size=batch_size,
                           lr=lr)
        clients[i].use_decay = False
        clients[i].save_model = False
        clients[i].workers = opt.workers
        clients[i].cfg = cfg

    # set w_start  train在判断state_dict不为None时就自动使用

    start_weights = torch.load(opt.startweights)

    for i in range(client_num):
        clients[i].state_dict = start_weights
        clients[i].use_ewc = True
        clients[i].lambda_ewc = lambda_ewc

    # fed --------------------------------------------------------------------------
    best_fi = 0
    best_epoch = 0

    for epoch in range(opt.startepoch, epochs):

        # cal_ewc -----------------
        if epoch % opt.ewccal == 0:

            if epoch == 0:
                cal_overlap_all(client_num=client_num,
                                task=opt.task,
                                epoch_train_before=ewc_epoch_train_before_cal_pri,
                                w=opt.startweights)
            else:
                cal_overlap_all(client_num=client_num,
                                task=opt.task,
                                epoch_train_before=ewc_epoch_train_before_cal_pri,
                                w='w.pt')  # 后续用上一回合的

            ewc_means = torch.load('./precision_matrices/mean.pr')

            ewc_client_precision_matrices = []

            for i in range(client_num):
                ewc_client_precision_matrices.append(
                    load_file(file_name=f'./precision_matrices_lap_all/overlap_all_{i}.pr'))

            # set
            for i in range(client_num):
                clients[i]._means = ewc_means
                clients[i].precision_matrices = ewc_client_precision_matrices[i]
        # --------------ewc end

        print(f'==   epoch: {epoch}     ===')
        # train
        # base.train()
        for i in range(client_num):
            clients[i].train()
            # fi  ===========================================

        # avg
        ws = []
        for i in range(client_num):
            ws.append(clients[i].state_dict)

        w_avg = avg(ws)

        # 存模型
        save_model(w_avg)

        # tset
        res_list = []
        if epoch % test_per_epochs == 0:
            print(f'==   test w_avg  epoch:{epoch}  ===')
            print('base test')

            fi = 0

            base_res = base_tester.test(w_avg)
            res_list.append(base_res)
            for i in range(client_num):
                print(f'client_{i} test')
                client_res = client_testers[i].test(w_avg)
                res_list.append(client_res)

                # fi=====
                fi = fi + fitness(np.array(client_res).reshape(1, -1))
                # print(f'client_{i}, fi  ', fitness(np.array(client_res).reshape(1, -1)))
                with open('fi.txt', "a") as f:
                    f.write('*  ' + str(epoch) + '  ' + str(fi) + '\n')

            write_res(epoch, res_list)
            res_list.clear()

            # fi=====
            avg_fi = fi / client_num
            if avg_fi > best_fi:
                best_fi = avg_fi
                best_epoch = epoch
            # fi=====
            if epoch - best_epoch >= 15:
                print(f'训练完成,最终轮次: {epoch}, 最优fi:{best_fi},最终fi:{avg_fi}')
                break

        # set
        for i in range(client_num):
            clients[i].state_dict = w_avg


if __name__ == '__main__':
    setup_seed(2021)

    fed_vehicle_ewc(client_num=4, lambda_ewc=0.01, ewc_epoch_train_before_cal_pri=50)

    # Chain
