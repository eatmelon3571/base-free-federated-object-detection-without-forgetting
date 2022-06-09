import torch
import argparse
from split_dog_data_few import write_data as write_dog_data
from split_all_dog_data_few import write_all_data as write_all_dog_data
from split_one_living_data import write_data as write_one_living_data
from split_all_one_living_data import write_data as write_all_one_living_data
from split_human_data import write_data as write_human_data
from split_vehicle_data import write_data as write_vehicle_data
import my_test
from utils.datasets import create_dataloader, my_create_dataloader
import yaml
from utils.loss import ComputeLoss
from models.yolo import Model

import warnings

warnings.filterwarnings('ignore')

import matplotlib#
matplotlib.use('Agg')#

from utils.metrics import fitness

import random#
import numpy as np#
import torch.backends.cudnn as cudnn#
import torch.nn.functional as F
import os#

def avg(ws: list,my_weight):
    k = len(ws)
    w_avg = {}
    my_weight=my_weight*k#与后面的k抵消
    for i, w in enumerate(ws):
        if i == 0:
            for key, values in w.items():
                w_avg[key] = values*my_weight[i]#乘上权重
        else:
            for key, values in w.items():
                # w_avg[key] += values#不能用+=
                w_avg[key] = w_avg[key]+values*my_weight[i]#乘上权重

    for key, values in w_avg.items():
        w_avg[key] = torch.div(w_avg[key], k)

    return w_avg


class Tester:
    def __init__(self, val_path, batch_size=8):
        self.val_path = val_path
        self.batch_size = batch_size
        hyp_path = 'data/hyp.scratch.yaml'
        cfg = 'models/yolov3.yaml'
        device = torch.device('cuda')
        nc = 1
        with open(hyp_path) as f:
            hyp = yaml.safe_load(f)  # load hyps
        self.model = Model(cfg, ch=3, nc=nc).to(device)  # create无初始参数
        self.testloader = my_create_dataloader(val_path, 640, batch_size, 32, None, hyp=hyp, workers=0)[0]

        self.conf_thres = 0.001

    def test(self, state_dict):
        self.model.load_state_dict(state_dict)
        results, maps, times = my_test.my_test(val_path=self.val_path,
                                               batch_size=self.batch_size,
                                               imgsz=640,
                                               model=self.model,
                                               dataloader=self.testloader,
                                               save_dir='.',
                                               conf_thres=self.conf_thres)
        #返回map50
        return results#[2]
        #返回map50



def load_model(file_name='w.pt'):
    return torch.load(file_name)


def fed_vehicle(epochs=2500,#500
                local_epochs=5,
                lr=0.002,
                batch_size=32,
                test_per_epochs=1,
                apply_weight=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)#32
    parser.add_argument('--weights', type=str, default='final_vehicle_base_model/last.pt')#设置base预训练模型
    num_client=3
    my_weight=torch.ones(num_client)#存储各个client的聚合权重
    # my_weight=torch.randn(num_client)#存储各个client的聚合权重
    parser.add_argument('--write_data', type=int, default=0)#
    opt = parser.parse_args()

    batch_size = opt.batch_size

    # 数据集
    if opt.write_data == 1:
        write_vehicle_data()

    # 初始权重
    w_start = torch.load(opt.weights)

    # tester
    base_tester = Tester(val_path=f'data/vehicle_{num_client}_1800/base/test_image.txt')#设置base测试集！！！
    client_testers = []
    for i in range(num_client):
        client_testers.append(Tester(val_path=f'data/vehicle_{num_client}_1800/client_{i}/test_image.txt'))

    # 使用训练集的图片测算map进行加权
    my_client_testers=[]
    for i in range(num_client):
        my_client_testers.append(Tester(val_path=f'data/vehicle_{num_client}_1800/client_{i}/train_image.txt'))

    clients = []
    for i in range(num_client):
        clients.append(Client(task=f'vehicle_{num_client}_1800', name=f'client_{i}'))

    for i in range(num_client):
        clients[i].set_all(epochs=local_epochs, weights='',
                           batch_size=batch_size,
                           lr=lr)
        clients[i].use_decay = False
        clients[i].save_model = False
        clients[i].workers = 8#8

    # set w_start  train在判断state_dict不为None时就自动使用
    for i in range(num_client):
        clients[i].state_dict = w_start

    w_avg= w_start # 动态加权代码

    best_fi=0
    best_epoch=0
    # fed
    for epoch in range(epochs):
        #获取初始mAP用于计算权重，开头
        if apply_weight==True:# ：链式  #正常：epoch==0 and apply_weight==True:
            print("------------head------------")
            # print('base test')
            # base_tester.test(w_start)
            for i in range(num_client):
                print(f'client_{i} test')
                my_mAP50=my_client_testers[i].test(w_avg)[2]#返回对应mAP值 # w_avg:动态加权 w_start:静态加权
                my_weight[i]=my_mAP50
            print("初始mAP:",my_weight)
            # my_weight=(torch.ones(num_client)-my_weight)*1.0#反向加权
            my_weight=torch.tensor([-1.0])*my_weight#反向加权
            my_weight=torch.exp(my_weight)#进行softmax
            my_weight=my_weight/torch.sum(my_weight)#进行softmax
            print("初始权重:",my_weight)
            print("------------tail------------")
        if epoch==0 and apply_weight==False:
            my_weight=my_weight/torch.tensor([num_client*1.0])#相同权重
        #获取初始mAP用于计算权重，结尾
        print(f'==   epoch: {epoch}     ===')
        # train
        # base.train()
        
        
        for i in range(num_client):
            clients[i].train()

        # avg
        ws = []
        for i in range(num_client):
            ws.append(clients[i].state_dict)

        w_avg = avg(ws,my_weight)#传入模型和对应的聚合权重

        # 存模型
        save_model(w_avg)

        fi=0
        # tset
        if epoch % test_per_epochs == 0:
            print(f'==   test w_avg  epoch:{epoch}  ===')
            print('base test')
            base_tester.test(w_avg)
            for i in range(num_client):
                print(f'client_{i} test')
                client_res = client_testers[i].test(w_avg)

                # fi=====
                fi = fi + fitness(np.array(client_res).reshape(1, -1))
                # print(f'client_{i}, fi  ', fitness(np.array(client_res).reshape(1, -1)))
                # with open('fi.txt', "a") as f:
                #     f.write('*  ' + str(epoch) + '  ' + str(fi) + '\n')

        # fi=====
        avg_fi = fi / num_client
        if avg_fi > best_fi:
            best_fi = avg_fi
            best_epoch = epoch
        # fi=====
 
        if epoch - best_epoch >= 15:
            print(f'训练完成,最终轮次: {epoch}, 最优fi:{best_fi},最终fi:{avg_fi}')
            break

        # set
        for i in range(num_client):
            clients[i].state_dict = w_avg

def fed_vehicle_low_communication(epochs=2500,#2500
                local_epochs=5,# 5改为1
                lr=0.002,
                batch_size=32,
                test_per_epochs=1,
                selected_num=2,
                use_bn=False,
                ct=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)#32
    # parser.add_argument('--weights', type=str, default='final_vehicle_base_model/last.pt')#设置base预训练模型
    parser.add_argument('--weights', type=str, default='final_vehicle_l2_model/final_fed_l2_lr=0.0001_alpha0.01_lc_vehicle_4_600_earlystop.pt')#设置base预训练模型
    
    num_client=4
    my_weight=torch.ones(selected_num)#存储各个client的聚合权重
    my_map=torch.zeros(num_client)
    # my_weight=torch.randn(num_client)#存储各个client的聚合权重
    parser.add_argument('--write_data', type=int, default=0)#
    opt = parser.parse_args()

    batch_size = opt.batch_size

    # 数据集
    if opt.write_data == 1:
        write_vehicle_data()

    # 初始权重  w_start = torch.load('./../task_human_base_2/last.pt')
    w_start = torch.load(opt.weights)

    # tester
    base_tester = Tester(val_path=f'data/vehicle_{num_client}_600/base/test_image.txt')#设置base测试集！！！
    client_testers = []
    for i in range(num_client):
        client_testers.append(Tester(val_path=f'data/vehicle_{num_client}_600/client_{i}/test_image.txt'))

    # 使用训练集的图片测算map进行加权
    my_client_testers=[]
    for i in range(num_client):
        my_client_testers.append(Tester(val_path=f'data/vehicle_{num_client}_600/client_{i}/train_image.txt'))

    clients = []
    for i in range(num_client):
        clients.append(Client(task=f'vehicle_{num_client}_600', name=f'client_{i}'))

    for i in range(num_client):
        clients[i].set_all(epochs=local_epochs, weights='',
                           batch_size=batch_size,
                           lr=lr)
        clients[i].use_decay = False
        clients[i].save_model = False
        clients[i].workers = 8#8
        clients[i].use_bn = use_bn

    # set w_start  train在判断state_dict不为None时就自动使用
    for i in range(num_client):
        clients[i].state_dict = w_start

    w_avg= w_start # 动态加权代码

    best_fi=0
    best_epoch=0
    # fed
    for epoch in range(epochs):
        #获取初始mAP用于计算权重，开头
        apply_weight=False#是否应用权重？
        if apply_weight==True:# ：链式  #正常：epoch==0 and apply_weight==True:
            print("------------head------------")
            # print('base test')
            # base_tester.test(w_start)
            for i in range(num_client):
                print(f'client_{i} test')
                my_mAP50=my_client_testers[i].test(w_avg)[2]#返回对应mAP值 # w_avg:动态加权 w_start:静态加权
                my_weight[i]=my_mAP50
            print("初始mAP:",my_weight)
            my_weight=(torch.ones(num_client)-my_weight)*1.0#反向加权
            # my_weight=torch.tensor([-1.0])*my_weight#反向加权
            my_weight=torch.exp(my_weight)#进行softmax
            my_weight=my_weight/torch.sum(my_weight)#进行softmax
            print("初始权重:",my_weight)
            print("------------tail------------")
        if epoch==0 and apply_weight==False:
            my_weight=my_weight/torch.tensor([selected_num*1.0])#相同权重
        #获取初始mAP用于计算权重，结尾

        if epoch==0:
            for i in range(num_client):
                print(f'client_{i} test')
                my_mAP50=my_client_testers[i].test(w_avg)[2]#返回对应mAP值 # w_avg:动态加权 w_start:静态加权
                my_map[i]=my_mAP50



        selected_clients=np.random.choice(num_client, selected_num, replace=False, p=F.softmax(-my_map).numpy())
        print(selected_clients)

        for i in selected_clients:
            print('更新mAP:')
            print(f'client_{i} test')
            my_mAP50=my_client_testers[i].test(w_avg)[2]#返回对应mAP值 # w_avg:动态加权 w_start:静态加权
            my_map[i]=my_mAP50
        
        print(f'==   epoch: {epoch}     ===')
        # train
        # base.train()
        
        
        for i in selected_clients:
            clients[i].train()

        # avg
        ws = []
        for i in selected_clients:
            ws.append(clients[i].state_dict)

        w_avg = avg(ws,my_weight)#传入模型和对应的聚合权重

        # 存模型
        save_model(w_avg)

        fi=0
        # tset
        if epoch % test_per_epochs == 0:
            print(f'==   test w_avg  epoch:{epoch}  ===')
            print('base test')
            
            # test前 修改bn-----------------
            # if use_bn:
            #     for key in w_avg.keys():
            #         if 'bn' in key:
            #             # print(torch.max(w_avg[key]), torch.max(w_start[key]))
            #             w_avg[key] = w_start[key]
            #             # print(torch.max(w_avg[key]), torch.max(w_start[key]))
            # -----------------------------

            base_tester.test(w_avg)
            for i in range(num_client):
                try:
                    print(f'client_{i} test')

                    # test前 修改bn-----------------
                    if use_bn:
                        for key in w_avg.keys():
                            if 'bn' in key:
                                
                                # print(torch.max(w_avg[key]), torch.max(clients[i].state_dict[key]))
                                w_avg[key] = clients[i].state_dict_bn[key]
                                # print(torch.max(w_avg[key]), torch.max(clients[i].state_dict[key]))
                    # -----------------------------

                    client_res = client_testers[i].test(w_avg)
                    
                    # fi=====
                    fi = fi + fitness(np.array(client_res).reshape(1, -1))
                    # print(f'client_{i}, fi  ', fitness(np.array(client_res).reshape(1, -1)))
                    # with open('fi.txt', "a") as f:
                    #     f.write('*  ' + str(epoch) + '  ' + str(fi) + '\n')
                except:
                    continue

        # fi=====
        avg_fi = fi / num_client
        if avg_fi > best_fi:
            best_fi = avg_fi
            best_epoch = epoch
        # fi=====
         
        if epoch - best_epoch >= 15:
            print(f'训练完成,最终轮次: {epoch}, 最优fi:{best_fi},最终fi:{avg_fi}')
            if ct:
                my_weight=torch.ones(num_client)/torch.tensor(num_client)
                # set
                for i in range(num_client):
                    clients[i].state_dict = w_avg

                for _ in range(5):
                    for i in range(num_client):
                        clients[i].train()

                    # avg
                    ws = []
                    for i in range(num_client):
                        ws.append(clients[i].state_dict)

                    w_avg = avg(ws,my_weight)#传入模型和对应的聚合权重

                    # 存模型
                    # 存模型前 修改bn-----------------
                    # if use_bn:
                    #     for key in w_avg.keys():
                    #         if 'bn' in key:
                    #             # print(torch.max(w_avg[key]), torch.max(w_start[key]))
                    #             w_avg[key] = w_start[key]
                    #             # print(torch.max(w_avg[key]), torch.max(w_start[key]))
                    # -----------------------------
                    save_model(w_avg,file_name='final_vehicle_l2_model/final_fed_l2_lr=0.0001_alpha0.01_lc_ct_vehicle_4_600_earlystop.pt')

                    fi=0
                    # tset
                    if epoch % test_per_epochs == 0:
                        print(f'==   test w_avg  epoch:{epoch}  ===')
                        print('base test')
                         # test前 修改bn-----------------
                        # if use_bn:
                        #     for key in w_avg.keys():
                        #         if 'bn' in key:
                        #             # print(torch.max(w_avg[key]), torch.max(w_start[key]))
                        #             w_avg[key] = w_start[key]
                        #             # print(torch.max(w_avg[key]), torch.max(w_start[key]))
                        # -----------------------------

                        base_tester.test(w_avg)
                        for i in range(num_client):
                            print(f'client_{i} test')
                            
                            # test前 修改bn-----------------
                            if use_bn:
                                for key in w_avg.keys():
                                    if 'bn' in key:
                                        
                                        # print(torch.max(w_avg[key]), torch.max(clients[i].state_dict[key]))
                                        w_avg[key] = clients[i].state_dict_bn[key]
                                        # print(torch.max(w_avg[key]), torch.max(clients[i].state_dict[key]))
                            # -----------------------------

                            client_res = client_testers[i].test(w_avg)
                            fi = fi + fitness(np.array(client_res).reshape(1, -1))

                    # fi=====
                    avg_fi = fi / num_client
                    if avg_fi > best_fi:
                        best_fi = avg_fi
                        best_epoch = epoch
                    # fi====
                    # set
                    for i in range(num_client):
                        clients[i].state_dict = w_avg
                print(f'继续训练完成,最终轮次: {epoch}, 最优fi:{best_fi},最终fi:{avg_fi}')
                break
            else:
                print(f'训练完成,最终轮次: {epoch}, 最优fi:{best_fi},最终fi:{avg_fi}')
                break

        # set
        for i in range(num_client):
            clients[i].state_dict = w_avg

# 设置固定随机种子的函数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] =str(seed)

def save_model(state_dict, file_name='final_vehicle_l2_model/final_fed_l2_lr=0.0001_alpha0.01_lc_vehicle_4_600_earlystop.pt'):
    torch.save(state_dict, file_name)

if __name__ == '__main__':
    setup_seed(2022)

    method = 'fedavg'   # method in ['fedavg', 'l2', 'l2chain']
    if method == 'fedavg':
        from client_fedavg import Client
    elif method == 'l2':
        from client_l2 import Client
    elif method == 'l2chain':
        from client_l2chain import Client

    # fed_vehicle(lr=0.0001,local_epochs=5)
    fed_vehicle_low_communication(lr=0.0001,use_bn=False,ct=True)

#######跑程序之前先修改save_model的file_name参数，指定模型保存位置，防止覆盖！！！

####### l2chain 4 600       使用fed_vehicle函数，method设定为l2chain
#python fed.py > final_vehicle_l2chain_model/final_fed_l2chain_lr=0.0001_alpha10_vehicle_4_600_earlystop.txt

####### l2 4 600            使用fed_vehicle函数，method设定为l2
#CUDA_VISIBLE_DEVICES=1 python fed.py > final_fed_l2_lr=0.0001_alpha0.01_vehicle_4_600_earlystop.txt

#########fed 4 600          使用fed_vehicle函数，method设定为fedavg
#CUDA_VISIBLE_DEVICES=1 python fed.py > final_fed_lr=0.0001_vehicle_4_600_earlystop.txt



######### l2chain W 4 600   使用fed_vehicle函数，method设定为l2chain,apply_weight参数设定为TRUE
#python fed.py > final_vehicle_l2chain_model/final_fed_l2chain_lr=0.0001_alpha10_-_vehicle_4_600_earlystop.txt

######### l2 W 4 600        使用fed_vehicle函数，method设定为l2,apply_weight参数设定为TRUE
#CUDA_VISIBLE_DEVICES=1 python fed.py > final_vehicle_l2_model/final_fed_l2_lr=0.0001_alpha0.01_-_vehicle_4_600_earlystop.txt



# comunicate vehicle 4 600
# python fed.py > final_vehicle_l2chain_model/final_fed_l2chain_lr=0.0001_alpha10_le=1_vehicle_4_600_earlystop.txt
#CUDA_VISIBLE_DEVICES=1 python fed.py > final_vehicle_l2chain_model/final_fed_l2chain_lr=0.0001_alpha10_le=10_vehicle_4_600_earlystop.txt
# python fed.py > final_vehicle_fed_model/final_fed_lr=0.0001_le=1_vehicle_4_600_earlystop.txt



# vehicle 4 600 lc bn ct  使用fed_vehicle_low_communication函数，设定相应的method以及相应的use_bn以及ct参数
# 注：直接运行lc-ct可以同时得到lc和ct的结果
#  python fed.py > final_vehicle_l2chain_model/final_fed_l2chain_lr=0.0001_alpha10_lc_vehicle_4_600_earlystop.txt
#  python fed.py > final_vehicle_l2_model/final_fed_l2_lr=0.0001_alpha0.01_lc_vehicle_4_600_earlystop.txt
#  CUDA_VISIBLE_DEVICES=1 python fed.py > final_vehicle_fed_model/final_fed_lr=0.0001_lc_vehicle_4_600_earlystop.txt

#  python fed.py > final_vehicle_fed_model/final_fed_lr=0.0001_lc_ct_vehicle_4_600_earlystop.txt
#  CUDA_VISIBLE_DEVICES=1 python fed.py > final_vehicle_l2chain_model/final_fed_l2chain_lr=0.0001_alpha10_lc_ct_vehicle_4_600_earlystop.txt
#  CUDA_VISIBLE_DEVICES=1 python fed.py > final_vehicle_l2_model/final_fed_l2_lr=0.0001_alpha0.01_lc_ct_vehicle_4_600_earlystop.txt

# python fed.py > final_vehicle_fed_model/final_fed_lr=0.0001_lc_bn_vehicle_4_600_earlystop.txt
# python fed.py > final_vehicle_l2chain_model/final_fed_l2chain_lr=0.0001_alpha10_lc_bn_vehicle_4_600_earlystop.txt
# python fed.py > final_vehicle_l2_model/final_fed_l2_lr=0.0001_alpha0.01_lc_bn_vehicle_4_600_earlystop.txt
# python fed.py > final_vehicle_fed_model/final_fed_lr=0.0001_lc_bn_ct_vehicle_4_600_earlystop.txt

