import numpy as np
import glob
import os

from nuimages import NuImages

seed_default = 1

# human.pedestrian
base_human = ['adult']
clients_human = [['child'],
                 ['construction_worker'],
                 ['personal_mobility'],
                 ['police_officer'],
                 ['wheelchair', 'stroller']]

# vehicle
base_vehicle = ['car']  # 250 088
clients_vehicle = [['truck'],  # 36 314
                   ['bicycle'],  # 17 060
                   ['motorcycle'],  # 16 779
                   ['trailer'],  # 3 771
                   ['bus']]  # 8 361 + 265

files = ['v1.0-train', 'v1.0-val']


# 设置打乱的随机种子
def set_seed(seed=1):
    np.random.seed(seed)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clear_path(path):
    """ 删除目录下文件及文件夹 """
    if os.path.exists(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))


def clear_all(path_name):
    clear_path(f'data/{path_name}/base')
    for i in range(5):
        clear_path(f'data/{path_name}/client_{i}')


def split(seed,
          dataroot='./../../data/nuimages-v1.0'):
    base_image_list = []
    clients_image_list = [[], [], [], [], []]

    nuim = NuImages(dataroot=dataroot, version='v1.0-train', verbose=True, lazy=True)
    for ann in nuim.object_ann:
        category_token = ann['category_token']

        sample_data_token = ann['sample_data_token']
        category = nuim.get('category', category_token)
        sample_data = nuim.get('sample_data', sample_data_token)

        image_path = os.path.join(dataroot, sample_data['filename'])  # 图片路径

        s = category['name'].split('.')
        # print(s)

        for h in base_vehicle:
            if h in s:
                # 去重
                if image_path not in base_image_list:
                    base_image_list.append(image_path)

        for i in range(5):
            for h in clients_vehicle[i]:
                if h in s:
                    if image_path not in clients_image_list[i]:
                        clients_image_list[i].append(image_path)

    # val部分
    nuim = NuImages(dataroot=dataroot, version='v1.0-val', verbose=True, lazy=True)
    for ann in nuim.object_ann:
        category_token = ann['category_token']

        sample_data_token = ann['sample_data_token']
        category = nuim.get('category', category_token)
        sample_data = nuim.get('sample_data', sample_data_token)

        image_path = os.path.join(dataroot, sample_data['filename'])  # 图片路径

        s = category['name'].split('.')
        # print(s)

        for h in base_vehicle:
            if h in s:
                # 去重
                if image_path not in base_image_list:
                    base_image_list.append(image_path)

        for i in range(5):
            for h in clients_vehicle[i]:
                if h in s:
                    if image_path not in clients_image_list[i]:
                        clients_image_list[i].append(image_path)
    """
    121200
    ['child']
    1683
    ['construction_worker']
    10465
    ['personal_mobility']
    1828
    ['police_officer']
    368
    ['wheelchair', 'stroller']
    326
    """
    pre_train = []
    pre_test = []

    clients_train = [[], [], [], [], []]
    clients_test = [[], [], [], [], []]

    # 判断，其它标签数<2的放训练集，其余放测试集-------------------
    # base
    for img in base_image_list:
        # 修改图片路径为实际使用路径
        img_vehicle = img.replace('samples', 'samples-vehicle')

        # label_txt = img.replace('jpg', 'txt').replace('samples', 'txt-annotations')
        label_txt_temp = img.replace('jpg', 'txt').replace('samples', 'txt-annotations-vehicle-temp')
        # print(label_txt)
        with open(label_txt_temp, 'r') as f:
            list1 = f.readlines()
            # print(list1)

        num = 0
        for i in range(len(list1)):
            t = list1[i].split(' ')
            if int(t[0]) != 0:
                num += 1
        if num < 1:
            pre_train.append(img_vehicle)
        else:
            pre_test.append(img_vehicle)

    # client
    for ci in range(5):
        for img in clients_image_list[ci]:
            # 修改图片路径为实际使用路径
            img_vehicle = img.replace('samples', 'samples-vehicle')

            # label_txt = img.replace('jpg', 'txt').replace('samples', 'txt-annotations')
            label_txt_temp = img.replace('jpg', 'txt').replace('samples', 'txt-annotations-vehicle-temp')
            # print(label_txt)
            with open(label_txt_temp, 'r') as f:
                list1 = f.readlines()
                # print(list1)
            # print('# ', ci)
            num = 0
            for k in range(len(list1)):
                t = list1[k].split(' ')
                if int(t[0]) != ci + 1:
                    # print(t[0], ci+1)
                    num += 1
            # print(ci, num)
            if num < 1:  # 没其它标签，即单标签数据
                # print(num, list1)
                clients_train[ci].append(img_vehicle)
            else:
                clients_test[ci].append(img_vehicle)
    # ===============================================================================
    temp_train = [[], [], [], [], []]
    temp_test = [[], [], [], [], []]

    print('处理前')
    print('len(pre_train)', len(pre_train))
    print('len(pre_test)', len(pre_test))
    for i in range(5):
        print(f'len(clients_train[{i}])', len(clients_train[i]))
        print(f'len(clients_test[{i}])', len(clients_test[i]))

    # 还是只分配0~1的数据，其余数据不适合做测试集----------------------
    # 打乱截取
    # pre_train = pre_train[:5000]  # 不截取
    for i in range(3):
        set_seed(seed)
        np.random.shuffle(clients_train[i])  # 先打乱再截取
        clients_train[i] = clients_train[i][:1800]

    set_seed(seed)
    np.random.shuffle(clients_train[4])
    clients_train[4] = clients_train[4][:600]

    # base相同分割
    set_seed(seed)
    np.random.shuffle(pre_train)  # 先打乱再截取
    ratio = 0.8
    num_file = len(pre_train)
    num_train = int(ratio * num_file)
    pre_test = pre_train[num_train:]
    pre_train = pre_train[:num_train]

    # 每600分割成480:120

    # 1800--------------------------------
    for i in range(3):
        temp_test[i] = np.hstack((clients_train[i][480:600], clients_train[i][1080:1200], clients_train[i][1680:1800]))
        temp_train[i] = np.hstack((clients_train[i][0:480], clients_train[i][600:1080], clients_train[i][1200:1680]))

    print('1800处理后')
    print('len(pre_train)', len(pre_train))
    print('len(pre_test)', len(pre_test))
    for i in range(3):
        print(f'len(clients_train[{i}])', len(temp_train[i]))
        print(f'len(clients_test[{i}])', len(temp_test[i]))

    # 写入分割数据
    path_name = 'vehicle_3_1800'
    path = f'data/{path_name}'
    # 写文件之前最好先清空一下
    clear_all(path_name=path_name)
    mkdir(path)
    # pre-train
    mkdir(f'data/{path_name}/base')
    write_txt(pre_train,
              image_txt_path=f'data/{path_name}/base/train_image.txt',
              label_txt_path=f'data/{path_name}/base/train_label.txt')
    write_txt(pre_test,
              image_txt_path=f'data/{path_name}/base/test_image.txt',
              label_txt_path=f'data/{path_name}/base/test_label.txt')

    # novel-train
    for i in range(3):
        mkdir(f'data/{path_name}/client_%d' % i)
        write_txt(temp_train[i],
                  image_txt_path=f'data/{path_name}/client_%d/train_image.txt' % i,
                  label_txt_path=f'data/{path_name}/client_%d/train_label.txt' % i)
        write_txt(temp_test[i],
                  image_txt_path=f'data/{path_name}/client_%d/test_image.txt' % i,
                  label_txt_path=f'data/{path_name}/client_%d/test_label.txt' % i)
    # ------------------------------------------

    # 1200--------------------------------
    for i in range(3):
        temp_test[i] = np.hstack((clients_train[i][480:600], clients_train[i][1080:1200]))
        temp_train[i] = np.hstack((clients_train[i][0:480], clients_train[i][600:1080]))

    print('1200处理后')
    print('len(pre_train)', len(pre_train))
    print('len(pre_test)', len(pre_test))
    for i in range(3):
        print(f'len(clients_train[{i}])', len(temp_train[i]))
        print(f'len(clients_test[{i}])', len(temp_test[i]))

    # 写入分割数据
    path_name = 'vehicle_3_1200'
    path = f'data/{path_name}'
    # 写文件之前最好先清空一下
    clear_all(path_name=path_name)
    mkdir(path)
    # pre-train
    mkdir(f'data/{path_name}/base')
    write_txt(pre_train,
              image_txt_path=f'data/{path_name}/base/train_image.txt',
              label_txt_path=f'data/{path_name}/base/train_label.txt')
    write_txt(pre_test,
              image_txt_path=f'data/{path_name}/base/test_image.txt',
              label_txt_path=f'data/{path_name}/base/test_label.txt')

    # novel-train
    for i in range(3):
        mkdir(f'data/{path_name}/client_%d' % i)
        write_txt(temp_train[i],
                  image_txt_path=f'data/{path_name}/client_%d/train_image.txt' % i,
                  label_txt_path=f'data/{path_name}/client_%d/train_label.txt' % i)
        write_txt(temp_test[i],
                  image_txt_path=f'data/{path_name}/client_%d/test_image.txt' % i,
                  label_txt_path=f'data/{path_name}/client_%d/test_label.txt' % i)
    # ------------------------------------------

    # 600--------------------------------
    for i in range(3):
        temp_test[i] = clients_train[i][480:600]
        temp_train[i] = clients_train[i][0:480]

    print('600处理后')
    print('len(pre_train)', len(pre_train))
    print('len(pre_test)', len(pre_test))
    for i in range(3):
        print(f'len(clients_train[{i}])', len(temp_train[i]))
        print(f'len(clients_test[{i}])', len(temp_test[i]))

    # 写入分割数据
    path_name = 'vehicle_3_600'
    path = f'data/{path_name}'
    # 写文件之前最好先清空一下
    clear_all(path_name=path_name)
    mkdir(path)
    # pre-train
    mkdir(f'data/{path_name}/base')
    write_txt(pre_train,
              image_txt_path=f'data/{path_name}/base/train_image.txt',
              label_txt_path=f'data/{path_name}/base/train_label.txt')
    write_txt(pre_test,
              image_txt_path=f'data/{path_name}/base/test_image.txt',
              label_txt_path=f'data/{path_name}/base/test_label.txt')

    # novel-train
    for i in range(3):
        mkdir(f'data/{path_name}/client_%d' % i)
        write_txt(temp_train[i],
                  image_txt_path=f'data/{path_name}/client_%d/train_image.txt' % i,
                  label_txt_path=f'data/{path_name}/client_%d/train_label.txt' % i)
        write_txt(temp_test[i],
                  image_txt_path=f'data/{path_name}/client_%d/test_image.txt' % i,
                  label_txt_path=f'data/{path_name}/client_%d/test_label.txt' % i)
    # ------------------------------------------

    # 4client 600--------------------------------
    for i in range(3):
        temp_test[i] = clients_train[i][480:600]
        temp_train[i] = clients_train[i][0:480]
    temp_test[3] = clients_train[4][480:600]  # client_4数据放到client_3文件夹里
    temp_train[3] = clients_train[4][0:480]

    print('3client 600处理后')
    print('len(pre_train)', len(pre_train))
    print('len(pre_test)', len(pre_test))
    for i in range(4):
        print(f'len(clients_train[{i}])', len(temp_train[i]))
        print(f'len(clients_test[{i}])', len(temp_test[i]))

    # 写入分割数据
    path_name = 'vehicle_4_600'
    path = f'data/{path_name}'
    # 写文件之前最好先清空一下
    clear_all(path_name=path_name)
    mkdir(path)
    # pre-train
    mkdir(f'data/{path_name}/base')
    write_txt(pre_train,
              image_txt_path=f'data/{path_name}/base/train_image.txt',
              label_txt_path=f'data/{path_name}/base/train_label.txt')
    write_txt(pre_test,
              image_txt_path=f'data/{path_name}/base/test_image.txt',
              label_txt_path=f'data/{path_name}/base/test_label.txt')

    # novel-train
    for i in range(4):
        mkdir(f'data/{path_name}/client_%d' % i)
        write_txt(temp_train[i],
                  image_txt_path=f'data/{path_name}/client_%d/train_image.txt' % i,
                  label_txt_path=f'data/{path_name}/client_%d/train_label.txt' % i)
        write_txt(temp_test[i],
                  image_txt_path=f'data/{path_name}/client_%d/test_image.txt' % i,
                  label_txt_path=f'data/{path_name}/client_%d/test_label.txt' % i)
    # ------------------------------------------


def write_txt(file_list,
              image_txt_path='data/vehicle/base/train_image.txt',
              label_txt_path='data/vehicle/base/train_image.txt'):
    """
    写 train_image.txt train_label test_image test_label
    image_path
    """
    with open(image_txt_path, "a") as f:
        for file in file_list:
            info = '%s\n' % file
            f.write(info)
    with open(label_txt_path, "a") as f:
        for file in file_list:
            info = '%s\n' % (file.replace('jpg', 'txt').replace('samples', 'txt-annotations'))
            f.write(info)


# 3 将配置文件写好：dogs.data等
def write_data(seed,
               dataroot='./../../data/nuimages-v1.0'
               ):
    # 分割、写入
    split(dataroot=dataroot, seed=seed)


if __name__ == '__main__':
    write_data(dataroot='./../../data/nuimages-v1.0',
               seed=1)
