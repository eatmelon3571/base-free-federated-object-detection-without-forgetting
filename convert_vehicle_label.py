import numpy as np
import shutil
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

vehicle_label = {'car': 0,
                 'truck': 1,
                 'bicycle': 2,
                 'motorcycle': 3,
                 'trailer': 4,
                 'bus': 5}

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


def clear_dog(path_name='dog'):
    clear_path(f'data/{path_name}/base')
    for i in range(5):
        clear_path(f'data/{path_name}/client_{i}')


def split(dataroot='./../../data/nuimages-v1.0',
          version='v1.0-train'):
    base_image_list = []
    clients_image_list = [[], [], [], [], []]
    base_label_list = []
    clients_label_list = [[], [], [], [], []]
    temp = 0

    count = 0

    nuim = NuImages(dataroot=dataroot, version=version, verbose=True, lazy=True)
    for ann in nuim.object_ann:
        write_label = False
        # print(i)
        category_token = ann['category_token']

        # label------------------------
        xmin, ymin, xmax, ymax = ann['bbox']
        w0 = 1600
        h0 = 900
        x = (xmin + xmax) / (2 * w0)
        y = (ymin + ymax) / (2 * h0)
        w = (xmax - xmin) / w0
        h = (ymax - ymin) / h0
        box = [0, x, y, w, h]
        # ------------------------------

        sample_data_token = ann['sample_data_token']
        category = nuim.get('category', category_token)
        sample_data = nuim.get('sample_data', sample_data_token)

        image_path = os.path.join(dataroot, sample_data['filename'])  # 图片路径

        # print(image_path)
        if 'sample' in image_path:
            temp += 1

        # if 'sweeps/CAM_FRONT_LEFT/' not in image_path \
        #         and 'sweeps/CAM_FRONT_RIGHT/' not in image_path:
        #     # print('continue')
        #     continue
        # else:
        #     print('not continue')

        # print(sample_data['filename'])
        # print(category['name'])
        s = category['name'].split('.')
        # print(s)

        for h in base_vehicle:
            if h in s:
                base_image_list.append(image_path)
                base_label_list.append(box)
                write_label = True
                box[0] = vehicle_label[h]

        for i in range(5):
            for h in clients_vehicle[i]:
                if h in s:
                    clients_image_list[i].append(image_path)
                    clients_label_list[i].append(box)
                    write_label = True
                    box[0] = vehicle_label[h]

        # 写label
        if write_label:
            # 改为临时存用位置，用于分割时判断
            label_path = image_path.replace('jpg', 'txt').replace('samples', 'txt-annotations-vehicle-temp')
            # print(image_path, label_path, box)
            with open(label_path, "a") as f:  # w写   a追加写
                f.write('%d %s %s %s %s\n' % (int(box[0]), box[1], box[2], box[3], box[4]))

        # 写label
        if write_label:
            # 训练时使用的标签位置
            label_path = image_path.replace('jpg', 'txt').replace('samples', 'txt-annotations-vehicle')
            # print(image_path, label_path, box)
            with open(label_path, "a") as f:  # w写   a追加写
                # 标签直接定成0
                f.write('%d %s %s %s %s\n' % (0, box[1], box[2], box[3], box[4]))

        # 复制图片至目标位置 /samples/  ->  /samples-vehicle/
        if write_label:
            source = image_path
            target = source.replace('samples', 'samples-vehicle')
            if not os.path.exists(target):
                # print('not exist')
                shutil.copy(source, target)

        if write_label:
            count += 1
            if count % 1000 == 0:
                print(f'写入{count}条数据')


# 3 将配置文件写好：dogs.data等
def write_data(dataroot='./../../data/nuimages-v1.0'):

    # 写文件之前最好先清空一下
    clear_path(f'{dataroot}/txt-annotations-vehicle-temp')
    clear_path(f'{dataroot}/txt-annotations-vehicle')

    # 清空图片目录
    clear_path(f'{dataroot}/samples-vehicle')

    # 建标签目录
    mkdir(f'{dataroot}/txt-annotations-vehicle-temp')
    mkdir(f'{dataroot}/txt-annotations-vehicle-temp/CAM_BACK')
    mkdir(f'{dataroot}/txt-annotations-vehicle-temp/CAM_BACK_LEFT')
    mkdir(f'{dataroot}/txt-annotations-vehicle-temp/CAM_BACK_RIGHT')
    mkdir(f'{dataroot}/txt-annotations-vehicle-temp/CAM_FRONT')
    mkdir(f'{dataroot}/txt-annotations-vehicle-temp/CAM_FRONT_LEFT')
    mkdir(f'{dataroot}/txt-annotations-vehicle-temp/CAM_FRONT_RIGHT')

    mkdir(f'{dataroot}/txt-annotations-vehicle')
    mkdir(f'{dataroot}/txt-annotations-vehicle/CAM_BACK')
    mkdir(f'{dataroot}/txt-annotations-vehicle/CAM_BACK_LEFT')
    mkdir(f'{dataroot}/txt-annotations-vehicle/CAM_BACK_RIGHT')
    mkdir(f'{dataroot}/txt-annotations-vehicle/CAM_FRONT')
    mkdir(f'{dataroot}/txt-annotations-vehicle/CAM_FRONT_LEFT')
    mkdir(f'{dataroot}/txt-annotations-vehicle/CAM_FRONT_RIGHT')

    # 建图片目录
    mkdir(f'{dataroot}/samples-vehicle')
    mkdir(f'{dataroot}/samples-vehicle/CAM_BACK')
    mkdir(f'{dataroot}/samples-vehicle/CAM_BACK_LEFT')
    mkdir(f'{dataroot}/samples-vehicle/CAM_BACK_RIGHT')
    mkdir(f'{dataroot}/samples-vehicle/CAM_FRONT')
    mkdir(f'{dataroot}/samples-vehicle/CAM_FRONT_LEFT')
    mkdir(f'{dataroot}/samples-vehicle/CAM_FRONT_RIGHT')

    # 分割、写入
    split(dataroot, version='v1.0-train')
    split(dataroot, version='v1.0-val')


if __name__ == '__main__':
    write_data(dataroot='./../../data/nuimages-v1.0')
