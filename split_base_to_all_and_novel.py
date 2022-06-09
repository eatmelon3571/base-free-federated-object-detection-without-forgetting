import numpy as np
import glob
import os


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


def form_to(from_file='data/dog/base/test_image.txt',
            to_file='data/all_dog/base/__test_image.txt'):
    with open(from_file, 'r') as f:
        list1 = f.readlines()

    with open(to_file, "a") as f:
        for info in list1:
            f.write(info)


def copy_train_test(from_file='data/dog/base',
                    to_file='data/all_dog/base'):
    form_to(from_file=f'{from_file}/train_image.txt',
            to_file=f'{to_file}/train_image.txt')
    form_to(from_file=f'{from_file}/train_label.txt',
            to_file=f'{to_file}/train_label.txt')
    form_to(from_file=f'{from_file}/test_image.txt',
            to_file=f'{to_file}/test_image.txt')
    form_to(from_file=f'{from_file}/test_label.txt',
            to_file=f'{to_file}/test_label.txt')


def copy_all():
    # --------------------------------------------------------------
    # all_vehicle_3_1800
    from_file = 'data/vehicle_3_1800'
    to_file = 'data/all_vehicle_3_1800'
    clear_path(f'{to_file}/base')
    mkdir(f'{to_file}/base')

    copy_train_test(from_file=f'{from_file}/base',
                    to_file=f'{to_file}/base')
    for i in range(3):
        copy_train_test(from_file=f'{from_file}/client_{i}',
                        to_file=f'{to_file}/base')

    # novel_vehicle_3_1800
    from_file = 'data/vehicle_3_1800'
    to_file = 'data/novel_vehicle_3_1800'
    clear_path(f'{to_file}/base')
    mkdir(f'{to_file}/base')

    for i in range(3):
        copy_train_test(from_file=f'{from_file}/client_{i}',
                        to_file=f'{to_file}/base')

    # --------------------------------------------------------------
    # all_vehicle_3_1200
    from_file = 'data/vehicle_3_1200'
    to_file = 'data/all_vehicle_3_1200'
    clear_path(f'{to_file}/base')
    mkdir(f'{to_file}/base')

    copy_train_test(from_file=f'{from_file}/base',
                    to_file=f'{to_file}/base')
    for i in range(3):
        copy_train_test(from_file=f'{from_file}/client_{i}',
                        to_file=f'{to_file}/base')

    # novel_vehicle_3_1200
    from_file = 'data/vehicle_3_1200'
    to_file = 'data/novel_vehicle_3_1200'
    clear_path(f'{to_file}/base')
    mkdir(f'{to_file}/base')

    for i in range(3):
        copy_train_test(from_file=f'{from_file}/client_{i}',
                        to_file=f'{to_file}/base')

    # --------------------------------------------------------------
    # all_vehicle_3_600
    from_file = 'data/vehicle_3_600'
    to_file = 'data/all_vehicle_3_600'
    clear_path(f'{to_file}/base')
    mkdir(f'{to_file}/base')

    copy_train_test(from_file=f'{from_file}/base',
                    to_file=f'{to_file}/base')
    for i in range(3):
        copy_train_test(from_file=f'{from_file}/client_{i}',
                        to_file=f'{to_file}/base')

    # novel_vehicle_3_600
    from_file = 'data/vehicle_3_600'
    to_file = 'data/novel_vehicle_3_600'
    clear_path(f'{to_file}/base')
    mkdir(f'{to_file}/base')

    for i in range(3):
        copy_train_test(from_file=f'{from_file}/client_{i}',
                        to_file=f'{to_file}/base')

    # --------------------------------------------------------------
    # all_vehicle_4_600
    from_file = 'data/vehicle_4_600'
    to_file = 'data/all_vehicle_4_600'
    clear_path(f'{to_file}/base')
    mkdir(f'{to_file}/base')

    copy_train_test(from_file=f'{from_file}/base',
                    to_file=f'{to_file}/base')
    for i in range(4):
        copy_train_test(from_file=f'{from_file}/client_{i}',
                        to_file=f'{to_file}/base')

    # novel_vehicle_4_600
    from_file = 'data/vehicle_4_600'
    to_file = 'data/novel_vehicle_4_600'
    clear_path(f'{to_file}/base')
    mkdir(f'{to_file}/base')

    for i in range(4):
        copy_train_test(from_file=f'{from_file}/client_{i}',
                        to_file=f'{to_file}/base')


if __name__ == '__main__':
    copy_all()
