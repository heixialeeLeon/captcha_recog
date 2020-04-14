import Augmentor
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as  T
import torch as t
import shutil

def get_distortion_pipline_single_image(src_path, dst_path, num):
    temp_dir = '/data/temp/test'
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    p = Augmentor.Pipeline(src_path, output_directory=temp_dir)
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.sample(num)
    # file_name = os.listdir(src_path)[0]
    # for index, item in enumerate(os.listdir(temp_dir)):
    #     label_name = file_name.split('.')[0]
    #     dst_file = "{}_{}.png".format(label_name,index)
    #     dst_file = os.path.join(dst_path,dst_file)
    #     temp_file = os.path.join(temp_dir,item)
    #     shutil.copyfile(temp_file,dst_file)
    # os.rmdir(temp_dir)

def get_distortion_pipline(src_path, dst_path,num):
    p = Augmentor.Pipeline(src_path,dst_path)
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.sample(num)
    return p


def get_skew_tilt_pipline(path, num):
    p = Augmentor.Pipeline(path)
    # p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.skew_tilt(probability=0.5, magnitude=0.02)
    p.skew_left_right(probability=0.5, magnitude=0.02)
    p.skew_top_bottom(probability=0.5, magnitude=0.02)
    p.skew_corner(probability=0.5, magnitude=0.02)
    p.sample(num)
    return p


def get_rotate_pipline(path, num):
    p = Augmentor.Pipeline(path)
    # p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.rotate(probability=1, max_left_rotation=1, max_right_rotation=1)
    p.sample(num)
    return p


if __name__ == "__main__":
    # times = 2
    # src_path = "/home/peizhao/data/captcha/huahang/train/train_raw"
    # dst_path = "/home/peizhao/data/captcha/huahang/train/train_augmentor"
    # num = len(os.listdir(src_path)) * times
    # p = get_distortion_pipline(src_path, dst_path, num)
    # p = get_rotate_pipline(path, num)
    # p.process()
    # augTrainDataset = augCaptcha("./data/auged_train", train=True)
    # trainDataset = Captcha("./data/train/", train=True)
    # testDataset = Captcha("./data/test/", train=False)
    # augTrainDataLoader = DataLoader(augTrainDataset, batch_size=1,
    #                                 shuffle=True, num_workers=4)
    # trainDataLoader = DataLoader(trainDataset, batch_size=1,
    #                              shuffle=True, num_workers=4)
    # testDataLoader = DataLoader(testDataset, batch_size=1,
    #                             shuffle=True, num_workers=4)

    # for data, label, data1, label1 in augTrainDataLoader,trainDataLoader:
    #     print(data.size(), label, data1.size(), label1)
    times = 2
    src_path = "/home/peizhao/data/captcha/huahang/train/train_raw"
    p = Augmentor.Pipeline(src_path)
    num = len(os.listdir(src_path)) * times
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    # p.sample(num)
    g = p.keras_generator(batch_size=1)
