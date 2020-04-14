import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.dataset import *
from model.res18 import res18
from utils.alphabets import Alphabets
from utils.tensor_utils import *
from utils.train_utils import *
from loss.center_loss import *
#import configs as config
import configs as config
from tqdm import *

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
# training parameters
# parser.add_argument("--image_size", type=int, default=256,help="training patch size")
parser.add_argument("--data_dir", type=str, default="/data/captcha/330/train",help="data dir location")
parser.add_argument("--batch_size", type=int, default=32,help="batch size")
parser.add_argument("--epochs", type=int, default=300,help="number of epochs")
parser.add_argument("--save_per_epoch", type=int, default=config.save_per_epoch,help="number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--steps_show", type=int, default=100,help="steps per epoch")
parser.add_argument("--scheduler_step", type=int, default=10,help="scheduler_step for epoch")
parser.add_argument("--weight", type=str, default=None,help="weight file for restart")
parser.add_argument("--output_path", type=str, default="checkpoint",help="checkpoint dir")
parser.add_argument("--devices", type=str, default="cuda",help="device description")
parser.add_argument("--image_channels", type=int, default=3,help="batch image_channels")
parser.add_argument("--resume_model", type=str, default=config.resume_model, help="resume model path")

args = parser.parse_args()
print(args)

def save_model_as(model, model_name):
    ckpt_name = '/'+model_name
    path = args.output_path
    if not os.path.exists(path):
        os.mkdir(path)
    path_final = path + ckpt_name
    print('Saving checkpoint to: {}\n'.format(path_final))
    torch.save(model.state_dict(), path_final)

def save_model(model,epoch):
    '''save model for eval'''
    ckpt_name = '/captcha_epoch_{}.pth'.format(epoch)
    path = args.output_path
    if not os.path.exists(path):
        os.mkdir(path)
    path_final = path + ckpt_name
    print('Saving checkpoint to: {}\n'.format(path_final))
    torch.save(model.state_dict(), path_final)

def resume_model(model, model_path):
    print("Resume model from {}".format(model_path))
    model.load_state_dict(torch.load(model_path))

def model_to_device(model):
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def tensor_to_device(tensor):
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")
    return tensor.to(device)

# prepare the data
test_dataset = DatasetFromFolder(config.test_folder)
test_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=1, shuffle=False, collate_fn=default_collate_fn)
test_loader_batch = DataLoader(dataset=test_dataset, num_workers=4, batch_size=1, shuffle=True, collate_fn=default_collate_fn)
print("test folder: {}".format(config.test_folder))

# prepare the network
net =res18(len(config.alphabets))

def eval_batch(net,loader):
    net.eval()
    net = model_to_device(net)
    correct = 0
    total_count=0
    for index, (data, label, label_length) in enumerate(loader):
        batch_size = data.size()[0]
        data = Variable(tensor_to_device(data))
        label = Variable(tensor_to_device(label)).long()
        label1, label2, label3, label4 = label[:, 0], label[:, 1], label[:, 2], label[:, 3]
        y1,y2,y3,y4 = net(data)
        y1 = y1.topk(1, dim=1)[1].view(batch_size, 1)
        y2 = y2.topk(1, dim=1)[1].view(batch_size, 1)
        y3 = y3.topk(1, dim=1)[1].view(batch_size, 1)
        y4 = y4.topk(1, dim=1)[1].view(batch_size, 1)
        y = torch.cat((y1, y2, y3, y4), dim=1)
        diff = (y != label)
        diff = diff.sum(1)
        diff = (diff != 0)
        res = diff.sum(0).item()
        correct += (batch_size - res)
        total_count += batch_size
    return float(correct)/total_count

def eval():
    resume_model(net, args.resume_model)
    eval_acc = eval_batch(net, test_loader_batch)
    print("eval acc is {}".format(eval_acc))

if __name__ == "__main__":
    eval()