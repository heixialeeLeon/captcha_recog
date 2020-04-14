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
from model.senet import senet
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
parser.add_argument("--batch_size", type=int, default=config.batch_size,help="batch size")
parser.add_argument("--epochs", type=int, default=300,help="number of epochs")
parser.add_argument("--save_per_epoch", type=int, default=config.save_per_epoch,help="number of epochs")
parser.add_argument("--lr", type=float, default=config.lr, help="learning rate")
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
train_dataset = DatasetFromFolder(config.train_folder, transform=train_transform)
test_dataset = DatasetFromFolder(config.test_folder)
train_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True, collate_fn=default_collate_fn)
test_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=1, shuffle=False, collate_fn=default_collate_fn)
test_loader_batch = DataLoader(dataset=test_dataset, num_workers=4, batch_size=16, shuffle=True, collate_fn=default_collate_fn)
print("train folder: {}".format(config.train_folder))
print("test folder: {}".format(config.test_folder))

# prepare the loss
criterion = nn.CrossEntropyLoss()
criterion_cent = CenterLoss(num_classes=62, feat_dim=62)

# prepare the network
net = res18(len(config.alphabets))
# net = senet(len(config.alphabets))

# prepare the optim
#optimizer = torch.optim.SGD(net.parameters(),args.lr)
#optimizer = torch.optim.SGD(parameters_settings, args.lr)
optimizer = torch.optim.Adam(net.parameters(), args.lr)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.step, gamma=0.1)

# # prepare the Alphabets
# alphabets = Alphabets(config.alphabets)
# # prepare the tensor process
# tensor_process = TensorProcess(alphabets)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_epoch(net, epoch):
    net = model_to_device(net)
    #warmup_scheduler = warmup_controller(optimizer)
    for e in range(epoch):
        #warmup_scheduler.step(epoch,optimizer)
        net.train()
        epoch_loss = 0.0
        correct_count = 0
        total_count = 0
        avg_loss = averager()
        # pbar = tqdm(total=len(train_loader))
        for index, (data, label, label_length) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = Variable(tensor_to_device(data))
            label = Variable(tensor_to_device(label)).long()
            label1, label2, label3, label4 = label[:,0],label[:,1],label[:,2],label[:,3]
            optimizer.zero_grad()
            y1,y2,y3,y4 = net(data)
            loss1 = criterion(y1, label1)
            loss2 = criterion(y2, label2)
            loss3 = criterion(y3, label3)
            loss4 = criterion(y4, label4)
            loss_cross = loss1 + loss2 + loss3+ loss4

            # loss_c1 = criterion_cent(y1, label1)
            # loss_c2 = criterion_cent(y2, label2)
            # loss_c3 = criterion_cent(y3, label3)
            # loss_c4 = criterion_cent(y4, label4)
            # loss_c = loss_c1 + loss_c2 + loss_c3 + loss_c4

            loss = 1*loss_cross
            loss.backward()
            epoch_loss += loss.item()
            avg_loss.add(loss / batch_size)
            optimizer.step()

            # if index % config.display_interval == 0:
            #     print("epoch:{} {}/{}  loss:{}".format(e, index, len(train_loader),avg_loss.val()))
            #     avg_loss.reset()
            y1 = y1.topk(1,dim=1)[1].view(batch_size,1)
            y2 = y2.topk(1, dim=1)[1].view(batch_size, 1)
            y3 = y3.topk(1, dim=1)[1].view(batch_size, 1)
            y4 = y4.topk(1, dim=1)[1].view(batch_size, 1)
            y = torch.cat((y1,y2,y3,y4),dim=1)
            # print(y.shape)
            # print(label.shape)
            diff = (y != label)
            diff = diff.sum(1)
            diff = (diff !=0)
            res = diff.sum(0).item()
            correct_count += (batch_size-res)
            total_count +=batch_size
            # pbar.update(1)
        # pbar.close()
        eval_acc = eval_batch(net,test_loader_batch)

        if index % 1 == 0:
            acc = float(correct_count)/total_count
            print("epoch: {}/{}, loss:{}, train_acc:{}, eval_acc:{},learning_rate: {}".format(e, epoch, epoch_loss, acc, eval_acc, get_learning_rate(optimizer)))
        scheduler.step()
        if e % args.save_per_epoch == 0 and e > 0 :
            save_model(net,e)

# def eval(net, epoch):
#     net.eval()
#     net = model_to_device(net)
#     correct = 0
#     for index, (data, label, label_length) in enumerate(test_loader):
#         data = Variable(tensor_to_device(data))
#         label = Variable(tensor_to_device(label)).int()
#         outputs = net(data)
#         predict = tensor_process.post_process(outputs)
#         gt = label.view(-1).detach().cpu().numpy()
#         gt = alphabets.encode(gt)
#         if predict == gt:
#             correct += 1
#     print("epoch: {}, eval acc: {}".format(epoch, float(correct)/len(test_loader)))

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

def main():
    if config.resume_model:
        resume_model(net, args.resume_model)
    train_epoch(net,config.epoch)

if __name__ == "__main__":
    main()