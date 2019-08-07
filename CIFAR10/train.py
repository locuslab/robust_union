import sys
sys.path.append('./models/')
import torch
from models import PreActResNet18
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
sys.path.append('./utils/')
from core import *
from torch_backend import *
from cifar_funcs import *
import ipdb
import sys 
args = sys.argv

device_id = args[1]
device = torch.device("cuda:{0}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(device_id))

torch.cuda.device_count() 


epochs = 50
batch_size = 128
DATA_DIR = './data'
dataset = cifar10(DATA_DIR)

train_set = list(zip(transpose(normalise2(pad(dataset['train']['data'], 4))), dataset['train']['labels']))
test_set = list(zip(transpose(normalise2(dataset['test']['data'])), dataset['test']['labels']))
train_set_x = Transform(train_set, [Crop(32, 32), FlipLR()])
train_batches = Batches(train_set_x, batch_size, shuffle=True, set_random_choices=True, num_workers=2, gpu_id = torch.cuda.current_device())
test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())


model = PreActResNet18().cuda()
for m in model.children(): 
    if not isinstance(m, nn.BatchNorm2d):
        m.half()   
        
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

import time

# lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, 0.1, 0.005, 0])[0]
lr_schedule = lambda t: np.interp([t], [0, 10, epochs*2//5, epochs], [0, 0.05, 0.005, 0])[0]

try:
    train_loss, train_acc = epoch(test_batches, lr_schedule, model, 0, criterion, opt = None, device = device, stop = True)
except:
    a =1


attack_list = [pgd_l1_topk, pgd_all, pgd_all_old, pgd_all_out, pgd_all]#TRIPLE DOENST HAVE A ATTACK NAME ANYTHING WORKS
attack_name = ["pgd_l1_topk", "pgd_all", "pgd_all_old", "pgd_all_out", "pgd_triple"]
folder_name = ["L1_topk", "msd_topk", "msd_old_topk", "worst_topk", "triple_topk"]
choice = int(args[2])
attack = attack_list[choice]
print(attack_name[choice])

# model.load_state_dict(torch.load("RobustModels/{0}/lr1_topk_rand_iter_10.pt".format(folder_name[choice]), map_location = device))

for epoch_i in range(1,epochs+1):  
    start_time = time.time()
    lr = lr_schedule(epoch_i + (epoch_i+1)/len(train_batches))
    if choice == 4:
        train_loss, train_acc = epoch_triple_adv(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device)
    else:
        train_loss, train_acc = epoch_adversarial(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device)

    total_loss, total_acc   = epoch(test_batches, lr_schedule, model, epoch_i, criterion, opt = None, device = "cuda:1")
    total_loss, total_acc_1 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_l1_topk, criterion, opt = None, device = device, stop = True)
    total_loss, total_acc_2 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_l2, criterion, opt = None, device = device, stop = True)
    total_loss, total_acc_3 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_linf, criterion, opt = None, device = device, stop = True)
    print('Epoch: {7}, Clean Acc: {6:.4f} Train Acc: {5:.4f}, Test Acc 1: {4:.4f}, Test Acc 2: {3:.4f}, Test Acc inf: {2:.4f}, Time: {1:.1f}, lr: {0:.4f}'.format(lr, time.time()-start_time, total_acc_3, total_acc_2,total_acc_1,train_acc, total_acc, epoch_i))    
    if epoch_i %5 == 0:
        torch.save(model.state_dict(), "RobustModels/{0}/lr2_topk_20_iter_{1}.pt".format(folder_name[choice], str(epoch_i)))
