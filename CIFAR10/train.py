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
import argparse

# python3 train.py -gpu_id 0 -model 3 -batch_size 128 -lr_schedule 1
parser = argparse.ArgumentParser(description='Adversarial Training for CIFAR10', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-gpu_id", help="Id of GPU to be used", type=int, default = 0)
parser.add_argument("-model", help="Type of Adversarial Training: \n\t 0: l_inf \n\t 1: l_1 \n\t 2: l_2 \n\t 3: msd \n\t 4: triple \n\t 5: worst \n\t 6: vanilla", type=int, default = 3)
parser.add_argument("-batch_size", help = "Batch Size for Train Set (Default = 128)", type = int, default = 128)


params = parser.parse_args()
device_id = params.gpu_id

device = torch.device("cuda:{0}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(device_id))

torch.cuda.device_count() 
batch_size = params.batch_size
choice = params.model


epochs = 50
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

lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, 0.1, 0.005, 0])[0]
#For clearing pytorch cuda inconsistency
try:
    train_loss, train_acc = epoch(test_batches, lr_schedule, model, 0, criterion, opt = None, device = device, stop = True)
except:
    a =1

attack_list = [ pgd_linf ,  pgd_l1_topk,   pgd_l2 ,  msd_v0 ,  triple_adv ,  pgd_worst_dir, triple_adv]#TRIPLE, VANILLA DON'T HAVE A ATTACK NAME ANYTHING WORKS
attack_name = ["pgd_linf", "pgd_l1_topk", "pgd_l2", "msd_v0", "triple_adv", "pgd_worst_dir", "vanilla"]
folder_name = ["LINF", "L1", "L2", "MSD_V0", "TRIPLE", "WORST", "VANILLA"]

model_dir = "Final/{0}".format(folder_name[choice])
import os
if(not os.path.exists(model_dir)):
    os.makedirs(model_dir)

file = open("{0}/logs.txt".format(model_dir), "w")

def myprint(a):
    print(a)
    file.write(a)
    file.write("\n")

attack = attack_list[choice]
print(attack_name[choice])

for epoch_i in range(1,epochs+1):  
    start_time = time.time()
    lr = lr_schedule(epoch_i + (epoch_i+1)/len(train_batches))
    if choice == 6:
        train_loss, train_acc = epoch(train_batches, lr_schedule, model, epoch_i, criterion, opt = opt, device = device)
    elif choice == 4:
        train_loss, train_acc = triple_adv(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_2 = 0.3)
    elif choice == 3:
        train_loss, train_acc = epoch_adversarial(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_2 = 0.3)
    elif choice == 5:
        train_loss, train_acc = epoch_adversarial(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_2 = 0.3, alpha_l_inf = 0.005)
    else:
        train_loss, train_acc = epoch_adversarial(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device)

    total_loss, total_acc   = epoch(test_batches, lr_schedule, model, epoch_i, criterion, opt = None, device = "cuda:1")
    total_loss, total_acc_1 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_l1_topk, criterion, opt = None, device = device, stop = True)
    total_loss, total_acc_2 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_l2, criterion, opt = None, device = device, stop = True)
    total_loss, total_acc_3 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_linf, criterion, opt = None, device = device, stop = True)
    myprint('Epoch: {7}, Clean Acc: {6:.4f} Train Acc: {5:.4f}, Test Acc 1: {4:.4f}, Test Acc 2: {3:.4f}, Test Acc inf: {2:.4f}, Time: {1:.1f}, lr: {0:.4f}'.format(lr, time.time()-start_time, total_acc_3, total_acc_2,total_acc_1,train_acc, total_acc, epoch_i))    
    if epoch_i %5 == 0:
        torch.save(model.state_dict(), "{0}/iter_{1}.pt".format(model_dir, str(epoch_i)))
